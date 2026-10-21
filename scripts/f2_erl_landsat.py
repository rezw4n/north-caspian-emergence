"""Landsat "then and now" composites for the ERL satellite figure (D-100, D-102): true-colour
median composites of Landsat Collection 2 Level-2 surface reflectance for one summer of the
1995-96 high stand (Landsat 5 TM) and one of the 2025-26 low stand (Landsat 8 and 9 OLI),
downsampled to 200 m on the analysis grid (Lambert azimuthal equal area, D-002).

Composite: every scene of the collection that meets the map box between 1 June and 30
September of the year; per scene, pixels flagged in QA_PIXEL as fill, dilated cloud, cloud
or cloud shadow (bits 0, 1, 3, 4), or saturated in any band (QA_RADSAT), are masked;
surface reflectance = DN x 0.0000275 - 0.2 (the Collection 2 scale factors); the median of
the unmasked values per band. Pixels with no clear value that summer take the median of
the adjacent summer (1995 for 1996, 2024 for 2025), and the share of such pixels is
recorded. Bands: red, green, blue (TM SR_B3, SR_B2, SR_B1; OLI SR_B4, SR_B3, SR_B2),
reflectance x 10,000 as int16, -32768 where no summer has a clear value.

Earth Engine (project ee-rezwanahmed), interactive computePixels only. Every request is
wrapped in ee.profilePrinting and its EECU-seconds appended to the ledger
results/inventory/erl_figures_ee_ledger.jsonl (which the month-to-date count reads).
Hard budget for this figure: 3 EECU-h (10,800 EECU-s) for pilot and fetch together.

Commands
  pilot   both composites over a 30 km box on the Volga delta front; writes
          results/inventory/erl_landsat_pilot.json with the EECU-s per composite and the
          projection to the full map box (scaled by pixel count). Stops if the projection
          plus the spend exceeds the budget.
  fetch   both composites over the map box in 4 x 4 tiles, metered request by request;
          stops before a tile that would break the budget at the pilot's rate. Writes
          data/raw/erl_landsat/<name>.npz and the acquisition record
          results/provenance/erl_landsat.json (sha256 of each file, scene ids, EECU).
Usage: python scripts/f2_erl_landsat.py pilot|fetch
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pyproj

from caspian import ee_io
from caspian.config import CRS_EQUAL_AREA, EE_PROJECT, RAW, RESULTS

INVENTORY = RESULTS / "inventory"
LEDGER = INVENTORY / "erl_figures_ee_ledger.jsonl"
PILOT = INVENTORY / "erl_landsat_pilot.json"
PROV = RESULTS / "provenance" / "erl_landsat.json"
OUT = RAW / "erl_landsat"
BUDGET_EECU_S = 3 * 3600.0
RES = 200.0
# Map box (equal-area metres): the hindcast grid's extent (results/hindcast/main/run.json),
# which covers the domain with a 0.1 degree margin, rounded out to the 200 m lattice.
MAP_BOX = (-297_400.0, -174_800.0, 377_600.0, 227_400.0)
PILOT_BOX = (-190_000.0, 10_000.0, -160_000.0, 40_000.0)
COMPOSITES = {
    "tm_1996": {"collections": ["LANDSAT/LT05/C02/T1_L2"], "year": 1996, "fill_year": 1995,
                "bands": ["SR_B3", "SR_B2", "SR_B1"], "sensor": "Landsat 5 TM"},
    "oli_2025": {"collections": ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"], "year": 2025,
                 "fill_year": 2024, "bands": ["SR_B4", "SR_B3", "SR_B2"], "sensor": "Landsat 8 and 9 OLI"},
}
RGB = ["red", "green", "blue"]
NODATA = -32768


def profile_eecu(text: str) -> float:
    tot = 0.0
    for line in text.splitlines()[1:]:
        m = re.match(r"\s*([\d.]+)\s", line)
        if m:
            tot += float(m.group(1))
    return tot


def metered(fn, *a, tries: int = 3, **k):
    """(result, EECU-s) of an Earth Engine call wrapped in ee.profilePrinting. The profile
    lookup sometimes fails ("Profile not found"); the call is then repeated, metered, up to
    ``tries`` times, and never run unmetered."""
    import ee
    for i in range(tries):
        buf = io.StringIO()
        try:
            with ee.profilePrinting(destination=buf):
                r = fn(*a, **k)
            return r, profile_eecu(buf.getvalue())
        except ee.EEException as e:
            if "profile" not in str(e).lower() or i == tries - 1:
                raise
            print(f"profile lookup failed ({e}); repeating the request", flush=True)
            time.sleep(5)


def spent_s() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(float(json.loads(ln).get("eecu_s") or 0.0) for ln in LEDGER.read_text(encoding="utf-8").splitlines()
               if ln.strip())


def ledger(step: str, eecu_s: float, **kw) -> None:
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "figure": "erl_fig2_then_now", "step": step,
                            "eecu_s": round(float(eecu_s), 3), **kw}) + "\n")


def lonlat_box(box):
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = box
    xs = np.r_[np.linspace(x0, x1, 21), np.full(21, x1), np.linspace(x1, x0, 21), np.full(21, x0)]
    ys = np.r_[np.full(21, y0), np.linspace(y0, y1, 21), np.full(21, y1), np.linspace(y1, y0, 21)]
    lon, lat = inv.transform(xs, ys)
    return [float(min(lon)), float(min(lat)), float(max(lon)), float(max(lat))]


def collection(spec: dict, year: int, region):
    import ee
    col = None
    for cid in spec["collections"]:
        c = ee.ImageCollection(cid).filterBounds(region).filterDate(f"{year}-06-01", f"{year}-10-01")
        col = c if col is None else col.merge(c)
    return col


def clear(img, bands):
    import ee
    qa = img.select("QA_PIXEL")
    bad = qa.bitwiseAnd(1 << 0).Or(qa.bitwiseAnd(1 << 1)).Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4))
    ok = bad.eq(0).And(img.select("QA_RADSAT").eq(0))
    return ee.Image(img.select(bands).multiply(0.0000275).add(-0.2).updateMask(ok).rename(RGB))


def composite(spec: dict, region):
    """(int16 image red/green/blue x 10,000 plus a 'filled' band, the two collections)."""
    import ee
    cols = {y: collection(spec, y, region) for y in (spec["year"], spec["fill_year"])}
    med = {y: ee.ImageCollection(c.map(lambda im: clear(im, spec["bands"]))).median() for y, c in cols.items()}
    main, fill = med[spec["year"]], med[spec["fill_year"]]
    filled = main.select("red").mask().Not().And(fill.select("red").mask()).rename("filled")
    img = main.unmask(fill).multiply(10000).round().clamp(-32767, 32767).toInt16()
    img = img.unmask(NODATA).addBands(filled.unmask(0).toInt16())
    return img, cols


def transform_of(box):
    x0, y0, x1, y1 = box
    return [RES, 0.0, x0, 0.0, -RES, y1], (int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES)))


def wkt() -> str:
    return pyproj.CRS(CRS_EQUAL_AREA).to_wkt(version="WKT1_GDAL")


def fetch_box(img, box, step: str, name: str, block: int | None = None) -> tuple[dict, float]:
    import ee
    tr, dims = transform_of(box)
    block = block or max(dims)
    out = {b: np.full((dims[1], dims[0]), NODATA if b != "filled" else 0, np.int16) for b in RGB + ["filled"]}
    tot = 0.0
    for c0, r0, w, h in ee_io.tiles(dims, block):
        sub = [tr[0], 0.0, tr[2] + tr[0] * c0, 0.0, tr[4], tr[5] + tr[4] * r0]
        spent = spent_s()
        if step == "fetch" and spent + tot > BUDGET_EECU_S:
            raise SystemExit(f"budget: {spent:.0f} EECU-s spent; stopping before tile {c0},{r0}")
        t0 = time.time()
        arr, eecu = metered(ee.data.computePixels, {"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                                    "grid": ee_io.grid(sub, (w, h), wkt())})
        tot += eecu
        ledger(step, eecu, composite=name, tile=[c0, r0, w, h], seconds=round(time.time() - t0, 1))
        for b in out:
            out[b][r0:r0 + h, c0:c0 + w] = arr[b]
        print(f"{step} {name} tile {c0},{r0} ({w}x{h}): {eecu:.1f} EECU-s", flush=True)
    return out, tot


def scene_ids(cols) -> tuple[dict, float, list]:
    """Scene ids per year (collection metadata). If the profile of this metadata-only
    request cannot be read after the retries, the ids are read unmetered and the ledger
    notes it (a metadata aggregation; its cost is not measured)."""
    import ee
    tot, ids, notes = 0.0, {}, []
    for y, c in cols.items():
        try:
            v, e = metered(c.aggregate_array("system:index").getInfo)
        except ee.EEException as err:
            if "profile" not in str(err).lower():
                raise
            v, e = c.aggregate_array("system:index").getInfo(), 0.0
            notes.append(f"{y}: scene-id request read unmetered (profile not found)")
        ids[str(y)] = sorted(v)
        tot += e
    return ids, tot, notes


def main(argv) -> int:
    import ee
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["pilot", "fetch"])
    a = ap.parse_args(argv)
    ee.Initialize(project=EE_PROJECT)
    if a.command == "pilot":
        spent0 = spent_s()
        lb = lonlat_box(PILOT_BOX)
        region = ee.Geometry.Rectangle(lb, None, False)
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "box_m": list(PILOT_BOX), "box_lonlat": lb, "res_m": RES,
               "composites": {}}
        _, dims_p = transform_of(PILOT_BOX)
        _, dims_f = transform_of(MAP_BOX)
        scale = (dims_f[0] * dims_f[1]) / (dims_p[0] * dims_p[1])
        proj_total = 0.0
        for name, spec in COMPOSITES.items():
            img, _ = composite(spec, region)
            arr, eecu = fetch_box(img, PILOT_BOX, "pilot", name)
            valid = float(np.mean(arr["red"] != NODATA))
            rec["composites"][name] = {"eecu_s": round(eecu, 2), "valid_share": round(valid, 4),
                                       "filled_share": round(float(np.mean(arr["filled"] == 1)), 4),
                                       "projected_full_eecu_s": round(eecu * scale, 1)}
            proj_total += eecu * scale
        rec["pixel_scale_factor"] = round(scale, 2)
        rec["projected_full_eecu_s"] = round(proj_total, 1)
        rec["spent_eecu_s"] = round(spent_s(), 1)
        rec["pilot_eecu_s"] = round(spent_s() - spent0, 1)
        rec["budget_eecu_s"] = BUDGET_EECU_S
        rec["go"] = bool(rec["spent_eecu_s"] + proj_total <= BUDGET_EECU_S)
        PILOT.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps(rec, indent=1))
        return 0 if rec["go"] else 2
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    if not pilot.get("go") or spent_s() + pilot["projected_full_eecu_s"] > BUDGET_EECU_S:
        raise SystemExit(f"pilot projects {pilot['projected_full_eecu_s']} EECU-s with {spent_s():.0f} spent: over the "
                         f"{BUDGET_EECU_S:.0f} EECU-s budget; not fetching")
    OUT.mkdir(parents=True, exist_ok=True)
    lb = lonlat_box(MAP_BOX)
    region = ee.Geometry.Rectangle(lb, None, False)
    prov = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "project": EE_PROJECT, "box_m": list(MAP_BOX),
            "box_lonlat": lb, "res_m": RES, "crs": CRS_EQUAL_AREA, "method": __doc__.split("Earth Engine (")[0].strip(),
            "files": {}}
    _, dims = transform_of(MAP_BOX)
    for name, spec in COMPOSITES.items():
        img, cols = composite(spec, region)
        ids, e_ids, notes = scene_ids(cols)
        ledger("fetch", e_ids, composite=name, what="scene ids", notes=notes)
        arr, eecu = fetch_box(img, MAP_BOX, "fetch", name, block=int(np.ceil(max(dims) / 4)))
        f = OUT / f"{name}.npz"
        tr, _ = transform_of(MAP_BOX)
        np.savez_compressed(f, transform=np.array(tr), **arr)
        prov["files"][name] = {"file": f"data/raw/erl_landsat/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                               "sensor": spec["sensor"], "collections": spec["collections"],
                               "window": f"{spec['year']}-06-01 to {spec['year']}-09-30",
                               "fill_window": f"{spec['fill_year']}-06-01 to {spec['fill_year']}-09-30",
                               "n_scenes": {y: len(v) for y, v in ids.items()}, "scene_ids": ids,
                               "valid_share": round(float(np.mean(arr["red"] != NODATA)), 4),
                               "filled_share": round(float(np.mean(arr["filled"] == 1)), 4),
                               "eecu_s": round(eecu + e_ids, 1)}
    prov["eecu_s_total_ledger"] = round(spent_s(), 1)
    PROV.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in prov.items() if k != "files"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
