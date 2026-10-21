"""GEDI L2A ground elevations as an additional independent height source for the
waterline DEM (PREREG Amendment 2; D-091). Builds a point table in the format of
scripts/a6_validate_dem.py, so that its ``blocksize`` and ``evaluate`` stages run on it
with ``--sources GEDI`` (same blocks, metrics and sectors as GLO-30; never pooled with
ICESat-2).

Data: Earth Engine ``LARSE/GEDI/GEDI02_A_002_MONTHLY`` (GEDI L2A V2 rasterised at 25 m,
one image per month and 6-degree UTM zone), project ee-rezwanahmed.

Stages
  pilot   One request type of ``export`` on a small box (``--box``) or on one region part
          (``--part``) for a few months, metered by ee.profilePrinting; prints the
          EECU-s per km² requested and the projected EECU-h of the whole export plan.
          Every request is logged; a part pilot's shots are kept as export's file.
  export  The plan (``plan``): for each monthly image, the 0.005-degree cells holding
          emerged-band cells (results/study_area/extents_v6_250m.npz) whose emergence
          date (A6 raster) is before the end of that month minus 30 days, so that no
          shot that could pass the dryness test is left out; grouped in 1-degree parts
          and clipped to the image's 6-degree zone. One single-pass reduceRegion per
          (image, part) returns the per-shot values as an aligned tuple list
          (``shots_lists``): elev_lowestmode, digital_elevation_model, quality_flag,
          degrade_flag, beam, solar_elevation, delta_time, and the lon/lat of the 25 m
          pixel centre (the collection has no lowest-mode location band). No filter is
          applied in Earth Engine except a valid elev_lowestmode, so every shot-selection
          step is counted here. A hard EECU-s budget covers all GEDI work (the log's
          total, pilots included). Writes data/interim/gedi/<image>_<part>.csv.gz.
  points  Shot selection of Amendment 2 (validation.gedi_steps: quality_flag = 1,
          degrade_flag = 0, BEAM0000/BEAM0001 before 2020-02-06 dropped,
          |elev_lowestmode - digital_elevation_model| <= 50 m), then a finite geoid
          and the dryness test (acquisition later than emergence + 30 days for every
          30 m DEM pixel within one pixel). Heights: h_orth = elev_lowestmode -
          N(EIGEN-6C4) (validation.EigenGeoid, D-078, as for ICESat-2). Then the DEM
          height, interpolation distance, slope, contour density and the five product
          heights at each shot, as in a6_validate_dem.prepare. ``night`` (solar
          elevation < 0) is kept for the Amendment 2 night-only sensitivity.

Outputs
  data/interim/gedi/*.csv.gz                          (git-ignored; raw shots)
  data/interim/validation/<tag>/GEDI_points.csv.gz    (git-ignored; the point table)
  results/inventory/gedi_interactive_log.jsonl        (every request: EECU-s, shots)
  results/inventory/gedi_export.json                  (regions, chunks, totals)
  results/validation/<tag>_gedi/prepare.json, GEDI_filter_cascade.csv

Usage
  python scripts/a6_gedi.py pilot [--box W,S,E,N] [--months 2020-06,2020-07,2020-08]
  python scripts/a6_gedi.py export --budget EECU_S [--workers N]
  python scripts/a6_gedi.py points <tag> --emergence DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, validation
from caspian.config import CRS_EQUAL_AREA, EE_PROJECT, INTERIM, PROCESSED, RESULTS

COLLECTION = "LARSE/GEDI/GEDI02_A_002_MONTHLY"
BANDS_FULL = ["elev_lowestmode", "digital_elevation_model", "quality_flag", "degrade_flag", "beam", "solar_elevation",
              "delta_time", "sensitivity", "elevation_bias_flag", "lat_highestreturn", "lon_highestreturn"]
# the bands Amendment 2's rules and the night sensitivity need; the location is the pixel centre
BANDS = ["elev_lowestmode", "digital_elevation_model", "quality_flag", "degrade_flag", "beam", "solar_elevation",
         "delta_time"]
LOCATION = "pixel"
TILE_SCALE = 1
SOURCE = "GEDI"
RAW_DIR = INTERIM / "gedi"
LOG = RESULTS / "inventory" / "gedi_interactive_log.jsonl"
STATE = RESULTS / "inventory" / "gedi_export.json"
HARD_BUDGET_EECU_S = 20 * 3600.0      # all GEDI work, pilot included (lead's instruction, 2026-10-21)
CELL_DEG = 0.005                       # region cells (about 0.39 x 0.56 km)
PART_DEG = 1.0                         # region parts: the cells inside one 1-degree box
ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _drv():
    spec = importlib.util.spec_from_file_location("a6_validate_dem", ROOT / "scripts" / "a6_validate_dem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spent_eecu_s() -> float:
    """EECU-s of every logged GEDI request so far."""
    if not LOG.exists():
        return 0.0
    tot = 0.0
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tot += float(json.loads(line).get("eecu_s") or 0)
    return tot


def _profile_eecu(text: str) -> float:
    """Total EECU-seconds of an ee.profilePrinting table (first column summed), as in
    scripts/a6_stage2_export.py."""
    tot = 0.0
    for line in text.splitlines()[1:]:
        m = re.match(r"\s*([\d.]+)\s", line)
        if m:
            tot += float(m.group(1))
    return tot


def shots_fc(image_id: str, rects, bands=None, location: str | None = None, tile_scale: int | None = None):
    """Per-shot values of one monthly image inside a list of [w, s, e, n] rectangles: the
    25 m pixels with a valid elev_lowestmode, sampled on the image's own grid. With
    ``location`` "pixel", the shot location is the 25 m pixel centre (ee.Image.pixelLonLat,
    computed, not read) as ``lon``/``lat``."""
    import ee

    bands = BANDS if bands is None else bands
    location = LOCATION if location is None else location
    src = ee.Image(image_id)
    img = src.select(bands)
    if location == "pixel":
        img = img.addBands(ee.Image.pixelLonLat().rename(["lon", "lat"]))
    region = ee.Geometry.MultiPolygon([ee.Geometry.Rectangle(r).coordinates() for r in rects], None, False)
    img = img.updateMask(src.select("elev_lowestmode").mask())
    return img.sample(region=region, projection=src.select("elev_lowestmode").projection(), dropNulls=True,
                      geometries=False, tileScale=TILE_SCALE if tile_scale is None else tile_scale)


def shots_lists(image_id: str, rects):
    """The same shots as ``shots_fc`` as one reduceRegion with a tuple list (BANDS, then
    lon and lat of the 25 m pixel centre), aligned by pixel: the other bands are unmasked
    to -9999 so that no shot drops. A single pass: computeFeatures pages re-evaluated the
    collection and cost 7 to 20 times more in the pilots (D-091)."""
    import ee

    src = ee.Image(image_id)
    names = BANDS + ["lon", "lat"]
    img = src.select(BANDS).addBands(ee.Image.pixelLonLat().rename(["lon", "lat"]))
    img = img.select(names).unmask(-9999).updateMask(src.select("elev_lowestmode").mask())
    region = ee.Geometry.MultiPolygon([ee.Geometry.Rectangle(r).coordinates() for r in rects], None, False)
    return img.reduceRegion(ee.Reducer.toList(len(names)), region, crs=src.select("elev_lowestmode").projection(),
                            maxPixels=1e10, tileScale=TILE_SCALE), names


def fetch(image_id: str, rects, depth: int = 0) -> tuple[pd.DataFrame, float, float]:
    """(shots, EECU-s, seconds) of one request. A request that fails (for example a
    response too large) is split into two halves of its rectangles and retried, up to
    three levels; the EECU of the failed attempt is not reported by the server and is not
    counted."""
    import ee

    d, names = shots_lists(image_id, rects)
    buf = io.StringIO()
    t0 = time.time()
    try:
        with ee.profilePrinting(destination=buf):
            v = d.getInfo()
    except ee.EEException:
        if depth >= 3 or len(rects) < 2:
            raise
        h = len(rects) // 2
        a, ea, sa = fetch(image_id, rects[:h], depth + 1)
        b, eb, sb = fetch(image_id, rects[h:], depth + 1)
        return pd.concat([a, b], ignore_index=True), ea + eb, sa + sb
    arr = np.asarray(v.get("list") or [], float).reshape(-1, len(names))
    return pd.DataFrame(arr, columns=names), _profile_eecu(buf.getvalue()), time.time() - t0


def _log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _images(bbox, months=None) -> list[str]:
    import ee

    col = ee.ImageCollection(COLLECTION).filterBounds(ee.Geometry.Rectangle(list(bbox)))
    ids = col.aggregate_array("system:id").getInfo()
    if months:
        keep = {m.replace("-", "") for m in months}
        ids = [i for i in ids if i.split("/")[-1][:6] in keep]
    return sorted(ids)


_BAND: dict = {}


def _band_emergence():
    """lon, lat and emergence date (days since 1970) of every emerged-band cell centre
    (D-071 band on the 250 m zone-v6 extents; A6 emergence raster), cached."""
    if not _BAND:
        import pyproj

        ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
        band = (ex["high"] == 1) & (ex["low"] == 0)
        t = ex["transform"]
        r, c = np.nonzero(band)
        x, y = t[2] + (c + 0.5) * t[0], t[5] + (r + 0.5) * t[4]
        em = validation.TileSet(PROCESSED / "dem" / "main" / "emergence", "emergence")
        inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
        lon, lat = inv.transform(x, y)
        _BAND.update(lon=np.asarray(lon), lat=np.asarray(lat), e=em.sample(x, y, 1))
    return _BAND["lon"], _BAND["lat"], _BAND["e"]


def month_cut(image_id: str) -> float:
    """Latest emergence date (days since 1970) at which a shot of this monthly image can
    pass the dryness test: the end of the image's month minus the 30-day lag."""
    ym = image_id.split("/")[-1][:6]
    y, m = int(ym[:4]), int(ym[4:])
    end = np.datetime64(f"{y + (m == 12):04d}-{m % 12 + 1:02d}-01")
    return float((end - np.datetime64("1970-01-01")).astype(int) - design.EMERGENCE_LAG_DAYS)


def region_cells(e_max_days: float | None = None) -> list[list[float]]:
    """[w, s, e, n] of the CELL_DEG cells that hold an emerged-band cell (D-071 band,
    250 m) with a finite emergence date before ``e_max_days`` (default: the end of the
    last GEDI month, 2025-07, minus the 30-day lag). Only shots in such cells can be in
    the band and pass the dryness test, so no shot of the H2 population lies outside."""
    lon, lat, e = _band_emergence()
    if e_max_days is None:
        e_max_days = month_cut("202507")
    ok = np.isfinite(e) & (e < e_max_days)
    lon, lat = lon[ok], lat[ok]
    # a 250 m cell spans up to about 0.0035 x 0.0023 degrees: include every cell its corners touch
    keys = set()
    for dlo in (-0.002, 0.002):
        for dla in (-0.0012, 0.0012):
            keys |= set(zip(np.floor((lon + dlo) / CELL_DEG).astype(int).tolist(),
                            np.floor((lat + dla) / CELL_DEG).astype(int).tolist()))
    return [[round(i * CELL_DEG, 4), round(j * CELL_DEG, 4), round((i + 1) * CELL_DEG, 4), round((j + 1) * CELL_DEG, 4)]
            for i, j in sorted(keys)]


def _merge_rows(cells: list[list[float]]) -> list[list[float]]:
    """Merge cells of one row (same s, n) that touch into longer rectangles."""
    rows: dict[tuple, list] = {}
    for w, s, e, n in cells:
        rows.setdefault((s, n), []).append((w, e))
    out = []
    for (s, n), seg in sorted(rows.items()):
        seg.sort()
        cw, ce = seg[0]
        for w, e in seg[1:]:
            if abs(w - ce) < 1e-9:
                ce = e
            else:
                out.append([cw, s, ce, n])
                cw, ce = w, e
        out.append([cw, s, ce, n])
    return out


_PARTS: dict = {}


def parts(e_max_days: float | None = None) -> dict[str, list[list[float]]]:
    """Region rectangles grouped by PART_DEG box: part key -> merged rectangles (cached
    per emergence cut-off)."""
    if e_max_days not in _PARTS:
        grp: dict[str, list] = {}
        for w, s, e, n in region_cells(e_max_days):
            key = f"{int(np.floor(s / PART_DEG + 1e-9)):02d}N{int(np.floor(w / PART_DEG + 1e-9)):03d}E"
            grp.setdefault(key, []).append([w, s, e, n])
        _PARTS[e_max_days] = {k: _merge_rows(v) for k, v in sorted(grp.items())}
    return _PARTS[e_max_days]


def _area_km2(rects) -> float:
    return float(sum((e - w) * 111.32 * np.cos(np.radians((s + n) / 2)) * (n - s) * 110.57 for w, s, e, n in rects))


def plan() -> list[tuple[str, str, list, float]]:
    """Every export request: (image id, part key, rectangles clipped to the image's
    6-degree zone, area km²). Each monthly image gets the region of its own emergence
    cut-off (``month_cut``). The zone images' footprints overlap at the zone edges, so
    the rectangles are clipped to the zone; shots are also de-duplicated on
    (delta_time, beam) in the points stage."""
    full = parts()
    bb = [min(r[0] for v in full.values() for r in v), min(r[1] for v in full.values() for r in v),
          max(r[2] for v in full.values() for r in v), max(r[3] for v in full.values() for r in v)]
    jobs = []
    for iid in _images(bb):
        z0 = int(re.search(r"_(\d{3})E_", iid).group(1))       # west edge of the image's 6-degree zone
        for key, rects in parts(month_cut(iid)).items():
            sub = [[max(r[0], z0), r[1], min(r[2], z0 + 6), r[3]] for r in rects if r[2] > z0 and r[0] < z0 + 6]
            if sub:
                jobs.append((iid, key, sub, _area_km2(sub)))
    return jobs


def pilot(a) -> int:
    import ee

    ee.Initialize(project=EE_PROJECT)
    months = {m.replace("-", "") for m in a.months.split(",")}
    if spent_eecu_s() + 3600 > HARD_BUDGET_EECU_S:
        raise SystemExit(f"GEDI budget: {spent_eecu_s():.0f} EECU-s already spent")
    jobs = plan()
    if a.part:
        sel = [j for j in jobs if j[1] == a.part and j[0].split("/")[-1][:6] in months]
        label = f"part {a.part}"
    else:
        box = [float(v) for v in a.box.split(",")]
        sel = [(i, "box", [box], _area_km2([box])) for i in _images(box, sorted(months))]
        label = f"box {box}"
    tot_e = tot_a = 0.0
    tot_n = 0
    for iid, key, rects, area in sel:
        df, eecu, secs = fetch(iid, rects)
        rec = {"time": _now(), "stage": "pilot", "image": iid, "part": key, "n_rects": len(rects),
               "area_km2": round(area, 1), "eecu_s": round(eecu, 3), "seconds": round(secs, 1), "n_shots": int(len(df))}
        _log(rec)
        if a.part:          # the same request as export's: kept, so export does not repeat it
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            out = RAW_DIR / f"{iid.split('/')[-1]}_{key}.csv.gz"
            tmp = out.with_name(out.name + ".part")
            df.to_csv(tmp, index=False, float_format="%.17g", compression="gzip")
            tmp.replace(out)
        tot_e += eecu
        tot_a += area
        tot_n += len(df)
        print(rec, flush=True)
    rate = tot_e / tot_a
    total_area = sum(j[3] for j in jobs)
    out = {"pilot": label, "months": sorted(months), "n_requests": len(sel), "eecu_s": round(tot_e, 2),
           "n_shots": tot_n, "eecu_s_per_km2_request": rate, "plan_n_requests": len(jobs),
           "plan_area_km2_summed": round(total_area), "projected_eecu_h": round(rate * total_area / 3600, 2),
           "spent_all_gedi_eecu_s": round(spent_eecu_s(), 1), "hard_budget_eecu_h": HARD_BUDGET_EECU_S / 3600}
    print(json.dumps(out, indent=1))
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    st.setdefault("pilots", []).append({"time": _now(), **out})
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return 0


def export(a) -> int:
    import ee
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ee.Initialize(project=EE_PROJECT)
    budget = min(float(a.budget), HARD_BUDGET_EECU_S)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    plan_all = plan()
    jobs = [(iid, key, sub, RAW_DIR / f"{iid.split('/')[-1]}_{key}.csv.gz") for iid, key, sub, _ in plan_all]
    todo = [j for j in jobs if not j[3].exists()]
    lock = threading.Lock()
    stop = threading.Event()
    spent = [spent_eecu_s()]
    print(f"{len(todo)} of {len(jobs)} requests to run; spent so far {spent[0]:.0f} EECU-s of {budget:.0f}", flush=True)
    if spent[0] >= budget:
        raise SystemExit("budget already spent")

    def run(job):
        iid, key, sub, out = job
        if stop.is_set():
            return None
        for k in range(3):
            try:
                df, eecu, secs = fetch(iid, sub)
                break
            except Exception as ex:  # noqa: BLE001 - logged and retried
                with lock:
                    _log({"time": _now(), "stage": "export", "image": iid, "part": key, "error": str(ex)[:300],
                          "eecu_s": 0.0})
                if k == 2:
                    return None
                time.sleep(10 * (k + 1))
        tmp = out.with_name(out.name + ".part")
        df.to_csv(tmp, index=False, float_format="%.17g", compression="gzip")
        tmp.replace(out)
        with lock:
            _log({"time": _now(), "stage": "export", "image": iid, "part": key, "n_rects": len(sub),
                  "area_km2": round(_area_km2(sub), 1), "eecu_s": round(eecu, 3), "seconds": round(secs, 1),
                  "n_shots": int(len(df))})
            spent[0] += eecu
            if spent[0] >= budget:
                stop.set()
        return len(df)

    done = 0
    with ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(run, j) for j in todo]
        for f in as_completed(futs):
            if f.result() is not None:
                done += 1
                if done % 50 == 0:
                    print(f"{done}/{len(todo)} requests; {spent[0]:.0f} EECU-s", flush=True)
    missing = [j[3].name for j in jobs if not j[3].exists()]
    st["export"] = {"updated": _now(), "collection": COLLECTION, "bands": BANDS, "cell_deg": CELL_DEG,
                    "region_rule": "per monthly image: CELL_DEG cells holding emerged-band cells whose emergence "
                                   "date is before the end of the month minus 30 days (month_cut)",
                    "n_requests": len(jobs), "n_missing": len(missing), "missing": missing[:50],
                    "area_km2_summed": round(sum(j[3] for j in plan_all)), "stopped_by_budget": stop.is_set(),
                    "spent_eecu_s_total": round(spent_eecu_s(), 1), "budget_eecu_s": budget}
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in st["export"].items() if k != "missing"}, indent=1))
    return 0 if not stop.is_set() and not missing else 4


def beam_name(b) -> np.ndarray:
    """The integer ``beam`` band as the L2A beam names: 0 -> BEAM0000, 5 -> BEAM0101,
    11 -> BEAM1011."""
    return np.array([f"BEAM{int(v):04b}" if np.isfinite(v) else "" for v in np.asarray(b, float)], dtype=object)


def points(a) -> int:
    import pyproj

    drv = _drv()
    if a.emergence is None or not any(Path(a.emergence).glob("emergence_*.tif")):
        raise SystemExit("Amendment 2 needs the A6 emergence map: pass --emergence DIR")
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"no exported shots in {RAW_DIR}: run the export stage first")
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if st.get("export", {}).get("stopped_by_budget"):
        raise SystemExit("the export stopped at the budget; the shot set is incomplete")
    geoid = validation.EigenGeoid()
    frames, cascades, n_raw, n_dup = [], [], 0, 0
    for f in files:        # file by file (memory): fill values to NaN, de-duplicate, Amendment 2 steps
        raw = pd.read_csv(f).replace(-9999.0, np.nan)
        n_raw += len(raw)
        u = raw.drop_duplicates(subset=["delta_time", "beam"])
        n_dup += len(raw) - len(u)
        dd = pd.DataFrame({"lat": u["lat"].to_numpy(float), "lon": u["lon"].to_numpy(float),
                           "elev_lowestmode": u["elev_lowestmode"].to_numpy(float),
                           "digital_elevation_model": u["digital_elevation_model"].to_numpy(float),
                           "quality_flag": u["quality_flag"].to_numpy(float),
                           "degrade_flag": u["degrade_flag"].to_numpy(float), "beam": beam_name(u["beam"]),
                           "solar_elevation": u["solar_elevation"].to_numpy(float),
                           "delta_time": u["delta_time"].to_numpy(float), "file": f.name[:-7]})
        dd["time"] = validation.gedi_time(dd["delta_time"].fillna(-1e12))
        dd.loc[dd["delta_time"].isna(), "elev_lowestmode"] = np.nan      # no time: fails the height step
        dd["geoid_h"] = geoid(dd["lon"].to_numpy(float), dd["lat"].to_numpy(float))
        keep, casc = validation.cascade({**validation.gedi_steps(dd), "geoid": np.isfinite(dd["geoid_h"].to_numpy(float))},
                                        len(dd))
        cascades.append(casc.set_index("step"))
        frames.append(dd.loc[keep])
    casc = sum(cascades[1:], cascades[0]).reset_index()
    d = pd.concat(frames, ignore_index=True)
    del frames
    n_cross = len(d)
    d = d.drop_duplicates(subset=["delta_time", "beam"]).reset_index(drop=True)      # zone-edge overlaps
    n_cross -= len(d)
    casc = pd.concat([casc, pd.DataFrame([{"step": "unique_across_images", "n_remaining": len(d),
                                           "n_fail_alone": n_cross}])], ignore_index=True)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    x, y = fwd.transform(d["lon"].to_numpy(float), d["lat"].to_numpy(float))
    x, y = np.asarray(x), np.asarray(y)
    em = validation.TileSet(Path(a.emergence), "emergence")
    rad = validation.DRY_RADIUS_PX[SOURCE]
    t = d["time"].to_numpy()
    dry = np.zeros(len(d), bool)
    for k in range(0, len(d), drv.CHUNK):
        s = slice(k, k + drv.CHUNK)
        dry[s] = validation.dry_at_acquisition(t[s], em.sample_window(x[s], y[s], 1, rad).reshape(len(t[s]), -1))
    casc = pd.concat([casc, pd.DataFrame([{"step": "dry_at_acquisition", "n_remaining": int(dry.sum()),
                                           "n_fail_alone": int((~dry).sum())}])], ignore_index=True)
    d = d.loc[dry].assign(x=x[dry], y=y[dry]).reset_index(drop=True)
    d["h_orth"] = validation.orthometric(d["elev_lowestmode"], d["geoid_h"])     # D-078 chain
    d["night"] = (d["solar_elevation"] < 0).astype(int)                           # Amendment 2 sensitivity
    d["source"] = SOURCE
    names, sample = drv._product_sampler(a.products, a.no_products)
    dem = validation.TileSet(PROCESSED / "dem" / a.tag, "dem")
    cx, cy = drv._contour_points(a.tag, a.epochs.split(","))
    d = drv.attach_model_columns(d, dem, cx, cy, names, sample)
    d["time"] = pd.Series(d["time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    path = drv._cache(a.tag, SOURCE)
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False, float_format="%.10g")
    res = RESULTS / "validation" / f"{a.tag}_{SOURCE.lower()}"
    res.mkdir(parents=True, exist_ok=True)
    casc.to_csv(res / f"{SOURCE}_filter_cascade.csv", index=False)
    meta = {"created": _now(), "source": SOURCE, "collection": COLLECTION, "dem_tag": a.tag,
            "emergence_dir": str(a.emergence), "epochs": a.epochs.split(","), "products": names,
            "product_interpolation": "bilinear, as distributed (caspian.products.sample)" if names else None,
            "n_files": len(files), "n_shots_exported": n_raw, "n_duplicates_within_files": n_dup,
            "shot_location": "centre of the 25 m pixel the shot is rasterised to (ee.Image.pixelLonLat); the collection has no lowest-mode location band",
            "shot_rules": "Amendment 2: quality_flag = 1, degrade_flag = 0, BEAM0000/BEAM0001 before "
                          f"{str(validation.GEDI_BEAM_FIX)[:10]} dropped, |elev_lowestmode - digital_elevation_model| "
                          f"<= {validation.GROSS_ERROR_M:g} m; dry at acquisition for every 30 m DEM pixel within {rad} pixel",
            "datum": "h_orth = elev_lowestmode (WGS84 ellipsoid) - N(EIGEN-6C4) (validation.EigenGeoid, D-078)",
            "emergence_lag_days": int(design.EMERGENCE_LAG_DAYS), "n_points": len(d),
            "n_night": int(d["night"].sum()), "n_with_dem": int(np.isfinite(d["dem_h"]).sum())}
    (res / "prepare.json").write_text(drv._json(meta), encoding="utf-8")
    print(SOURCE, "->", len(d), "dry shots,", meta["n_night"], "at night,", meta["n_with_dem"], "with a DEM height",
          flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "export", "points"])
    ap.add_argument("tag", nargs="?", default="main")
    ap.add_argument("--box", default="53.0,46.0,53.25,46.25", help="pilot: W,S,E,N degrees")
    ap.add_argument("--part", default=None, help="pilot: a region part (e.g. 46N052E) instead of a box")
    ap.add_argument("--months", default="2020-06,2020-07,2020-08", help="pilot: months YYYY-MM")
    ap.add_argument("--budget", type=float, default=0.0, help="export: hard EECU-s budget (capped at 20 EECU-h)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--emergence", default=None)
    ap.add_argument("--products", default=None)
    ap.add_argument("--no-products", action="store_true")
    ap.add_argument("--epochs", default="E2,E3")
    a = ap.parse_args(argv)
    if a.stage == "export" and a.budget <= 0:
        ap.error("export needs --budget (EECU-s)")
    return {"pilot": pilot, "export": export, "points": points}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
