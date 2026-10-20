"""A6 stage 2 (D-015, D-027): per-pixel water and valid counts per group, exported
as Earth Engine image assets per tile of the stage-2 cover, then fetched as arrays.

Group = (epoch, level bin, acquisition parity) for the main export, or (bin, parity,
setup class) for the screening-sensitivity export. For each group, the scenes in the
plan are:
1. masked as in stage 1 (caspian.masks, with the D-063 snow rules) and clipped to the
   tile's part of the cover;
2. classified with their acquisition's threshold (pipeline_ee.water_valid);
3. masked in the sectors where the plan excludes them (|setup| > tau);
4. mosaicked per acquisition and summed (pipeline_ee.group_counts).

Grid: the equal-area projection (D-002) at RES m, aligned to multiples of RES.

Usage:
  python scripts/a6_stage2_export.py upload <plan.csv> <tag>
  python scripts/a6_stage2_export.py submit <tag> [--tiles i,j,...] [--groups-from <plan.csv>]
  python scripts/a6_stage2_export.py status <tag>
  python scripts/a6_stage2_export.py fetch <tag>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import caspian  # noqa: F401
import ee
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

from caspian import ee_io, masks, pipeline_ee
from caspian.config import CRS_EQUAL_AREA, EE_PROJECT, INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER, sectors

ROOT = f"projects/{EE_PROJECT}/assets/caspian"
RES = 60
TILE_M = 100_000
CHUNK = 4000
COVER = RESULTS / "study_area" / "zone_v6_cover_buffered.geojson"
STATE = RESULTS / "inventory"
SENSORS = ["TM_L5", "ETM_L7", "OLI_L8", "OLI_L9", "MSI_S2"]
FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform
INV = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform


def ee_planar(geom, max_error: float = 1.0):
    """A shapely lon/lat (Multi)Polygon as an ee.Geometry with planar edges and an
    explicit error margin, which Earth Engine needs to reproject planar geometry."""
    def rings(poly):
        return [list(map(list, poly.exterior.coords))] + [list(map(list, r.coords)) for r in poly.interiors]
    def polys(g):  # intersections can return collections with line or point fragments
        if g.geom_type == "Polygon":
            return [g]
        if hasattr(g, "geoms"):
            return [q for h in g.geoms for q in polys(h)]
        return []
    ps = polys(geom)
    if len(ps) == 1:
        return ee.Geometry.Polygon(rings(ps[0]), None, False, max_error)
    return ee.Geometry.MultiPolygon([rings(g) for g in ps], None, False, max_error)


def group_of(r, kind):
    if kind == "main":
        return f"{r['epoch']}_b{int(r['bin']):03d}_p{int(r['parity'])}"
    raise ValueError(kind)


def cover_laea():
    g = json.loads(COVER.read_text(encoding="utf-8"))["features"][0]["geometry"]
    return transform(FWD, shape(g))


def block_tiles(n_per_sector: int):
    """D-064: a stratified random sample of training blocks (n_per_sector per sector,
    seed design.SEED), each clipped to the stage-2 cover, as export regions."""
    from caspian import design
    cov = cover_laea()
    b = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    b = b[b["split"] == "train"].copy()
    b["part"] = [cov.intersection(box(x, y, x + 25_000, y + 25_000)) for x, y in zip(b["x0"], b["y0"])]
    b["area"] = [g.area for g in b["part"]]
    b = b[b["area"] > 25e6]   # at least 25 km2 of the block inside the cover
    rng = np.random.default_rng(design.SEED)
    out = []
    for sec in sorted(b["sector"].unique()):
        g = b[b["sector"] == sec].sort_values(["i", "j"])
        pick = g.iloc[rng.permutation(len(g))[:n_per_sector]]
        out += [(int(r.i), int(r.j), r.part, SECTOR_ORDER.index(sec) + 1) for r in pick.itertuples()]
    return out


def tiles():
    cov = cover_laea()
    x0, y0, x1, y1 = cov.bounds
    out = []
    for i in range(math.floor(x0 / TILE_M), math.ceil(x1 / TILE_M)):
        for j in range(math.floor(y0 / TILE_M), math.ceil(y1 / TILE_M)):
            part = cov.intersection(box(i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M))
            if not part.is_empty and part.area > 1e6:
                out.append((i, j, part))
    return out


def sector_image():
    feats = [ee.Feature(ee_planar(poly), {"code": k + 1})
             for k, (name, (_, poly)) in enumerate((n, sectors()[n]) for n in SECTOR_ORDER)]
    return ee.Image(0).paint(ee.FeatureCollection(feats), "code").rename("sector")


def upload(plan_csv: str, tag: str) -> int:
    d = pd.read_csv(plan_csv)
    d["sectors"] = d["sectors"].fillna("").astype(str)
    cols = ["scene", "sensor", "acq", "threshold", "group", "sectors", "mask_mode"]
    st = {"tag": tag, "plan": str(plan_csv), "n_rows": len(d), "groups": sorted(d["group"].unique()), "chunks": []}
    for k in range(0, len(d), CHUNK):
        part = d.iloc[k:k + CHUNK]
        # table exports need a geometry; a dummy point carries the plan rows
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([0, 0]), {c: (float(v) if c == "threshold" else str(v)) for c, v in zip(cols, row)})
                                   for row in part[cols].itertuples(index=False)])
        aid = f"{ROOT}/stage2_plan_{tag}_{k // CHUNK:02d}"
        t = ee.batch.Export.table.toAsset(collection=fc, description=f"stage2_plan_{tag}_{k // CHUNK:02d}", assetId=aid)
        t.start()
        st["chunks"].append({"asset": aid, "task": t.id, "n": len(part)})
        print("submitted", aid, t.id)
    (STATE / f"stage2_{tag}.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return 0


def build_image(tag: str, groups, region_ll, sector_code=None):
    st = json.loads((STATE / f"stage2_{tag}.json").read_text(encoding="utf-8"))
    plan = ee.FeatureCollection([ee.FeatureCollection(c["asset"]) for c in st["chunks"]]).flatten()
    sec = sector_image()
    bands = []
    for g in groups:
        pg = plan.filter(ee.Filter.eq("group", g))
        if sector_code is not None:   # D-064: a block lies in one sector; only rows that include it
            pg = pg.filter(ee.Filter.stringContains("sectors", str(sector_code)))
        thr = ee.Dictionary.fromLists(pg.aggregate_array("scene"), pg.aggregate_array("threshold"))
        exc = ee.Dictionary.fromLists(pg.aggregate_array("scene"), pg.aggregate_array("sectors"))
        mode = ee.String(ee.Feature(pg.first()).get("mask_mode"))
        cols = []
        for s in SENSORS:
            ids = pg.filter(ee.Filter.eq("sensor", s)).aggregate_array("scene")
            if s == "MSI_S2":
                c = masks.sentinel2(region_ll, "2015-06-01", "2027-01-01", bands=masks.CLASSIFY_BANDS, ids=ids)
            else:
                c = masks.landsat(s, region_ll, "1984-01-01", "2027-01-01", bands=masks.CLASSIFY_BANDS, ids=ids)
            cols.append(c)
        col = cols[0]
        for c in cols[1:]:
            col = col.merge(c)

        def wv(im, thr=thr, exc=exc, mode=mode):
            sid = im.get("system:index")
            # merged collections prefix system:index; the plan holds the original index
            key = ee.String(sid).split("_").slice(-3).join("_")
            key = ee.String(ee.Algorithms.If(thr.contains(sid), sid, key))
            e = ee.String(exc.get(key, ""))
            codes = ee.List(ee.Algorithms.If(e.length().gt(0), e.split(",").map(lambda x: ee.Number.parse(x)), ee.List([])))
            # exclude: 0 in the listed sectors; include: 1 only in the listed sectors
            ok_ex = ee.Image(ee.Algorithms.If(codes.size().gt(0),
                                              sec.remap(codes, ee.List.repeat(0, codes.size()), 1), ee.Image(1)))
            ok_in = ee.Image(ee.Algorithms.If(codes.size().gt(0),
                                              sec.remap(codes, ee.List.repeat(1, codes.size()), 0), ee.Image(0)))
            ok = ee.Image(ee.Algorithms.If(mode.equals("include"), ok_in, ok_ex))
            return (pipeline_ee.water_valid(im.clip(region_ll), thr.get(key)).updateMask(ok)
                    .copyProperties(im, ["system:time_start", "acq"]))

        bands.append(pipeline_ee.group_counts(col.map(wv), g))
    return ee.Image.cat(bands)


def submit(tag: str, only=None, blocks: int = 0) -> int:
    st = json.loads((STATE / f"stage2_{tag}.json").read_text(encoding="utf-8"))
    ee.Initialize(project=EE_PROJECT)
    st.setdefault("tiles", {})
    plan_df = pd.read_csv(st["plan"]) if blocks else None
    for item in (block_tiles(blocks) if blocks else tiles()):
        i, j, part = item[:3]
        code = item[3] if blocks else None
        key = f"{i}_{j}"
        if only and key not in only:
            continue
        region_ll = ee_planar(transform(INV, part))
        x0, y0, x1, y1 = part.bounds
        x0, y1 = math.floor(x0 / RES) * RES, math.ceil(y1 / RES) * RES
        dims = (int(math.ceil((part.bounds[2] - x0) / RES)), int(math.ceil((y1 - part.bounds[1]) / RES)))
        tr = [RES, 0, x0, 0, -RES, y1]
        if blocks:
            has = plan_df["sectors"].fillna("").astype(str).str.contains(str(code), regex=False)
            groups = sorted(plan_df.loc[has, "group"].unique())
        else:
            groups = st["groups"]
        img = build_image(tag, groups, region_ll, code)
        name = f"stage2_{tag}_t{i}_{j}".replace("-", "m")
        t = ee.batch.Export.image.toAsset(image=img, description=name, assetId=f"{ROOT}/{name}", region=region_ll,
                                          crs=pipeline_ee.LAEA_WKT, crsTransform=tr, maxPixels=1e11)
        t.start()
        st["tiles"][key] = {"asset": f"{ROOT}/{name}", "task": t.id, "transform": tr, "dims": dims,
                            "area_km2": round(part.area / 1e6, 1), "groups": groups, "sector_code": code}
        print("submitted", name, t.id, st["tiles"][key]["area_km2"], "km2", flush=True)
        (STATE / f"stage2_{tag}.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return 0


def status(tag: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    st = json.loads((STATE / f"stage2_{tag}.json").read_text(encoding="utf-8"))
    tot = 0.0
    for k, t in list(st.get("tiles", {}).items()) + [(c["asset"], c) for c in st["chunks"]]:
        md = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{t['task']}")["metadata"]
        t.update(state=md.get("state"), eecu_s=md.get("batchEecuUsageSeconds"))
        tot += float(t["eecu_s"] or 0)
        print(k, t["state"], t["eecu_s"])
    print("total reported EECU-s:", round(tot, 1))
    (STATE / f"stage2_{tag}.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return 0


def fetch(tag: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    st = json.loads((STATE / f"stage2_{tag}.json").read_text(encoding="utf-8"))
    out = INTERIM / "stage2" / tag
    out.mkdir(parents=True, exist_ok=True)
    for k, t in st["tiles"].items():
        bands = [f"{p}_{g}" for g in t.get("groups", st["groups"]) for p in ("w", "v")]
        f = out / f"tile_{k}.npz"
        if f.exists():
            continue
        arr = ee_io.fetch(t["asset"], bands, t["transform"], t["dims"], pipeline_ee.LAEA_WKT, block=512)
        np.savez_compressed(f, transform=np.array(t["transform"]), **arr)
        print("fetched", k, f.stat().st_size // 1_000_000, "MB", flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--tiles", default=None)
    ap.add_argument("--blocks", type=int, default=0, help="D-064: sample this many training blocks per sector")
    a = ap.parse_args(argv)
    if a.cmd == "upload":
        ee.Initialize(project=EE_PROJECT)
        return upload(a.args[0], a.args[1])
    if a.cmd == "submit":
        return submit(a.args[0], a.tiles.split(",") if a.tiles else None, a.blocks)
    if a.cmd == "status":
        return status(a.args[0])
    if a.cmd == "fetch":
        return fetch(a.args[0])
    raise SystemExit(f"unknown command {a.cmd}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
