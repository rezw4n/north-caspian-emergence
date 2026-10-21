"""A9 generality test, the hindcast per arm (PREREG Amendment 16 items 7 and 8; D-094).

The Caspian driver (scripts/a7_hindcast.py: run, record_low_water, coast_band,
train_mask) is called unchanged on the basin's own grid and static layers:
- grid: 60 m on the basin's equal-area projection (caspian.generality.grid);
- domain: the arm (scripts/g1_static.py), rasterised by cell centre; one sector;
- zone for the outside states: the stage-2 cover in the arm plus the 2 km band around
  the shorelines of the near-maximum and near-minimum extents (D-083);
- open-water seed: the arm's largest open-water component inside the domain less its
  2 km rim (it may lie inside the stage-2 cover, which in these lakes takes in nearly
  all open water; the outside states then use its part outside the zone, or have no
  water state at all); land seed: cells more than 2 km from the near-maximum water
  (hindcast.beyond_land_state, D-067), outside the zone and rim; no beyond-domain state;
- salt ponds: unknown, observed (counts zeroed) and outside states alike;
- stage-2 counts: the arm's tiles, with counts zeroed outside the arm and in ponds;
- record-low envelope (D-082) from the arm's three lowest E3 bins;
- products: the five global grids, bilinear at cell centres, NaN outside the arm (so
  the flood fill stays in it); total variant + H(NAVD 88) - H(EGM2008) per cell (GSL);
  shape variant: the D-070 fit on training-block cells (25 km blocks of the basin grid,
  split 0.6/0.2/0.2 stratified by arm, seed 20261020);
- positive-control substitute (GSL): 3DEP averaged to 60 m, on NAVD 88 (no correction),
  hindcast only at the bins whose lower edge lies at or above the arm's hydro-flattened
  2016 lidar surface (its shape constant is fitted on those units only).

Outputs: results/generality/<basin>/<arm>/{hindcast,hindcast_dep3}/ (the Caspian tables)
Usage: python scripts/g1_hindcast.py <basin> [--arms north,south] [--products ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape as as_shape
from shapely.ops import unary_union

from caspian import blocking, design, hindcast, levels, products
from caspian import generality as G
from caspian import hindcast_tiled as ht
from caspian.config import INTERIM, RAW, RESULTS

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a7_hindcast", ROOT / "scripts" / "a7_hindcast.py")
A7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A7)

RES = 60.0
RIM_M = 2000          # D-083
FILES = {"gebco_2026": ("gebco_2026.nc", "elevation"), "gebco_2022": ("gebco_2022.nc", "elevation"),
         "etopo_2022": ("etopo_2022.nc", "elevation"), "srtm15p_v2.7": ("srtm15p_v2.7.nc", "elevation"),
         "etopo1": ("etopo1.tif", "bedrock")}
LIDAR_SURFACE_M = {"north": 1277.650024, "south": 1278.660034}   # Root (2023) metadata; 3DEP over the lake


def load_native(path: Path, layer: str) -> products.Native:
    if path.suffix == ".tif":
        import rasterio
        with rasterio.open(path) as r:
            band = list(r.descriptions).index(layer) + 1
            z = r.read(band, masked=True).astype(np.float32).filled(np.nan)
            t = r.transform
            lon = t.c + t.a * (np.arange(r.width) + 0.5)
            lat = t.f + t.e * (np.arange(r.height) + 0.5)
        return products._ascending(lon, lat, z)
    import xarray as xr
    with xr.open_dataset(path) as ds:
        da = ds[layer].transpose("lat", "lon")
        return products._ascending(ds["lon"].values, ds["lat"].values, da.values)


def burn(geom, transform, shape) -> np.ndarray:
    if geom.is_empty:
        return np.zeros(shape, bool)
    return rasterize([(geom, 1)], out_shape=shape, transform=Affine(*transform[:6]), fill=0, dtype="uint8") > 0


def blocks_table(basin, arms: dict, transform, shape) -> pd.DataFrame:
    """25 km blocks of the basin grid with the arm of their centre and the seeded split
    (PREREG section 6, unchanged rule; Amendment 16 item 7)."""
    a, _, c, _, e, f = transform
    bounds = (c, f + e * shape[0], c + a * shape[1], f)
    b = blocking.block_grid(bounds, design.BLOCK_SIZE_M)
    b = blocking.assign_sectors(b, arms)
    b["split"] = blocking.assign_splits(b)
    return b


def masked_stage2(basin, tag, arm, keep, transform) -> Path:
    """The arm's stage-2 tiles with every count zeroed where ``keep`` is False (outside
    the arm, salt ponds), written once under data/interim/generality/<basin>/."""
    src = INTERIM / "generality" / basin / f"stage2_{tag}"
    dst = INTERIM / "generality" / basin / f"stage2_{tag}_{arm}_masked"
    dst.mkdir(exist_ok=True)
    x0, y1 = transform[2], transform[5]
    n = 0
    for f in sorted(src.glob(f"tile_{arm}_*.npz")):
        out = dst / f.name
        n += 1
        if out.exists():
            continue
        with np.load(f) as z:
            tr = z["transform"]
            arrs = {k: z[k] for k in z.files if k != "transform"}
        h, w = next(iter(arrs.values())).shape
        c0, r0 = int(round((tr[2] - x0) / RES)), int(round((y1 - tr[5]) / RES))
        k = keep[r0:r0 + h, c0:c0 + w]
        if k.shape != (h, w):
            raise ValueError(f"{f.name}: tile outside the grid")
        for name, v in arrs.items():
            v = v.copy()
            v[~k] = 0
            arrs[name] = v
        np.savez_compressed(out, transform=tr, **arrs)
    if n == 0:
        raise SystemExit(f"no stage-2 tiles for arm {arm} in {src}")
    return dst


def nearest250(arr, tr250, transform, shape, fill):
    return A7.nearest(np.asarray(arr), tr250, transform, shape, fill)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("basin")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--products", default=",".join(G.GLOBAL_PRODUCTS))
    ap.add_argument("--no-dep3", action="store_true")
    a = ap.parse_args(argv)
    basin = a.basin
    B = G.BASINS[basin]
    crs = B["crs"]
    A7.CRS_EQUAL_AREA = crs                   # run.json records the basin grid's CRS
    transform, shape = G.grid(B["bbox"], crs, RES)
    d = RESULTS / "generality" / basin
    arms_gj = json.loads((d / "arms_laea.geojson").read_text(encoding="utf-8"))
    arms = {f["properties"]["arm"]: as_shape(f["geometry"]) for f in arms_gj["features"]
            if f["properties"]["role"] == "sector"}
    cover = unary_union([as_shape(f["geometry"]) for f in json.loads((d / "cover_laea.geojson").read_text())["features"]])
    ponds = unary_union([as_shape(f["geometry"]) for f in json.loads((d / "ponds_laea.geojson").read_text())["features"]])
    st = np.load(INTERIM / "generality" / basin / "static_250m.npz")
    tr250 = [float(v) for v in st["transform"]]
    blocks = blocks_table(basin, arms, transform, shape)
    blocks.to_csv(d / "blocks_25km.csv", index=False)
    pond = burn(ponds, transform, shape)
    corr_native, corr_rec = products.load_correction(RAW / "generality" / basin / "vdatum_navd88_minus_egm2008.npz")
    corr = products.to_grid(corr_native, transform, shape, "bilinear", crs=crs)
    natives = {p: load_native(RAW / "generality" / basin / "bathymetry" / FILES[p][0], FILES[p][1])
               for p in a.products.split(",")}
    dep3 = np.load(INTERIM / "generality" / basin / "dep3_60m.npy") if not a.no_dep3 else None
    rim_cells = int(round(RIM_M / RES))
    for arm in (a.arms.split(",") if a.arms else list(arms)):
        t0 = time.time()
        dom = burn(arms[arm], transform, shape)
        k = 1 + list(arms).index(arm)
        armlab = np.asarray(st["armlab"]) == k
        hi250 = np.asarray(st["high"]) & armlab
        lo250 = np.asarray(st["low"]) & armlab
        zone = burn(cover.intersection(arms[arm]), transform, shape) & dom
        zone |= A7.coast_band({"high": hi250.astype(np.int8), "low": lo250.astype(np.int8), "transform": tr250},
                              transform, shape) & dom
        rim_in = ht.eroded(dom, rim_cells)
        # the open-water seed may lie inside the stage-2 cover: in these lakes the 5 km cover and
        # its ring take in nearly all open water, which is then observed rather than labelled
        sea_seed = (nearest250(np.asarray(st["seed"]) & armlab, tr250, transform, shape, False) > 0) & rim_in
        land250 = hindcast.beyond_land_state(np.asarray(st["high"]), np.ones(hi250.shape, bool), 250.0, 2000.0)
        b = nearest250(np.nan_to_num(land250, nan=-1).astype(np.int8), tr250, transform, shape, -1)
        land_seed = (b == 0) & ~zone & rim_in
        if (sea_seed & ~zone).any():
            outside = ht.outside_state(zone, sea_seed & ~zone, land_seed, domain=dom, beyond_state=None,
                                       domain_margin=rim_cells)
        else:   # no open water outside the zone: only the land states of Amendment 6 item 2 exist
            land = ht.components_any(~zone & rim_in, [land_seed])[0]
            outside = np.full(shape, -1, np.int8)
            outside[land] = 0
        outside[pond] = -1
        sector = np.ones(shape, np.uint8)
        static = {"domain": dom, "sector": sector, "sea_seed": sea_seed, "outside": outside,
                  "counts": {"domain": int(dom.sum()), "zone": int(zone.sum()), "sea_seed": int(sea_seed.sum()),
                             "land_seed": int(land_seed.sum()), "ponds_unknown": int((pond & dom).sum()),
                             "outside_water": int((outside == 1).sum()), "outside_land": int((outside == 0).sum()),
                             "outside_unknown": int((outside == -1).sum())}}
        print(arm, "static", static["counts"], flush=True)
        s2 = masked_stage2(basin, a.tag, arm, dom & ~pond, transform)
        units = A7.stage2_index(s2)
        A7.ENVELOPE = A7.record_low_water(units, transform, shape)
        env = {"rule": "D-082: cells water at the record low (3 lowest E3 bins) are water at every level",
               "cells": int(A7.ENVELOPE.sum())}
        fit_region = dom & A7.train_mask(blocks, transform, shape)
        heights, flood = {}, {}
        for p, nat in natives.items():
            h = products.to_grid(nat, transform, shape, "bilinear", crs=crs)
            h[~dom] = np.nan
            heights[p] = products.Corrected(h, corr)
            flood[p] = ht.flood_level_map(heights[p], sea_seed)
        meta = {"basin": basin, "arm": arm, "tag": a.tag, "resampling": "bilinear", "envelope": env,
                "vertical_correction": {**corr_rec, "meaning": "H(NAVD 88) - H(EGM2008), m, added to product heights"},
                "level_datum": "bins on NAVD 88 minus %.1f m (Amendment 16 item 5)" % B["level_offset_m"],
                "seeds": {"sea": "largest open-water component of the arm (5 km, >= 80%), inside the domain less "
                                 "its rim; it may lie inside the stage-2 cover (observed there)",
                          "land": "observed cells > 2 km from near-maximum water (D-067 construction), outside zone and rim"},
                "ponds": "OpenStreetMap salt ponds + 120 m: unknown (counts zeroed, outside state unknown)"}
        out = d / arm / "hindcast"
        A7.run(s2, out, static, heights, transform, level_datum_offset=B["level_offset_m"],
               level_datum_source="NAVD 88 working datum (Amendment 16): bin centres + level_offset_m",
               shape_offsets="fit", fit_region=fit_region, flood_maps=flood, sector_names=[arm], blocks=blocks,
               block_m=design.BLOCK_SIZE_M, meta=meta)
        print(arm, "global products done", round(time.time() - t0), "s", flush=True)
        if dep3 is not None and basin == "great_salt_lake":
            h = dep3.copy()
            h[~dom] = np.nan
            bmin = int(np.ceil(round((LIDAR_SURFACE_M[arm] - B["level_offset_m"] - design.BIN_ORIGIN_M)
                                     / design.BIN_WIDTH_M, 6)))
            orig = A7.stage2_index
            A7.stage2_index = lambda src, _o=orig, _b=bmin: {k: v for k, v in _o(src).items() if k >= _b}
            try:
                fl = {"dep3": ht.flood_level_map(h, sea_seed)}
                meta3 = {**meta, "product": "USGS 3DEP 1/3 arc-second, mean over each 60 m cell (positive-control "
                                            "substitute, Amendment 16 item 7)", "vertical_correction": "none (NAVD 88)",
                         "bins": f">= {bmin} (lower edge >= hydro-flattened 2016 surface {LIDAR_SURFACE_M[arm]} m)"}
                A7.run(s2, d / arm / "hindcast_dep3", static, {"dep3": h}, transform,
                       level_datum_offset=B["level_offset_m"], level_datum_source=meta["level_datum"],
                       shape_offsets="fit", fit_region=fit_region, flood_maps=fl, sector_names=[arm], blocks=blocks,
                       block_m=design.BLOCK_SIZE_M, meta=meta3)
            finally:
                A7.stage2_index = orig
            print(arm, "dep3 done", round(time.time() - t0), "s", flush=True)
        del heights, flood
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
