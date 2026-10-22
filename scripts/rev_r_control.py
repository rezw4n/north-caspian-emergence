"""Revision R, A18 item 6: the "perfect but degraded" control.

The waterline DEM (main) is degraded as each product stores the seabed and audited
exactly as the products are (scripts/a7_hindcast.run: its own area constant,
total and shape variants, the same units, static layers, envelope and blocks).

Construction, on the 60 m hindcast grid unless stated:
1. Working-frame surface. The DEM aggregated to 60 m (a8_emergence.dem_to_grid). Cells
   without a DEM value are filled so that the surface is defined wherever a product is:
   LOW = lowest audited level - 1 m where the cell is water at the record low (record-low
   envelope) or outside-state water; HIGH = highest audited level + 1 m where it is land
   at the 1995-96 high stand (zone-v6 "high" extent) or outside-state land; elsewhere in
   the zone the nearest DEM value within 256 cells; NaN otherwise (unknown, as for a
   product's NaN).
2. Product frame: minus the per-cell correction N(EGM2008) - N(EIGEN-6C4) (the inverse of
   the total variant, caspian.products.Corrected).
3. Mean of the 60 m values in each native cell of the product (15", or 1' for ETOPO1).
4. Rounded to whole metres where the product stores whole metres (GEBCO_2026, GEBCO_2022,
   ETOPO1: every cell; ETOPO 2022: cells whose source is not FABDEM, source_id != 10;
   SRTM15+: cells whose stored value lies within 0.01 m of a whole metre). Skipped for the
   unrounded controls.
5. Sampled back to 60 m bilinearly (caspian.products.to_grid) and the correction
   added back (working frame).

Distinct surfaces (the four 15" products share one native grid and one correction):
  ctrl_r15_gebco   15", rounded everywhere          GEBCO_2026, GEBCO_2022
  ctrl_r15_srtm    15", rounded where SRTM15+ is    SRTM15+
  ctrl_r15_etopo22 15", rounded outside FABDEM      ETOPO 2022
  ctrl_u15         15", not rounded                 the four 15" products
  ctrl_r60_etopo1  1', rounded everywhere           ETOPO1
  ctrl_u60_etopo1  1', not rounded                  ETOPO1

Commands:
  build    the surfaces and their flood-level maps (data/interim/rev_r/control/, flood/)
  run      the audit (scripts/a7_hindcast.run) -> results/revision/r/control/hindcast/
           [--pilot-bins N: only the N highest bins, to time it; output .../pilot/]
  refit    sensitivity: each control's constant refitted on the units within the DEM's
           range -> results/revision/r/control/refit_dem_range.json (audited by
           scripts/rev_r_audit.py pass --surfaces controls, out control_refit)
  decide   excess and stop rules against results/hindcast/main ->
           results/revision/r/control/{levels.csv, excess.csv, stop_rule.json, summary.json}

Usage: python scripts/rev_r_control.py build|run|decide [--pilot-bins N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from caspian import h1, products
from caspian import hindcast_tiled as ht
from caspian import rev_r
from caspian.config import CRS_EQUAL_AREA, INTERIM, PROCESSED, RAW, RESULTS
from caspian.geography import DOMAIN

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a7 = _load("a7_hindcast")
a8 = _load("a8_emergence")

TAG = "main"
TILE = 2048
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
WORK = INTERIM / "rev_r" / "control"
FLOOD = INTERIM / "rev_r" / "flood"
OUT = RESULTS / "revision" / "r" / "control"
FILL_MARGIN_M = 1.0          # LOW / HIGH fills: 1 m beyond the audited level range
NEAREST_HALO = 256           # cells (15 km) for the nearest-DEM fill in the zone
SRTM_TOL_M = 0.01            # SRTM15+ "stores whole metres": within 1 cm of a whole metre
FABDEM_ID = 10               # ETOPO 2022 source_id of FABDEM (MacFerrin et al. 2025, Table 1)

# name: (native-grid product, rounding rule or None, products it stands for)
CONTROLS = {
    "ctrl_r15_gebco": ("gebco_2026", "all", ["gebco_2026", "gebco_2022"]),
    "ctrl_r15_srtm": ("srtm15p_v2.7", "srtm", ["srtm15p_v2.7"]),
    "ctrl_r15_etopo22": ("etopo_2022", "etopo22", ["etopo_2022"]),
    "ctrl_u15": ("gebco_2026", None, ["gebco_2026", "gebco_2022", "srtm15p_v2.7", "etopo_2022"]),
    "ctrl_r60_etopo1": ("etopo1", "all", ["etopo1"]),
    "ctrl_u60_etopo1": ("etopo1", None, ["etopo1"]),
}
ROUNDED_FOR = {"gebco_2026": "ctrl_r15_gebco", "gebco_2022": "ctrl_r15_gebco", "srtm15p_v2.7": "ctrl_r15_srtm",
               "etopo_2022": "ctrl_r15_etopo22", "etopo1": "ctrl_r60_etopo1"}
UNROUNDED_FOR = {"gebco_2026": "ctrl_u15", "gebco_2022": "ctrl_u15", "srtm15p_v2.7": "ctrl_u15",
                 "etopo_2022": "ctrl_u15", "etopo1": "ctrl_u60_etopo1"}
MISPREDICTING = ["gebco_2026", "gebco_2022", "etopo1", "srtm15p_v2.7"]
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]


def setup():
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    static = a7.production_static(transform, shape, TILE)
    units = a7.stage2_index(INTERIM / "stage2" / TAG)
    return transform, shape, static, units


def round_mask(product: str, rule, native: products.Native):
    if rule is None:
        return None
    if rule == "all":
        return np.ones(native.z.shape, bool)
    if rule == "srtm":
        return rev_r.whole_metre_mask(native.z, SRTM_TOL_M)
    if rule == "etopo22":
        with xr.open_dataset(RAW / "bathymetry" / products.PRODUCTS[product]["file"]) as ds:
            sid = ds["source_id"].transpose("lat", "lon").values
            lat, lon = ds["lat"].values, ds["lon"].values
        if lat[0] > lat[-1]:
            sid = sid[::-1]
        if lon[0] > lon[-1]:
            sid = sid[:, ::-1]
        if sid.shape != native.z.shape:
            raise ValueError("ETOPO 2022 source_id and elevation grids differ")
        return np.isfinite(sid) & (sid != FABDEM_ID)
    raise ValueError(rule)


def _rows(shape, block=256):
    for r0 in range(0, shape[0], block):
        yield r0, min(r0 + block, shape[0])


def base_surface(transform, shape, static, units, corr) -> tuple[np.ndarray, dict]:
    """Step 1: the working-frame 60 m surface (memory map), and its record."""
    WORK.mkdir(parents=True, exist_ok=True)
    dem_paths = sorted((PROCESSED / "dem" / TAG).glob("dem_*.tif"))
    levels = [a7._bin_level(b) for b in units]
    lo_v, hi_v = min(levels) - FILL_MARGIN_M, max(levels) + FILL_MARGIN_M
    key = hashlib.sha256(json.dumps({"dem": [p.name for p in dem_paths],
                                     "dem_bytes": sum(p.stat().st_size for p in dem_paths),
                                     "lo": lo_v, "hi": hi_v, "halo": NEAREST_HALO, "shape": list(shape)},
                                    sort_keys=True).encode()).hexdigest()[:12]
    path = WORK / f"base_{key}.npy"
    rec_path = path.with_suffix(".json")
    if path.exists() and rec_path.exists():
        return np.load(path, mmap_mode="r"), json.loads(rec_path.read_text(encoding="utf-8"))
    dem = a8.dem_to_grid(dem_paths, transform, shape)
    env = a7.record_low_water(units, transform, shape)
    ext = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    high = a7.nearest(np.asarray(ext["high"]).astype(np.int8), ext["transform"], transform, shape, -1)
    out = static["outside"]
    fin = np.isfinite(dem)
    low_m = ~fin & (env | (out == 1))
    high_m = ~fin & ~low_m & ((high == 0) | (out == 0))
    z = np.where(low_m, np.float32(lo_v), np.where(high_m, np.float32(hi_v), dem)).astype(np.float32)
    del dem, env, high
    near_m = static["zone"] & ~np.isfinite(z)
    z = rev_r.fill_nearest(z, near_m, tile=TILE, halo=NEAREST_HALO)
    rec = {"rule": "D-116 step 1", "low_value_m": lo_v, "high_value_m": hi_v,
           "cells": {"dem": int(fin.sum()), "low_fill": int(low_m.sum()), "high_fill": int(high_m.sum()),
                     "nearest_fill_needed": int(near_m.sum()),
                     "nearest_fill_left_nan": int((near_m & ~np.isfinite(z)).sum()),
                     "finite_total": int(np.isfinite(z).sum())}}
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, z)
    tmp.replace(path)
    rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return np.load(path, mmap_mode="r"), rec


def degraded(name, base, corr, transform, shape, log=print) -> tuple[np.ndarray, dict]:
    """Steps 2-5 for one control; float32 memory map of a cached .npy."""
    prod, rule, _ = CONTROLS[name]
    native = products.load_native(prod)
    rmask = round_mask(prod, rule, native)
    key = hashlib.sha256(json.dumps({"name": name, "base": Path(base.filename).name, "rule": rule,
                                     "srtm_tol": SRTM_TOL_M, "lon": [float(native.lon[0]), float(native.lon[-1]),
                                                                    len(native.lon)],
                                     "lat": [float(native.lat[0]), float(native.lat[-1]), len(native.lat)],
                                     "mask": None if rmask is None else int(rmask.sum())},
                                    sort_keys=True).encode()).hexdigest()[:12]
    path = WORK / f"{name}_{key}.npy"
    rec_path = path.with_suffix(".json")
    if path.exists() and rec_path.exists():
        return np.load(path, mmap_mode="r"), json.loads(rec_path.read_text(encoding="utf-8"))
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    xs = c + a * (np.arange(shape[1]) + 0.5)
    acc = rev_r.NativeMean(native.z.shape)
    for r0, r1 in _rows(shape):
        ys = f + e * (np.arange(r0, r1) + 0.5)
        lon, lat = inv.transform(*np.meshgrid(xs, ys))
        i, j = rev_r.native_cell_index(lon, lat, native.lon, native.lat)
        zp = np.asarray(base[r0:r1], np.float64) - np.asarray(corr[r0:r1], np.float64)
        acc.add(zp, i, j)
    zn = acc.mean()
    n_round = 0
    if rmask is not None:
        n_round = int((rmask & np.isfinite(zn)).sum())
        zn = rev_r.round_half_up(zn, rmask)
    nat = products.Native(native.lon, native.lat, zn.astype(np.float32))
    tmp = path.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
    products.to_grid(nat, transform, shape, "bilinear", out=mm)
    for r0, r1 in _rows(shape, 1024):
        mm[r0:r1] = np.asarray(mm[r0:r1], np.float32) + np.asarray(corr[r0:r1], np.float32)
    mm.flush()
    del mm
    tmp.replace(path)
    rec = {"control": name, "native_grid_of": prod, "rounding": rule or "none",
           "native_cells_with_value": int(np.isfinite(zn).sum()), "native_cells_rounded": n_round,
           "native_cells_round_mask": None if rmask is None else int(rmask.sum()),
           "native_shape": list(native.z.shape), "stands_for": CONTROLS[name][2]}
    rec_path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log(f"{name}: {rec}", flush=True)
    return np.load(path, mmap_mode="r"), rec


def build(transform, shape, static, units, log=print):
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    base, base_rec = base_surface(transform, shape, static, units, corr)
    log(f"base surface: {base_rec}", flush=True)
    surfaces, recs, flood = {}, {}, {}
    for name in CONTROLS:
        surfaces[name], recs[name] = degraded(name, base, corr, transform, shape, log)
        flood[name] = a7.flood_map_cached(name, surfaces[name], static["sea_seed"],
                                          {"heights": Path(surfaces[name].filename).name, "correction": "in surface"},
                                          TILE, cache=FLOOD)
        log(f"flood-level map {name}", flush=True)
    return surfaces, flood, {"base": base_rec, "controls": recs, "correction": corr_rec}


def cmd_run(a) -> int:
    t0 = time.time()
    transform, shape, static, units = setup()
    surfaces, flood, rec = build(transform, shape, static, units)
    for k in ("zone", "land_seed"):
        static.pop(k)
    blocks = pd.read_csv(a7.BLOCKS)
    fit_region = static["domain"] & a7.train_mask(blocks, transform, shape)
    stage2 = INTERIM / "stage2" / TAG
    a7.ENVELOPE = a7.record_low_water(units, transform, shape)
    out = OUT / "hindcast"
    if a.pilot_bins:
        keep = sorted(units)[-a.pilot_bins:]
        orig = a7.stage2_index
        a7.stage2_index = lambda src: {b: u for b, u in orig(src).items() if b in keep}
        out = OUT / "pilot"
    meta = {"tag": TAG, "variant": "A18 item 6 perfect-but-degraded control (D-116)", "resampling": "bilinear",
            "tile": TILE, "envelope": {"cells": int(a7.ENVELOPE.sum())}, "construction": rec,
            "pilot_bins": a.pilot_bins or None}
    a7.run(stage2, out, static, surfaces, transform, product_offsets={}, shape_offsets="fit",
           fit_region=fit_region, flood_maps=flood, blocks=blocks, tile=TILE, meta=meta)
    print(f"control audit written to {out} in {time.time() - t0:.0f} s", flush=True)
    return 0


def cmd_build(a) -> int:
    transform, shape, static, units = setup()
    _, _, rec = build(transform, shape, static, units)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "construction.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    return 0


def top_dem_bin() -> int:
    """The highest bin of the DEM epochs (E2, E3): above it the DEM holds no contour."""
    so = json.loads((RESULTS / "hindcast" / TAG / "shape_offsets.json").read_text(encoding="utf-8"))
    return max(u["bin"] for u in so["units"] if u["epoch"] in ("E2", "E3"))


def cmd_refit(a) -> int:
    """Sensitivity: each control's area constant refitted on the units
    within the DEM's range (bins up to the highest E2/E3 bin)."""
    transform, shape, static, units = setup()
    surfaces, flood, _ = build(transform, shape, static, units)
    for k in ("zone", "land_seed"):
        static.pop(k)
    blocks = pd.read_csv(a7.BLOCKS)
    fit_region = static["domain"] & a7.train_mask(blocks, transform, shape)
    a7.ENVELOPE = a7.record_low_water(units, transform, shape)
    top = top_dem_bin()
    sub = {b: u for b, u in units.items() if b <= top}
    finite = {n: (lambda h: lambda r0, r1, c0, c1: np.isfinite(np.asarray(h[r0:r1, c0:c1], float)))(h)
              for n, h in surfaces.items()}
    rec, _ = a7.fit_shape(sub, transform, static, flood, finite, fit_region, level_datum_offset=0.0,
                          product_offsets={}, tile=TILE)
    out = {n: {"offset_m": r["offset_m"], "source": f"D-116 addendum: D-070 fit on the {len(rec['units'])} units "
                                                    f"with bin <= {top}"} for n, r in rec["products"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "refit_dem_range.json").write_text(json.dumps({"constants": out, "top_dem_bin": top,
                                                          "n_units": len(rec["units"])}, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def level_table(hc_dir: Path, variant: str = "shape") -> pd.DataFrame:
    """Per product, sector and bin of a hindcast run: the H1 detail table (caspian.h1)."""
    disp = pd.read_csv(hc_dir / "displacement.csv")
    noise = pd.read_csv(RESULTS / "hindcast" / TAG / "noise_floor.csv")
    prod = disp[(disp["epoch"] == "pooled") & (disp["sector"] != "all") & (disp["variant"] == variant)]
    prod = prod.rename(columns={"median_m": "median"})
    nz = noise[(noise["kind"] == "pooled_level") & (noise["sector"] != "all")].rename(columns={"p95_m": "p95"})
    _, detail = h1.mispredicts(prod, nz[["sector", "bin", "n", "p95"]])
    return detail


def cmd_decide(a) -> int:
    ctrl_dir = OUT / "hindcast"
    main = level_table(RESULTS / "hindcast" / TAG)
    ctrl = level_table(ctrl_dir)
    # the control's own noise floor must equal the primary's (same units and code)
    nf_c = pd.read_csv(ctrl_dir / "noise_floor.csv")
    nf_m = pd.read_csv(RESULTS / "hindcast" / TAG / "noise_floor.csv")
    key = ["kind", "epoch", "bin", "sector"]
    mm = nf_m.merge(nf_c, on=key, suffixes=("_m", "_c"))
    floor_max_diff = float(np.nanmax(np.abs(mm["p95_m_m"].astype(float) - mm["p95_m_c"].astype(float))))
    # DEM contour range: bins of the DEM epochs (E2, E3); higher bins have no DEM contour
    so = json.loads((RESULTS / "hindcast" / TAG / "shape_offsets.json").read_text(encoding="utf-8"))
    dem_bins = sorted({u["bin"] for u in so["units"] if u["epoch"] in ("E2", "E3")})
    top_dem_bin = max(dem_bins)
    cshape = json.loads((ctrl_dir / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    rows = []
    for which, table in (("rounded", ROUNDED_FOR), ("unrounded", UNROUNDED_FOR)):
        for p, cname in table.items():
            m = main[main["product"] == p][["product", "sector", "bin", "level_m", "median", "p95_noise",
                                            "evaluable", "exceeds"]]
            c = ctrl[ctrl["product"] == cname][["sector", "bin", "median", "evaluable"]].rename(
                columns={"median": "D_control", "evaluable": "evaluable_control"})
            d = m.merge(c, on=["sector", "bin"], how="left")
            d["evaluable_control"] = d["evaluable_control"].fillna(False).astype(bool)
            d["in_dem_range"] = d["bin"] <= top_dem_bin
            d = d.rename(columns={"median": "D", "p95_noise": "Q", "evaluable": "evaluable_product",
                                  "exceeds": "exceeds_floor"})
            d["control"], d["control_kind"] = cname, which
            d["control_c_m"] = float(cshape[cname]["offset_m"])
            rows.append(d)
    lv = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lv.to_csv(OUT / "levels.csv", index=False)
    res = {}
    ex_all = []
    for which in ("rounded", "unrounded"):
        for restrict in ("dem_range", "all_levels"):
            d = lv[lv["control_kind"] == which].copy()
            d["evaluable"] = d["evaluable_product"].astype(bool) & d["evaluable_control"]
            if restrict == "dem_range":
                d["evaluable"] &= d["in_dem_range"]
            ex = rev_r.excess_rule(d)
            ex["control_kind"], ex["levels"] = which, restrict
            ex_all.append(ex)
            res[f"{which}|{restrict}"] = rev_r.stop_rule(ex, MISPREDICTING, VALIDATED)
    ex = pd.concat(ex_all, ignore_index=True)
    ex.to_csv(OUT / "excess.csv", index=False)
    sens = OUT.parent / "audit" / "control_refit" / "levels_D.csv.gz"
    if sens.exists():                       # sensitivity: the controls at the refitted constant
        L = pd.read_csv(sens)
        L = L[L["config"] == "replica"].rename(columns={"median_m": "D_control_refit", "product": "control"})
        nzr = pd.read_csv(RESULTS / "hindcast" / TAG / "noise_floor.csv")
        nzr = nzr[(nzr["kind"] == "pooled_level") & (nzr["sector"] != "all")][["sector", "bin", "n"]].rename(
            columns={"n": "n_noise"})
        L = L.merge(nzr, on=["sector", "bin"], how="left")
        L["evaluable_control_refit"] = L["observed_level"].astype(bool) & (L["n"] > 0) & (L["n_noise"].fillna(0) > 0)
        lr = lv.merge(L[["control", "sector", "bin", "D_control_refit", "evaluable_control_refit"]],
                      on=["control", "sector", "bin"], how="left")
        lr["evaluable_control_refit"] = lr["evaluable_control_refit"].fillna(False).astype(bool)
        lr.to_csv(OUT / "levels_refit.csv", index=False)
        for which in ("rounded", "unrounded"):
            d = lr[lr["control_kind"] == which].copy()
            d["evaluable"] = d["evaluable_product"].astype(bool) & d["evaluable_control_refit"] & d["in_dem_range"]
            ex = rev_r.excess_rule(d.drop(columns=["D_control"]).rename(columns={"D_control_refit": "D_control"}))
            ex["control_kind"], ex["levels"] = which, "dem_range_refit"
            ex_all.append(ex)
            res[f"{which}|dem_range_refit"] = rev_r.stop_rule(ex, MISPREDICTING, VALIDATED)
        pd.concat(ex_all, ignore_index=True).to_csv(OUT / "excess.csv", index=False)
    decision = {"rule": "A18 item 6; deciding: rounded control, levels in the DEM's contour range (D-116)",
                "deciding": res["rounded|dem_range"], "all": res, "top_dem_bin": top_dem_bin,
                "control_constants_m": {k: v["offset_m"] for k, v in cshape.items()},
                "floor_reproduced_max_abs_diff_m": floor_max_diff,
                "products": MISPREDICTING, "sectors": VALIDATED}
    (OUT / "stop_rule.json").write_text(json.dumps(decision, indent=1, default=str), encoding="utf-8")
    print(json.dumps(decision, indent=1, default=str))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build", "run", "refit", "decide"])
    ap.add_argument("--pilot-bins", type=int, default=0)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "run": cmd_run, "refit": cmd_refit, "decide": cmd_decide}[a.command](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
