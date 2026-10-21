"""A8 independent check: ICESat-2 ATL24 seabed heights against the near-term emergence
surfaces (PREREG section 2 "ATL24 bathymetry where available", section 10 and Amendment
17; brief section 8 A8, "Use ICESat-2 ATL24 as an independent check where turbidity
allows"; D-095). Rules and definitions: caspian.atl24 and Amendment 17.

Stages (one run does all three):
1. photons: the class-40 photons saved by scripts/a8_atl24_fetch.py
   (data/raw/icesat2/atl24/*.npz, read one granule at a time) pass the ordered steps of
   caspian.atl24.photon_steps; every step is counted (photon_cascade.csv). Kept photons
   lie inside the domain polygon. H = ellipse_h - N(EIGEN-6C4) (validation.EigenGeoid).
   The photons are aggregated per 60 m hindcast cell and track (granule x beam) for the
   primary selection and each photon-level sensitivity (night only; all beams; depth at
   most 3 m) -> data/interim/atl24/celltracks_<variant>.csv.gz.
2. surfaces: on the hindcast grid of scripts/a7_hindcast.py (the hindcast run "main"),
   for every cell holding a cell-track: the sector and domain flag; per product the
   uncorrected heights (products.Corrected with geoid_correction_primary.npz, the H1
   total variant), the corrected product (exposure.CorrectedSurface of the final
   linear_hold models of results/emergence/main/error_models.json, nothing refitted),
   the fused surface (exposure.FusedSurface with the DEM aggregated to 60 m as
   scripts/a8_emergence.dem_to_grid does, the observed unit at the lowest fitted level
   and the outside states, all rebuilt with the a7/a8 functions), the shape variant
   (uncorrected minus the D-070 constant of results/hindcast/main/shape_offsets.json)
   and the DEM. Check: the fused surface's cell counts per sector and source reproduce
   results/emergence/main/fused_summary.csv (fused_check.csv).
3. evaluate: the populations of Amendment 17 item 4 on y_ref = H + c; per product the
   paired comparison uncorrected against fused (caspian.atl24.compare, block bootstrap
   over the 50 km H2 lattice) and the verdict (caspian.atl24.verdict), overall and per
   sector; the same for the sensitivities (constants, photon variants); the waterline
   DEM in the emerged band; the water-surface constant.

Outputs (results/validation/atl24/): photon_cascade.csv, granules.json, celltracks_summary.csv,
fused_check.csv, accuracy_<population>.csv, differences_<population>.csv,
verdicts.json, depth_classes.csv, correction_at_cells.csv, water_surface_constant.json,
error_model_range.csv (a surface-only diagnostic of the H3 error models, error_model_range()),
run.json. If there is no usable ATL24 photon in the domain, the tables are written
empty and every verdict is "not evaluable: no ATL24 data".

Usage: python scripts/a8_atl24_check.py [--tile 2048] [--skip-fused-check]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import atl24, blocking, products, validation
from caspian import exposure as ex
from caspian import hindcast_tiled as ht
from caspian.config import CRS_EQUAL_AREA, INTERIM, PROCESSED, RAW, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TAG = "main"
RAWDIR = RAW / "icesat2" / "atl24"
WORK = INTERIM / "atl24"
OUT = RESULTS / "validation" / "atl24"
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
DAHITI = RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv"
LEVEL_GAP_DAYS = 60.0            # DAHITI interpolation for the water-surface constant (caspian.levels default)
BLOCK_M = 50_000.0               # the H2 block lattice (D-085)
PHOTON_VARIANTS = {"primary": {"strong_only": True, "night_only": False, "max_depth_m": atl24.MAX_DEPTH_M},
                   "night": {"strong_only": True, "night_only": True, "max_depth_m": atl24.MAX_DEPTH_M},
                   "all_beams": {"strong_only": False, "night_only": False, "max_depth_m": atl24.MAX_DEPTH_M},
                   "shallow3m": {"strong_only": True, "night_only": False, "max_depth_m": 3.0}}
DEPTH_CLASSES = ((0.0, 2.0), (2.0, 4.0), (4.0, atl24.MAX_DEPTH_M))


def _clean(obj):
    """JSON-safe copy: non-finite floats become null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _dump(obj, indent=1) -> str:
    return json.dumps(_clean(obj), indent=indent, default=str)


# --- 1. photons ---------------------------------------------------------------------------------

def grid():
    a7 = _load("a7_hindcast")
    w, s, e, n = DOMAIN.bounds
    return a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))


def photons(transform, log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stage 1. Returns (kept photons of every variant, primary cascade, per-granule
    counts). Kept photons pass every step except strong_beam (column ``strong``) with
    depth at most MAX_DEPTH_M; the cascade counts the primary steps in order, the last
    being the domain polygon."""
    import pyproj
    import shapely

    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    geoid = validation.EigenGeoid()
    shapely.prepare(DOMAIN)
    files = sorted(RAWDIR.glob("*.npz"))
    casc_tot, per_gran, kept = None, {}, []
    for k, f in enumerate(files):
        with np.load(f) as z:
            arr = {key: z[key] for key in z.files}
        n = int(arr["beam"].size)
        if n == 0:
            continue
        d = pd.DataFrame({v: arr[v] for v in atl24.PHOTON_VARS})
        d["beam"] = np.asarray(atl24.BEAMS)[arr["beam"].astype(int)]
        d["sc_orient"] = int(arr["sc_orient"])
        d["geoid"] = geoid(d["lon_ph"].to_numpy(float), d["lat_ph"].to_numpy(float))
        steps = atl24.photon_steps(d)
        steps["domain"] = shapely.contains_xy(DOMAIN, d["lon_ph"].to_numpy(float), d["lat_ph"].to_numpy(float))
        keep, casc = validation.cascade(steps, n)
        casc_tot = casc if casc_tot is None else casc_tot.assign(
            n_remaining=casc_tot["n_remaining"] + casc["n_remaining"].to_numpy(),
            n_fail_alone=casc_tot["n_fail_alone"] + casc["n_fail_alone"].to_numpy())
        loose = np.ones(n, bool)
        for name, m in steps.items():
            if name != "strong_beam":
                loose &= m
        per_gran[f.stem] = {"n_class40_box": n, "n_primary": int(keep.sum()), "n_any_beam": int(loose.sum()),
                            "invalid_kd": int(d["invalid_kd"].to_numpy()[loose].sum()),
                            "invalid_wind_speed": int(d["invalid_wind_speed"].to_numpy()[loose].sum())}
        if not loose.any():
            continue
        s = d.loc[loose]
        x, y = fwd.transform(s["lon_ph"].to_numpy(float), s["lat_ph"].to_numpy(float))
        H = s["ellipse_h"].to_numpy(float) - s["geoid"].to_numpy(float)
        kept.append(pd.DataFrame({
            "x": x, "y": y, "granule": f.stem, "track": f.stem + ":" + s["beam"].astype(str).to_numpy(),
            "H": H, "depth": atl24.depth(s["surface_h"], s["ortho_h"]),
            "surface": atl24.surface_working(s["surface_h"], s["ellipse_h"], s["ortho_h"], s["geoid"], 0.0),
            "night": s["night_flag"].to_numpy(float), "strong": steps["strong_beam"][loose],
            "time": atl24.photon_time(s["delta_time"])}))
        if (k + 1) % 50 == 0:
            log(f"photons: {k + 1}/{len(files)} granules", flush=True)
    ph = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["x", "y", "granule", "track", "H", "depth", "surface", "night", "strong", "time"])
    if casc_tot is None:
        casc_tot = pd.DataFrame({"step": ["input"], "n_remaining": [0], "n_fail_alone": [0]})
    return ph, casc_tot, per_gran


def variant_photons(ph: pd.DataFrame, spec: dict) -> pd.DataFrame:
    m = ph["depth"].to_numpy(float) <= spec["max_depth_m"]
    if spec["strong_only"]:
        m &= ph["strong"].to_numpy(bool)
    if spec["night_only"]:
        m &= ph["night"].to_numpy(float) == 1
    return ph.loc[m]


def celltracks(ph: pd.DataFrame, transform) -> dict[str, pd.DataFrame]:
    """Cell-tracks per photon variant; y_ref here is H (no constant yet)."""
    out = {}
    for v, spec in PHOTON_VARIANTS.items():
        s = variant_photons(ph, spec).rename(columns={"H": "y_ref"})
        out[v] = atl24.aggregate(s, transform)
    return out


# --- 2. surfaces --------------------------------------------------------------------------------

def sample(surface, rows, cols, shape, tile: int) -> np.ndarray:
    """Values of a window-readable surface (array, memory map, Corrected, CorrectedSurface,
    FusedSurface) at cells, read tile by tile."""
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    out = np.full(rows.size, np.nan)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    tr, tc = rows // tile, cols // tile
    for (a, b), idx in pd.DataFrame({"a": tr[ok], "b": tc[ok]}).groupby(["a", "b"]).indices.items():
        k = np.flatnonzero(ok)[idx]
        rr0, rr1 = int(rows[k].min()), int(rows[k].max()) + 1
        cc0, cc1 = int(cols[k].min()), int(cols[k].max()) + 1
        win = np.asarray(surface[(slice(rr0, rr1), slice(cc0, cc1))], float)
        out[k] = win[rows[k] - rr0, cols[k] - cc0]
    return out


def surfaces(cells: pd.DataFrame, transform, shape, tile: int, fused_check: bool, log=print):
    """Stage 2 (module docstring). Returns (cells with surface columns, fused check table,
    record)."""
    a8 = _load("a8_emergence")
    a7 = a8.a7      # the a7 instance a8.observed_unit reads (its ENVELOPE, D-082), not a second copy
    hdir = RESULTS / "hindcast" / TAG
    hrun = json.loads((hdir / "run.json").read_text(encoding="utf-8"))
    if [float(v) for v in transform[:6]] != hrun["grid"]["transform"][:6] or list(shape) != hrun["grid"]["shape"]:
        raise SystemExit("the hindcast grid differs from this one")
    erun = json.loads((RESULTS / "emergence" / TAG / "run.json").read_text(encoding="utf-8"))
    models = json.loads((RESULTS / "emergence" / TAG / "error_models.json").read_text(encoding="utf-8"))
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != erun["vertical_correction"]["sha256"]:
        raise SystemExit("the correction differs from the H3 run's")
    log("surfaces: static layers", flush=True)
    static = a7.production_static(transform, shape, tile)
    units = a7.stage2_index(INTERIM / "stage2" / TAG)
    a7.ENVELOPE = a7.record_low_water(units, transform, shape)
    fit_epochs = tuple(erun["fit_epochs"])
    ulist = [(ep, b) for b in sorted(units) for ep in sorted(units[b])]
    low_b = min(b for ep, b in ulist if ep in fit_epochs)
    low_ep = max(ep for ep, b in ulist if ep in fit_epochs and b == low_b)
    log(f"surfaces: observed unit {low_ep} bin {low_b}", flush=True)
    code_low = a8.observed_unit(units, low_b, low_ep, transform, shape, static, tile)
    a7.ENVELOPE = None
    WORK.mkdir(parents=True, exist_ok=True)
    dem_path = WORK / "dem60.npy"
    dem_paths = sorted((PROCESSED / "dem" / erun["dem_tag"]).glob("dem_*.tif"))
    if not dem_path.exists():
        dem = a8.dem_to_grid(dem_paths, transform, shape)
        np.save(dem_path, dem)
        del dem
    dem = np.load(dem_path, mmap_mode="r")
    sector_names = list(SECTOR_ORDER)
    code_of = {s_: k + 1 for k, s_ in enumerate(sector_names)}
    lab_in = np.where(np.asarray(static["domain"], bool), np.asarray(static["sector"]), 0).astype(np.uint8)
    rows, cols = cells["row"].to_numpy(np.int64), cells["col"].to_numpy(np.int64)
    out = cells.copy()
    lab = sample(lab_in, rows, cols, shape, tile)
    out["domain"] = lab >= 1
    out["sector"] = [sector_names[int(v) - 1] if v >= 1 else None for v in np.nan_to_num(lab, nan=0)]
    out["dem"] = sample(dem, rows, cols, shape, tile)
    shape_c = {p: float(r["offset_m"]) for p, r in
               json.loads((hdir / "shape_offsets.json").read_text(encoding="utf-8"))["products"].items()}
    checks, rec = [], {"fit_epochs": list(fit_epochs), "lowest_unit": [low_ep, int(low_b)], "products": {}}
    ref = pd.read_csv(RESULTS / "emergence" / TAG / "fused_summary.csv")
    for p in erun["products"]:
        h = products.heights(p, transform, shape, erun.get("resampling", "bilinear"))
        unc = products.Corrected(h, corr)
        mrec = models[p][ex.PRIMARY_FORM]["final"]
        mods = {code_of[r]: m for r, m in mrec["models"].items() if r in code_of}
        corr_s = ex.CorrectedSurface(unc, static["sector"], mods, mrec["lowest_level_m"])
        fused = ex.FusedSurface(dem, corr_s, code_low, static["outside"], mrec["lowest_level_m"])
        out[f"{p}|uncorrected"] = sample(unc, rows, cols, shape, tile)
        out[f"{p}|corrected"] = sample(corr_s, rows, cols, shape, tile)
        out[f"{p}|fused"] = sample(fused, rows, cols, shape, tile)
        out[f"{p}|source"] = _sample_source(fused, rows, cols, shape, tile)
        out[f"{p}|shape"] = out[f"{p}|uncorrected"] - shape_c[p]
        rec["products"][p] = {"lowest_level_m": mrec["lowest_level_m"], "shape_offset_m": shape_c[p],
                              "heights_cache": Path(h.filename).name}
        if fused_check:
            fs = fused.summary(lab_in, len(sector_names), tile=tile).assign(product=p, region=sector_names)
            r = ref[ref["product"] == p].set_index("region")
            for _, row in fs.iterrows():
                for col in ("n_none", "n_dem", "n_product", "n_extrapolated", "n_clamped_sea", "n_clamped_land"):
                    checks.append({"product": p, "region": row["region"], "count": col, "here": int(row[col]),
                                   "h3": int(r.loc[row["region"], col]),
                                   "equal": int(row[col]) == int(r.loc[row["region"], col])})
        log(f"surfaces: {p} done", flush=True)
    rec.update(dem_tiles=len(dem_paths), correction=corr_rec["file"], correction_sha256=corr_rec["sha256"])
    del static, code_low, lab_in
    return out, pd.DataFrame(checks), rec


def _sample_source(fused, rows, cols, shape, tile):
    class _Src:
        def __getitem__(self, key):
            return fused.source(key)
    return sample(_Src(), rows, cols, shape, tile)


# --- 3. evaluate ------------------------------------------------------------------------------

def dahiti_at(times) -> np.ndarray:
    """DAHITI level linearly interpolated to ``times``; NaN where the bracketing records
    are more than LEVEL_GAP_DAYS apart or outside the record."""
    d = pd.read_csv(DAHITI, sep=";")
    t = pd.to_datetime(d["datetime"]).to_numpy("datetime64[ns]").astype("int64") / 86_400e9
    o = np.argsort(t)
    t, v = t[o], d["wse"].to_numpy(float)[o]
    q = np.asarray(times, "datetime64[ns]").astype("int64") / 86_400e9
    k = np.searchsorted(t, q)
    lo, hi = np.clip(k - 1, 0, t.size - 1), np.clip(k, 0, t.size - 1)
    ok = (k > 0) & (k < t.size) & ((t[hi] - t[lo]) <= LEVEL_GAP_DAYS)
    return np.where(ok, np.interp(q, t, v), np.nan)


def population(ct: pd.DataFrame, c: float, level_2026: float, record_high: float, which: str) -> pd.DataFrame:
    y = ct["y_ref"].to_numpy(float) + c
    if which == "near_term":
        m = (y >= level_2026 - atl24.NEAR_TERM_DROP_M) & (y < level_2026)
    elif which == "emerged_band":
        m = (y >= level_2026) & (y <= record_high)
    else:
        raise ValueError(which)
    m &= ct["domain"].to_numpy(bool)
    return ct.loc[m].assign(y_ref=y[m])


def evaluate(ct: pd.DataFrame, prods, c: float, level_2026: float, record_high: float, pop: str,
             n_boot: int = atl24.BOOT_N, extra_pairs: dict | None = None):
    t = population(ct, c, level_2026, record_high, pop)
    pairs = {p: (f"{p}|uncorrected", f"{p}|fused") for p in prods}
    pairs.update(extra_pairs or {})
    acc, dif = atl24.compare(t, pairs, SECTOR_ORDER, n_boot=n_boot)
    verd = {}
    for name in pairs:
        for scope in ["overall", *SECTOR_ORDER]:
            rows = dif[(dif["pair"] == name) & (dif["scope"] == scope)]
            verd.setdefault(name, {})[scope] = atl24.verdict(rows) if len(rows) else {
                "verdict": "not evaluable", "reason": "no cell-tracks", "partial": None}
    return t, acc, dif, verd


def error_model_range(level_2026: float) -> pd.DataFrame:
    """Surface-only diagnostic of the H3 error models (no ATL24 involved; Amendment 17,
    "What had been seen"; STOP memo of 2026-10-21): per product and region, the range of
    the matching shifts the final model was fitted on (usable E2 + E3 units of
    results/emergence/main/shifts.csv), the number of distinct fitted depths, the shift
    the model applies to product heights in the near-term zone [L2026 - 5 m, L2026]
    (0.01 m steps), whether that leaves the fitted range, and the emerged area at the
    2 m scenario per surface (emergence.csv, "emergence" functional)."""
    em = RESULTS / "emergence" / TAG
    models = json.loads((em / "error_models.json").read_text(encoding="utf-8"))
    sh = pd.read_csv(em / "shifts.csv")
    e2 = pd.read_csv(em / "emergence.csv")
    e2 = e2[(e2["functional"] == "emergence") & np.isclose(e2["scenario_m"], 2.0)]
    z = np.round(np.arange(level_2026 - atl24.NEAR_TERM_DROP_M, level_2026 + 1e-9, 0.01), 4)
    rows = []
    for p, rec in models.items():
        f = rec[ex.PRIMARY_FORM]["final"]
        s = sh[(sh["product"] == p) & sh["epoch"].isin(f["epochs"]) & sh["usable"].astype(bool)]
        for reg, mod in f["models"].items():
            if mod is None:
                continue
            r = s[s["region"] == reg]
            d = ex.unit_depths(r["level_m"].to_numpy(float), r["shift_m"].to_numpy(float), f["lowest_level_m"])
            pred = ex.predict_shift(mod, f["lowest_level_m"] - z)
            area = e2[(e2["product"] == p) & (e2["region"] == reg)].set_index("surface")["estimate_km2"]
            lo, hi = (float(r["shift_m"].min()), float(r["shift_m"].max())) if len(r) else (np.nan, np.nan)
            rows.append({"product": p, "region": reg, "form": mod["form"], "used_domain": mod.get("used_domain"),
                         "n_units": int(len(r)), "n_distinct_depths": int(np.unique(np.round(d, 2)).size),
                         "slope_b": mod["b"], "fit_shift_min_m": lo, "fit_shift_max_m": hi,
                         "applied_near_term_min_m": float(np.min(pred)), "applied_near_term_max_m": float(np.max(pred)),
                         "outside_fit_range": bool(np.max(pred) > hi + 0.01 or np.min(pred) < lo - 0.01),
                         **{f"emerged_2m_{k}_km2": float(area.get(k, np.nan))
                            for k in ("uncorrected", "corrected", "fused")}})
    return pd.DataFrame(rows)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=ht.TILE)
    ap.add_argument("--skip-fused-check", action="store_true")
    ap.add_argument("--n-boot", type=int, default=atl24.BOOT_N)
    ap.add_argument("--check-surfaces", type=int, default=0, metavar="N",
                    help="with no ATL24 cell-track, still build the surfaces at N random domain cells (seed "
                         "20261020) and run the fused check, to verify the surface stage")
    a = ap.parse_args(argv)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    transform, shape = grid()
    erun = json.loads((RESULTS / "emergence" / TAG / "run.json").read_text(encoding="utf-8"))
    level_2026 = float(erun["level_2026"]["level_m"])
    record_high = float(json.loads((RESULTS / "qc" / "dahiti_summary.json").read_text(encoding="utf-8"))
                        ["smoothed"]["level_max_m"])
    consts = {"atl08": json.loads((RESULTS / "validation" / "main" / "ATL08_datum_constant.json").read_text(
        encoding="utf-8"))["constant"], "atl06": json.loads((RESULTS / "validation" / "main" / "ATL06_datum_constant.json")
                                                             .read_text(encoding="utf-8"))["constant"]}
    prov = json.loads((RESULTS / "provenance" / "icesat2_atl24.json").read_text(encoding="utf-8"))
    prods = list(erun["products"])
    emr = error_model_range(level_2026)
    emr.to_csv(OUT / "error_model_range.csv", index=False)

    # 1. photons
    ph, casc, per_gran = photons(transform)
    casc.to_csv(OUT / "photon_cascade.csv", index=False)
    cts = celltracks(ph, transform)
    nonempty = {v: c for v, c in cts.items() if len(c)}
    # water-surface constant (primary photons)
    pp = variant_photons(ph, PHOTON_VARIANTS["primary"])
    ws = atl24.water_surface_constant(pp["surface"].to_numpy(float), dahiti_at(pp["time"]),
                                      pp["granule"].to_numpy()) if len(pp) else {"constant_m": float("nan"),
                                                                                 "n_overpasses": 0}
    ws["note"] = "sensitivity of Amendment 17 item 3; L_DAHITI linear between records at most 60 days apart"
    (OUT / "water_surface_constant.json").write_text(_dump(ws), encoding="utf-8")
    consts["water_surface"] = {"constant_m": ws["constant_m"]}

    # 2. surfaces
    fused_check, srec = pd.DataFrame(), {"status": "not run: no ATL24 cell-tracks"}
    if nonempty:
        cells = pd.concat([c[["row", "col"]] for c in nonempty.values()]).drop_duplicates().reset_index(drop=True)
        cells_s, fused_check, srec = surfaces(cells, transform, shape, a.tile, not a.skip_fused_check)
        x, y = atl24.cell_centres(cells_s["row"], cells_s["col"], transform)
        bi, bj = blocking.block_index(x, y, BLOCK_M)
        cells_s["block"] = [f"{i}_{j}" for i, j in zip(bi, bj)]
        cells_s.to_csv(WORK / "cells_surfaces.csv.gz", index=False, float_format="%.4f")
        for v in nonempty:
            cts[v] = cts[v].merge(cells_s, on=["row", "col"], how="left")
            cts[v].to_csv(WORK / f"celltracks_{v}.csv.gz", index=False, float_format="%.4f")
    elif a.check_surfaces:
        import shapely
        rng = np.random.default_rng(atl24.SEED)
        w, s, e, n = DOMAIN.bounds
        lon, lat = rng.uniform(w, e, 20 * a.check_surfaces), rng.uniform(s, n, 20 * a.check_surfaces)
        inside = shapely.contains_xy(DOMAIN, lon, lat)
        import pyproj
        x, y = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform(
            lon[inside][:a.check_surfaces], lat[inside][:a.check_surfaces])
        r, c = atl24.cell_index(x, y, transform)
        cells = pd.DataFrame({"row": r, "col": c}).drop_duplicates().reset_index(drop=True)
        cells_s, fused_check, srec = surfaces(cells, transform, shape, a.tile, not a.skip_fused_check)
        srec["status"] = f"surface stage verified on {len(cells)} random domain cells (no ATL24 cell-track)"
        cells_s.to_csv(OUT / "surfaces_check_cells.csv", index=False, float_format="%.4f")
    fused_check.to_csv(OUT / "fused_check.csv", index=False)

    # 3. evaluate
    c0 = float(consts["atl08"]["constant_m"])
    summary_rows, verdicts, depth_rows, corr_rows = [], {}, [], []
    runs = [("primary", "atl08"), ("night", "atl08"), ("all_beams", "atl08"), ("shallow3m", "atl08"),
            ("primary", "atl06"), ("primary", "water_surface")]
    for v, cname in runs:
        c = float(consts[cname]["constant_m"])
        key = v if cname == "atl08" else f"{v}_c_{cname}"
        for pop in ("near_term", "emerged_band"):
            ct = cts.get(v)
            if ct is None or not len(ct) or not np.isfinite(c):
                verdicts.setdefault(key, {})[pop] = {p: {"overall": {"verdict": "not evaluable",
                                                                     "reason": "no ATL24 data", "partial": None}}
                                                     for p in prods}
                summary_rows.append({"variant": key, "population": pop, "n_celltracks": 0, "n_cells": 0,
                                     "n_blocks": 0, "n_tracks": 0, "n_granules": 0})
                continue
            extra = {f"{p}|dem_vs_uncorrected": (f"{p}|uncorrected", "dem") for p in prods} if pop == "emerged_band" else {}
            extra.update({f"{p}|corrected_vs_uncorrected": (f"{p}|uncorrected", f"{p}|corrected") for p in prods})
            extra.update({f"{p}|fused_vs_shape": (f"{p}|shape", f"{p}|fused") for p in prods})
            t, acc, dif, verd = evaluate(ct, prods, c, level_2026, record_high, pop, a.n_boot, extra)
            acc.to_csv(OUT / f"accuracy_{key}_{pop}.csv", index=False)
            dif.to_csv(OUT / f"differences_{key}_{pop}.csv", index=False)
            verdicts.setdefault(key, {})[pop] = verd
            summary_rows.append({"variant": key, "population": pop, "n_celltracks": int(len(t)),
                                 "n_cells": int(t[["row", "col"]].drop_duplicates().shape[0]),
                                 "n_blocks": int(t["block"].nunique()), "n_tracks": int(t["track"].nunique()),
                                 "n_granules": int(t["track"].str.split(":").str[0].nunique()),
                                 "n_photons": int(t["n_photons"].sum()),
                                 **{f"n_celltracks.{s}": int((t["sector"] == s).sum()) for s in SECTOR_ORDER}})
            if key == "primary" and pop == "near_term":
                for lo, hi in DEPTH_CLASSES:
                    dm = (t["depth"] > lo) & (t["depth"] <= hi)
                    if dm.any():
                        a2, d2 = atl24.compare(t.loc[dm], {p: (f"{p}|uncorrected", f"{p}|fused") for p in prods},
                                               [], n_boot=a.n_boot)
                        depth_rows.append(a2.assign(depth_lo_m=lo, depth_hi_m=hi))
                for p in prods:
                    dcor = (t[f"{p}|fused"] - t[f"{p}|uncorrected"]).to_numpy(float)
                    for scope in ["overall", *SECTOR_ORDER]:
                        m = np.isfinite(dcor) & ((t["sector"] == scope).to_numpy() if scope != "overall" else True)
                        if not np.any(m):
                            continue
                        dd = dcor[m]
                        corr_rows.append({"product": p, "scope": scope, "n": int(dd.size),
                                          "median_m": float(np.median(dd)), "p05_m": float(np.percentile(dd, 5)),
                                          "p95_m": float(np.percentile(dd, 95)),
                                          "share_abs_gt_5m": float(np.mean(np.abs(dd) > 5.0)),
                                          "share_source_dem": float(np.mean(t.loc[m, f"{p}|source"] == ex.SRC_DEM)),
                                          "share_source_extrapolated": float(np.mean(t.loc[m, f"{p}|source"]
                                                                                     == ex.SRC_EXTRAPOLATED))})
    pd.DataFrame(summary_rows).to_csv(OUT / "celltracks_summary.csv", index=False)
    (pd.concat(depth_rows, ignore_index=True) if depth_rows else pd.DataFrame()).to_csv(OUT / "depth_classes.csv",
                                                                                          index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "correction_at_cells.csv", index=False)
    (OUT / "verdicts.json").write_text(_dump(verdicts), encoding="utf-8")
    (OUT / "granules.json").write_text(_dump(per_gran, indent=0), encoding="utf-8")
    sens = {}
    for p in prods:
        prim = verdicts.get("primary", {}).get("near_term", {}).get(p, {}).get("overall", {}).get("verdict")
        others = {k: verdicts[k]["near_term"].get(p, {}).get("overall", {}).get("verdict")
                  for k in verdicts if k != "primary" and "near_term" in verdicts[k]}
        sens[p] = {"primary": prim, "sensitivities": others,
                   "sensitive_to_datum_constant": any(others.get(k) != prim for k in others if "_c_" in k)}
    run = {"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "prereg": "Amendment 17",
           "decision": "D-095", "level_2026_m": level_2026, "record_high_m": record_high,
           "near_term_range_m": [level_2026 - atl24.NEAR_TERM_DROP_M, level_2026],
           "constants": consts, "photon_variants": PHOTON_VARIANTS, "min_photons": atl24.MIN_PHOTONS,
           "min_blocks": atl24.MIN_BLOCKS, "max_depth_m": atl24.MAX_DEPTH_M,
           "confidence_min": atl24.CONFIDENCE_MIN, "block_m": BLOCK_M, "n_boot": a.n_boot, "seed": atl24.SEED,
           "granules": {"n_found": prov.get("n_granules_found"), "n_read": len(prov.get("granules", {})),
                        "n_extent_meets_box": prov.get("summary", {}).get("n_extent_intersects_box"),
                        "n_with_class40_in_box": sum(1 for r in prov.get("granules", {}).values()
                                                     if int(r.get("n_saved", 0)) > 0),
                        "n_with_primary_photons": sum(1 for r in per_gran.values() if r["n_primary"] > 0),
                        "spot_check": prov.get("spot_check")},
           "n_photons_primary": int(len(variant_photons(ph, PHOTON_VARIANTS["primary"]))) if len(ph) else 0,
           "n_celltracks": {v: int(len(c)) for v, c in cts.items()},
           "surfaces": srec, "fused_check_all_equal": (bool(fused_check["equal"].all()) if len(fused_check)
                                                       else None),
           "verdict_primary_near_term": sens,
           "error_model_range": {"n_models": int(len(emr)), "n_outside_fit_range": int(emr["outside_fit_range"].sum()),
                                 "max_applied_shift_m": float(emr["applied_near_term_max_m"].max()),
                                 "min_applied_shift_m": float(emr["applied_near_term_min_m"].min()),
                                 "fit_shift_range_m": [float(emr["fit_shift_min_m"].min()),
                                                       float(emr["fit_shift_max_m"].max())]},
           "seconds": round(time.time() - t0, 1)}
    (OUT / "run.json").write_text(_dump(run), encoding="utf-8")
    print(json.dumps({"n_celltracks": run["n_celltracks"], "verdicts": {p: s["primary"] for p, s in sens.items()}}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
