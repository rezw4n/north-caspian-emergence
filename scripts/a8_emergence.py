"""A8: near-term seabed emergence and exposure (H3; PREREG section 10; brief section 8
A8). Definitions and the proposed choices are in the docstring of caspian.exposure.

Inputs (what this driver needs, and where it reads them):
- the stage-2 counts the hindcast read: data/interim/stage2/<tag>/tile_*.npz (the
  units, observed extents and sea connection are rebuilt with the functions of
  scripts/a7_hindcast.py, so they are the hindcast's own);
- the hindcast run of the same tag, results/hindcast/<tag>/:
  * run.json: products, grid, vertical correction (file name and sha256, checked
    against --vertical-correction), level and product offsets (both must be 0, the
    DAHITI working datum of D-070);
  * extent.csv: the total-variant area bias per product, unit and region; the
    "domain" rows are recomputed here and the largest difference is recorded
    (cross-check, expected 0);
  * displacement.csv: observed_level per sector and unit (PREREG Amendment 6 item 5),
    which selects the calibration scores of a region;
- product heights and the vertical correction as caspian.products reads them, with
  the flood-level maps from the hindcast's cache (data/interim/hindcast/flood);
- the waterline DEM, data/processed/dem/<dem tag>/dem_<i>_<j>.tif (band 1, m on the
  DAHITI reference), aggregated to the 60 m grid (mean of the finite 30 m pixels);
- the DAHITI series (the 2026 level, caspian.exposure.reference_level);
- exposure layers: data/raw/osm/osm_<layer>.gpkg (scripts/a8_osm_exposure.py) and any
  polygon layers given with --polygons;
- results/levels/projection_pathways.csv (scenario dates), if present.

Steps:
1. Shifts (pass 1). Per unit and product: the counts of known cells below L_u + c for
   every c of the D-070 grid, per sector inside the domain, in one pass
   (exposure.unit_region_counts); the matching shift per sector and for the domain
   (exposure.shift_table) -> shifts.csv.
2. Error models (exposure.fit_models): the final model on E2 + E3, and one per
   leave-one-epoch-out fold (fit on the other epochs) -> error_models.json.
3. Transfer and calibration (pass 2). Per held-out epoch: the products corrected by
   the fold's models (exposure.CorrectedSurface) and their flood-level maps; per held-
   out unit and region, the hindcast area errors of the corrected and the uncorrected
   product, for the "sea_area" (total area bias) and "emergence" (emerged area from
   the epoch's highest level, over the cells known in both) functionals ->
   loeo_errors.csv; the vertical transfer error of the shifts and a summary ->
   transfer.csv.
4. Surfaces at the scenario levels. Per product: uncorrected (heights as the hindcast
   reads them), corrected (final model), fused (DEM inside the observed range,
   exposure.FusedSurface) -> fused_summary.csv. Sea cells per region on a 0.01 m level
   grid, emerged area per scenario (design.SCENARIOS_M below the 2026 level) and its
   conformal interval (exposure.interval; corrected and fused use the corrected
   product's scores) and the outward equivalent levels of the interval bounds; as a
   sensitivity, the interval from the scores of the extrapolating fold alone (the
   held-out epoch with the lowest levels, E3) -> emergence.csv.
5. Exposure per layer, surface, region and scenario at the scenario level and at the
   equivalent levels of the bounds -> exposure.csv.
6. Scenario dates from the projection pathways -> scenario_dates.csv. run.json records
   inputs and options.

Not computed here: any H3 outcome (caspian.h3 applies the rule to emergence.csv).

Usage:
  python scripts/a8_emergence.py <tag> --dem-tag <tag> [--vertical-correction <tif|npz>]
      [--products ...] [--forms linear_hold[,constant,linear]] [--fit-region all|train]
      [--level-2026 <m> | --level-method annual_mean|latest_running_median]
      [--polygons name=path[,name=path]] [--pathways <csv>] [--tile 2048]
Outputs: results/emergence/<tag>/*.csv, error_models.json, run.json; flood-level maps
cached under data/interim/emergence/<tag>/.
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

from caspian import design, products
from caspian import exposure as ex
from caspian import hindcast_tiled as ht
from caspian.config import CRS_EQUAL_AREA, INTERIM, PROCESSED, RAW, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

_spec = importlib.util.spec_from_file_location("a7_hindcast", Path(__file__).resolve().parent / "a7_hindcast.py")
a7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a7)

RES = a7.RES
CELL_KM2 = a7.CELL_KM2
PORT_ACCESS_M = 1_000.0      # proposed: a port is on the sea when the sea lies within 1 km
COAST_M = 5_000.0            # proposed: a coastal settlement has the sea within 5 km
# OSM layer -> (rule, threshold). Roads are not an emergence exposure and are skipped.
# OSM wetlands (reed beds etc.) are an open-licence habitat stand-in, not an authoritative
# habitat map. OSM protected areas are not used: in the domain they hold four features,
# and the Astrakhan Biosphere Reserve is only a point (fetched 2026-10-21); protected
# areas enter with --polygons once the authoritative layers are obtained.
# Access layers also get a search radius, so that the distance to the sea is measured
# (not only compared with the threshold) up to that radius.
OSM_RULES = {"platforms": ("onshore", None, None), "pipelines": ("length", None, None),
             "channels": ("length", None, None), "ports": ("access", PORT_ACCESS_M, 10_000.0),
             "settlements": ("access", COAST_M, 20_000.0), "wetlands": ("area", None, None)}
LEVEL_MARGIN_M = 5.0         # the level grid reaches this far below the deepest scenario
PATHWAYS = RESULTS / "levels" / "projection_pathways.csv"
DAHITI = RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv"


# --- helpers -------------------------------------------------------------------------------

def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def flood_map(surface, seed, tile: int, cache: Path | None, name: str):
    """Flood-level map of ``surface`` from ``seed`` (hindcast_tiled.flood_level_map), as a
    float32 memory map cached under ``cache`` (or an array when ``cache`` is None)."""
    if cache is None:
        return ht.flood_level_map(surface, seed, tile=tile)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}.npy"
    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(np.shape(surface)))
        ht.flood_level_map(surface, seed, tile=tile, out=mm)
        mm.flush()
        del mm
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def finite_of(h):
    return lambda r0, r1, c0, c1: np.isfinite(np.asarray(h[r0:r1, c0:c1], float))


def observed_unit(units, b, ep, transform, shape, static, tile):
    raw, _ = a7.observed_raw(a7.groups_for(units, b, [ep]), transform, shape)
    code = a7.classify(raw, static, tile)
    del raw
    return code


def observed_levels(hindcast_tables: dict | None, sector_names) -> dict | None:
    """{(sector, epoch, bin): observed_level} from the hindcast's displacement table
    (product-independent; the first product's total-variant rows), with the domain the
    OR of its sectors; None without the table."""
    if not hindcast_tables or "displacement" not in hindcast_tables:
        return None
    d = hindcast_tables["displacement"]
    d = d[(d["variant"] == "total") & d["epoch"].isin(["E1", "E2", "E3"]) & d["sector"].isin(list(sector_names))]
    if d.empty:
        return None
    d = d[d["product"] == sorted(d["product"].unique())[0]]
    out = {(r.sector, r.epoch, int(r.bin)): bool(r.observed_level) for r in d.itertuples()}
    for (ep, b), g in d.groupby(["epoch", "bin"]):
        out[("domain", ep, int(b))] = bool(g["observed_level"].astype(bool).any())
    return out


def dem_to_grid(paths, transform, shape) -> np.ndarray:
    """30 m DEM tiles (band 1) on the 60 m hindcast grid: each cell the mean of its
    finite 30 m pixels, NaN if none. Tiles must lie on the 30 m lattice of the grid."""
    import rasterio
    x0, y1 = float(transform[2]), float(transform[5])
    out = np.full(shape, np.nan, np.float32)
    for p in paths:
        with rasterio.open(p) as r:
            a7.check_crs(r.crs.to_wkt(), Path(p).name)
            t = r.transform
            z = r.read(1, masked=True).astype(np.float32).filled(np.nan)
        if t.a != RES / 2 or t.e != -RES / 2 or t.b or t.d:
            raise ValueError(f"{Path(p).name}: not a north-up {RES // 2} m grid")
        c0, r0 = (t.c - x0) / RES, (y1 - t.f) / RES
        if c0 != round(c0) or r0 != round(r0) or z.shape[0] % 2 or z.shape[1] % 2:
            raise ValueError(f"{Path(p).name}: not aligned with the {RES} m hindcast grid")
        r0, c0 = int(round(r0)), int(round(c0))
        zz = z.reshape(z.shape[0] // 2, 2, z.shape[1] // 2, 2)
        n = np.isfinite(zz).sum(axis=(1, 3))
        s = np.nansum(zz, axis=(1, 3))
        agg = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
        rr0, cc0 = max(r0, 0), max(c0, 0)
        rr1, cc1 = min(r0 + agg.shape[0], shape[0]), min(c0 + agg.shape[1], shape[1])
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        sub = agg[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        dst = out[rr0:rr1, cc0:cc1]
        put = np.isfinite(sub) & ~np.isfinite(dst)
        dst[put] = sub[put]
    return out


# --- driver --------------------------------------------------------------------------------

def run(stage2_dir, out_dir, static: dict, heights: dict, transform, *, dem, level_2026: float,
        level_2026_record: dict | None = None, flood_maps: dict | None = None, hindcast_tables: dict | None = None,
        sector_names=SECTOR_ORDER, forms=(ex.PRIMARY_FORM,), fit_epochs=("E2", "E3"), fit_mask=None,
        scenarios=design.SCENARIOS_M, layers: dict | None = None, pathways: pd.DataFrame | None = None,
        extra_curves: dict | None = None, sensitivity: bool = False,
        cache_dir: Path | None = None, tile: int = ht.TILE, meta: dict | None = None, log=print) -> dict:
    """Compute and write the H3 tables (module docstring). ``heights``: {product: heights
    on the grid as the hindcast read them (total variant)}; ``dem``: the waterline DEM on
    the grid; ``layers``: {name: {"geoms": [shapely, grid CRS], "rule", "threshold_m"}};
    ``fit_mask``: optional cells the shifts are counted on (e.g. training blocks; default
    every cell); ``extra_curves``: {id: {"curve": (years, change), "role", "source", ...}}
    dated beside ``pathways`` (eq1_curves). ``sensitivity``: allow a first (final) form other
    than the primary, for the D-081 form sensitivities (written to a separate folder)."""
    t_start = time.time()
    forms = tuple(forms)
    if forms[0] != ex.PRIMARY_FORM and not sensitivity:
        raise ValueError(f"the first form must be the primary one, {ex.PRIMARY_FORM}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sector, outside, seed, domain = static["sector"], static["outside"], static["sea_seed"], static["domain"]
    shape = tuple(sector.shape)
    K = len(sector_names)
    code_of = {s: k + 1 for k, s in enumerate(sector_names)}
    regions = list(sector_names) + ["domain"]
    lab_in = np.where(np.asarray(domain, bool), np.asarray(sector), 0).astype(np.uint8)
    units = a7.stage2_index(stage2_dir)
    if not units:
        raise ValueError(f"no stage-2 tiles with main-export groups in {stage2_dir}")
    ulist = [(ep, b) for b in sorted(units) for ep in sorted(units[b])]
    epochs = sorted({ep for ep, _ in ulist})
    if not set(fit_epochs) <= set(epochs):
        raise ValueError(f"fit epochs {list(fit_epochs)} not all present in {epochs}")
    level = {b: a7._bin_level(b) for b in units}
    finite = {p: finite_of(h) for p, h in heights.items()}
    if flood_maps is None:
        flood_maps = {p: flood_map(h, seed, tile, None if cache_dir is None else cache_dir / "flood",
                                   f"uncorrected_{p}") for p, h in heights.items()}
    obs_lv = observed_levels(hindcast_tables, sector_names)
    fit_known = None if fit_mask is None else (lambda r0, r1, c0, c1: np.asarray(fit_mask[r0:r1, c0:c1], bool))

    # 1. shifts ---------------------------------------------------------------------------------
    rows = []
    for i, (ep, b) in enumerate(ulist):
        code = observed_unit(units, b, ep, transform, shape, static, tile)
        T = products.flood_level(level[b], 0.0, 0.0, ex.C_GRID)
        for p, F in flood_maps.items():
            cnt = ex.unit_region_counts(F, code, outside, finite[p], lab_in, K, T, shape=shape, tile=tile,
                                        mask=fit_known)
            t = ex.shift_table(cnt, round(level[b], 4), sector_names)
            rows.append(t.assign(product=p, epoch=ep, bin=b))
        del code
        log(f"shifts: unit {ep} bin {b} ({i + 1}/{len(ulist)})", flush=True)
    shifts = pd.concat(rows, ignore_index=True)
    shifts["bias0_km2"] = shifts["bias0_cells"] * CELL_KM2
    if obs_lv is not None:
        shifts["observed_level"] = [obs_lv.get((r.region, r.epoch, int(r.bin)), False) for r in shifts.itertuples()]
        olv_source = "hindcast displacement.csv (Amendment 6 item 5)"
    else:
        shifts["observed_level"] = (shifts["n_water"] > 0) & (shifts["n_water"] < shifts["n_known"])
        olv_source = "fallback: the region holds known water and known land at the unit"
    shifts = shifts[["product", "epoch", "bin", "level_m", "region", "shift_m", "matched", "usable", "n_tied",
                     "residual_cells", "n_known", "n_water", "bias0_cells", "bias0_km2", "observed_level"]]
    shifts.to_csv(out_dir / "shifts.csv", index=False)
    check = cross_check(shifts, hindcast_tables) if fit_mask is None else "not run: shifts counted on fit_mask"

    # 2. models ---------------------------------------------------------------------------------
    folds = {e: [x for x in epochs if x != e] for e in epochs if len(epochs) > 1}
    models = {}
    for p in heights:
        sp = shifts[shifts["product"] == p]
        models[p] = {f: {"final": ex.fit_models(sp, regions, epochs=fit_epochs, form=f),
                         "folds": {e: ex.fit_models(sp, regions, epochs=fe, form=f) for e, fe in folds.items()}}
                     for f in forms}
    (out_dir / "error_models.json").write_text(json.dumps(models, indent=1, default=float), encoding="utf-8")

    def corrected(p, rec):
        mods = {code_of[r]: m for r, m in rec["models"].items() if r in code_of}
        return ex.CorrectedSurface(heights[p], sector, mods, rec["lowest_level_m"])

    def cache(sub):
        return None if cache_dir is None else cache_dir / sub

    # 3. leave-one-epoch-out errors ----------------------------------------------------------------
    err_rows = []
    for e in folds:
        fmaps = {(p, f): flood_map(corrected(p, models[p][f]["folds"][e]), seed, tile, cache("loeo"),
                                   f"{p}_{f}_wo{e}_{_sha(models[p][f]['folds'][e])}")
                 for p in heights for f in forms}
        eu = sorted(b for ep, b in ulist if ep == e)
        top = eu[-1]
        code_top = observed_unit(units, top, e, transform, shape, static, tile)
        for b in eu:
            code = code_top if b == top else observed_unit(units, b, e, transform, shape, static, tile)
            surf = [(p, "uncorrected", "none", flood_maps[p]) for p in heights]
            surf += [(p, "corrected", f, fmaps[(p, f)]) for p in heights for f in forms]
            for p, s, f, F in surf:
                c1 = ex.unit_region_counts(F, code, outside, finite[p], lab_in, K, [level[b]], shape=shape, tile=tile)
                c2 = ex.pair_region_counts(F, level[top], level[b], code_top, code, outside, finite[p], lab_in, K,
                                           shape=shape, tile=tile)
                for k, reg in enumerate(regions):
                    sel = slice(None) if reg == "domain" else slice(k, k + 1)
                    pred1, obs1 = int(c1["below"][sel, 0].sum()), int(c1["water"][sel].sum())
                    n1 = int(c1["known"][sel].sum())
                    pc = {kk: int(v[sel].sum()) for kk, v in c2.items()}
                    obs2, pred2 = pc["obs_top"] - pc["obs_u"], pc["pred_top"] - pc["pred_u"]
                    base = {"product": p, "surface": s, "form": f, "fold": e, "epoch": e, "bin": b,
                            "level_m": round(level[b], 4), "region": reg,
                            "observed_level": (obs_lv.get((reg, e, b), False) if obs_lv is not None
                                               else bool(0 < obs1 < n1))}
                    err_rows.append({**base, "functional": "sea_area", "anchor_bin": np.nan, "n_cells": n1,
                                     "pred_km2": pred1 * CELL_KM2, "obs_km2": obs1 * CELL_KM2,
                                     "error_km2": (pred1 - obs1) * CELL_KM2, "is_anchor": False})
                    err_rows.append({**base, "functional": "emergence", "anchor_bin": top, "n_cells": pc["n_common"],
                                     "pred_km2": pred2 * CELL_KM2, "obs_km2": obs2 * CELL_KM2,
                                     "error_km2": (pred2 - obs2) * CELL_KM2, "is_anchor": b == top})
            if b != top:
                del code
        del code_top, fmaps
        log(f"leave-one-epoch-out: {e} held out, {len(eu)} units", flush=True)
    errors = pd.DataFrame(err_rows)
    if len(errors):
        errors["score"] = errors["observed_level"].astype(bool) & ~errors["is_anchor"].astype(bool)
    errors.to_csv(out_dir / "loeo_errors.csv", index=False)
    transfer = transfer_table(shifts, models, errors, forms, folds)
    transfer.to_csv(out_dir / "transfer.csv", index=False)

    # 4. scenario surfaces ---------------------------------------------------------------------
    deltas = [float(d) for d in scenarios]
    lv = ex.level_grid(level_2026, max(deltas) + LEVEL_MARGIN_M)
    final_f = forms[0]
    em_rows, fused_rows, bounds = [], [], {}
    surfaces = {}
    # the fold whose held-out epoch reaches the lowest level: its errors come from a correction
    # extrapolated below the fitted levels, as at the scenarios
    ext_fold = min(folds, key=lambda e: min(b for ep, b in ulist if ep == e)) if folds else None
    low_b = min(b for ep, b in ulist if ep in fit_epochs)          # the lowest observed level of the fit
    low_ep = max(ep for ep, b in ulist if ep in fit_epochs and b == low_b)
    code_low = observed_unit(units, low_b, low_ep, transform, shape, static, tile)
    for p in heights:
        rec = models[p][final_f]["final"]
        lowest = rec["lowest_level_m"]
        corr = corrected(p, rec)
        fused = ex.FusedSurface(dem, corr, code_low, outside, lowest)
        fs = fused.summary(lab_in, K, tile=tile).assign(product=p, region=list(sector_names))
        fused_rows.append(fs)
        key = {"model": _sha(rec), "dem": _sha([float(np.nansum(np.asarray(dem[r0:r1, c0:c1], float)))
                                                 for r0, r1, c0, c1 in ht.tiles(shape, tile)]),
               "low": [low_ep, low_b]}
        F_corr = flood_map(corr, seed, tile, cache("final"), f"{p}_{final_f}_corrected_{_sha(key['model'])}")
        F_fused = flood_map(fused, seed, tile, cache("final"), f"{p}_{final_f}_fused_{_sha(key)}")
        src = fused.source
        ex_mask = lambda r0, r1, c0, c1, s=src: s((slice(r0, r1), slice(c0, c1))) == ex.SRC_EXTRAPOLATED  # noqa: E731
        surfaces[p] = {"uncorrected": flood_maps[p], "corrected": F_corr, "fused": F_fused}
        sea = {s: ex.sea_cells_by_region(F, lab_in, K, lv, shape=shape, tile=tile) for s, F in surfaces[p].items()}
        sea_ex = ex.sea_cells_by_region(F_fused, lab_in, K, lv, mask=ex_mask, shape=shape, tile=tile)
        k26 = int(np.flatnonzero(np.isclose(lv, level_2026, atol=1e-6))[0])
        for s, cells in sea.items():
            scores_s = "uncorrected" if s == "uncorrected" else "corrected"
            for k, reg in enumerate(regions):
                a = cells.sum(axis=0) if reg == "domain" else cells[k]
                a_fused = sea["fused"].sum(axis=0) if reg == "domain" else sea["fused"][k]
                for fn in ex.FUNCTIONALS:
                    base_cells = a[k26] if fn == "emergence" else a_fused[k26]
                    E = ex.emerged_curve(a, lv, level_2026, None if fn == "emergence" else base_cells) * CELL_KM2
                    sc = errors[(errors["product"] == p) & (errors["surface"] == scores_s) & (errors["region"] == reg)
                                & (errors["functional"] == fn) & errors["score"]] if len(errors) else errors
                    if scores_s == "corrected" and len(sc):
                        sc = sc[sc["form"] == final_f]
                    for D in deltas:
                        L = round(level_2026 - D, 6)
                        kL = int(np.flatnonzero(np.isclose(lv, L, atol=1e-6))[0])
                        iv = ex.interval(float(E[kL]), sc["error_km2"].to_numpy(float) if len(sc) else [],
                                         lower=0.0, upper=float(base_cells) * CELL_KM2)
                        eq = ex.equivalent_levels(lv, E, iv["lo"], iv["hi"]) if fn == "emergence" else {}
                        if fn == "emergence":
                            bounds[(p, s, reg, D)] = eq
                        row = {"product": p, "surface": s, "form": final_f if s != "uncorrected" else "none",
                               "region": reg, "functional": fn, "scenario_m": D, "level_m": L,
                               "level_2026_m": level_2026, "lowest_level_m": lowest,
                               "estimate_km2": iv["estimate"], "lo_km2": iv["lo"], "hi_km2": iv["hi"], "q_km2": iv["q"],
                               "n_scores": iv["n_scores"], "baseline_km2": float(base_cells) * CELL_KM2,
                               "scores_from": scores_s, **eq}
                        if ext_fold is not None:            # sensitivity: the extrapolating fold's scores only
                            se = sc[sc["fold"] == ext_fold] if len(sc) else sc
                            ie = ex.interval(float(E[kL]), se["error_km2"].to_numpy(float) if len(se) else [],
                                             lower=0.0, upper=float(base_cells) * CELL_KM2)
                            row.update({"extrapolating_fold": ext_fold, "lo_extrap_km2": ie["lo"],
                                        "hi_extrap_km2": ie["hi"], "q_extrap_km2": ie["q"],
                                        "n_scores_extrap": ie["n_scores"]})
                        if s == "fused":
                            ae = sea_ex.sum(axis=0) if reg == "domain" else sea_ex[k]
                            emerged_ex = (ae[k26] - ae[kL]) * CELL_KM2
                            row["extrapolated_share"] = (emerged_ex / iv["estimate"]) if iv["estimate"] > 0 else np.nan
                        em_rows.append(row)
        log(f"scenarios: {p} done", flush=True)
    del code_low
    emergence = pd.DataFrame(em_rows)
    emergence.to_csv(out_dir / "emergence.csv", index=False)
    fused_summary = pd.concat(fused_rows, ignore_index=True) if fused_rows else pd.DataFrame()
    fused_summary.to_csv(out_dir / "fused_summary.csv", index=False)

    # 5. exposure -----------------------------------------------------------------------------
    exposure = exposure_table(layers or {}, surfaces, bounds, lab_in, sector_names, transform, shape,
                              level_2026, deltas)
    exposure.to_csv(out_dir / "exposure.csv", index=False)

    # 6. scenario dates ------------------------------------------------------------------------
    dates = pd.DataFrame()
    curves = {} if pathways is None or not len(pathways) else ex.pathway_curves(pathways)
    curves.update({k: v["curve"] for k, v in (extra_curves or {}).items()})
    if curves:
        dates = ex.scenario_dates(curves, deltas, base_year=2026.0)
        info = pd.DataFrame(columns=["pathway_id"]) if pathways is None else pathways.drop_duplicates("pathway_id")[
            [c for c in ("pathway_id", "role", "source", "scenario", "statistic", "kind", "ref_year")
             if c in pathways.columns]]
        extra = pd.DataFrame([{"pathway_id": k, **{c: v.get(c) for c in ("role", "source", "scenario", "statistic",
                                                                         "kind", "ref_year")}}
                              for k, v in (extra_curves or {}).items()])
        dates = dates.merge(pd.concat([info, extra], ignore_index=True), on="pathway_id", how="left")
    dates.to_csv(out_dir / "scenario_dates.csv", index=False)

    record = {**(meta or {}), "stage2_dir": str(stage2_dir), "products": list(heights), "forms": list(forms),
              "primary_form": ex.PRIMARY_FORM, "fit_epochs": list(fit_epochs), "folds": folds,
              "fit_region": "every cell" if fit_mask is None else "fit_mask (e.g. training blocks)",
              "level_2026": level_2026_record or {"level_m": level_2026},
              "scenarios_m": deltas, "conformal_level": design.CONFORMAL_LEVEL,
              "observed_level_source": olv_source, "cross_check_extent_domain": check,
              "units": [{"epoch": ep, "bin": b, "level_m": round(level[b], 4)} for ep, b in ulist],
              "constants": {"C_GRID": "D-070, 0.01 m over +-5 m", "MIN_UNITS": ex.MIN_UNITS,
                            "MIN_MONOTONE": ex.MIN_MONOTONE, "PORT_ACCESS_M": PORT_ACCESS_M, "COAST_M": COAST_M,
                            "LEVEL_MARGIN_M": LEVEL_MARGIN_M},
              "layers": {n: {"rule": v["rule"], "threshold_m": v.get("threshold_m"), "radius_m": v.get("radius_m"),
                             "file": v.get("file"), "source": v.get("source"),
                             "n_features": (int(v["samples"]["fid"].nunique()) if "samples" in v else len(v["geoms"]))}
                         for n, v in (layers or {}).items()},
              "grid": {"transform": [float(v) for v in transform[:6]], "shape": list(shape), "crs": CRS_EQUAL_AREA},
              "seconds": round(time.time() - t_start, 1)}
    (out_dir / "run.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return {"shifts": shifts, "models": models, "loeo_errors": errors, "transfer": transfer,
            "emergence": emergence, "fused_summary": fused_summary, "exposure": exposure,
            "scenario_dates": dates, "run": record}


def cross_check(shifts: pd.DataFrame, hindcast_tables: dict | None):
    """Largest |difference| between the domain total area bias here (c = 0) and the
    hindcast's extent.csv "domain" rows of the total variant, in cells (expected 0) and
    in km2 (rounding only)."""
    if not hindcast_tables or "extent" not in hindcast_tables:
        return "not run: no extent table"
    e = hindcast_tables["extent"]
    e = e[(e["variant"] == "total") & (e["sector"] == "domain")][["product", "epoch", "bin", "area_bias_total_km2"]]
    s = shifts[shifts["region"] == "domain"][["product", "epoch", "bin", "bias0_km2"]]
    m = s.merge(e, on=["product", "epoch", "bin"], how="inner")
    if m.empty:
        return "not run: no matching rows"
    cells = np.rint(m["area_bias_total_km2"].to_numpy(float) / CELL_KM2)
    s_cells = m["bias0_km2"].to_numpy(float) / CELL_KM2
    return {"n_rows": int(len(m)), "max_abs_diff_cells": float(np.max(np.abs(np.rint(s_cells) - cells))),
            "max_abs_diff_km2": float((m["bias0_km2"] - m["area_bias_total_km2"]).abs().max())}


def transfer_table(shifts, models, errors, forms, folds) -> pd.DataFrame:
    """Per product, form, fold and region: the vertical transfer error of the shifts
    (predicted minus held-out matching shift, m, over the held-out usable units) and the
    median absolute area errors of the corrected and uncorrected product at the scored
    units of each functional."""
    rows = []
    for p, mf in models.items():
        for f in forms:
            for e in folds:
                rec = mf[f]["folds"][e]
                sp = shifts[(shifts["product"] == p) & (shifts["epoch"] == e) & shifts["usable"]]
                for reg, m in rec["models"].items():
                    r = sp[sp["region"] == reg]
                    row = {"product": p, "form": f, "fold": e, "region": reg, "model_form": m and m["form"],
                           "used_domain_model": m and m.get("used_domain"), "n_heldout_usable": int(len(r))}
                    if m is not None and len(r):
                        d = ex.unit_depths(r["level_m"].to_numpy(float), r["shift_m"].to_numpy(float),
                                           rec["lowest_level_m"])
                        te = ex.predict_shift(m, d) - r["shift_m"].to_numpy(float)
                        row.update({"shift_transfer_median_m": float(np.median(te)),
                                    "shift_transfer_median_abs_m": float(np.median(np.abs(te))),
                                    "share_extrapolated": float(np.mean(d > m["d_hi"]))})
                    if len(errors):
                        for fn in ex.FUNCTIONALS:
                            sel = errors[(errors["product"] == p) & (errors["fold"] == e) & (errors["region"] == reg)
                                         & (errors["functional"] == fn) & errors["score"]]
                            c = sel[(sel["surface"] == "corrected") & (sel["form"] == f)]["error_km2"].abs()
                            u = sel[sel["surface"] == "uncorrected"]["error_km2"].abs()
                            row[f"{fn}_n"] = int(len(c))
                            row[f"{fn}_median_abs_corrected_km2"] = float(c.median()) if len(c) else np.nan
                            row[f"{fn}_median_abs_uncorrected_km2"] = float(u.median()) if len(u) else np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def exposure_table(layers: dict, surfaces: dict, bounds: dict, lab_in, sector_names, transform, shape,
                   level_2026: float, deltas) -> pd.DataFrame:
    """Exposure per layer, product, surface (uncorrected and fused), region and scenario,
    with the interval from the equivalent levels of the area bounds (``bounds``)."""
    rows = []
    regions = list(sector_names) + ["domain"]
    for name, spec in layers.items():
        samples = spec["samples"] if "samples" in spec else ex.feature_samples(spec["geoms"], transform, shape)
        if samples.empty:
            continue
        freg = ex.feature_regions(samples, lab_in)
        n_outside = int((freg == 0).sum())
        rule, thr = spec["rule"], spec.get("threshold_m")
        radius = max(float(spec.get("radius_m") or 0.0), float(thr or 0.0))
        for p, sf in surfaces.items():
            for s in ("uncorrected", "fused"):
                F = sf[s]
                fv = ex.sample_values(F, samples["row"], samples["col"])
                prof = ex.distance_profiles(F, samples["row"], samples["col"], RES, radius) if rule == "access" else None
                cache = {}

                def at(level, fv=fv, prof=prof, cache=cache):
                    if level not in cache:
                        cache[level] = ex.exposure_at(rule, samples, fv, level_2026, level, profiles=prof,
                                                      threshold_m=thr).set_index("fid")
                    return cache[level]

                for k, reg in enumerate(regions):
                    fids = freg.index[freg >= 1] if reg == "domain" else freg.index[freg == k + 1]
                    for D in deltas:
                        L = round(level_2026 - D, 6)
                        eq = bounds.get((p, s, reg, D), {})
                        vals = {}
                        for tag, lev in (("estimate", L), ("lo", eq.get("level_lo_bound")),
                                         ("hi", eq.get("level_hi_bound"))):
                            if lev is None:
                                vals[tag] = {"amount": np.nan, "n_changed": np.nan, "n_considered": np.nan}
                                continue
                            vals[tag] = ex.summarise_exposure(at(min(float(lev), level_2026)).loc[fids], rule)
                        rows.append({"layer": name, "rule": rule, "threshold_m": thr, "product": p, "surface": s,
                                     "region": reg, "scenario_m": D, "level_m": L,
                                     "n_features": int(len(fids)), "n_features_outside_domain": n_outside,
                                     "n_considered": vals["estimate"]["n_considered"],
                                     "amount": vals["estimate"]["amount"], "amount_lo": vals["lo"]["amount"],
                                     "amount_hi": vals["hi"]["amount"], "n_changed": vals["estimate"]["n_changed"],
                                     "n_changed_lo": vals["lo"]["n_changed"], "n_changed_hi": vals["hi"]["n_changed"],
                                     "level_lo_bound": eq.get("level_lo_bound"),
                                     "level_hi_bound": eq.get("level_hi_bound"),
                                     **({"radius_m": radius,
                                         "median_distance_increase_m": vals["estimate"].get("median_distance_increase_m"),
                                         "n_beyond_radius": vals["estimate"].get("n_beyond_radius")}
                                        if rule == "access" else {})})
    return pd.DataFrame(rows)


# --- production entry point -----------------------------------------------------------------

# Pathways that date the scenarios (proposed). Primary: Tier A papers' own projections with a
# stated reference year (Samant & Prange 2023 best-fit totals; their observed 1996-2021
# trend carried forward, which they add to the projections; Hoseini et al. 2025 medians).
# Range: the same papers' inter-model ranges and quartiles. Sensitivity: the faster
# 2006-2021 trend (which Samant & Prange warn may reflect interdecadal variability) and
# Nandini-Weiss et al. 2020 (venue tier undecided). Every other row of the table (no
# stated reference year, secondary citations, upper bounds, absolute levels, rates not
# carried forward by their paper) is not used for dates. Rates need a reference year and
# a horizon: the end of the observed period and the paper's own 2100.
# (pathway_id, role, ref_year override, horizon override)
DATE_PATHWAYS = [
    ("SAM23-07", "primary", None, None), ("SAM23-10", "primary", None, None),
    ("SAM23-14", "primary", 2021, 2100),
    ("HOS25-01", "primary", None, None), ("HOS25-04", "primary", None, None),
    ("HOS25-07", "primary", None, None), ("HOS25-10", "primary", None, None),
    ("SAM23-08", "range", None, None), ("SAM23-09", "range", None, None),
    ("SAM23-11", "range", None, None), ("SAM23-12", "range", None, None),
    ("HOS25-02", "range", None, None), ("HOS25-03", "range", None, None),
    ("HOS25-05", "range", None, None), ("HOS25-06", "range", None, None),
    ("HOS25-08", "range", None, None), ("HOS25-09", "range", None, None),
    ("HOS25-11", "range", None, None), ("HOS25-12", "range", None, None),
    ("SAM23-16", "sensitivity", 2021, 2100),
    ("NW20-01", "sensitivity", None, None), ("NW20-02", "sensitivity", None, None),
]


def read_pathways(path: Path, selection=DATE_PATHWAYS) -> pd.DataFrame | None:
    """The pathways of ``selection`` from results/levels/projection_pathways.csv, in the
    form exposure.pathway_curves reads (pathway_id, kind, ref_year, year, value_m, plus
    role, source, scenario, statistic). The reference year is the override or the year
    the ``reference`` column starts with; the target year the override or ``year``.
    Raises if a selected id is missing or has no reference year or target year."""
    if not path.exists():
        return None
    p = pd.read_csv(path)
    sel = pd.DataFrame(selection, columns=["pathway_id", "role", "ref_override", "year_override"])
    missing = sorted(set(sel["pathway_id"]) - set(p["pathway_id"]))
    if missing:
        raise ValueError(f"{path.name}: selected pathways missing: {missing}")
    m = sel.merge(p, on="pathway_id", how="left")
    ref = pd.to_numeric(m["reference"].astype(str).str.extract(r"^\s*(\d{4})\b")[0], errors="coerce")
    m["ref_year"] = m["ref_override"].astype(float).fillna(ref)
    m["year"] = m["year_override"].astype(float).fillna(pd.to_numeric(m["year"], errors="coerce"))
    m["value_m"] = pd.to_numeric(m["value_m"], errors="coerce")
    bad = m[m[["ref_year", "year", "value_m"]].isna().any(axis=1)]["pathway_id"].tolist()
    if bad:
        raise ValueError(f"{path.name}: no reference year, target year or value for {bad}")
    cols = ["pathway_id", "role", "kind", "ref_year", "year", "value_m", "source", "scenario", "statistic"]
    return m[[c for c in cols if c in m.columns]].reset_index(drop=True)


def eq1_curves(path: Path) -> dict:
    """Sensitivity (proposed): the Samant & Prange 2023 totals with the time path their
    Eq. 1 implies (exposure.trend_plus_quadratic_curve): the best-fit P - E parts by 2100
    (SAM23-05, SAM23-06) growing with the square of the time since 2020, plus the observed
    5.7 cm per year (SAM23-14) that they add, linear from 2020. Their sums by 2100 (8.35
    and 14.11 m) match the stated totals of about 8 and 14 m (SAM23-07, SAM23-10). The shape
    is our reading of Eq. 1; the paper states no intermediate year."""
    if not path.exists():
        return {}
    p = pd.read_csv(path).set_index("pathway_id")
    need = ["SAM23-05", "SAM23-06", "SAM23-14"]
    if any(k not in p.index for k in need):
        return {}
    rate = float(p.loc["SAM23-14", "value_m"])
    out = {}
    for pid, scen in (("SAM23-05", "SSP245"), ("SAM23-06", "SSP585")):
        out[f"{pid}+SAM23-14 (Eq. 1 shape)"] = {
            "curve": ex.trend_plus_quadratic_curve(float(p.loc[pid, "value_m"]), rate, 2020.0, 2100.0),
            "role": "sensitivity", "source": "samant2023", "scenario": scen, "kind": "projection", "ref_year": 2020.0,
            "statistic": "best-fit P-E part (quadratic in time, Eq. 1) + observed 1996-2021 trend (linear)"}
    return out


def _polygons_only(g):
    """Polygon and MultiPolygon rows of a GeoDataFrame, made valid; polygon parts of
    geometry collections are kept, other geometry types dropped (area rule)."""
    import shapely
    geoms = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            geoms.append(None)
            continue
        geom = shapely.make_valid(geom)
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            geoms.append(geom)
        elif hasattr(geom, "geoms"):
            parts = [p for p in shapely.get_parts(geom) if p.geom_type in ("Polygon", "MultiPolygon")]
            geoms.append(shapely.union_all(parts) if parts else None)
        else:
            geoms.append(None)
    g = g.copy()
    g["geometry"] = geoms
    return g[g.geometry.notna()]


def osm_layers(osm_dir: Path) -> dict:
    import geopandas as gpd
    out = {}
    for name, (rule, thr, radius) in OSM_RULES.items():
        f = osm_dir / f"osm_{name}.gpkg"
        if f.exists():
            g = gpd.read_file(f).to_crs(CRS_EQUAL_AREA)
            if rule == "area":
                g = _polygons_only(g)
            out[f"osm_{name}"] = {"geoms": list(g.geometry), "rule": rule, "threshold_m": thr, "radius_m": radius,
                                  "file": f.name}
    return out


# Habitat rasters (CC BY 4.0; retrieved 2026-10-21 into data/raw/habitat/, provenance.json in
# each folder). Class codes as the products document them: WorldCover from the legend tag
# of its GeoTIFFs ("90  Herbaceous wetland"); GWL_FCS30 from Zhang et al. 2023, ESSD 15,
# 265-293, section 7 (181 swamp, 182 marsh, 183 flooded flat, 184 saline); GLWD v2 from
# GLWD_Legend_v2_0.csv (8-32: the wetland classes, rice paddies 33 left out). The choice
# of classes is proposed (habitat stand-ins, not an authoritative habitat map).
HABITAT_RASTERS = {
    "worldcover2021_herbaceous_wetland": {"glob": "esa_worldcover_2021_v200/*_Map_NCaspian_subset.tif",
                                          "classes": (90,), "factor": 3},
    "gwl_fcs30_2020_wetland": {"glob": "gwl_fcs30_2020/GWL_FCS30_2020_*.tif", "classes": (181, 182, 183, 184),
                               "factor": 1},
    "glwd_v2_wetland": {"glob": "glwd_v2/GLWD_v2_0_main_class.tif", "classes": tuple(range(8, 33)), "factor": 1},
}


def raster_layers(transform, shape, root: Path = RAW / "habitat", names=None) -> dict:
    """Habitat share rasters on the grid (exposure.raster_class_fraction) as "area" layers
    of 8-connected patches (exposure.fraction_samples). Missing products are skipped."""
    w, s, e, n = DOMAIN.bounds
    bounds = (w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG)
    out = {}
    for name, spec in HABITAT_RASTERS.items():
        if names is not None and name not in names:
            continue
        paths = sorted(root.glob(spec["glob"]))
        if not paths:
            continue
        frac = ex.raster_class_fraction(paths, spec["classes"], transform, shape, factor=spec["factor"],
                                        bounds=bounds)
        out[name] = {"samples": ex.fraction_samples(frac, transform), "rule": "area", "threshold_m": None,
                     "file": ", ".join(p.name for p in paths), "classes": list(spec["classes"])}
    return out


def polygon_layers(spec: str | None) -> dict:
    import geopandas as gpd
    out = {}
    for item in (spec or "").split(","):
        if not item:
            continue
        name, path = item.split("=", 1)
        g = _polygons_only(gpd.read_file(path).to_crs(CRS_EQUAL_AREA))
        g = g[g.intersects(gpd.GeoSeries([DOMAIN], crs="EPSG:4326").to_crs(CRS_EQUAL_AREA).iloc[0])]
        out[name] = {"geoms": list(g.geometry), "rule": "area", "threshold_m": None, "file": Path(path).name}
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("--dem-tag", required=True)
    ap.add_argument("--vertical-correction", default=None)
    ap.add_argument("--products", default=None, help="default: the products of the hindcast run")
    ap.add_argument("--forms", default=ex.PRIMARY_FORM)
    ap.add_argument("--fit-region", default="all", choices=["all", "train"])
    ap.add_argument("--level-2026", type=float, default=None)
    ap.add_argument("--level-method", default="annual_mean", choices=["annual_mean", "latest_running_median"])
    ap.add_argument("--polygons", default=None, help="name=path[,name=path]: polygon layers (area rule)")
    ap.add_argument("--no-habitat-rasters", action="store_true", help="skip the habitat rasters of HABITAT_RASTERS")
    ap.add_argument("--pathways", default=str(PATHWAYS))
    ap.add_argument("--resampling", default="bilinear", choices=products.METHODS)
    ap.add_argument("--tile", type=int, default=ht.TILE)
    ap.add_argument("--sensitivity-form", default=None, choices=[f for f in ex.FORMS if f != ex.PRIMARY_FORM],
                    help="D-081 sensitivity: this form is the final (fused) one; writes results/emergence/<tag>_<form>")
    a = ap.parse_args(argv)
    if a.sensitivity_form:
        a.forms = a.sensitivity_form
    hdir = RESULTS / "hindcast" / a.tag
    hrun = json.loads((hdir / "run.json").read_text(encoding="utf-8"))
    if float(hrun["level_datum"]["offset_m"]) != 0.0 or hrun.get("product_offsets") not in (None, {},
                                                                                             "none: no constant offset"):
        raise SystemExit("the hindcast run used a level or product offset; H3 needs the DAHITI working datum (D-070)")
    if hrun.get("resampling", "bilinear") != a.resampling:
        raise SystemExit(f"the hindcast used {hrun.get('resampling')} resampling")
    names = a.products.split(",") if a.products else list(hrun["products"])
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    if [float(v) for v in transform[:6]] != hrun["grid"]["transform"][:6] or list(shape) != hrun["grid"]["shape"]:
        raise SystemExit("the hindcast grid differs from this one")
    static = a7.production_static(transform, shape, a.tile)
    # the record-low water envelope of the hindcast's observed extents (D-082)
    a7.ENVELOPE = a7.record_low_water(a7.stage2_index(INTERIM / "stage2" / a.tag), transform, shape)
    corr, corr_rec = None, None
    hcorr = hrun.get("vertical_correction")
    if a.vertical_correction:
        corr, corr_rec = products.correction(a.vertical_correction, transform, shape)
        if not isinstance(hcorr, dict) or hcorr.get("sha256") != corr_rec["sha256"]:
            raise SystemExit("--vertical-correction differs from the hindcast run's correction")
    elif isinstance(hcorr, dict):
        raise SystemExit("the hindcast run used a vertical correction; pass the same file")
    heights, flood_maps = {}, {}
    for p in names:
        h = products.heights(p, transform, shape, a.resampling)
        heights[p] = h if corr is None else products.Corrected(h, corr)
        flood_maps[p] = a7.flood_map_cached(p, heights[p], static["sea_seed"],
                                            {"heights": Path(h.filename).name,
                                             "correction": corr_rec["sha256"] if corr_rec else None}, a.tile)
    fit_mask = None
    if a.fit_region == "train":
        fit_mask = static["domain"] & a7.train_mask(pd.read_csv(a7.BLOCKS), transform, shape)
    dem_paths = sorted((PROCESSED / "dem" / a.dem_tag).glob("dem_*.tif"))
    if not dem_paths:
        raise SystemExit(f"no DEM tiles in {PROCESSED / 'dem' / a.dem_tag}")
    dem = dem_to_grid(dem_paths, transform, shape)
    if a.level_2026 is not None:
        lrec = {"level_m": a.level_2026, "method": "given on the command line"}
    else:
        d = pd.read_csv(DAHITI, sep=";")
        lrec = ex.reference_level(d["datetime"], d["wse"], 2026, a.level_method)
    lrec["datum"] = "DAHITI working datum (D-070)"
    tables = {k: pd.read_csv(hdir / f"{k}.csv") for k in ("extent", "displacement") if (hdir / f"{k}.csv").exists()}
    pathways = read_pathways(Path(a.pathways))
    layers = {**osm_layers(RAW / "osm"), **polygon_layers(a.polygons),
              **({} if a.no_habitat_rasters else raster_layers(transform, shape))}
    meta = {"tag": a.tag, "dem_tag": a.dem_tag, "dem_tiles": len(dem_paths), "hindcast_run": str(hdir / "run.json"),
            "vertical_correction": corr_rec or "none", "resampling": a.resampling,
            "pathways": a.pathways if pathways is not None else "none",
            "sensitivity_form": a.sensitivity_form or "none (primary run)"}
    out_name = a.tag + (f"_{a.sensitivity_form}" if a.sensitivity_form else "")
    run(INTERIM / "stage2" / a.tag, RESULTS / "emergence" / out_name, static, heights, transform, dem=dem,
        level_2026=round(float(lrec["level_m"]), 4), level_2026_record=lrec, flood_maps=flood_maps,
        hindcast_tables=tables, forms=a.forms.split(","), fit_mask=fit_mask, layers=layers, pathways=pathways,
        extra_curves=eq1_curves(Path(a.pathways)), sensitivity=bool(a.sensitivity_form),
        cache_dir=INTERIM / "emergence" / a.tag, tile=a.tile, meta=meta)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
