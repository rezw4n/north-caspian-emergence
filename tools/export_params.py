"""Write results/params.json from the code constants that define the analysis, so
that Methods text quotes the values the code actually uses (via \metric{params.*})."""

import ast
import calendar
import inspect
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import caspian  # noqa: E402,F401
import pyproj  # noqa: E402
from caspian import design, masks, pipeline_ee, water  # noqa: E402
from caspian.config import CRS_EQUAL_AREA  # noqa: E402

S2_CS = inspect.signature(masks.sentinel2).parameters["cs_threshold"].default
with warnings.catch_warnings():  # to_dict() warns that a PROJ string is lossy; only lat_0/lon_0 are read
    warnings.simplefilter("ignore", UserWarning)
    _LAEA = pyproj.CRS(CRS_EQUAL_AREA).to_dict()


def _years(period) -> str:
    """'1995--96' for a (start, end) pair of ISO dates in consecutive years."""
    y0, y1 = int(period[0][:4]), int(period[1][:4])
    return f"{y0}--{str(y1)[2:]}" if y1 == y0 + 1 else f"{y0}--{y1}"


SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import a4_scene_levels  # noqa: E402

params = {
    "params.s2_cloudscore_threshold": (S2_CS, "cs_cdf", "caspian.masks.sentinel2"),
    "params.ndvi_veg": (water.NDVI_VEG, "NDVI", "caspian.water.NDVI_VEG"),
    "params.otsu_bins": (water.OTSU_BINS, "bins", "caspian.water.OTSU_BINS"),
    "params.zone_dilate_m": (pipeline_ee.ZONE_DILATE_M, "m", "caspian.pipeline_ee.ZONE_DILATE_M"),
    "params.zone_scale_m": (pipeline_ee.ZONE_SCALE, "m", "caspian.pipeline_ee.ZONE_SCALE"),
    "params.hist_scale_m": (pipeline_ee.HIST_SCALE, "m", "caspian.pipeline_ee.HIST_SCALE"),
    "params.occ_lo_pct": (pipeline_ee.OCC_LO, "%", "caspian.pipeline_ee.OCC_LO"),
    "params.occ_hi_pct": (pipeline_ee.OCC_HI, "%", "caspian.pipeline_ee.OCC_HI"),
    "params.max_gap_days": (a4_scene_levels.MAX_GAP_DAYS, "days", "scripts/a4_scene_levels.MAX_GAP_DAYS"),
    "params.highstand_start": (pipeline_ee.HIGHSTAND[0], "date", "caspian.pipeline_ee.HIGHSTAND"),
    "params.highstand_end": (pipeline_ee.HIGHSTAND[1], "date", "caspian.pipeline_ee.HIGHSTAND"),
    "params.lowstand_start": (pipeline_ee.LOWSTAND[0], "date", "caspian.pipeline_ee.LOWSTAND"),
    "params.lowstand_end": (pipeline_ee.LOWSTAND[1], "date", "caspian.pipeline_ee.LOWSTAND"),
    # zone v6 (D-029), the zone used by stage 1 and stage 2: water in at least this
    # share of clear May-September observations (zone v5 used 100 - HIGH_PCT and
    # 100 - LOW_PCT, the same 10% and 90%, with MNDWI > 0 as the only water test)
    "params.high_extent_min_water_pct": (round(100 * pipeline_ee.HIGH_MIN_FRAC), "%",
                                         "100 * caspian.pipeline_ee.HIGH_MIN_FRAC"),
    "params.low_extent_min_water_pct": (round(100 * pipeline_ee.LOW_MIN_FRAC), "%",
                                        "100 * caspian.pipeline_ee.LOW_MIN_FRAC"),
    "params.dark_water_reflectance": (pipeline_ee.DARK_WATER, "reflectance", "caspian.pipeline_ee.DARK_WATER"),
    "params.conn_scale_km": (pipeline_ee.CONN_SCALE // 1000, "km", "caspian.pipeline_ee.CONN_SCALE"),
    "params.sea_radius_km": (pipeline_ee.SEA_RADIUS_M // 1000, "km", "caspian.pipeline_ee.SEA_RADIUS_M"),
    "params.sea_min_water_pct": (round(100 * pipeline_ee.SEA_MIN_FRAC), "%", "100 * caspian.pipeline_ee.SEA_MIN_FRAC"),
    "params.conn_max_km": (pipeline_ee.CONN_MAX_M // 1000, "km", "caspian.pipeline_ee.CONN_MAX_M"),
    "params.highstand_label": (_years(pipeline_ee.HIGHSTAND), "years", "caspian.pipeline_ee.HIGHSTAND"),
    "params.lowstand_label": (_years(pipeline_ee.LOWSTAND), "years", "caspian.pipeline_ee.LOWSTAND"),
    "params.min_obs_high": (pipeline_ee.MIN_OBS_HIGH, "observations", "caspian.pipeline_ee.MIN_OBS_HIGH"),
    "params.zone_tile_km": (pipeline_ee.ZONE_TILE_M // 1000, "km", "caspian.pipeline_ee.ZONE_TILE_M"),
    "params.mndwi_min": (int(water.MNDWI_RANGE[0]), "MNDWI", "caspian.water.MNDWI_RANGE"),
    "params.mndwi_max": (int(water.MNDWI_RANGE[1]), "MNDWI", "caspian.water.MNDWI_RANGE"),
    # snow and ice (PREREG section 3; LIS pass-1 test, Gascoin et al. 2019)
    "params.ndsi_snow": (masks.NDSI_SNOW, "NDSI", "caspian.masks.NDSI_SNOW"),
    "params.red_snow": (masks.RED_SNOW, "reflectance", "caspian.masks.RED_SNOW"),
    "params.s2_scl_snow": (masks.S2_SCL_SNOW, "SCL class", "caspian.masks.S2_SCL_SNOW"),
    # months in which snow/ice pixels are masked (PREREG Amendments 3 and 9; D-048)
    "params.snow_season_first_month": (calendar.month_name[masks.NDSI_MONTHS[0]], "month",
                                       "caspian.masks.NDSI_MONTHS"),
    "params.snow_season_last_month": (calendar.month_name[masks.NDSI_MONTHS[-1]], "month",
                                      "caspian.masks.NDSI_MONTHS"),
    "params.snow_rule_max_pct": (design.SNOW_RULE_MAX_PCT, "%", "caspian.design.SNOW_RULE_MAX_PCT", 0),
    # D-002 equal-area projection centre
    "params.laea_lat0_deg": (_LAEA["lat_0"], "degrees N", "caspian.config.CRS_EQUAL_AREA", 1),
    "params.laea_lon0_deg": (_LAEA["lon_0"], "degrees E", "caspian.config.CRS_EQUAL_AREA"),
}
# pre-registered design constants (caspian.design; PREREG sections 4-10)
params.update({
    "params.bin_width_m": (design.BIN_WIDTH_M, "m", "caspian.design.BIN_WIDTH_M"),
    "params.bin_width_sens_lo_m": (design.BIN_WIDTH_SENS_M[0], "m", "caspian.design.BIN_WIDTH_SENS_M"),
    "params.bin_width_sens_hi_m": (design.BIN_WIDTH_SENS_M[1], "m", "caspian.design.BIN_WIDTH_SENS_M"),
    "params.bin_origin_m": (design.BIN_ORIGIN_M, "m", "caspian.design.BIN_ORIGIN_M"),
    "params.contour_level": (design.CONTOUR_LEVEL, "fraction", "caspian.design.CONTOUR_LEVEL", 1),
    "params.min_valid_obs": (design.MIN_VALID_OBS, "observations", "caspian.design.MIN_VALID_OBS"),
    "params.dem_res_m": (design.DEM_RES_M, "m", "caspian.design.DEM_RES_M"),
    "params.e2_end": (design.E2_END, "date", "caspian.design.E2_END"),
    "params.emergence_lag_days": (design.EMERGENCE_LAG_DAYS, "days", "caspian.design.EMERGENCE_LAG_DAYS"),
    "params.block_size_km": (design.BLOCK_SIZE_M / 1000, "km", "caspian.design.BLOCK_SIZE_M", 0),
    "params.split_train": (design.SPLIT["train"], "fraction", "caspian.design.SPLIT"),
    "params.split_calib": (design.SPLIT["calib"], "fraction", "caspian.design.SPLIT"),
    "params.split_test": (design.SPLIT["test"], "fraction", "caspian.design.SPLIT"),
    "params.seed": (design.SEED, "", "caspian.design.SEED"),
    "params.bootstrap_n": (design.BOOTSTRAP_N, "resamples", "caspian.design.BOOTSTRAP_N"),
    "params.conformal_level": (design.CONFORMAL_LEVEL, "", "caspian.design.CONFORMAL_LEVEL"),
    "params.mondrian_min_n": (design.MONDRIAN_MIN_N, "points", "caspian.design.MONDRIAN_MIN_N"),
    "params.permutations_n": (design.PERMUTATIONS_N, "permutations", "caspian.design.PERMUTATIONS_N"),
    "params.alpha": (design.ALPHA, "", "caspian.design.ALPHA"),
    "params.edge_otsu_sample_n": (design.EDGE_OTSU_SAMPLE_N, "scenes", "caspian.design.EDGE_OTSU_SAMPLE_N"),
    "params.block_size_sens_lo_km": (design.BLOCK_SIZE_SENS_M[0] / 1000, "km", "caspian.design.BLOCK_SIZE_SENS_M", 1),
    "params.block_size_sens_hi_km": (design.BLOCK_SIZE_SENS_M[-1] / 1000, "km", "caspian.design.BLOCK_SIZE_SENS_M", 0),
})


# --- Methods constants of A3 (threshold rule), A5 (setup screening), A6 (waterlines,
# DEM, validation) and A7 (hindcast), read from the scripts and modules that apply
# them. Scripts that import Earth Engine or read data at import are parsed, not run.
def _const(rel: str, name: str):
    """Module-level literal ``name`` of the script ``rel`` (parsed with ast, never executed)."""
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} is not a module-level literal of {rel}")


def _and_list(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _min_bins_rho() -> int:
    """Fewest valid level bins for which a6_waterlines._rho returns a value (found by
    running it on a monotonic one-pixel stack, so the text follows the code)."""
    for k in range(2, 50):
        stack = [(b, np.full((1, 1), b / k)) for b in range(k)]
        if np.isfinite(a6_waterlines._rho(stack)[0, 0]):
            return k
    raise RuntimeError("rho_level never defined")


def _pressure_window_h() -> int:
    """Length of the pressure-anomaly window of caspian.surge.forcing_features, read
    from the feature names it returns (dp_<a>_<b>)."""
    t = pd.date_range("2020-01-01", periods=72, freq="h")
    cols = pd.MultiIndex.from_product([["s"], ["u", "v", "msl"]])
    f = surge.forcing_features(pd.DataFrame(1.0, index=t, columns=cols), "s", t[-1:])
    (name,) = [c for c in f.columns if c.startswith("dp_")]
    a, b = (int(x) for x in name.split("_")[1:])
    return b - a


import a3_thresholds  # noqa: E402
import a5_setup_model  # noqa: E402
import a6_dem  # noqa: E402
import a6_screening  # noqa: E402
import a6_stage2_plan  # noqa: E402
import a6_static_rule  # noqa: E402
import a6_waterlines  # noqa: E402
import a7_beyond_state  # noqa: E402
import a7_hindcast  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from caspian import blocking, conformal, h1, surge, thresholds, validation, waterlines  # noqa: E402

if tuple(a3_thresholds.ICE_MONTHS) != tuple(a6_stage2_plan.ICE_MONTHS):
    raise SystemExit("ice months differ between a3_thresholds.py and a6_stage2_plan.py")
_taus = [t for t in a6_screening.TAUS if np.isfinite(t)]
if sorted(_taus) != sorted(a6_stage2_plan.SETUP_CLASS_EDGES) or sorted(_taus) != sorted(a6_screening.EDGES):
    raise SystemExit("screening candidates and setup-class edges differ (a6_screening.py, a6_stage2_plan.py)")
_stage2_res = _const("scripts/a6_stage2_export.py", "RES")
if _stage2_res != a7_hindcast.RES:
    raise SystemExit("stage-2 grid of a6_stage2_export.py and a7_hindcast.py differ")
if (waterlines.STATIC_MIN_ETA, waterlines.STATIC_MAX_VALLEY) != (a3_thresholds.MIN_ETA, a3_thresholds.MAX_VALLEY):
    raise SystemExit("static-contour split validity differs from the threshold rule (D-075 says they are equal)")
if _const("scripts/a6_vegetation.py", "VEG_SECTOR_FRACTION") != design.VEGETATED_SECTOR_FRACTION:
    raise SystemExit("vegetated-sector fraction differs between a6_vegetation.py and caspian.design")
_em = validation.error_metrics(np.arange(101.0))
_err_pct = (round(_em["p05"]), round(_em["p95"]))
_ci = {inspect.signature(f).parameters["conf"].default
       for f in (validation.percentile_interval, validation.coverage_on_test, conformal.coverage)}
if len(_ci) != 1:
    raise SystemExit("confidence levels of the validation intervals differ")
_ci_pct = round(100 * _ci.pop())
_win = surge.WINDOWS_H
_cg = a7_hindcast.C_GRID
params.update({
    # A3 per-acquisition threshold rule (D-061) and ice season (D-062)
    "params.thr_min_pixels": (a3_thresholds.MIN_N, "pixels", "scripts/a3_thresholds.MIN_N"),
    "params.thr_min_class_pct": (round(100 * a3_thresholds.MIN_SHARE), "%", "100 * scripts/a3_thresholds.MIN_SHARE"),
    "params.thr_min_eta": (a3_thresholds.MIN_ETA, "", "scripts/a3_thresholds.MIN_ETA", 2),
    "params.thr_max_valley": (a3_thresholds.MAX_VALLEY, "", "scripts/a3_thresholds.MAX_VALLEY", 2),
    "params.thr_lo": (a3_thresholds.THR_RANGE[0], "MNDWI", "scripts/a3_thresholds.THR_RANGE", 0),
    "params.thr_hi": (a3_thresholds.THR_RANGE[1], "MNDWI", "scripts/a3_thresholds.THR_RANGE", 1),
    "params.thr_smooth_bins": (thresholds.SMOOTH, "bins", "caspian.thresholds.SMOOTH"),
    "params.ice_first_month": (calendar.month_name[a3_thresholds.ICE_MONTHS[0]], "month",
                               "scripts/a3_thresholds.ICE_MONTHS"),
    "params.ice_last_month": (calendar.month_name[a3_thresholds.ICE_MONTHS[-1]], "month",
                              "scripts/a3_thresholds.ICE_MONTHS"),
    # A5 setup model (D-065, D-066) and screening threshold (D-064)
    "params.setup_windows_h": (_and_list([f"{a}--{b}" for a, b in _win]), "h", "caspian.surge.WINDOWS_H"),
    "params.setup_lag_max_h": (max(b for _, b in _win), "h", "caspian.surge.WINDOWS_H"),
    "params.setup_pressure_window_h": (_pressure_window_h(), "h", "caspian.surge.forcing_features (dp feature)"),
    "params.setup_pass_gap_min": (a5_setup_model.PASS_GAP_MIN, "minutes", "scripts/a5_setup_model.PASS_GAP_MIN"),
    "params.setup_min_pass_px": (a5_setup_model.MIN_PASS_PX, "pixels", "scripts/a5_setup_model.MIN_PASS_PX"),
    "params.setup_min_block_passes": (a5_setup_model.MIN_BLOCK_PASSES, "passes",
                                      "scripts/a5_setup_model.MIN_BLOCK_PASSES"),
    "params.setup_max_block_nmad_m": (a5_setup_model.MAX_BLOCK_NMAD, "m", "scripts/a5_setup_model.MAX_BLOCK_NMAD", 1),
    "params.setup_max_robust_z": (a5_setup_model.MAX_ROBUST_Z, "", "scripts/a5_setup_model.MAX_ROBUST_Z", 0),
    "params.setup_ridge_n": (len(a5_setup_model.RIDGE_GRID), "values", "scripts/a5_setup_model.RIDGE_GRID"),
    "params.setup_ridge_max": (max(a5_setup_model.RIDGE_GRID), "", "scripts/a5_setup_model.RIDGE_GRID", 0),
    "params.setup_cv_folds": (a5_setup_model.N_FOLDS, "folds", "scripts/a5_setup_model.N_FOLDS"),
    "params.swot_min_water_frac": (_const("scripts/a5_swot_sector_levels.py", "MIN_WATER_FRAC"), "fraction",
                                   "scripts/a5_swot_sector_levels.MIN_WATER_FRAC", 1),
    "params.swot_min_pixels": (_const("scripts/a5_swot_sector_levels.py", "MIN_PIXELS"), "pixels",
                               "scripts/a5_swot_sector_levels.MIN_PIXELS"),
    "params.sea_mask_res_km": (_const("scripts/a5_sea_mask.py", "SCALE") // 1000, "km", "scripts/a5_sea_mask.SCALE"),
    "params.tau_candidates_m": (_and_list([f"{t:.2f}" for t in _taus]), "m", "scripts/a6_screening.TAUS"),
    "params.tau_n_candidates": (len(_taus), "thresholds", "scripts/a6_screening.TAUS (finite values)"),
    # A6 stage 2, waterlines, static-contour rule (D-075), DEM tiling (D-069)
    "params.max_scene_cloud_pct": (_const("scripts/a3_stage1_interactive.py", "MAX_CLOUD"), "%",
                                   "scripts/a3_stage1_interactive.MAX_CLOUD (stage-1 scene selection, D-028)"),
    "params.stage2_res_m": (_stage2_res, "m", "scripts/a6_stage2_export.RES (= scripts/a7_hindcast.RES)"),
    "params.rho_min_bins": (_min_bins_rho(), "bins", "scripts/a6_waterlines._rho (run on a test stack)"),
    "params.static_rho_lo": (waterlines.STATIC_RHO_RANGE[0], "", "caspian.waterlines.STATIC_RHO_RANGE", 1),
    "params.static_rho_hi": (waterlines.STATIC_RHO_RANGE[1], "", "caspian.waterlines.STATIC_RHO_RANGE", 1),
    "params.static_rho_fallback": (waterlines.STATIC_RHO_FALLBACK, "", "caspian.waterlines.STATIC_RHO_FALLBACK", 1),
    "params.static_steep_m": (a6_static_rule.STEEP_M, "m", "scripts/a6_static_rule.STEEP_M", 0),
    "params.veg_cell_share": (_const("scripts/a6_vegetation.py", "VEG_CELL_SHARE"), "fraction",
                              "scripts/a6_vegetation.VEG_CELL_SHARE", 1),
    "params.veg_sector_pct": (round(100 * design.VEGETATED_SECTOR_FRACTION), "%",
                              "100 * caspian.design.VEGETATED_SECTOR_FRACTION (= scripts/a6_vegetation.VEG_SECTOR_FRACTION)"),
    "params.dem_tile_km": (a6_dem.TILE_M // 1000, "km", "scripts/a6_dem.TILE_M"),
    "params.dem_overlap_km": (a6_dem.OVERLAP_M // 1000, "km", "scripts/a6_dem.OVERLAP_M"),
    # A6 validation (D-069; PREREG sections 2 and 5-8)
    "params.gross_error_m": (validation.GROSS_ERROR_M, "m", "caspian.validation.GROSS_ERROR_M", 0),
    "params.dry_radius_px_atl06": (validation.DRY_RADIUS_PX["ATL06"], "pixels", "caspian.validation.DRY_RADIUS_PX"),
    "params.dry_radius_px_atl08": (validation.DRY_RADIUS_PX["ATL08"], "pixels", "caspian.validation.DRY_RADIUS_PX"),
    "params.geoid_max_gap_s": (validation.GEOID_MAX_GAP_S, "s", "caspian.validation.GEOID_MAX_GAP_S", 1),
    "params.contour_density_radius_m": (validation.CONTOUR_RADIUS_M, "m", "caspian.validation.CONTOUR_RADIUS_M", 0),
    "params.abs_resid_floor_mm": (1000 * validation.ABS_RESID_FLOOR_M, "mm", "caspian.validation.ABS_RESID_FLOOR_M", 0),
    "params.vario_bin_km": (validation.VARIO_BIN_M / 1000, "km", "caspian.validation.VARIO_BIN_M", 1),
    "params.vario_max_lag_km": (validation.VARIO_MAX_LAG_M / 1000, "km", "caspian.validation.VARIO_MAX_LAG_M", 0),
    "params.vario_short_lag_km": (validation.VARIO_SHORT_LAG_M / 1000, "km", "caspian.validation.VARIO_SHORT_LAG_M", 1),
    "params.vario_max_pairs": (validation.VARIO_MAX_PAIRS, "pairs", "caspian.validation.VARIO_MAX_PAIRS"),
    "params.nmad_scale": (validation.NMAD_SCALE, "", "caspian.validation.NMAD_SCALE", 4),
    "params.err_pct_lo": (_err_pct[0], "percentile", "caspian.validation.error_metrics (p05, run on 0..100)"),
    "params.err_pct_hi": (_err_pct[1], "percentile", "caspian.validation.error_metrics (p95, run on 0..100)"),
    "params.ci_pct": (_ci_pct, "%", "caspian.validation.percentile_interval and caspian.conformal.coverage (conf default)"),
    "params.structure_perm_n": (inspect.signature(blocking.spatial_structure_test).parameters["n_perm"].default,
                                "permutations", "caspian.blocking.spatial_structure_test (n_perm default)"),
    # robustness variant of the dryness test fixed in D-069; no code constant yet
    "params.emergence_lag_sens_days": (365, "days", "docs/DECISIONS.md D-069 (robustness check; not yet in code)"),
    # A7 hindcast (D-067, D-070, D-072)
    "params.hindcast_margin_deg": (a7_hindcast.MARGIN_DEG, "degrees", "scripts/a7_hindcast.MARGIN_DEG", 1),
    "params.shape_step_m": (float(np.round(np.min(np.diff(_cg)), 6)), "m", "scripts/a7_hindcast.C_GRID", 2),
    "params.shape_max_m": (float(np.max(np.abs(_cg))), "m", "scripts/a7_hindcast.C_GRID", 0),
    "params.h1_min_block_n": (h1.MIN_BLOCK_N, "distances", "caspian.h1.MIN_BLOCK_N"),
    "params.beyond_inner_km": (a7_beyond_state.INNER_M // 1000, "km", "scripts/a7_beyond_state.INNER_M"),
    # display constants quoted in the figure captions
    "params.fig_scale_bar_km": (_const("figures/fig_study_area.py", "SCALE_KM"), "km", "figures/fig_study_area.SCALE_KM"),
    "params.fig_graticule_deg": (_const("figures/fig_study_area.py", "GRATICULE_DEG"), "degrees",
                                 "figures/fig_study_area.GRATICULE_DEG"),
})
out = {}
for k, rec in params.items():
    v, u, src = rec[:3]
    out[k] = {"value": v, "unit": u, "source": src}
    if len(rec) > 3:  # decimal places for the macro (e.g. 45.5, not 45.50)
        out[k]["decimals"] = rec[3]
(ROOT / "results" / "params.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"wrote {len(out)} parameters")
