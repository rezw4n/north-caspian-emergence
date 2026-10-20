"""Write results/params.json from the code constants that define the analysis, so
that Methods text quotes the values the code actually uses (via \metric{params.*})."""

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
    "params.snow_rule_max_pct": (design.SNOW_RULE_MAX_PCT, "%", "caspian.design.SNOW_RULE_MAX_PCT"),
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
    "params.contour_level": (design.CONTOUR_LEVEL, "fraction", "caspian.design.CONTOUR_LEVEL"),
    "params.min_valid_obs": (design.MIN_VALID_OBS, "observations", "caspian.design.MIN_VALID_OBS"),
    "params.dem_res_m": (design.DEM_RES_M, "m", "caspian.design.DEM_RES_M"),
    "params.e2_end": (design.E2_END, "date", "caspian.design.E2_END"),
    "params.emergence_lag_days": (design.EMERGENCE_LAG_DAYS, "days", "caspian.design.EMERGENCE_LAG_DAYS"),
    "params.block_size_km": (design.BLOCK_SIZE_M / 1000, "km", "caspian.design.BLOCK_SIZE_M"),
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
})
out = {}
for k, rec in params.items():
    v, u, src = rec[:3]
    out[k] = {"value": v, "unit": u, "source": src}
    if len(rec) > 3:  # decimal places for the macro (e.g. 45.5, not 45.50)
        out[k]["decimals"] = rec[3]
(ROOT / "results" / "params.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"wrote {len(out)} parameters")
