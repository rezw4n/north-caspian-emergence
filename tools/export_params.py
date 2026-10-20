"""Write results/params.json from the code constants that define the analysis, so
that Methods text quotes the values the code actually uses (via \metric{params.*})."""

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import caspian  # noqa: E402,F401
from caspian import masks, pipeline_ee, water  # noqa: E402

S2_CS = inspect.signature(masks.sentinel2).parameters["cs_threshold"].default
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
    "params.lowstand_start": (pipeline_ee.LOWSTAND[0], "date", "caspian.pipeline_ee.LOWSTAND"),
    "params.lowstand_end": (pipeline_ee.LOWSTAND[1], "date", "caspian.pipeline_ee.LOWSTAND"),
}
out = {k: {"value": v, "unit": u, "source": s} for k, (v, u, s) in params.items()}
(ROOT / "results" / "params.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"wrote {len(out)} parameters")
