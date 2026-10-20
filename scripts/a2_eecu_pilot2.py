"""A2 pilot, redesigned pipeline (D-015): stage-2 aggregation for Landsat 8 path 165
row 28, 2020, using the stage-1 thresholds already exported
(assets/caspian/pilot_L8_165028_2020_v2_thr) and the stage-0 active zone.
As in the first pilot, grouping is by calendar month (a cost proxy only).

Two variants over the same export footprint (D-024 decision rule):
  full    : the whole v3 zone
  quarter : the v3 zone cut to the footprint's south-west quadrant
If cost scales with zone area, quarter costs about a quarter of full.

Usage: python scripts/a2_eecu_pilot2.py submit [full|quarter] | status [full|quarter]
"""

import json
import sys

import caspian  # noqa: F401
import ee

from caspian import masks, pipeline_ee
from caspian.config import EE_PROJECT, RESULTS

ROOT = f"projects/{EE_PROJECT}/assets/caspian"
NAMES = {"full": "pilot2_L8_165028_2020_stage2", "quarter": "pilot2_L8_165028_2020_stage2_q"}


def state_file(variant: str):
    suffix = "" if variant == "full" else f"_{variant}"
    return RESULTS / "inventory" / f"eecu_pilot2_task{suffix}.json"


def build(variant: str):
    dom = pipeline_ee.domain_geometry()
    zone = ee.Image(f"{ROOT}/active_zone_v3").eq(1)
    thr = ee.FeatureCollection(f"{ROOT}/pilot_L8_165028_2020_v2_thr").filter(
        ee.Filter.eq("threshold_source", "otsu"))
    col = (masks.landsat("OLI_L8", dom, "2020-01-01", "2021-01-01")
           .filter(ee.Filter.eq("WRS_PATH", 165)).filter(ee.Filter.eq("WRS_ROW", 28)))
    joined = ee.Join.inner().apply(col, thr, ee.Filter.equals(leftField="system:index", rightField="id"))
    joined = ee.ImageCollection(joined.map(
        lambda f: ee.Image(ee.Feature(f).get("primary")).set(
            "thr", ee.Feature(ee.Feature(f).get("secondary")).get("otsu_threshold"))))
    classified = joined.map(lambda im: pipeline_ee.classify_with_threshold(im, im.get("thr"))
                            .copyProperties(im, ["system:time_start"]))
    footprint = ee.Image(col.first()).geometry().intersection(dom, 1)
    if variant == "quarter":
        ring = footprint.bounds(1).coordinates().get(0).getInfo()
        (x0, y0), (x1, y1) = ring[0], ring[2]
        q = ee.Geometry.Rectangle([x0, y0, (x0 + x1) / 2, (y0 + y1) / 2])
        zone = zone.updateMask(ee.Image(1).clip(q))
    bands = [pipeline_ee.group_counts(classified.filter(ee.Filter.calendarRange(m, m, "month")),
                                      zone, f"m{m:02d}") for m in range(1, 13)]
    return ee.Image.cat(bands).clip(footprint), footprint, joined.size()


def main(cmd: str, variant: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    name, st_path = NAMES[variant], state_file(variant)
    if cmd == "submit":
        img, fp, n = build(variant)
        t = ee.batch.Export.image.toAsset(image=img, description=name, assetId=f"{ROOT}/{name}",
                                          region=fp, scale=30, crs="EPSG:4326", maxPixels=1e11)
        t.start()
        st_path.write_text(json.dumps({"task": t.id, "variant": variant, "n_scenes": n.getInfo()}, indent=1),
                           encoding="utf-8")
        print("submitted", variant, t.id)
    else:
        st = json.loads(st_path.read_text(encoding="utf-8"))
        op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{st['task']}")
        st.update(state=op["metadata"].get("state"), eecu_s=op["metadata"].get("batchEecuUsageSeconds"),
                  error=op.get("error", {}).get("message"))
        st_path.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "status", sys.argv[2] if len(sys.argv) > 2 else "full"))
