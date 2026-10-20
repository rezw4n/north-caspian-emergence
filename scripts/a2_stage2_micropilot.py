"""Stage-2 cost micro-pilot (D-026, D-027): the redesigned stage-2 graph
(caspian.pipeline_ee.water_valid, per-acquisition mosaics, one-pass uint16 counts,
equal-area export grid) on a 0.3 x 0.3 degree box inside Landsat-8 path 165 row 28,
2020, grouped by calendar month (a cost proxy, as in the earlier pilots).

Variants (same box and scenes):
  a30  : 30 m
  b60  : 60 m
  d90  : 90 m
  c30q : 30 m, inputs clipped to the box's south-west quadrant (export region is still
         the whole box). If clipping skips the computation outside the quadrant,
         c30q costs about a quarter of a30.
  s60  : Sentinel-2, 60 m, the first 20 scenes of April-October 2020 with scene cloud
         < 60%, at a fixed MNDWI threshold of 0.3 (the threshold value does not affect
         cost; no stage-1 rows exist yet for these scenes).

Thresholds come from the stage-1 pilot table on zone v3; they matter only for cost.

Usage: python scripts/a2_stage2_micropilot.py submit <variant> | status
"""

import json
import sys

import caspian  # noqa: F401
import ee

from caspian import masks, pipeline_ee
from caspian.config import EE_PROJECT, RESULTS

ROOT = f"projects/{EE_PROJECT}/assets/caspian"
BOX = (52.8, 46.2, 53.1, 46.5)
THR = f"{ROOT}/stage1_pilot_L8_165028_2020"
VARIANTS = {"a30": (30, False), "b60": (60, False), "d90": (90, False), "c30q": (30, True), "s60": (60, False)}
STATE = RESULTS / "inventory" / "stage2_micropilot.json"


def build(variant: str):
    res, quarter = VARIANTS[variant]
    box = ee.Geometry.Rectangle(list(BOX))
    w, s, e, n = BOX
    geom = ee.Geometry.Rectangle([w, s, (w + e) / 2, (s + n) / 2]) if quarter else box
    if variant == "s60":
        imgs = (masks.sentinel2(box, "2020-04-01", "2020-11-01", bands=masks.CLASSIFY_BANDS)
                .filter(ee.Filter.lt("cloud_scene", 60)).limit(20).map(lambda im: im.set("thr", 0.3)))
        return _counts(imgs, geom), box, res, imgs.size()
    thr = ee.FeatureCollection(THR).filter(ee.Filter.notNull(["otsu_threshold"]))
    col = (masks.landsat("OLI_L8", box, "2020-01-01", "2021-01-01", bands=masks.CLASSIFY_BANDS)
           .filter(ee.Filter.eq("WRS_PATH", 165)).filter(ee.Filter.eq("WRS_ROW", 28)))
    joined = ee.Join.inner().apply(col, thr, ee.Filter.equals(leftField="system:index", rightField="scene"))
    imgs = ee.ImageCollection(joined.map(lambda f: ee.Image(ee.Feature(f).get("primary")).set(
        "thr", ee.Feature(ee.Feature(f).get("secondary")).get("otsu_threshold"))))
    return _counts(imgs, geom), box, res, imgs.size()


def _counts(imgs, geom):
    wv = imgs.map(lambda im: pipeline_ee.water_valid(im.clip(geom), im.get("thr"))
                  .copyProperties(im, ["system:time_start", "acq"]))
    bands = [pipeline_ee.group_counts(wv.filter(ee.Filter.calendarRange(m, m, "month")), f"m{m:02d}")
             for m in range(1, 13)]
    return ee.Image.cat(bands)


def main(cmd: str, variant: str | None) -> int:
    ee.Initialize(project=EE_PROJECT)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if cmd == "submit":
        img, box, res, n = build(variant)
        transform, dims = pipeline_ee.laea_transform(res, BOX)
        name = f"s2pilot_{variant}"
        t = ee.batch.Export.image.toAsset(image=img, description=name, assetId=f"{ROOT}/{name}",
                                          region=box, crs=pipeline_ee.LAEA_WKT, crsTransform=transform,
                                          maxPixels=1e10)
        t.start()
        state[variant] = {"task": t.id, "res_m": res, "dims": dims, "n_scenes": n.getInfo(), "box": BOX}
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print("submitted", name, t.id, state[variant])
    else:
        for v, st in state.items():
            op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{st['task']}")
            st.update(state=op["metadata"].get("state"), eecu_s=op["metadata"].get("batchEecuUsageSeconds"),
                      error=op.get("error", {}).get("message"))
            print(v, st["state"], st["eecu_s"], st.get("error") or "")
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "status", sys.argv[2] if len(sys.argv) > 2 else None))
