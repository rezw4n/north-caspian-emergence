"""A2 pilot: run the A3 classification on one Landsat path/row-year and one
Sentinel-2 tile-season as Earth Engine batch exports. Each finished task reports
the EECU it consumed, which scales to the full-run estimate (G2).

The pilot aggregates per-scene classes by calendar month. The grouping is only a
cost proxy for the level-bin aggregation of A6, which needs the DAHITI levels; no
scientific result is drawn from the pilot rasters.

Usage: python scripts/a2_eecu_pilot.py submit | status
"""

from __future__ import annotations

import json
import sys
import time

import caspian  # noqa: F401
import ee
from shapely.geometry import mapping

from caspian import masks, water
from caspian.config import EE_PROJECT, RESULTS
from caspian.geography import DOMAIN

ASSET_ROOT = f"projects/{EE_PROJECT}/assets/caspian"
STATE = RESULTS / "inventory" / "eecu_pilot_tasks.json"
PILOTS = [
    {"name": "pilot_L8_165028_2020", "sensor": "OLI_L8", "path": 165, "row": 28,
     "start": "2020-01-01", "end": "2021-01-01", "scale": 30},
    {"name": "pilot_S2_39TXM_2020JJA", "sensor": "MSI_S2", "tile": "39TXM",
     "start": "2020-06-01", "end": "2020-09-01", "scale": 20},
]


def gsw_zone():
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    return gsw.select("max_extent").focalMax(2000, "circle", "meters").gt(0)


def build(p: dict):
    domain = ee.Geometry(mapping(DOMAIN))
    if p["sensor"] == "MSI_S2":
        col = masks.sentinel2(domain, p["start"], p["end"]).filter(ee.Filter.eq("MGRS_TILE", p["tile"]))
    else:
        col = (masks.landsat(p["sensor"], domain, p["start"], p["end"])
               .filter(ee.Filter.eq("WRS_PATH", p["path"])).filter(ee.Filter.eq("WRS_ROW", p["row"])))
    footprint = ee.Image(col.first()).geometry().intersection(domain, 1)
    zone = gsw_zone()
    classified = col.map(lambda im: water.ee_classify(im, footprint, zone, scale=p["scale"]))
    bands = []
    for m in range(1, 13):
        sub = classified.filter(ee.Filter.calendarRange(m, m, "month")).select("class")
        w = sub.map(lambda im: im.eq(water.WATER)).sum().unmask(0).uint8().rename(f"w{m:02d}")
        v = sub.map(lambda im: im.lte(water.WATER)).sum().unmask(0).uint8().rename(f"v{m:02d}")
        bands += [w, v]
    img = ee.Image.cat(bands).clip(footprint)
    thresholds = classified.map(lambda im: ee.Feature(None, {
        "id": im.get("system:index"), "time": im.get("system:time_start"),
        "otsu_threshold": im.get("otsu_threshold"), "otsu_n": im.get("otsu_n"),
        "threshold_source": im.get("threshold_source")}))
    return img, ee.FeatureCollection(thresholds), footprint, col.size()


def submit() -> None:
    ee.Initialize(project=EE_PROJECT)
    try:
        ee.data.getAsset(ASSET_ROOT)
    except ee.EEException:
        ee.data.createAsset({"type": "FOLDER"}, ASSET_ROOT)
    state = {"submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": []}
    for p in PILOTS:
        img, table, footprint, n = build(p)
        n = n.getInfo()
        t1 = ee.batch.Export.image.toAsset(image=img, description=p["name"],
                                           assetId=f"{ASSET_ROOT}/{p['name']}", region=footprint,
                                           scale=p["scale"], crs="EPSG:4326", maxPixels=1e11)
        t2 = ee.batch.Export.table.toAsset(collection=table, description=p["name"] + "_thr",
                                           assetId=f"{ASSET_ROOT}/{p['name']}_thr")
        t1.start()
        t2.start()
        state["tasks"].append({**p, "n_scenes": n, "image_task": t1.id, "table_task": t2.id})
        print(f"submitted {p['name']}: {n} scenes; tasks {t1.id}, {t2.id}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def status(wait: bool = False) -> int:
    ee.Initialize(project=EE_PROJECT)
    state = json.loads(STATE.read_text(encoding="utf-8"))
    while True:
        done = True
        for t in state["tasks"]:
            for key in ("image_task", "table_task"):
                op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{t[key]}")
                md = op.get("metadata", {})
                t[key + "_state"] = md.get("state")
                t[key + "_eecu_s"] = md.get("batchEecuUsageSeconds")
                t[key + "_error"] = op.get("error", {}).get("message")
                if md.get("state") not in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    done = False
        if done or not wait:
            break
        time.sleep(60)
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    for t in state["tasks"]:
        print(t["name"], t["n_scenes"], "scenes |",
              "image:", t.get("image_task_state"), t.get("image_task_eecu_s"), "EECU-s |",
              "table:", t.get("table_task_state"), t.get("table_task_eecu_s"), "EECU-s",
              t.get("image_task_error") or "", t.get("table_task_error") or "")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "submit":
        submit()
    else:
        sys.exit(status(wait=(cmd == "wait")))
