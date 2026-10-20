"""A3 stage 1 (D-015, D-027): per-scene table of the Otsu MNDWI threshold (fixed-bin
histogram inside active zone v6 (D-029) at pipeline_ee.HIST_SCALE, 240 m since D-028) and
in-zone clear / snow-ice pixel counts,
for every Landsat and Sentinel-2 scene over the domain. Snow and ice come from the
product QA (Landsat QA_PIXEL bit 5, Sentinel-2 SCL class 11) or the LIS pass-1 NDSI
test (caspian.masks); zone_snow_ndsi is the NDSI test alone. Exported as Earth
Engine table assets in sensor x period chunks; fetched as CSV. Not used for production:
batch table exports under-report EECU, so stage 1 runs as profiled interactive batches
(scripts/a3_stage1_interactive.py, D-028).

Usage: python scripts/a3_stage1_thresholds.py pilot | submit | status | fetch
  pilot  : Landsat-8 path 165 row 28, 2020 only (cost measurement)
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import ee
import requests

from caspian import masks, pipeline_ee
from caspian.config import EE_PROJECT, INTERIM, RESULTS

ROOT = f"projects/{EE_PROJECT}/assets/caspian"
ZONE = f"{ROOT}/active_zone_v6"
STATE = RESULTS / "inventory" / "stage1_tasks.json"
CHUNKS = (
    [("TM_L5", a, b) for a, b in [(1984, 1989), (1989, 1994), (1994, 1999), (1999, 2004), (2004, 2008), (2008, 2012)]]
    + [("ETM_L7", a, b) for a, b in [(1999, 2004), (2004, 2009), (2009, 2014), (2014, 2019), (2019, 2024)]]
    + [("OLI_L8", a, b) for a, b in [(2013, 2018), (2018, 2022), (2022, 2027)]]
    + [("OLI_L9", 2021, 2027)]
    + [("MSI_S2", y, y + 1) for y in range(2015, 2027)]
)
COLUMNS = ["scene", "sensor", "acq", "time", "otsu_threshold", "otsu_n", "zone_px", "zone_clear",
           "zone_snow", "zone_snow_ndsi", "cloud_scene"]


def collection(sensor: str, start: str, end: str):
    dom = pipeline_ee.domain_geometry()
    if sensor == "MSI_S2":
        return masks.sentinel2(dom, start, end, bands=masks.CLASSIFY_BANDS)
    return masks.landsat(sensor, dom, start, end, bands=masks.CLASSIFY_BANDS)


def export(name: str, col) -> str:
    zone = ee.Image(ZONE).select("zone").eq(1)
    table = pipeline_ee.scene_table(col, zone)
    t = ee.batch.Export.table.toAsset(collection=table, description=name, assetId=f"{ROOT}/{name}")
    t.start()
    return t.id


def main(cmd: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    if cmd == "pilot":
        col = (collection("OLI_L8", "2020-01-01", "2021-01-01")
               .filter(ee.Filter.eq("WRS_PATH", 165)).filter(ee.Filter.eq("WRS_ROW", 28)))
        tid = export("stage1_pilot_v5_L8_165028_2020", col)
        STATE.with_name("stage1_pilot_task.json").write_text(json.dumps({"task": tid}), encoding="utf-8")
        print("submitted pilot", tid)
    elif cmd == "submit":
        state = {"tasks": []}
        for sensor, a, b in CHUNKS:
            name = f"stage1_{sensor}_{a}_{b - 1}"
            tid = export(name, collection(sensor, f"{a}-01-01", f"{b}-01-01"))
            state["tasks"].append({"name": name, "task": tid})
            print("submitted", name, tid)
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    elif cmd == "status":
        state = json.loads(STATE.read_text(encoding="utf-8"))
        tot = 0.0
        for t in state["tasks"]:
            md = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{t['task']}")["metadata"]
            t.update(state=md.get("state"), eecu_s=md.get("batchEecuUsageSeconds"))
            tot += float(t["eecu_s"] or 0)
            print(t["name"], t["state"], t["eecu_s"])
        print("total EECU-s:", round(tot, 1))
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    elif cmd == "fetch":
        out = INTERIM / "stage1"
        out.mkdir(parents=True, exist_ok=True)
        for t in json.loads(STATE.read_text(encoding="utf-8"))["tasks"]:
            md = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{t['task']}")["metadata"]
            if md.get("state") != "SUCCEEDED" or (out / f"{t['name']}.csv").exists():
                continue
            url = ee.FeatureCollection(f"{ROOT}/{t['name']}").getDownloadURL("CSV", COLUMNS, t["name"])
            r = requests.get(url, timeout=3600)
            r.raise_for_status()
            (out / f"{t['name']}.csv").write_bytes(r.content)
            print("fetched", t["name"], len(r.content) // 1024, "KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "status"))
