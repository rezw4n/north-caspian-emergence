"""A5 input: hourly ERA5 10 m wind (u, v) and mean sea-level pressure averaged over
each frozen sector (caspian.geography), 1984-01-01 to the end of the ERA5 record in
Earth Engine (ECMWF/ERA5/HOURLY; Hersbach et al. 2020). Exported as Earth Engine
table assets in decade chunks; downloaded as CSV with `fetch`.

Usage: python scripts/a5_era5_winds.py submit | status | fetch
Outputs: data/interim/era5/era5_sector_winds_<chunk>.csv (git-ignored)
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import ee
import requests
from shapely.geometry import mapping

from caspian.config import EE_PROJECT, INTERIM, RESULTS
from caspian.geography import sectors

ROOT = f"projects/{EE_PROJECT}/assets/caspian"
STATE = RESULTS / "inventory" / "era5_tasks.json"
CHUNKS = [("1984", "1994"), ("1994", "2004"), ("2004", "2014"), ("2014", "2024"), ("2024", "2027")]
BANDS = ["u_component_of_wind_10m", "v_component_of_wind_10m", "mean_sea_level_pressure"]


def regions():
    return ee.FeatureCollection([ee.Feature(ee.Geometry(mapping(g)), {"sector": k})
                                 for k, (_, g) in sectors().items()])


def table(start: str, end: str):
    fc = regions()
    col = ee.ImageCollection("ECMWF/ERA5/HOURLY").filterDate(f"{start}-01-01", f"{end}-01-01").select(BANDS)

    def per_image(img):
        t = img.get("system:time_start")
        red = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=27830, tileScale=1)
        return red.map(lambda f: ee.Feature(None, f.toDictionary(["sector"] + BANDS)).set("t", t))

    return ee.FeatureCollection(col.map(per_image)).flatten()


def main(cmd: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    if cmd == "submit":
        state = {"tasks": []}
        for a, b in CHUNKS:
            name = f"era5_sector_winds_{a}_{b}"
            fc = table(a, b).map(lambda f: f.setGeometry(ee.Geometry.Point([50, 45.5])))
            t = ee.batch.Export.table.toAsset(collection=fc, description=name, assetId=f"{ROOT}/{name}")
            t.start()
            state["tasks"].append({"name": name, "task": t.id})
            print("submitted", name, t.id)
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    elif cmd == "status":
        state = json.loads(STATE.read_text(encoding="utf-8"))
        for t in state["tasks"]:
            op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{t['task']}")
            t.update(state=op["metadata"].get("state"), eecu_s=op["metadata"].get("batchEecuUsageSeconds"),
                     error=(op.get("error") or {}).get("message"))
            print(t)
        STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    elif cmd == "fetch":
        out = INTERIM / "era5"
        out.mkdir(parents=True, exist_ok=True)
        for t in json.loads(STATE.read_text(encoding="utf-8"))["tasks"]:
            op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{t['task']}")
            if op["metadata"].get("state") != "SUCCEEDED" or (out / f"{t['name']}.csv").exists():
                continue
            fc = ee.FeatureCollection(f"{ROOT}/{t['name']}")
            url = fc.getDownloadURL("CSV", ["t", "sector"] + BANDS, t["name"])
            r = requests.get(url, timeout=3600)
            r.raise_for_status()
            (out / f"{t['name']}.csv").write_bytes(r.content)
            print("fetched", t["name"], len(r.content) // 1024, "KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "status"))
