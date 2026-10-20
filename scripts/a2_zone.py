"""Stage 0 (D-015): export the active zone (where 1984-2026 waterlines can occur)
as an Earth Engine asset at 60 m, and report its area.

Usage: python scripts/a2_zone.py submit | area
"""

import json
import sys

import caspian  # noqa: F401
import ee

from caspian import pipeline_ee
from caspian.config import EE_PROJECT, RESULTS

ASSET = f"projects/{EE_PROJECT}/assets/caspian/active_zone_v1"
STATE = RESULTS / "inventory" / "zone_task.json"


def main(cmd: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    dom = pipeline_ee.domain_geometry()
    if cmd == "submit":
        zone = pipeline_ee.active_zone(dom).unmask(0).uint8()
        t = ee.batch.Export.image.toAsset(image=zone, description="active_zone_v1", assetId=ASSET,
                                          region=dom, scale=pipeline_ee.ZONE_SCALE, crs="EPSG:4326",
                                          maxPixels=1e11)
        t.start()
        STATE.write_text(json.dumps({"task": t.id, "asset": ASSET}, indent=1), encoding="utf-8")
        print("submitted", t.id)
    elif cmd == "area":
        z = ee.Image(ASSET)
        a = z.eq(1).multiply(ee.Image.pixelArea()).reduceRegion(
            ee.Reducer.sum(), dom, 120, maxPixels=1e10).getInfo()
        st = json.loads(STATE.read_text(encoding="utf-8"))
        op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{st['task']}")
        st["eecu_s"] = op["metadata"].get("batchEecuUsageSeconds")
        st["zone_area_km2"] = round(list(a.values())[0] / 1e6, 1)
        STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "area"))
