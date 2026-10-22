"""Stage 0: export the active zone (where 1992-2026 waterlines
can occur) as an Earth Engine asset on the equal-area grid at 250 m, its 5 km tile
cover as a table asset, and report areas.

Usage: python scripts/a2_zone.py pilot | submit | tiles | area [pilot|submit|v6]
                                 extents | v6preview <png> | v6
  pilot, submit : zone v5 (1 x 1 degree NE box, or the domain)
  extents       : the 1995-96 and 2025-26 extents of zone v6, exported once
  v6preview     : 1 km thumbnail of zone v6 built from the extents asset (profiled)
  v6            : zone v6 at 250 m from the extents asset
"""

import json
import sys

import caspian  # noqa: F401
import ee

from caspian import pipeline_ee
from caspian.config import EE_PROJECT, RESULTS
from caspian.geography import DOMAIN

ROOT = f"projects/{EE_PROJECT}/assets/caspian"
ASSET = f"{ROOT}/active_zone_v5"
PILOT_ASSET = f"{ROOT}/active_zone_v5_pilot"
TILES = f"{ROOT}/active_zone_v6_tiles"
EXTENTS = f"{ROOT}/extents_v6"
ASSET_V6 = f"{ROOT}/active_zone_v6"
PILOT_BOX = (52.4, 45.8, 53.4, 46.8)
STATE = RESULTS / "inventory"


def main(cmd: str) -> int:
    ee.Initialize(project=EE_PROJECT)
    dom = pipeline_ee.domain_geometry()
    if cmd in ("pilot", "submit"):
        bounds = PILOT_BOX if cmd == "pilot" else DOMAIN.bounds
        region = ee.Geometry.Rectangle(list(bounds)) if cmd == "pilot" else dom
        zone = pipeline_ee.active_zone_v5(region).unmask(0).uint8()
        name = "active_zone_v5_pilot" if cmd == "pilot" else "active_zone_v5"
        transform, _ = pipeline_ee.laea_transform(pipeline_ee.ZONE_SCALE, bounds)
        t = ee.batch.Export.image.toAsset(image=zone, description=name,
                                          assetId=PILOT_ASSET if cmd == "pilot" else ASSET,
                                          region=region, crs=pipeline_ee.LAEA_WKT, crsTransform=transform,
                                          maxPixels=1e11)
        t.start()
        (STATE / f"zone_v5_task_{cmd}.json").write_text(json.dumps({"task": t.id, "name": name}, indent=1),
                                                        encoding="utf-8")
        print("submitted", name, t.id)
    elif cmd in ("extents", "v6"):
        transform, _ = pipeline_ee.laea_transform(pipeline_ee.ZONE_SCALE, DOMAIN.bounds)
        if cmd == "extents":
            img, name, aid = pipeline_ee.extents_v6(dom).uint8(), "extents_v6", EXTENTS
        else:
            img = pipeline_ee.active_zone_v6(ee.Image(EXTENTS), dom).unmask(0).uint8()
            name, aid = "active_zone_v6", ASSET_V6
        t = ee.batch.Export.image.toAsset(image=img, description=name, assetId=aid, region=dom,
                                          crs=pipeline_ee.LAEA_WKT, crsTransform=transform, maxPixels=1e11)
        t.start()
        (STATE / f"zone_v6_task_{cmd}.json").write_text(json.dumps({"task": t.id, "name": name}, indent=1),
                                                        encoding="utf-8")
        print("submitted", name, t.id)
    elif cmd == "v6preview":
        import io, re, requests
        ext = ee.Image(EXTENTS)
        z = pipeline_ee.active_zone_v6(ext, dom, scale=1000).select("zone").unmask(0)
        hi, lo = ext.select("high"), ext.select("low")
        vis = (ee.Image(0).where(hi.And(lo.Not()), 1).where(lo, 2).where(z.eq(1), 3).where(z.eq(1).And(lo), 4))
        buf = io.StringIO()
        with ee.profilePrinting(destination=buf):
            url = vis.visualize(min=0, max=4, palette=["ffffff", "c6dbef", "08306b", "fd8d3c", "a63603"]).getThumbURL(
                {"region": dom.bounds(), "dimensions": 1400, "format": "png"})
            open(sys.argv[2], "wb").write(requests.get(url, timeout=600).content)
            a = z.multiply(ee.Image.pixelArea()).reduceRegion(ee.Reducer.sum(), dom, 1000, crs=pipeline_ee.LAEA_WKT,
                                                              maxPixels=1e9).getInfo()
        tot = sum(float(m.group(1)) for line in buf.getvalue().splitlines() if (m := re.match(r"\s*([\d.]+)\s", line)))
        print("zone v6 at 1 km:", round(list(a.values())[0] / 1e6), "km2; profiled EECU-s", round(tot, 1))
    elif cmd == "tiles":
        tiles = pipeline_ee.zone_tiles(ee.Image(ASSET_V6).selfMask(), dom)
        t = ee.batch.Export.table.toAsset(collection=tiles, description="active_zone_v6_tiles", assetId=TILES)
        t.start()
        (STATE / "zone_v5_task_tiles.json").write_text(json.dumps({"task": t.id}, indent=1), encoding="utf-8")
        print("submitted tiles", t.id)
    elif cmd == "area":
        which = sys.argv[2] if len(sys.argv) > 2 else "submit"
        asset = {"pilot": PILOT_ASSET, "submit": ASSET, "v6": ASSET_V6}[which]
        region = ee.Geometry.Rectangle(list(PILOT_BOX)) if which == "pilot" else dom
        z = ee.Image(asset)
        px = ee.Image.pixelArea()
        a = (ee.Image.cat([z.select("zone").eq(1).multiply(px).rename("zone"),
                           z.select("zone").eq(1).And(z.select("filled").eq(1)).multiply(px).rename("filled"),
                           px.rename("region")])
             .reduceRegion(ee.Reducer.sum(), region, crs=pipeline_ee.LAEA_WKT, scale=pipeline_ee.ZONE_SCALE,
                           maxPixels=1e10).getInfo())
        path = STATE / ("zone_v6_task_v6.json" if which == "v6" else f"zone_v5_task_{which}.json")
        st = json.loads(path.read_text(encoding="utf-8"))
        op = ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{st['task']}")
        st["eecu_s"] = op["metadata"].get("batchEecuUsageSeconds")
        st["zone_area_km2"] = round(a["zone"] / 1e6, 1)
        st["gsw_filled_km2"] = round(a["filled"] / 1e6, 1)
        st["region_km2"] = round(a["region"] / 1e6, 1)
        st["asset"] = asset
        path.write_text(json.dumps(st, indent=1), encoding="utf-8")
        print(json.dumps(st, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "area"))
