"""The stage-2 computation cover, the zone-v6 5 km tile cover extended by one
more 5 km cell on its seaward side only (into the open sea of
results/study_area/sea_mask_1km.npz), as insurance for the lowest-level waterlines.
Buffering every edge added 31% of area; the low-level waterlines can only lie
seaward. Built locally on the equal-area grid; clipped to the domain.

Output (committed): results/study_area/zone_v6_cover_buffered.geojson
"""

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pyproj
import rasterio.features
from affine import Affine
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union

from caspian import pipeline_ee
from caspian.config import CRS_EQUAL_AREA, RESULTS
from caspian.geography import DOMAIN


def main() -> int:
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True).transform
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform
    src = json.loads((RESULTS / "study_area" / "zone_v6_tiles.geojson").read_text(encoding="utf-8"))
    cover = unary_union([transform(fwd, shape(f["geometry"])) for f in src["features"]])
    sea_npz = np.load(RESULTS / "study_area" / "sea_mask_1km.npz")
    a, b, c, d, e, f = sea_npz["transform"]
    sea = unary_union([shape(g) for g, v in rasterio.features.shapes(
        sea_npz["mask"].astype(np.uint8), mask=sea_npz["mask"].astype(bool), transform=Affine(a, b, c, d, e, f)) if v == 1])
    ring = cover.buffer(pipeline_ee.ZONE_TILE_M, cap_style="square", join_style="mitre").difference(cover)
    dom = transform(fwd, DOMAIN.segmentize(0.01))
    buf = cover.union(ring.intersection(sea)).intersection(dom).buffer(0).simplify(50)
    out = RESULTS / "study_area" / "zone_v6_cover_buffered.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"area_km2": round(buf.area / 1e6, 1)},
         "geometry": mapping(transform(inv, buf))}]}), encoding="utf-8")
    print("cover km2:", round(cover.area / 1e6), "-> buffered:", round(buf.area / 1e6), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
