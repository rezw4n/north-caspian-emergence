"""Export the frozen study domain and sectors (caspian.geography) with their
equal-area sizes. Outputs (committed): results/study_area/sectors.geojson,
results/study_area/sector_areas.json"""

import hashlib
import json

import caspian  # noqa: F401
import pyproj
import shapely

from caspian import config
from caspian.geography import DOMAIN, sectors, to_geojson

to_laea = pyproj.Transformer.from_crs(config.CRS_EXCHANGE, config.CRS_EQUAL_AREA, always_xy=True)


def area_km2(g):
    return shapely.transform(g, lambda x, y: to_laea.transform(x, y), interleaved=False).area / 1e6


out = config.RESULTS / "study_area"
gj = to_geojson(out / "sectors.geojson")
areas = {"domain_km2": round(area_km2(DOMAIN), 1),
         "sectors_km2": {k: round(area_km2(g), 1) for k, (_, g) in sectors().items()},
         "names": {k: n for k, (n, _) in sectors().items()},
         "geojson_sha256": hashlib.sha256(gj.read_bytes()).hexdigest()}
(out / "sector_areas.json").write_text(json.dumps(areas, indent=1), encoding="utf-8")
print(json.dumps(areas, indent=1))
