"""A8 input: OpenStreetMap infrastructure inside the study domain, via the Overpass
API (data (c) OpenStreetMap contributors, ODbL 1.0).

Layers: ports/harbours, piers and ferry terminals; settlements (place nodes);
pipelines; offshore platforms and wells; navigation channels and canals;
major roads. Each layer is clipped to the domain polygon and saved as GeoPackage.

Outputs (git-ignored): data/raw/osm/osm_<layer>.gpkg
Provenance (committed): results/provenance/osm_exposure.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import caspian  # noqa: F401
import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, Polygon

from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN

ENDPOINT = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "caspian-research/0.1 (rezwan.ahmed@du.ac.bd)"}
W, S, E, N = DOMAIN.bounds
BBOX = f"{S},{W},{N},{E}"
LAYERS = {
    "ports": f"""(nwr["harbour"]({BBOX}); nwr["industrial"="port"]({BBOX}); nwr["landuse"="port"]({BBOX});
                 nwr["seamark:type"="harbour"]({BBOX}); nwr["amenity"="ferry_terminal"]({BBOX});
                 nwr["man_made"="pier"]({BBOX}););""",
    "settlements": f"""(node["place"~"^(city|town|village|hamlet)$"]({BBOX}););""",
    "pipelines": f"""(way["man_made"="pipeline"]({BBOX}););""",
    "platforms": f"""(nwr["man_made"="offshore_platform"]({BBOX}); nwr["man_made"="petroleum_well"]({BBOX});
                     nwr["seamark:type"="platform"]({BBOX}););""",
    "channels": f"""(way["waterway"~"^(canal|fairway)$"]({BBOX}); way["seamark:type"="fairway"]({BBOX});
                    way["seamark:type"="dredged_area"]({BBOX}););""",
    "roads": f"""(way["highway"~"^(motorway|trunk|primary|secondary)$"]({BBOX}););""",
}


def overpass(body: str) -> dict:
    q = f"[out:json][timeout:600];{body}out geom tags;"
    for attempt in range(4):
        r = requests.post(ENDPOINT, data={"data": q}, headers=UA, timeout=900)
        if r.status_code == 200:
            return r.json()
        time.sleep(30 * (attempt + 1))
    r.raise_for_status()
    return {}


def to_gdf(js: dict) -> gpd.GeoDataFrame:
    rows = []
    for el in js.get("elements", []):
        tags = el.get("tags", {})
        geom = None
        if el["type"] == "node":
            geom = Point(el["lon"], el["lat"])
        elif "geometry" in el and el["geometry"]:
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            closed = len(pts) > 3 and pts[0] == pts[-1]
            geom = Polygon(pts) if closed and el["type"] == "way" and not tags.get("highway") \
                and not tags.get("man_made") == "pipeline" and not tags.get("waterway") else LineString(pts)
        elif el["type"] == "relation" and "center" in el:
            geom = Point(el["center"]["lon"], el["center"]["lat"])
        if geom is None or geom.is_empty:
            continue
        rows.append({"osm_type": el["type"], "osm_id": el["id"], "name": tags.get("name"),
                     "name_en": tags.get("name:en"), "tags": json.dumps(tags, ensure_ascii=False),
                     "geometry": geom})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if len(gdf):
        gdf = gdf[gdf.intersects(DOMAIN)].reset_index(drop=True)
    return gdf


def main() -> int:
    out = RAW / "osm"
    out.mkdir(parents=True, exist_ok=True)
    prov = {"source": "OpenStreetMap via Overpass API", "endpoint": ENDPOINT, "licence": "ODbL 1.0",
            "attribution": "(c) OpenStreetMap contributors", "bbox_swne": BBOX,
            "queried_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "layers": {}}
    for name, body in LAYERS.items():
        js = overpass(body)
        gdf = to_gdf(js)
        path = out / f"osm_{name}.gpkg"
        if len(gdf):
            gdf.to_file(path, driver="GPKG")
        prov["layers"][name] = {"n_features_in_domain": int(len(gdf)),
                                "osm_timestamp": js.get("osm3s", {}).get("timestamp_osm_base"),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
        print(name, prov["layers"][name]["n_features_in_domain"], flush=True)
        time.sleep(5)
    (RESULTS / "provenance").mkdir(parents=True, exist_ok=True)
    (RESULTS / "provenance" / "osm_exposure.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
