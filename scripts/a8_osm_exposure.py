"""A8 input: OpenStreetMap infrastructure inside the study domain, via the Overpass
API (data (c) OpenStreetMap contributors, ODbL 1.0).

Layers: ports/harbours, piers and ferry terminals; settlements (place nodes);
pipelines; offshore platforms and wells; navigation channels and canals;
major roads; protected areas (boundary=protected_area, leisure=nature_reserve,
boundary=national_park) and wetlands (natural=wetland). Each layer is clipped to the
domain polygon and saved as GeoPackage. OpenStreetMap is not an authoritative source
for protected-area boundaries or habitats; the two added layers are an open-licence
stand-in until the authoritative layers (WDPA, Ramsar) are obtained by the PI, and
are labelled as such.

Multipolygon and boundary relations are assembled from their member ways (outer
rings polygonised, inner rings removed); before 2026-10-21 relations without a
centre were skipped, which matters only for layers fetched after that date.

Outputs (git-ignored): data/raw/osm/osm_<layer>.gpkg
Provenance (committed): results/provenance/osm_exposure.json (layers fetched in a run
are added or replaced; the others are kept)

Usage: python scripts/a8_osm_exposure.py [--only layer[,layer]]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import caspian  # noqa: F401
import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from caspian.config import RAW, RESULTS
from caspian.geography import DOMAIN

ENDPOINT = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "caspian-research/0.1 (rezwan490@gmail.com)"}
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
    "protected_areas": f"""(nwr["boundary"="protected_area"]({BBOX}); nwr["leisure"="nature_reserve"]({BBOX});
                           nwr["boundary"="national_park"]({BBOX}););""",
    "wetlands": f"""(nwr["natural"="wetland"]({BBOX}););""",
}


def relation_polygon(el: dict):
    """A multipolygon or boundary relation as a (multi)polygon: its outer member ways
    polygonised, the inner ones removed; None if no ring closes."""
    outer, inner = [], []
    for m in el.get("members", []):
        pts = [(q["lon"], q["lat"]) for q in (m.get("geometry") or []) if q]
        if m.get("type") != "way" or len(pts) < 2:
            continue
        (inner if m.get("role") == "inner" else outer).append(LineString(pts))
    if not outer:
        return None
    polys = list(polygonize(unary_union(outer)))
    if not polys:
        return None
    shape = unary_union(polys)
    holes = list(polygonize(unary_union(inner))) if inner else []
    if holes:
        shape = shape.difference(unary_union(holes))
    return None if shape.is_empty else shape


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
        elif el["type"] == "relation" and el.get("members"):
            geom = relation_polygon(el)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated layers to fetch (default: all)")
    a = ap.parse_args(argv)
    names = a.only.split(",") if a.only else list(LAYERS)
    unknown = set(names) - set(LAYERS)
    if unknown:
        raise SystemExit(f"unknown layers: {sorted(unknown)}")
    out = RAW / "osm"
    out.mkdir(parents=True, exist_ok=True)
    prov_path = RESULTS / "provenance" / "osm_exposure.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {"layers": {}}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prov.update({"source": "OpenStreetMap via Overpass API", "endpoint": ENDPOINT, "licence": "ODbL 1.0",
                 "attribution": "(c) OpenStreetMap contributors", "bbox_swne": BBOX})
    prov.setdefault("queried_utc", now)
    for name in names:
        js = overpass(LAYERS[name])
        gdf = to_gdf(js)
        path = out / f"osm_{name}.gpkg"
        if len(gdf):
            gdf.to_file(path, driver="GPKG")
        prov["layers"][name] = {"n_features_in_domain": int(len(gdf)),
                                "osm_timestamp": js.get("osm3s", {}).get("timestamp_osm_base"),
                                "queried_utc": now,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
        print(name, prov["layers"][name]["n_features_in_domain"], flush=True)
        time.sleep(5)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
