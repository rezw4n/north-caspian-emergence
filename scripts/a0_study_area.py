"""Export the frozen study domain and sectors (caspian.geography) with their
equal-area sizes. Outputs (committed): results/study_area/sectors.geojson,
results/study_area/sector_areas.json

Edge convention: the domain and sectors are defined, clipped and tested in
longitude/latitude with straight edges in that plane, so the box sides follow
parallels and meridians. Every edge is densified to DENSIFY_DEG before the vertices
are projected to the equal-area CRS. Projecting the vertices alone turns the long
box edges (8.2 degrees along 47.35 N) into chords, which made the domain about 1.3%
larger than the sum of its sectors (review finding 36).

Earth Engine reads the same GeoJSON polygons with geodesic edges: ee.Geometry(geojson)
defaults to geodesic=True for EPSG:4326 (caspian.pipeline_ee.domain_geometry,
scripts/a2_scene_inventory.py, scripts/a5_era5_winds.py). The scene inventory, the
active zone and the ERA5 sector winds therefore use geodesic edges. ``geodesic_reading``
in the output records, for that second reading, the area of every polygon (pyproj.Geod,
exact for geodesic edges) and, for every edge, the largest distance between the
geodesic and the lon/lat line (measured in the equal-area projection). Which ground
lies between the two readings is measured against the relief grids in
scripts/a0_sector_edges.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys

import caspian  # noqa: F401
import numpy as np
import pyproj
import shapely
from shapely.geometry import LineString, Polygon

from caspian import config
from caspian.geography import DOMAIN, sectors, to_geojson

DENSIFY_DEG = 0.01  # areas change by < 0.01 km2 between 0.01 and 0.001 degrees
GEODESIC_STEP_M = 250.0  # spacing of the points that trace a geodesic edge
LINE_DENSIFY_DEG = 0.0005  # densification of the lon/lat line it is compared with

to_laea = pyproj.Transformer.from_crs(config.CRS_EXCHANGE, config.CRS_EQUAL_AREA, always_xy=True)
_geod = pyproj.Geod(ellps="WGS84")


def area_km2(g, densify_deg: float | None = DENSIFY_DEG) -> float:
    """Equal-area size (km2) of a lon/lat polygon whose edges are straight in lon/lat."""
    if densify_deg:
        g = shapely.segmentize(g, densify_deg)
    return shapely.transform(g, lambda x, y: to_laea.transform(x, y), interleaved=False).area / 1e6


def geodesic_area_km2(g: Polygon) -> float:
    """Ellipsoidal size (km2) of a polygon whose edges are geodesics between its vertices
    (the reading Earth Engine gives a lon/lat GeoJSON polygon)."""
    if g.geom_type != "Polygon" or len(g.interiors):
        raise ValueError("expected a polygon without holes")
    x, y = g.exterior.coords.xy
    a, _ = _geod.polygon_area_perimeter(list(x), list(y))
    return abs(a) / 1e6


def geodesic_points(p, q, step_m: float = GEODESIC_STEP_M) -> np.ndarray:
    """(n, 2) lon/lat points along the geodesic from p to q, ends included."""
    length = _geod.inv(p[0], p[1], q[0], q[1])[2]
    n = max(int(math.ceil(length / step_m)) - 1, 0)
    mid = _geod.npts(p[0], p[1], q[0], q[1], n) if n else []
    return np.asarray([tuple(p)] + [tuple(m) for m in mid] + [tuple(q)], dtype=float)


def geodesic_polygon(g: Polygon, step_m: float = GEODESIC_STEP_M) -> Polygon:
    """``g`` with each edge replaced by its geodesic, traced in lon/lat every ``step_m``.
    Straight lon/lat segments between such close points depart from the geodesic by
    millimetres, so the result can be tested with ordinary (lon/lat) predicates."""
    c = list(g.exterior.coords)
    pts = [geodesic_points(p, q, step_m)[:-1] for p, q in zip(c[:-1], c[1:])]
    return Polygon(np.vstack(pts))


def edge_offset_km(p, q) -> float:
    """Largest distance (km, equal-area projection) between the geodesic from p to q and
    the straight lon/lat line between them."""
    g = geodesic_points(p, q)
    gx, gy = to_laea.transform(g[:, 0], g[:, 1])
    line = shapely.segmentize(LineString([tuple(p), tuple(q)]), LINE_DENSIFY_DEG)
    line = shapely.transform(line, lambda x, y: to_laea.transform(x, y), interleaved=False)
    return float(np.max(shapely.distance(line, shapely.points(gx, gy)))) / 1e3


def edge_offsets(g: Polygon) -> list[dict]:
    c = list(g.exterior.coords)
    out = []
    for p, q in zip(c[:-1], c[1:]):
        out.append({"from": [round(p[0], 4), round(p[1], 4)], "to": [round(q[0], 4), round(q[1], 4)],
                    "length_km": round(_geod.inv(p[0], p[1], q[0], q[1])[2] / 1e3, 1),
                    "offset_km": round(edge_offset_km(p, q), 3)})
    return out


def compute_geodesic_reading(secs: dict | None = None) -> dict:
    secs = secs or sectors()
    dom_edges = edge_offsets(DOMAIN)
    north = max(dom_edges, key=lambda e: e["offset_km"])
    polys = {"domain": DOMAIN, **{k: g for k, (_, g) in secs.items()}}
    per_poly = {}
    for k, g in polys.items():
        e = edge_offsets(g)
        per_poly[k] = {"area_km2": round(geodesic_area_km2(g), 1),
                       "area_km2_lonlat_reading": round(area_km2(g), 1),
                       "max_edge_offset_km": max(x["offset_km"] for x in e), "edges": e}
    # every edge of every polygon, apart from those that lie along the parallel of the
    # domain's northern edge (the domain's own and the sectors' pieces of it)
    lat_n = north["from"][1]
    others = [x["offset_km"] for v in per_poly.values() for x in v["edges"]
              if not (abs(x["from"][1] - lat_n) < 1e-9 and abs(x["to"][1] - lat_n) < 1e-9)]
    return {"reading": "edges are geodesics between the vertices (Earth Engine default for EPSG:4326)",
            "domain_km2": per_poly["domain"]["area_km2"],
            "sectors_sum_km2": round(sum(per_poly[k]["area_km2"] for k in secs), 1),
            "north_edge": north,
            "north_edge_offset_km": north["offset_km"],
            "other_edges_max_offset_km": max(others),
            "polygons": per_poly}


def compute_areas() -> dict:
    secs = sectors()
    sec_km2 = {k: round(area_km2(g), 1) for k, (_, g) in secs.items()}
    return {"domain_km2": round(area_km2(DOMAIN), 1),
            "sectors_km2": sec_km2,
            "sectors_sum_km2": round(sum(sec_km2.values()), 1),
            "edge_convention": f"straight in lon/lat (box sides along parallels and meridians); "
                               f"edges densified to {DENSIFY_DEG} deg before projection",
            "crs_equal_area": config.CRS_EQUAL_AREA,
            "names": {k: n for k, (n, _) in secs.items()},
            "geodesic_reading": compute_geodesic_reading(secs)}


def main() -> int:
    out = config.RESULTS / "study_area"
    gj = to_geojson(out / "sectors.geojson")
    areas = compute_areas()
    areas["geojson_sha256"] = hashlib.sha256(gj.read_bytes()).hexdigest()
    (out / "sector_areas.json").write_text(json.dumps(areas, indent=1), encoding="utf-8")
    g = areas["geodesic_reading"]
    print(json.dumps({k: v for k, v in areas.items() if k != "geodesic_reading"}, indent=1))
    print("geodesic reading:", json.dumps({k: v for k, v in g.items() if k != "polygons"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
