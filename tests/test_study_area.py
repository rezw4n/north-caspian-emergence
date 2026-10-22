"""Study-area sizes and the sector-layout diagnostics (scripts/a0_study_area.py,
scripts/a0_sector_edges.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pyproj
import pytest
import shapely
from shapely.geometry import Polygon

from caspian.geography import DOMAIN, sectors

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sa():
    return _load("a0_study_area")


@pytest.fixture(scope="module")
def edges():
    return _load("a0_sector_edges")


def _geod_area_km2(poly, step=0.001):
    """Ellipsoidal area of a lon/lat polygon with edges along its lon/lat lines."""
    dense = shapely.segmentize(poly, step)
    x, y = dense.exterior.coords.xy
    a, _ = pyproj.Geod(ellps="WGS84").polygon_area_perimeter(list(x), list(y))
    return abs(a) / 1e6


def test_box_area_follows_parallels(sa):
    # Shaped like the domain: a southern edge with many vertices and a single 8.2-degree
    # edge along 47.35 N. Projecting the vertices alone turns that edge into a chord and
    # overstates the area by about 1%.
    south = [(float(x), 45.0) for x in np.linspace(46.4, 54.6, 60)]
    box = Polygon(south + [(54.6, 47.35), (46.4, 47.35)])
    ref = _geod_area_km2(box)
    assert sa.area_km2(box) == pytest.approx(ref, rel=1e-4)
    assert abs(sa.area_km2(box, densify_deg=None) - ref) / ref > 5e-3  # the old method


def test_sector_areas_sum_to_domain(sa):
    a = sa.compute_areas()
    assert a["sectors_sum_km2"] == pytest.approx(a["domain_km2"], abs=0.5)
    assert sum(a["sectors_km2"].values()) == pytest.approx(a["domain_km2"], abs=0.5)


def test_domain_area_matches_ellipsoid(sa):
    assert sa.area_km2(DOMAIN) == pytest.approx(_geod_area_km2(DOMAIN), rel=1e-4)


def test_sea_mask_excludes_closed_depressions(edges):
    z = np.full((7, 9), 5.0)
    z[1:6, 0:4] = -10.0   # open sea, deepest at [3, 0]
    z[3, 0] = -50.0
    z[2:5, 6:8] = -10.0   # a closed depression below the level, not connected
    sea = edges.sea_mask(z, level=-1.0)
    assert sea[3, 1] and sea[1, 3]
    assert not sea[3, 6], "a depression below the level but cut off from the sea is land"
    assert not sea[0, 0]


def test_sea_mask_is_four_connected(edges):
    z = np.full((5, 5), 5.0)
    z[0:2, 0:2] = -20.0
    z[2, 2] = -10.0       # touches the sea only at a corner
    z[0, 0] = -30.0
    assert not edges.sea_mask(z, level=0.0)[2, 2]


def test_coast_cells_are_land_next_to_sea(edges):
    sea = np.zeros((5, 5), bool)
    sea[:, :2] = True
    c = edges.coast_cells(sea)
    assert c[:, 2].all() and not c[:, 3:].any() and not (c & sea).any()


def test_edge_points_lie_on_shared_boundary(edges):
    s = sectors()
    pts = edges.edge_points(s["volga"][1], s["open_shelf"][1], step_km=5.0)
    assert len(pts) > 10
    shared = s["volga"][1].boundary.intersection(s["open_shelf"][1].boundary)
    d = shapely.distance(shared, shapely.points(pts))
    assert float(np.max(d)) < 1e-6


# --- the two readings of the polygon edges (the geodesic reading of Earth Engine)

def test_geodesic_reading_of_a_parallel_bows_poleward(sa):
    # An 8.2-degree edge along 47.35 N, as the domain's northern edge. On the sphere the
    # great circle's vertex is at atan(tan(47.35)/cos(4.1)) = 47.4245 N, about 8.3 km north.
    off = sa.edge_offset_km((46.4, 47.35), (54.6, 47.35))
    assert 7.5 < off < 9.0
    # a meridian and the equator are geodesics: both readings coincide
    assert sa.edge_offset_km((50.0, 44.0), (50.0, 47.0)) < 1e-3
    assert sa.edge_offset_km((46.0, 0.0), (54.0, 0.0)) < 1e-3


def test_geodesic_polygon_traces_the_geodesic_area(sa):
    box = Polygon([(46.4, 45.0), (54.6, 45.0), (54.6, 47.35), (46.4, 47.35)])
    geo = sa.geodesic_polygon(box)
    # the traced polygon, read with straight lon/lat edges, has the geodesic area
    assert _geod_area_km2(geo) == pytest.approx(sa.geodesic_area_km2(box), rel=1e-4)
    # With its south edge on the equator (a geodesic), only the north edge bows: the
    # geodesic reading adds about 2/3 x 8.2 km x 619 km = 3,400 km2.
    eq = Polygon([(46.4, 0.0), (54.6, 0.0), (54.6, 47.35), (46.4, 47.35)])
    gain = sa.geodesic_area_km2(eq) - sa.area_km2(eq)
    assert 2500 < gain < 4500


def test_geodesic_reading_of_the_domain(sa):
    g = sa.compute_geodesic_reading()
    assert g["north_edge"]["from"][1] == g["north_edge"]["to"][1] == 47.35
    assert g["north_edge_offset_km"] > 5 * g["other_edges_max_offset_km"]
    assert g["domain_km2"] > sa.area_km2(DOMAIN)
    # each polygon's geodesic area is Geod's area of its vertex list
    x, y = DOMAIN.exterior.coords.xy
    a, _ = pyproj.Geod(ellps="WGS84").polygon_area_perimeter(list(x), list(y))
    assert g["domain_km2"] == pytest.approx(abs(a) / 1e6, abs=0.1)


def test_reading_differences_flags_the_band_beyond_the_north_edge(edges):
    s = sectors()
    geo = edges.geodesic_polygons(s)
    lon = np.arange(49.0, 49.2001, 0.005)
    lat = np.arange(47.30, 47.4501, 0.0025)
    LON, LAT = np.meshgrid(lon, lat)
    in_dom = shapely.contains_xy(DOMAIN, LON, LAT)
    member = np.full(LON.shape, "", dtype=object)
    for k, (_, g) in s.items():
        member[shapely.contains_xy(g, LON, LAT) & in_dom] = k
    ones = np.ones(LON.shape)
    everywhere = np.ones(LON.shape, bool)
    r = edges.reading_differences(LON, LAT, ones, everywhere, everywhere, member, in_dom, geo, 47.35)
    # at 49.0-49.2 E the domain's geodesic north edge lies about 7 km (0.06 deg) north
    n_between = int(((LAT > 47.35) & (LAT < 47.40)).sum())
    assert r["north_edge_sea_at_max_km2"] >= n_between
    assert r["band_changed_km2"] == pytest.approx(r["north_edge_sea_at_max_km2"])
    assert r["band_changed_km2"] < int((LAT > 47.35).sum())  # beyond the bulge nothing changes
    # nothing south of the parallel changes: both readings put it in the Volga sector
    south = (LAT < 47.345)
    assert (member[south] == "volga").all()
