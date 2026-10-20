"""The study domain and sectors must partition the domain exactly (brief section 6)."""

import itertools

import pyproj
import pytest
import shapely

from caspian import config
from caspian.geography import CAPE_TYUB_KARAGAN, CHECHEN_ISLAND, DOMAIN, SECTOR_ORDER, sectors

_to_laea = pyproj.Transformer.from_crs(config.CRS_EXCHANGE, config.CRS_EQUAL_AREA, always_xy=True)


def _area_km2(geom):
    return shapely.transform(geom, lambda x, y: _to_laea.transform(x, y), interleaved=False).area / 1e6


def test_all_sectors_present_and_valid():
    s = sectors()
    assert list(s) == SECTOR_ORDER
    for key, (name, geom) in s.items():
        assert geom.is_valid and not geom.is_empty, key
        assert geom.geom_type == "Polygon", f"{key} split into several parts"


def test_sectors_are_disjoint():
    s = sectors()
    for (ka, (_, a)), (kb, (_, b)) in itertools.combinations(s.items(), 2):
        assert _area_km2(a.intersection(b)) < 1e-3, f"{ka} overlaps {kb}"


def test_sectors_cover_domain():
    union = shapely.union_all([g for _, g in sectors().values()])
    assert _area_km2(DOMAIN.symmetric_difference(union)) < 1e-3


def test_boundary_anchors_on_domain_edge():
    for pt in (CHECHEN_ISLAND, CAPE_TYUB_KARAGAN):
        assert DOMAIN.exterior.distance(shapely.Point(pt)) < 1e-9


def test_domain_is_north_of_boundary_line():
    # A point just south of the Chechen-Tyub-Karagan line in the open sea is outside.
    assert not DOMAIN.contains(shapely.Point(49.0, 44.15))
    assert DOMAIN.contains(shapely.Point(49.0, 44.60))
