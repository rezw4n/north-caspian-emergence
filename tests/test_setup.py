"""Tests for configuration and secret handling."""

import os

import pyproj
import shapely
import pytest

import caspian
from caspian import config, credentials


def test_equal_area_crs_is_laea_centred_on_north_caspian():
    crs = pyproj.CRS.from_user_input(config.CRS_EQUAL_AREA)
    params = {p.name: p.value for p in crs.coordinate_operation.params}
    assert crs.coordinate_operation.method_name == "Lambert Azimuthal Equal Area"
    assert params["Latitude of natural origin"] == pytest.approx(45.5)
    assert params["Longitude of natural origin"] == pytest.approx(50.0)


def test_equal_area_crs_preserves_area():
    # A 1-degree cell near the origin has the same projected area as on the ellipsoid.
    from shapely.geometry import box

    geod = pyproj.Geod(ellps="WGS84")
    cell = box(49.5, 45.0, 50.5, 46.0)
    area_ellipsoid = abs(geod.geometry_area_perimeter(cell)[0])
    to_laea = pyproj.Transformer.from_crs(config.CRS_EXCHANGE, config.CRS_EQUAL_AREA, always_xy=True)
    area_laea = shapely.transform(cell, lambda x, y: to_laea.transform(x, y), interleaved=False).area
    assert area_laea == pytest.approx(area_ellipsoid, rel=1e-4)


def test_describe_never_reveals_secret():
    secret = "abc123-very-secret"
    text = credentials.describe(secret)
    assert secret not in text and "abc" not in text
    assert credentials.describe(None) == "absent"


def test_get_secret_prefers_process_env(monkeypatch):
    monkeypatch.setenv("CASPIAN_TEST_SECRET", "x" * 5)
    assert credentials.get_secret("CASPIAN_TEST_SECRET") == "xxxxx"


def test_foreign_gdal_dirs_removed_from_path(tmp_path, monkeypatch):
    foreign = tmp_path / "osgeo" / "bin"
    foreign.mkdir(parents=True)
    (foreign / "gdal999.dll").write_bytes(b"")
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setattr(caspian._sys, "platform", "win32")
    monkeypatch.setenv("PATH", os.pathsep.join([str(plain), str(foreign)]))
    caspian._drop_foreign_gdal_from_path()
    entries = os.environ["PATH"].split(os.pathsep)
    assert str(plain) in entries and str(foreign) not in entries
