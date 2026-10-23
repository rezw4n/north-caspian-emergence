"""Tests of caspian.lakebed (PREREG Amendment 19 item 13; D-150)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import caspian  # noqa: F401
from caspian import lakebed as LB


def test_contour_feet_to_navd88():
    assert LB.ft_ngvd29_to_navd88_m(4193.0, 0.977) == pytest.approx(1279.0034, abs=1e-4)
    assert LB.ft_ngvd29_to_navd88_m(4200.0, 0.930) == pytest.approx(1281.09, abs=1e-4)
    assert LB.ft_ngvd29_to_navd88_m(4167.0, 0.0) == pytest.approx(1270.1016, abs=1e-4)


def test_to_navd88_constant_and_field():
    z = np.array([[1270.0, np.nan], [1280.5, 1281.0]], np.float32)
    out = LB.to_navd88(z, 0.93)
    assert out.dtype == np.float32
    assert np.isnan(out[0, 1])
    assert out[1, 0] == pytest.approx(1281.43, abs=1e-3)
    field = np.array([[0.9, 0.95], [0.97, 1.0]], np.float32)
    out = LB.to_navd88(z, field)
    assert out[1, 1] == pytest.approx(1282.0, abs=1e-3)
    assert np.isnan(out[0, 1])


def test_lattice_interpolator_is_exact_for_planes_and_extrapolates():
    lon = np.array([-113.2, -113.0, -112.8])
    lat = np.array([40.4, 40.6])
    LON, LAT = np.meshgrid(lon, lat)
    v = 0.9 + 0.1 * (LON + 113.2) + 0.05 * (LAT - 40.4)
    f = LB.lattice_interpolator(lon, lat, v)
    q = f(np.array([-113.1, -112.7]), np.array([40.5, 40.7]))
    assert q[0] == pytest.approx(0.9 + 0.01 + 0.005)
    assert q[1] == pytest.approx(0.9 + 0.05 + 0.015)   # beyond the lattice: linear extrapolation
    with pytest.raises(ValueError):
        LB.lattice_interpolator(lon, lat, v[:, :2])


def test_average_to_grid_is_the_block_mean_without_nodata():
    rasterio = pytest.importorskip("rasterio")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    nd = -3.4028234663852886e38
    a = np.arange(36 * 36, dtype=np.float32).reshape(36, 36)
    a[0, 0] = nd                        # one no-data cell in the first block
    a[6:12, 6:12] = nd                  # a block with no valid cell
    crs = "+proj=laea +lat_0=41.2 +lon_0=-112.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
    prof = {"driver": "GTiff", "width": 36, "height": 36, "count": 1, "dtype": "float32", "crs": crs,
            "transform": from_origin(0.0, 360.0, 10.0, 10.0), "nodata": nd}
    with MemoryFile() as mem:
        with mem.open(**prof) as w:
            w.write(a, 1)
        with mem.open() as src:
            out = LB.average_to_grid(src, [60.0, 0.0, 0.0, 0.0, -60.0, 360.0], (6, 6), crs)
    blk = a[:6, :6].astype(float)
    assert out[0, 0] == pytest.approx(blk[blk > -1e30].mean(), rel=1e-5)
    assert out[0, 1] == pytest.approx(a[:6, 6:12].mean(), rel=1e-5)
    assert out[5, 5] == pytest.approx(a[30:, 30:].mean(), rel=1e-5)
    assert np.isnan(out[1, 1])


def test_bins_against_boundaries():
    # the 3DEP rule of scripts/g1_hindcast.py: bins >= 77 (north) and >= 87 (south)
    assert LB.min_bin_at_or_above(1277.650024, 1300.0) == 77
    assert LB.min_bin_at_or_above(1278.660034, 1300.0) == 87
    # upper edge at or below the top of the soundings
    assert LB.max_bin_at_or_below(1279.0034, 1300.0) == 89
    assert LB.max_bin_at_or_below(1279.0, 1300.0) == 89      # upper edge of bin 89 is 1,279.0
    assert LB.max_bin_at_or_below(1278.99, 1300.0) == 88
    assert LB.max_bin_at_or_below(1279.2612, 1300.0) == 91
    lo, hi = LB.bin_edges(89, 1300.0)
    assert (float(lo), float(hi)) == (1278.9, 1279.0)


def test_source_class():
    c = LB.source_class([88, 89, 90, 92, 109, 110, 111], 1279.0, 1281.05, 1300.0)
    assert list(c) == ["sounded", "sounded", "map_contours", "map_contours", "map_contours", "mixed", "ned"]
    assert list(LB.source_class([110, 111], 1279.0, 1281.1, 1300.0)) == ["map_contours", "ned"]


def test_survey_timing():
    d = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    lv = pd.Series(1281.0, index=d)
    lv[(d >= "2003-01-01") & (d <= "2003-12-31")] = 1279.6        # the survey year
    lv[d >= "2015-01-01"] = 1278.0                                  # later low stand
    lv[(d >= "1995-01-01") & (d <= "1995-01-10")] = 1279.15         # an earlier dip
    t = LB.survey_timing(lv, [81, 85, 91, 96], ("2003-01-01", "2003-12-31"), level_offset_m=1300.0)
    t = t.set_index("bin")
    assert t.loc[81, "upper_m"] == pytest.approx(1278.2)
    assert bool(t.loc[81, "under_water_in_survey"]) and bool(t.loc[81, "within_sounding_depth"])
    assert t.loc[81, "first_reached"] == pd.Timestamp("2015-01-01")
    assert bool(t.loc[81, "first_reached_after_survey"])
    assert bool(t.loc[85, "under_water_in_survey"]) and not bool(t.loc[85, "within_sounding_depth"])
    assert t.loc[91, "first_reached"] == pd.Timestamp("1995-01-01")   # upper edge 1,279.2 reached before
    assert not bool(t.loc[91, "first_reached_after_survey"])
    assert not bool(t.loc[96, "under_water_in_survey"])               # upper edge 1,279.7 above the survey low
    with pytest.raises(ValueError):
        LB.survey_timing(lv, [81], ("2030-01-01", "2030-02-01"), level_offset_m=1300.0)
