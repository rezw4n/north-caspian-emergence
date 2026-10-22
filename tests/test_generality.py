"""caspian.generality: the generality-test helpers (PREREG Amendment 16), on
synthetic inputs only."""

import math

import numpy as np
import pandas as pd
import pyproj
import pytest

from caspian import generality as G


# --- levels ----------------------------------------------------------------------------------

def test_ngvd29_feet_to_navd88_metres():
    assert G.ngvd29_ft_to_navd88_m(4200.0, 0.977) == pytest.approx(4200 * 0.3048 + 0.977)
    np.testing.assert_allclose(G.ngvd29_ft_to_navd88_m([0.0, 1.0], 1.0), [1.0, 1.3048])


def _series(days, values):
    return pd.Series(values, index=pd.to_datetime(days))


def test_level_on_dates_exact_interpolated_and_gap():
    s = _series(["2000-01-01", "2000-01-11", "2000-02-20"], [10.0, 11.0, 20.0])
    out = G.level_on_dates(s, pd.to_datetime(["2000-01-01", "2000-01-06", "2000-01-25", "1999-12-31", "2000-03-01"]),
                           max_gap_days=16)
    assert out[0] == 10.0                       # the day's own value
    assert out[1] == pytest.approx(10.5)        # 5 of 10 days
    assert np.isnan(out[2])                     # bracketing values 40 days apart
    assert np.isnan(out[3]) and np.isnan(out[4])   # outside the record


def test_level_on_dates_semi_monthly_records_bridge_at_16_days():
    s = _series(["1985-06-01", "1985-06-15", "1985-07-01"], [1.0, 2.0, 3.0])
    out = G.level_on_dates(s, pd.to_datetime(["1985-06-08", "1985-06-23"]), max_gap_days=16)
    assert out[0] == pytest.approx(1.5)
    assert out[1] == pytest.approx(2.5)
    assert np.isnan(G.level_on_dates(s, pd.to_datetime(["1985-06-23"]), max_gap_days=10))[0]


def test_level_on_dates_ignores_nan_values():
    s = _series(["2000-01-01", "2000-01-02", "2000-01-03"], [1.0, np.nan, 3.0])
    assert G.level_on_dates(s, pd.to_datetime(["2000-01-02"]))[0] == pytest.approx(2.0)


def test_epochs_boundaries():
    e = G.epochs(pd.to_datetime(["1986-05-31", "1986-06-01", "1986-06-02", "2010-12-31", "2011-01-01"]), "1986-06-01")
    assert list(e) == ["E1", "E1", "E2", "E2", "E3"]


def test_parity_alternates_acquisitions_in_time_order():
    df = pd.DataFrame({"epoch": ["E2"] * 5, "bin": [7, 7, 7, 7, 8], "acq": ["a", "b", "b", "c", "d"],
                       "t": pd.to_datetime(["2001-01-03", "2001-01-01", "2001-01-01", "2001-01-05", "2001-01-01"])})
    p = G.parity(df)
    assert list(p) == [1, 0, 0, 0, 0]          # b (first), a (second), c (third); d alone in its bin
    assert list(p.index) == list(df.index)


# --- grid and zone ---------------------------------------------------------------------------

def test_grid_is_aligned_and_covers_the_box():
    b = G.BASINS["great_salt_lake"]
    tr, shape = G.grid(b["bbox"], b["crs"], 60)
    assert tr[0] == 60 and tr[4] == -60 and tr[2] % 60 == 0 and tr[5] % 60 == 0
    t = pyproj.Transformer.from_crs("EPSG:4326", b["crs"], always_xy=True)
    w, s, e, n = b["bbox"]
    for lon, lat in ((w, s), (w, n), (e, s), (e, n), ((w + e) / 2, n)):
        x, y = t.transform(lon, lat)
        assert tr[2] <= x <= tr[2] + 60 * shape[1]
        assert tr[5] - 60 * shape[0] <= y <= tr[5]


def test_open_water_needs_surroundings():
    low = np.zeros((41, 41), bool)
    low[5:36, 5:36] = True                      # a 31-cell lake
    ow = G.open_water(low, 5, 0.8)
    assert ow[20, 20] and not ow[5, 5]          # centre qualifies, a corner does not
    assert not (ow & ~low).any()


def test_largest_component():
    m = np.zeros((10, 10), bool)
    m[0:2, 0:2] = True
    m[5:9, 5:9] = True
    lc = G.largest_component(m)
    assert lc.sum() == 16 and not lc[0, 0]
    assert not G.largest_component(np.zeros((3, 3), bool)).any()


def test_zone_v6_band_connected_and_dilated():
    high = np.zeros((30, 30), bool)
    high[5:25, 5:25] = True
    high[0:3, 27:30] = True                     # an unconnected pond
    low = np.zeros_like(high)
    low[10:20, 10:20] = True
    seed = np.zeros_like(high)
    seed[15, 15] = True
    z = G.zone_v6(high, low, seed, 1)
    assert z[5, 5] and z[9, 9] and z[4, 5]      # band and its dilation
    assert not z[15, 15]                        # inside the low extent, beyond the dilation
    assert not z[1, 28]                         # the pond is not lake-connected


def test_tile_cover_with_buffer_ring():
    zone = np.zeros((10, 10), bool)
    zone[0, 0] = True                           # cell centre at (30, -30) with the transform below
    tiles = G.tile_cover(zone, [60, 0, 0, 0, -60, 0], 5000, 1)
    assert (0, -1) in tiles and len(tiles) == 9
    assert G.tile_cover(np.zeros((2, 2), bool), [60, 0, 0, 0, -60, 0], 5000) == []


# --- arm geometry -----------------------------------------------------------------------------

def test_medial_line_follows_a_peninsula_and_stops_at_its_base():
    land = np.zeros((20, 30), bool)
    for r in range(5, 20):
        c = 10 + (r - 5) // 3                  # a 3-cell strip drifting east southward
        land[r, c:c + 3] = True
    land[0:5, :] = True                         # the mainland (a full-width run)
    line = G.medial_line(land, (19, 15), direction=-1, max_shift=2, max_width=10)
    rows = [r for r, _ in line]
    assert rows[0] == 19 and rows[-1] == 5      # stops before the 30-cell-wide mainland rows
    for r, c in line:
        assert land[r, int(round(c))]


def test_split_box_by_a_line_crossing_it():
    parts = G.split_box([(1, 5), (9, 5)], (0, 0, 10, 10))
    assert len(parts) == 2
    assert sum(p.area for p in parts) == pytest.approx(100)
    assert parts[0].area == pytest.approx(50)


# --- verdict (Amendment 16 item 9) ----------------------------------------------------------------

def test_in_basin_majority_of_evaluable_sectors():
    assert G.in_basin([True, True]) is True
    assert G.in_basin([True, False]) is False   # strictly more than half
    assert G.in_basin([True, None]) is True     # one evaluable arm
    assert G.in_basin([None, None]) is None
    assert G.in_basin([pd.NA, False]) is False


@pytest.mark.parametrize("flags,pc,expected", [
    ([True, True, True, False, False], False, "recurs"),
    ([True, True, True, True, True], None, "recurs"),
    ([True, False, False, False, False], False, "does not recur"),
    ([False] * 5, None, "does not recur"),
    ([True, True, False, False, False], False, "partial"),
    ([True, True, True, True, True], True, "partial"),       # the positive control fails
    ([False] * 5, True, "partial"),
])
def test_verdict(flags, pc, expected):
    v = G.verdict({f"p{i}": f for i, f in enumerate(flags)}, pc)
    assert v["verdict"] == expected
    assert v["positive_control"] == {None: "not evaluable", True: "fail", False: "pass"}[pc]


def test_verdict_not_run_without_evaluable_products():
    v = G.verdict({"a": None, "b": None}, None)
    assert v["verdict"].startswith("not run")
