"""Tests of caspian.rev3 (PREREG Amendment 20 helpers)."""

import math

import numpy as np
import pandas as pd
import pytest

from caspian import rev3
from caspian.geography import SECTOR_ORDER


# --- item 1 ----------------------------------------------------------------------------------

def test_marine_codes_drop_the_deltaic_sectors():
    codes = rev3.marine_codes(SECTOR_ORDER, ("volga", "ural"))
    assert [SECTOR_ORDER[k - 1] for k in codes] == ["west", "interfluve", "northeast", "komsomolets", "open_shelf"]
    with pytest.raises(ValueError):
        rev3.marine_codes(SECTOR_ORDER, ("volga", "nowhere"))


def test_area_bias_rel_matches_metrics_from_counts():
    from caspian import hindcast_tiled as ht
    c = {"tp": 900, "fp": 150, "fn": 100, "fp_out": 30, "fn_out": 10}
    assert rev3.area_bias_rel(c) == pytest.approx(ht.metrics_from_counts(c)["area_bias_rel_total"])
    assert math.isnan(rev3.area_bias_rel({"tp": 0, "fp": 5, "fn": 0, "fp_out": 0, "fn_out": 0}))


def test_band_display_parts_samples_the_class_raster_at_cell_centres(tmp_path):
    import pyproj
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from caspian.config import CRS_EQUAL_AREA

    # a 250 m extents grid of 40 x 60 cells around the projection centre (50 E, 45.5 N)
    tr = np.array([250.0, 0.0, -7500.0, 0.0, -250.0, 5000.0])
    high = np.ones((40, 60), np.uint8)
    low = np.zeros((40, 60), np.uint8)
    low[:, :10] = 1                       # still water in the low stand: not band
    low[:, 10:12] = 255                   # no low-stand observation: not band either
    high[:5, :] = 0                       # not water in the high stand: not band
    ext = tmp_path / "extents.npz"
    np.savez(ext, high=high, low=low, transform=tr, crs=np.array(CRS_EQUAL_AREA))
    # a class raster in lon/lat at 0.004 degrees whose origin is not a multiple of the window's
    # bounds: class 2 west of 50.01 E, class 9 east of it
    res, lon0, lat0 = 0.004, 49.7013, 45.7007
    cls = np.full((120, 160), 9, np.uint8)
    lon_centres = lon0 + res * (np.arange(160) + 0.5)
    cls[:, lon_centres < 50.01] = 2
    tif = tmp_path / "classes.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=120, width=160, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(lon0, lat0, res, res)) as ds:
        ds.write(cls, 1)
    polys = {"a": box(49.0, 45.0, 50.02, 46.0), "b": box(50.02, 45.0, 51.0, 46.0)}
    p = rev3.band_display_parts(ext, tif, polys, deltaic=("a",))
    assert np.array_equal(p["band"], (high == 1) & (low == 0))
    # the expected class at each cell centre, computed directly
    X, Y = np.meshgrid(tr[2] + 250.0 * (np.arange(60) + 0.5), tr[5] - 250.0 * (np.arange(40) + 0.5))
    lon, lat = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform(X, Y)
    col = np.floor((lon - lon0) / res).astype(int)
    assert np.array_equal(p["lake2000"], cls[0, col] == 2)
    assert p["lake2000"].any() and (~p["lake2000"]).any()
    sec = np.where(lon < 50.02, 1, 2)
    assert np.array_equal(p["sector"], sec)
    assert np.array_equal(p["not_seabed"], p["band"] & (sec == 1) & ~p["lake2000"])
    assert p["not_seabed"].any()
    # without an outline mask the raster is the outline
    assert np.array_equal(p["lake2000_raster"], p["lake2000"]) and np.array_equal(p["not_seabed_raster"], p["not_seabed"])
    # with one, the mask decides and the raster form is kept beside it
    lake = np.zeros((40, 60), bool)
    lake[:, :20] = True
    out = tmp_path / "outline.npz"
    np.savez(out, lake2000=lake, transform=tr)
    q = rev3.band_display_parts(ext, tif, polys, deltaic=("a",), outline_npz=out)
    assert np.array_equal(q["lake2000"], lake) and np.array_equal(q["lake2000_raster"], p["lake2000"])
    assert np.array_equal(q["not_seabed"], q["band"] & (sec == 1) & ~lake)
    assert np.array_equal(q["not_seabed_raster"], p["not_seabed"])
    assert not np.array_equal(q["not_seabed"], q["not_seabed_raster"])
    np.savez(tmp_path / "wrong.npz", lake2000=lake[:, :50], transform=tr)
    with pytest.raises(ValueError):       # an outline mask on another grid
        rev3.band_display_parts(ext, tif, polys, deltaic=("a",), outline_npz=tmp_path / "wrong.npz")
    with pytest.raises(ValueError):       # a class raster that does not cover the grid
        with rasterio.open(tmp_path / "small.tif", "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(49.99, 45.51, res, res)) as ds:
            ds.write(np.zeros((4, 4), np.uint8), 1)
        rev3.band_display_parts(ext, tmp_path / "small.tif", polys, deltaic=("a",))


def test_lake_mask_from_polygon_marks_the_cells_whose_centre_is_inside():
    import pyproj
    from shapely.geometry import Polygon

    from caspian.config import CRS_EQUAL_AREA

    tr = np.array([250.0, 0.0, -5000.0, 0.0, -250.0, 5000.0])       # 40 x 40 cells of 250 m around the centre
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    ring = lambda h: [inv.transform(x, y) for x, y in ((-h, -h), (h, -h), (h, h), (-h, h))]
    # a square of half-side 2,600 m with a square hole of half-side 600 m; no cell centre lies on an edge
    m = rev3.lake_mask_from_polygon(Polygon(ring(2600.0), [ring(600.0)]), tr, (40, 40), CRS_EQUAL_AREA)
    X, Y = np.meshgrid(tr[2] + 250.0 * (np.arange(40) + 0.5), tr[5] - 250.0 * (np.arange(40) + 0.5))
    inside = (np.abs(X) < 2600) & (np.abs(Y) < 2600)
    hole = (np.abs(X) < 600) & (np.abs(Y) < 600)
    assert np.array_equal(m, inside & ~hole)
    assert m.sum() == 20 * 20 - 4 * 4


# --- item 2 ----------------------------------------------------------------------------------

def test_quantile_from_counts_equals_numpy_percentile():
    rng = np.random.default_rng(1)
    pixel = 60.0
    for n in (1, 2, 7, 100, 1001):
        # distances between lattice points: sqrt(i^2 + j^2) * pixel
        ij = rng.integers(0, 40, (n, 2))
        d = np.hypot(ij[:, 0], ij[:, 1]) * pixel
        k = rev3.lattice_code(d, pixel)
        kk, cc = np.unique(k, return_counts=True)
        for q in (0.0, 0.25, 0.5, 0.95, 1.0):
            assert rev3.quantile_from_counts(kk, cc, q, pixel) == pytest.approx(np.percentile(d, 100 * q), abs=1e-9)


def test_quantile_from_counts_with_infinite_distances():
    pixel = 60.0
    d = np.array([60.0, 120.0, np.inf, np.inf])
    kk, cc = np.unique(rev3.lattice_code(d, pixel), return_counts=True)
    assert rev3.quantile_from_counts(kk, cc, 0.0, pixel) == 60.0
    assert rev3.quantile_from_counts(kk, cc, 1 / 3, pixel) == pytest.approx(120.0)   # exactly on a finite value
    assert math.isinf(rev3.quantile_from_counts(kk, cc, 0.5, pixel))                 # between 120 and inf
    assert math.isinf(rev3.quantile_from_counts(kk, cc, 0.95, pixel))
    assert math.isnan(rev3.quantile_from_counts(np.empty(0), np.empty(0), 0.5, pixel))
    with pytest.raises(ValueError):
        rev3.lattice_code([np.nan], pixel)


def test_blockdist_pools_units_and_matches_direct_statistics():
    rng = np.random.default_rng(2)
    pixel = 60.0
    bd = rev3.BlockDist(pixel)
    pool: dict = {}
    for _ in range(4):                                   # four "units"
        n = 500
        bi, bj = rng.integers(0, 3, n), rng.integers(0, 2, n)
        ij = rng.integers(0, 30, (n, 2))
        d = np.hypot(ij[:, 0], ij[:, 1]) * pixel * rng.choice([-1.0, 1.0], n)       # signed
        bd.add("g", bi, bj, d)
        for i, j, v in zip(bi, bj, np.abs(d)):
            pool.setdefault((int(i), int(j)), []).append(v)
    t = bd.table()
    assert len(t) == len(pool)
    for r in t.itertuples():
        v = np.asarray(pool[(r.block_i, r.block_j)])
        assert r.n == v.size
        assert r.median_abs_m == pytest.approx(np.median(v), abs=1e-9)
        assert r.p95_abs_m == pytest.approx(np.percentile(v, 95), abs=1e-9)
    bd.add("empty", [], [], [])
    assert "empty" not in bd.s


def test_boundary_blocks():
    b = pd.DataFrame({"i": [0, 1, 2, 3, 0, 1, 2, 3], "j": [0, 0, 0, 0, 1, 1, 1, 1],
                      "sector": ["a", "a", "a", "b", "a", "a", "a", "b"]})
    got = rev3.boundary_blocks(b)
    # columns 2 and 3 touch the a|b boundary; columns 0 and 1 do not
    assert got.tolist() == [False, False, True, True, False, False, True, True]


# --- item 4 ----------------------------------------------------------------------------------

def test_surface_stratum_and_height_band():
    s = rev3.surface_stratum([0.0, 0.0999, 0.1, 0.4999, 0.5, 1.0, np.nan])
    assert s.tolist() == ["bare", "bare", "mixed", "mixed", "vegetated", "vegetated", ""]
    hb = rev3.height_band([-29.5, -29.0, -28.51, -28.5, -26.0, -20.0, np.nan], lowest=-29.0, step=0.5, n_bands=6)
    assert hb.tolist() == [-1, 0, 0, 1, 5, 5, -1]


def test_histboot_matches_the_exact_block_bootstrap():
    from caspian import rev2_d
    rng = np.random.default_rng(4)
    # dense bins (tens of points per millimetre near the median), as in the lidar tables
    blocks = np.repeat(np.array(["b1", "b2", "b3", "b4", "b5"]), [40000, 15000, 90000, 6000, 30000])
    shift = {"b1": 0.0, "b2": 0.6, "b3": -0.2, "b4": 1.5, "b5": 0.1}
    e = np.array([shift[b] for b in blocks]) + rng.normal(scale=0.3, size=blocks.size)
    hb = rev3.HistBoot(e, blocks, width=1e-3)
    ones = np.ones(hb.k)
    mean, med, nmad = hb.stats(ones)
    assert mean == pytest.approx(e.mean(), abs=1e-12)
    assert med == pytest.approx(np.median(e), abs=1e-3)
    assert nmad == pytest.approx(1.4826 * np.median(np.abs(e - np.median(e))), abs=2e-3)
    got = hb.replicates(100, 11)
    want = rev2_d.block_bootstrap_stats(e, blocks, 100, 11)          # columns: mean, median, NMAD, RMSE
    assert np.allclose(got[:, 0], want[:, 0], atol=1e-12)
    assert np.abs(got[:, 1] - want[:, 1]).max() < 1e-3
    assert np.abs(got[:, 2] - want[:, 2]).max() < 2e-3
    lo, hi = rev3.interval(got[:, 1])
    assert lo <= np.median(got[:, 1]) <= hi
    with pytest.raises(ValueError):
        rev3.HistBoot([1.0, np.nan], ["a", "b"])
    one = rev3.HistBoot([0.5, 0.5, 0.5], ["a", "a", "a"])
    assert one.stats([1])[0] == pytest.approx(0.5) and one.stats([1])[1] == pytest.approx(0.5, abs=1e-3)


# --- item 5 ----------------------------------------------------------------------------------

def test_distance_to_blocks():
    d = rev3.distance_to_blocks([5.0, 15.0, 13.0, -3.0], [5.0, 5.0, 14.0, -4.0], [0.0], [0.0], 10.0)
    assert d.tolist() == pytest.approx([0.0, 5.0, 5.0, 5.0])
    two = rev3.distance_to_blocks([15.0], [5.0], [0.0, 18.0], [0.0, 0.0], 10.0)
    assert two.tolist() == pytest.approx([3.0])
    assert np.isinf(rev3.distance_to_blocks([1.0], [1.0], [], [], 10.0)).all()


# --- item 6 ----------------------------------------------------------------------------------

def test_native_cell_and_half_metre_distance():
    assert rev3.native_cell_m(15.0) == pytest.approx(463.3, abs=0.1)        # 15 arc-seconds of latitude
    assert rev3.native_cell_m(60.0) == pytest.approx(1853.2, abs=0.2)       # one arc-minute: a nautical mile
    assert rev3.half_metre_distance_m(0.4) == pytest.approx(1250.0)
    assert math.isinf(rev3.half_metre_distance_m(0.0))


def test_tier_levels_and_sector_class():
    #            below   resol(ctrl)  resol(cell)  resol(0.5m)  material  no-control  inf      not evaluable
    D = np.array([400.0, 2000.0,      450.0,       1000.0,      3000.0,   3000.0,     np.inf,  5000.0])
    Q = np.array([500.0, 500.0,       300.0,       500.0,       500.0,    500.0,      500.0,   np.nan])
    C = np.array([100.0, 2500.0,      100.0,       100.0,       800.0,    np.nan,     800.0,   100.0])
    ev = np.array([True, True,        True,        True,        True,     True,       True,    False])
    t = rev3.tier_levels(D, Q, C, ev, cell_m=463.0, dist_m=1250.0)
    assert t.tolist() == ["below_floor", "resolution", "resolution", "resolution", "material", "resolution",
                          "material", "not_evaluable"]
    s = rev3.tier_sector(t)
    assert s == {"n_evaluable": 7, "n_above_floor": 6, "n_material": 2, "n_resolution": 4, "class": "resolution"}
    assert rev3.tier_sector(["material", "material", "below_floor"])["class"] == "material"
    assert rev3.tier_sector(["below_floor", "below_floor", "material"])["class"] == "pass"
    assert rev3.tier_sector(["resolution", "below_floor"])["class"] == "pass"          # exactly half: not a majority
    assert rev3.tier_sector(["not_evaluable"])["class"] is None


def test_mbb_median_interval_is_seeded_and_bounded():
    v = np.arange(12, dtype=float)
    m = v >= 4                                              # 8 of 12 material
    a = rev3.mbb_median_interval(v, m, 500, 7)
    b = rev3.mbb_median_interval(v, m, 500, 7)
    assert a == b
    assert a["median"] == 5.5 and a["lo"] <= 5.5 <= a["hi"] and 0 <= a["p_material_majority"] <= 1
    const = rev3.mbb_median_interval(np.full(9, 2.0), np.ones(9, bool), 200, 1)
    assert const["lo"] == const["hi"] == 2.0 and const["p_material_majority"] == 1.0
    assert rev3.mbb_median_interval([], [], 10, 1)["n_levels"] == 0


# --- item 7 ----------------------------------------------------------------------------------

def test_icc_oneway_extremes_and_kish():
    # identical values inside each cluster, different between: rho = 1
    v = np.repeat([0.0, 1.0, 2.0], 50)
    c = np.repeat(["a", "b", "c"], 50)
    r = rev3.icc_oneway(v, c)
    assert r["rho"] == pytest.approx(1.0) and r["k"] == 3 and r["n"] == 150
    # independent noise: rho near zero (floored at zero)
    rng = np.random.default_rng(3)
    r0 = rev3.icc_oneway(rng.normal(size=6000), np.repeat(np.arange(30), 200))
    assert 0.0 <= r0["rho"] < 0.02
    assert math.isnan(rev3.icc_oneway([1.0, 2.0], ["a", "a"])["rho"])
    # balanced design against the textbook estimator
    k, m = 8, 25
    means = rng.normal(size=k)
    vv = (means[:, None] + rng.normal(scale=0.5, size=(k, m))).ravel()
    cc = np.repeat(np.arange(k), m)
    msb = m * np.var(vv.reshape(k, m).mean(axis=1), ddof=1)
    msw = np.mean(np.var(vv.reshape(k, m), axis=1, ddof=1))
    assert rev3.icc_oneway(vv, cc)["rho"] == pytest.approx((msb - msw) / (msb + (m - 1) * msw))
    e = rev3.kish_effective_n([50, 50, 50], 1.0)
    assert e["n_eff"] == pytest.approx(3.0) and e["deff"] == pytest.approx(50.0)
    assert rev3.kish_effective_n([50, 50, 50], 0.0)["n_eff"] == pytest.approx(150.0)
    assert math.isnan(rev3.kish_effective_n([10, 10], float("nan"))["n_eff"])


# --- item 8 ----------------------------------------------------------------------------------

def test_abs_summary():
    s = rev3.abs_summary([-0.3, -0.05, 0.0, 0.05, 0.15, np.nan])
    assert s["n"] == 5 and s["abs_median"] == pytest.approx(0.05)
    assert s["share_above_0.1"] == pytest.approx(0.4) and s["share_above_0.2"] == pytest.approx(0.2)
    assert rev3.abs_summary([])["n"] == 0


# --- the comparisons the text states in words -------------------------------------------------

def test_text_claims_guard_passes_on_the_results_and_stops_on_an_altered_one():
    """tools/metrics_rev_x3.text_claims: every comparison the article states in words holds on
    results/metrics.json, and the guard stops the metrics build when a result is altered."""
    import copy
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "metrics.json"
    if not path.exists():
        pytest.skip("results/metrics.json has not been built")
    m = json.loads(path.read_text(encoding="utf-8"))
    if "rev3.marine.n_changed_checked" not in m:
        pytest.skip("the revision analyses have not been run")
    spec = importlib.util.spec_from_file_location("metrics_rev_x3", root / "tools" / "metrics_rev_x3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.text_claims(m)
    for key, bad in (("rev3.marine.n_changed_checked", 1), ("rev3.lidar.worst.coasts.over_d_min.max", 0.7),
                     ("rev3.tier.srtm15p.openshelf.class", "material")):
        altered = copy.deepcopy(m)
        altered[key]["value"] = bad
        with pytest.raises(SystemExit):
            mod.text_claims(altered)
    mod.text_claims({})          # without the revision results the guard does nothing
