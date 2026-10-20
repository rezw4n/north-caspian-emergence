"""caspian.exposure (H3, PREREG section 10) on synthetic inputs only: counting, the
area-matching shift, the error model, corrected and fused surfaces, emerged areas and
their conformal intervals, equivalent levels, exposure rules and scenario dates."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from shapely.geometry import LineString, Point, box

from caspian import exposure as ex
from caspian import hindcast

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a7_hindcast", ROOT / "scripts" / "a7_hindcast.py")
a7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a7)


def rng():
    return np.random.default_rng(20261021)


# --- counting -----------------------------------------------------------------------------

def test_below_counts_match_brute_force_across_tiles():
    g = rng()
    F = g.normal(-28, 1, (37, 53)).astype(np.float32)
    F[3, 4], F[5, 6] = np.inf, np.nan
    lab = g.integers(0, 4, F.shape)
    mask = g.random(F.shape) > 0.3
    T = np.array([-29.0, -28.5, -28.0, -27.0])
    below, total = ex.below_counts(F, lab, 3, T, mask=mask, tile=8)
    for k in (1, 2, 3):
        sel = (lab == k) & mask
        assert total[k - 1] == sel.sum()
        for j, t in enumerate(T):
            assert below[k - 1, j] == np.sum(sel & (F.astype(float) < t))
    with pytest.raises(ValueError):
        ex.below_counts(F, lab, 3, [1.0, 1.0])


def _code_outside(shape, g):
    code = g.choice(np.array([1, 0, -1, -2], np.int8), size=shape, p=[0.35, 0.35, 0.1, 0.2])
    outside = g.choice(np.array([1, 0, -1], np.int8), size=shape)
    return code, outside


def test_unit_region_counts_equal_the_hindcast_unit_counts():
    g = rng()
    shape = (40, 30)
    F = g.normal(-28, 0.7, shape).astype(np.float32)
    h = F + g.normal(0, 0.01, shape).astype(np.float32)
    h[0, :5] = np.nan
    code, outside = _code_outside(shape, g)
    lab = g.integers(0, 3, shape)
    T = -28.2 + ex.C_GRID[::50]
    finite = np.isfinite(h)
    c = ex.unit_region_counts(F, code, outside, finite, lab, 2, T, tile=7)
    for k in (1, 2):
        cnt, water = a7.unit_counts(F, code, outside, finite, lab == k, T, tile=7)
        assert np.array_equal(c["below"][k - 1], cnt)
        assert c["water"][k - 1] == water
    m = g.random(shape) > 0.5
    c2 = ex.unit_region_counts(F, code, outside, finite, lab, 2, T, mask=m, tile=7)
    cnt, water = a7.unit_counts(F, code, outside, finite, (lab == 1) & m, T, tile=7)
    assert np.array_equal(c2["below"][0], cnt) and c2["water"][0] == water


def test_pair_region_counts_brute_force():
    g = rng()
    shape = (25, 31)
    F = g.normal(-28, 0.5, shape)
    code_t, outside = _code_outside(shape, g)
    code_u, _ = _code_outside(shape, g)
    lab = g.integers(0, 3, shape)
    finite = g.random(shape) > 0.1
    c = ex.pair_region_counts(F, -27.9, -28.3, code_t, code_u, outside, finite, lab, 2, tile=6)
    kt, wt = ex.known_masks(code_t, outside)
    ku, wu = ex.known_masks(code_u, outside)
    for k in (1, 2):
        com = kt & ku & finite & (lab == k)
        assert c["n_common"][k - 1] == com.sum()
        assert c["obs_top"][k - 1] == (com & wt).sum() and c["obs_u"][k - 1] == (com & wu).sum()
        assert c["pred_top"][k - 1] == (com & (F < -27.9)).sum()
        assert c["pred_u"][k - 1] == (com & (F < -28.3)).sum()


# --- matching shift and error model -------------------------------------------------------

def test_matching_shift_ties_and_unmatched():
    c = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    r = ex.matching_shift([-4, -1, 1, 1, 3], c)
    assert r["shift_m"] == 0.0 and r["matched"] and r["n_tied"] == 3        # |-1| = |1|: smallest |c| wins
    r = ex.matching_shift([-3, -1, 2, 2, 5], c)
    assert r["shift_m"] == -0.01 and r["residual_cells"] == -1
    r = ex.matching_shift([-3, -1, 2, 1, 3], c)                             # tie at -0.01 and +0.01
    assert r["shift_m"] == -0.01
    r = ex.matching_shift([2, 3, 4, 5, 6], c)
    assert not r["matched"] and r["shift_m"] == -0.02
    with pytest.raises(ValueError):
        ex.matching_shift([1, 2], c)


def test_matching_shift_recovers_a_product_offset():
    """A product 0.3 m too high floods at L + 0.3 what the truth floods at L."""
    z = np.add.outer(np.zeros(20), np.linspace(-31, -26, 200))
    seed = np.zeros(z.shape, bool)
    seed[:, 0] = True
    L = -28.4
    obs = hindcast.flood_fill(z, L, seed)
    F = ht_flood(z + 0.3, seed)
    code = np.where(obs, 1, 0).astype(np.int8)
    outside = np.full(z.shape, -1, np.int8)
    cnt = ex.unit_region_counts(F, code, outside, np.ones(z.shape, bool), np.ones(z.shape, int), 1,
                                L + ex.C_GRID)
    t = ex.shift_table(cnt, L, ["only"])
    assert abs(t.loc[0, "shift_m"] - 0.3) <= 0.0251 and t.loc[0, "usable"]


def ht_flood(z, seed):
    from caspian import hindcast_tiled as ht
    return ht.flood_level_map(np.asarray(z, np.float32), seed, tile=64)


def test_fit_model_forms_and_fallbacks():
    d = np.linspace(-2, 0, 11)
    y = 0.4 + 0.25 * d
    m = ex.fit_model(d, y, "linear_hold")
    assert m["form"] == "linear_hold" and m["a"] == pytest.approx(0.4) and m["b"] == pytest.approx(0.25)
    assert m["d_lo"] == -2 and m["d_hi"] == 0
    assert ex.predict_shift(m, 3.0) == pytest.approx(0.4)                  # held beyond the fitted depths
    assert ex.predict_shift(m, -5.0) == pytest.approx(0.4 - 0.5)
    lin = ex.fit_model(d, y, "linear")
    assert ex.predict_shift(lin, 2.0) == pytest.approx(0.9)
    const = ex.fit_model(d, y, "constant")
    assert const["b"] == 0 and const["a"] == pytest.approx(np.median(y))
    steep = ex.fit_model(d, 0.1 - 0.9 * d, "linear_hold")                  # 1 + b = 0.1 < MIN_MONOTONE
    assert steep["form"] == "constant" and steep["fallback"] == "monotone"
    few = ex.fit_model(d[:3], y[:3], "linear_hold")
    assert few["form"] == "constant" and few["fallback"] == "few_units"
    flat = ex.fit_model(np.zeros(6), np.arange(6.0), "linear_hold")
    assert flat["fallback"] == "no_depth_spread"
    assert ex.fit_model([np.nan], [1.0]) is None
    with pytest.raises(ValueError):
        ex.fit_model(d, y, "cubic")


def test_fit_models_lowest_level_and_domain_fallback():
    rows = []
    for ep, bins in (("E2", range(10, 20)), ("E3", range(0, 10))):
        for b in bins:
            L = -29.0 + 0.1 * b
            rows.append({"epoch": ep, "level_m": L, "region": "north", "shift_m": 0.3, "usable": True})
            rows.append({"epoch": ep, "level_m": L, "region": "south", "shift_m": -0.2, "usable": b < 3})
            rows.append({"epoch": ep, "level_m": L, "region": "domain", "shift_m": 0.1, "usable": True})
    rows.append({"epoch": "E1", "level_m": -35.0, "region": "north", "shift_m": 9.0, "usable": True})
    s = pd.DataFrame(rows)
    r = ex.fit_models(s, ["north", "south", "domain"], epochs=("E2", "E3"))
    assert r["lowest_level_m"] == pytest.approx(-29.0)                     # E1 is not a fit epoch
    assert r["models"]["north"]["a"] == pytest.approx(0.3) and not r["models"]["north"]["used_domain"]
    assert r["models"]["south"]["used_domain"] and r["models"]["south"]["a"] == pytest.approx(0.1)
    assert r["models"]["south"]["n_own"] == 3
    with pytest.raises(ValueError):
        ex.fit_models(s, ["north"], epochs=("E9",))


def test_corrected_surface_per_sector_monotone_and_extrapolation():
    z = np.tile(np.linspace(-33, -27, 61, dtype=np.float32), (4, 1))
    z[0, 0] = np.nan
    lab = np.array([[1], [1], [2], [3]]) * np.ones((1, 61), int)
    lowest = -29.0
    models = {1: {"form": "constant", "a": 0.3, "b": 0.0, "d_lo": -2.0, "d_hi": 0.2},
              2: {"form": "linear_hold", "a": -0.2, "b": 0.5, "d_lo": -2.0, "d_hi": 0.0},
              3: None}
    cs = ex.CorrectedSurface(z, lab, models, lowest)
    out = cs[:, :]
    assert np.isnan(out[0, 0])
    assert np.allclose(out[1], z[1] - 0.3)
    assert np.array_equal(out[3], z[3])                                     # no model: unchanged
    d = lowest - z[2].astype(float)
    assert np.allclose(out[2], z[2] - (-0.2 + 0.5 * np.clip(d, -2, 0)), atol=1e-5)
    assert np.all(np.diff(out[2]) > 0)                                       # monotone in z
    ext = cs.extrapolated((slice(None), slice(None)))
    assert np.array_equal(ext[2], d > 0) and not ext[3].any() and not ext[0, 0]
    assert np.allclose(cs[1:2, 10:20], out[1:2, 10:20])                      # windows agree


# --- fused surface ---------------------------------------------------------------------------

def test_fused_surface_uses_dem_inside_and_clamps_against_the_lowest_extent():
    z = np.full((3, 6), -30.0, np.float32)
    corr = ex.CorrectedSurface(z, np.ones(z.shape, int), {1: {"form": "constant", "a": 0.0, "b": 0.0,
                                                             "d_lo": -1.0, "d_hi": 0.5}}, -29.0)
    dem = np.full(z.shape, np.nan, np.float32)
    dem[0, :] = -28.5
    low = np.array([[0, 0, 0, 0, 0, 0],          # known land (DEM there)
                    [1, 1, 0, -2, -2, -1],        # sea, sea, land (no DEM: raised), outside, outside, unknown
                    [1, 1, 1, 1, 1, 1]], np.int8)
    outside = np.array([[-1] * 6, [-1, -1, -1, 1, 0, -1], [-1] * 6], np.int8)
    dem[2, 0] = -28.0                             # DEM on known sea: not used
    f = ex.FusedSurface(dem, corr, low, outside, -29.0)
    out = f[:, :]
    assert np.all(out[0] == np.float32(-28.5))
    assert out[1, 2] >= -29.0 and out[1, 2] == ex.float32_at_or_above(-29.0)   # known land: raised to L_min
    assert out[1, 4] == ex.float32_at_or_above(-29.0)                         # outside land state: raised
    assert out[1, 3] == -30.0 and out[1, 5] == -30.0                           # sea / unknown: unchanged
    assert out[2, 0] == -30.0                                                  # product, not the DEM
    src = f.source((slice(None), slice(None)))
    assert np.all(src[0] == ex.SRC_DEM) and src[2, 0] == ex.SRC_EXTRAPOLATED     # d = 1 m > d_hi = 0.5 m
    summ = f.summary(np.ones(z.shape, int), 1, tile=2)
    assert summ.loc[0, "n_dem"] == 6 and summ.loc[0, "n_clamped_land"] == 2 and summ.loc[0, "n_clamped_sea"] == 0
    high = ex.CorrectedSurface(np.full(z.shape, -28.0, np.float32), np.ones(z.shape, int),
                               {1: {"form": "constant", "a": 0.0, "b": 0.0, "d_lo": -1.0, "d_hi": 0.5}}, -29.0)
    out2 = ex.FusedSurface(dem, high, low, outside, -29.0)[:, :]
    assert out2[2, 3] < -29.0 and out2[2, 3] == ex.float32_below(-29.0)        # known sea: just below L_min
    assert float(ex.float32_below(-29.15)) < -29.15 <= float(ex.float32_at_or_above(-29.15))


# --- areas and intervals ---------------------------------------------------------------------

def test_level_grid_emerged_curve_and_interval():
    lv = ex.level_grid(-28.9, 2.0, above=0.3)
    assert np.any(np.isclose(lv, -28.9, atol=1e-9)) and lv[0] == pytest.approx(-30.9) and np.all(np.diff(lv) > 0)
    sea = 1000 + 100 * (lv - lv[0])
    E = ex.emerged_curve(sea, lv, -28.9)
    k = int(np.flatnonzero(np.isclose(lv, -28.9))[0])
    assert E[k] == 0 and E[0] == pytest.approx(200) and np.all(np.diff(E) <= 1e-9)
    assert ex.emerged_curve(sea, lv, -28.9, baseline_cells=5000)[k] == pytest.approx(5000 - sea[k])
    scores = np.r_[np.arange(1, 20), -50.0]
    iv = ex.interval(100.0, scores, alpha=0.1)
    assert iv["q"] == 19 and iv["n_scores"] == 20 and iv["lo"] == 81 and iv["hi"] == 119
    assert ex.interval(5.0, scores, alpha=0.1)["lo"] == 0.0                    # clipped at zero
    assert ex.interval(5.0, scores, alpha=0.1, upper=10.0)["hi"] == 10.0
    small = ex.interval(5.0, [1, 2, 3])
    assert small["q"] == np.inf and small["hi"] == np.inf
    assert ex.interval(5.0, [])["n_scores"] == 0


def test_equivalent_levels_round_outward():
    lv = np.round(np.arange(-31.0, -28.9 + 1e-9, 0.1), 6)
    E = np.maximum(0.0, (-28.9 - lv) * 100)                                     # 100 per m below -28.9
    eq = ex.equivalent_levels(lv, E, 45.0, 155.0)
    assert eq["level_lo_bound"] == pytest.approx(-29.3)                        # E = 40 <= 45 (not -29.4: 50)
    assert eq["level_hi_bound"] == pytest.approx(-30.5)                        # E = 160 >= 155 (-30.4: 150)
    eq2 = ex.equivalent_levels(lv, E, 0.0, 1e9)
    assert eq2["level_lo_bound"] == pytest.approx(-28.9) and eq2["hi_beyond_grid"]
    with pytest.raises(ValueError):
        ex.equivalent_levels(lv, -E, 1, 2)


# --- exposure ---------------------------------------------------------------------------------

TR = [60, 0, 0, 0, -60, 600]                     # 10 rows, x from 0


def test_feature_samples():
    s = ex.feature_samples([Point(90, 570), LineString([(0, 30), (600, 30)]), box(60, 60, 240, 240),
                            box(1, 1, 5, 5), Point(-100, 0)], TR, (10, 12))
    p = s[s["fid"] == 0]
    assert len(p) == 1 and p.iloc[0]["row"] == 0 and p.iloc[0]["col"] == 1
    line = s[s["fid"] == 1]
    assert line["weight"].sum() == pytest.approx(600) and (line["row"] == 9).all()
    poly = s[s["fid"] == 2]
    assert len(poly) == 9 and poly["weight"].sum() == pytest.approx(9 * 3600)
    tiny = s[s["fid"] == 3]
    assert len(tiny) == 1 and tiny.iloc[0]["weight"] == pytest.approx(16)
    assert not (s["fid"] == 4).any()                                            # off the grid
    with pytest.raises(ValueError):
        from shapely.geometry import GeometryCollection
        ex.feature_samples([GeometryCollection([Point(1, 1)])], TR, (10, 12))


def test_distance_profiles_match_a_distance_transform():
    g = rng()
    F = np.cumsum(g.normal(0, 1, (30, 40)), axis=1) - 5
    F[4, 7] = np.inf
    pts_r, pts_c = np.array([3, 15, 29, 0]), np.array([5, 20, 39, 0])
    prof = ex.distance_profiles(F, pts_r, pts_c, 60.0, 600.0)
    for L in (-8.0, -3.0, 0.0, 2.5):
        sea = F < L
        d = ndimage.distance_transform_edt(~sea) * 60.0 if sea.any() else np.full(F.shape, np.inf)
        want = d[pts_r, pts_c]
        want = np.where(want <= 600.0, want, np.inf)
        assert np.allclose(ex.distance_at(prof, L), want)


def test_exposure_rules_and_monotonicity():
    F = np.tile(np.linspace(-31, -28, 12), (10, 1))                            # sea floods from the west
    geoms = [Point(30, 570), Point(330, 570), Point(690, 570),                 # cols 0, 5, 11
             LineString([(0, 30), (720, 30)]), box(0, 0, 600, 120)]
    s = ex.feature_samples(geoms, TR, F.shape)
    fv = ex.sample_values(F, s["row"], s["col"])
    pts = s[s["fid"] < 3]
    r = ex.exposure_at("onshore", pts, fv[pts.index], -28.0, -29.0)
    # col 5 lies at -29.636 (sea at -28 and at -29), col 0 at -31 (sea at both), col 11 at -28 (land at -28)
    assert r.set_index("fid")["changed"].tolist() == [False, False, False]
    r = ex.exposure_at("onshore", pts, fv[pts.index], -28.0, -30.0)
    assert r.set_index("fid")["changed"].tolist() == [False, True, False]
    assert r.set_index("fid")["considered"].tolist() == [True, True, False]
    ln = s[s["fid"] == 3]
    r = ex.exposure_at("length", ln, fv[ln.index], -28.0, -29.5)
    cols = np.arange(12)
    em = (F[9] >= -29.5) & (F[9] < -28.0)
    assert r.iloc[0]["amount"] == pytest.approx(em.sum() * 60 / 1000)
    assert r.iloc[0]["considered"] and not r.iloc[0]["changed"]
    pg = s[s["fid"] == 4]
    a1 = ex.exposure_at("area", pg, fv[pg.index], -28.0, -29.5).iloc[0]["amount"]
    a2 = ex.exposure_at("area", pg, fv[pg.index], -28.0, -30.5).iloc[0]["amount"]
    assert 0 < a1 < a2
    prof = ex.distance_profiles(F, s["row"], s["col"], 60.0, 200.0)
    acc = ex.exposure_at("access", s, fv, -28.0, -29.8, profiles=prof, threshold_m=200.0).set_index("fid")
    # col 11 point: the sea at -28 is 60 m away (col 10 lies at -28.27), at -29.8 the coast is at col 4 (> 200 m)
    assert acc.loc[2, "considered"] and acc.loc[2, "changed"]
    assert acc.loc[1, "considered"] and not acc.loc[1, "changed"]
    summ = ex.summarise_exposure(acc, "access")
    assert summ["n_changed"] == int(acc["changed"].sum())
    prev = -1
    for L in (-28.0, -28.5, -29.0, -29.5, -30.0, -31.0):                        # monotone in the level
        v = ex.summarise_exposure(ex.exposure_at("length", ln, fv[ln.index], -28.0, L), "length")["amount"]
        assert v >= prev
        prev = v
    with pytest.raises(ValueError):
        ex.exposure_at("length", ln, fv[ln.index], -28.0, -27.0)
    with pytest.raises(ValueError):
        ex.exposure_at("access", ln, fv[ln.index], -28.0, -29.0)
    del cols


def test_feature_regions_majority_by_weight():
    s = pd.DataFrame({"fid": [0, 0, 0, 1], "row": [0, 0, 1, 1], "col": [0, 1, 0, 1], "weight": [1, 1, 5, 1]})
    lab = np.array([[1, 1], [2, 3]])
    r = ex.feature_regions(s, lab)
    assert r.loc[0] == 2 and r.loc[1] == 3


# --- scenario dates ---------------------------------------------------------------------------

def _pw(rows):
    return pd.DataFrame(rows, columns=["pathway_id", "kind", "ref_year", "year", "value_m"])


def test_scenario_dates_base_anchor():
    p = _pw([("lin", "projection", 2020, 2100, -8.0),
             ("rise", "projection", 2021, 2100, 0.3),
             ("trend", "rate", 2021, 2100, -0.057),
             ("pw", "projection", 2020, 2050, -1.0), ("pw", "projection", 2020, 2100, -11.0)])
    c = ex.pathway_curves(p)
    assert np.allclose(c["trend"][1], [0.0, -0.057 * 79])
    d = ex.scenario_dates(c, [0.5, 1.0, 5.0, 8.0], base_year=2026.0).set_index(["pathway_id", "delta_m"])
    assert d.loc[("lin", 0.5), "date"] == pytest.approx(2031.0)               # 0.1 m per year from 2026
    assert d.loc[("lin", 5.0), "date"] == pytest.approx(2076.0)
    assert not d.loc[("lin", 8.0), "reached"] and np.isnan(d.loc[("lin", 8.0), "date"])
    assert not d.loc[("rise", 0.5), "reached"]
    assert d.loc[("trend", 1.0), "date"] == pytest.approx(2026 + 1 / 0.057, abs=1e-3)
    # piecewise: -0.2 m by 2026 (1/30 m per year), then 0.2 m per year after 2050
    assert d.loc[("pw", 0.5), "date"] == pytest.approx(2026 + 0.5 * 30)
    assert d.loc[("pw", 1.0), "date"] == pytest.approx(2050 + (1.0 - 0.8) / 0.2)
    with pytest.raises(ValueError):
        ex.pathway_curves(_pw([("x", "projection", 2020, 2100, -1.0), ("x", "projection", 2021, 2050, -1.0)]))
    with pytest.raises(ValueError):
        ex.pathway_curves(_pw([("x", "projection", 2020, 2010, -1.0)]))


def test_scenario_dates_reference_anchor():
    c = ex.pathway_curves(_pw([("lin", "projection", 2020, 2100, -8.0)]))
    # observed 2020 level -27.5, 2026 level -28.9: already 1.4 m down, the pathway only 0.6 m
    d = ex.scenario_dates(c, [0.5, 1.0], anchor="reference", level_2026=-28.9,
                          ref_levels={"lin": -27.5}).set_index("delta_m")
    assert d.loc[0.5, "date"] == pytest.approx(2020 + 1.9 / 0.1)
    assert d.loc[1.0, "date"] == pytest.approx(2020 + 2.4 / 0.1)
    early = ex.scenario_dates(c, [0.1], anchor="reference", level_2026=-27.0, ref_levels={"lin": -27.5})
    assert early.iloc[0]["note"] == "already_passed" and early.iloc[0]["date"] == pytest.approx(2020.0)
    with pytest.raises(ValueError):
        ex.scenario_dates(c, [1.0], anchor="reference")
    out = ex.scenario_dates(ex.pathway_curves(_pw([("late", "projection", 2030, 2100, -8.0)])), [1.0])
    assert out.iloc[0]["note"] == "base year outside the pathway"


def test_reference_level():
    t = ["2025-12-20", "2026-01-29", "2026-05-01", "2026-09-14"]
    v = [-28.8, -28.9, -28.7, -28.6]
    r = ex.reference_level(t, v)
    assert r["level_m"] == pytest.approx(np.mean(v[1:])) and r["n_records"] == 3
    r = ex.reference_level(t, v, method="latest_running_median")
    from caspian.levels import running_median
    assert r["level_m"] == pytest.approx(running_median(np.array(v), 3)[-1]) and r["date"] == "2026-09-14"
    with pytest.raises(ValueError):
        ex.reference_level(t, v, year=2030)


# --- habitat rasters --------------------------------------------------------------------------

def _class_raster(path, data, west, north, res, nodata=None):
    import rasterio
    from rasterio.transform import Affine
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=Affine(res, 0, west, 0, -res, north),
                       nodata=nodata) as w:
        w.write(data, 1)


def test_raster_class_fraction_preserves_area_and_is_block_independent(tmp_path):
    import pyproj
    from caspian.config import CRS_EQUAL_AREA
    res = 0.0005                                                   # about 40-55 m
    data = np.zeros((240, 300), np.uint8)
    data[:, 150:] = 90                                             # east half habitat
    data[20:60, 20:80] = 90                                        # a patch
    data[200:, :40] = 255                                          # nodata corner
    a = tmp_path / "a.tif"
    _class_raster(a, data[:, :170], 50.0, 46.0, res, nodata=255)  # two tiles sharing an edge
    b = tmp_path / "b.tif"
    _class_raster(b, data[:, 170:], 50.0 + 170 * res, 46.0, res, nodata=255)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    x0, y1 = fwd.transform(50.0, 46.0)
    x1, y0 = fwd.transform(50.0 + 300 * res, 46.0 - 240 * res)
    tr = [60, 0, np.floor(x0 / 60) * 60 - 600, 0, -60, np.ceil(y1 / 60) * 60 + 600]
    shape = (int((tr[5] - y0) / 60) + 20, int((x1 - tr[2]) / 60) + 20)
    f = ex.raster_class_fraction([a, b], [90], tr, shape, block=64)
    f2 = ex.raster_class_fraction([a, b], [90], tr, shape, block=4096)
    with pytest.raises(ValueError, match="larger than a grid cell"):
        ex.raster_class_fraction([a, b], [90], tr, shape, factor=2, block=64)
    fine = tmp_path / "fine.tif"                                    # the same map at 1/4 of the pixel
    _class_raster(fine, np.repeat(np.repeat(data, 4, axis=0), 4, axis=1), 50.0, 46.0, res / 4, nodata=255)
    f3 = ex.raster_class_fraction([fine], [90], tr, shape, factor=4, block=256)
    assert np.nanmin(f) >= 0 and np.nanmax(f) <= 1 and np.isnan(f).any()
    assert np.array_equal(np.isnan(f), np.isnan(f2)) and np.allclose(np.nan_to_num(f), np.nan_to_num(f2))
    whole = tmp_path / "c.tif"
    _class_raster(whole, data, 50.0, 46.0, res, nodata=255)
    fw = ex.raster_class_fraction([whole], [90], tr, shape)
    assert np.array_equal(np.isnan(f), np.isnan(fw)) and np.allclose(np.nan_to_num(f), np.nan_to_num(fw))
    geod = pyproj.Geod(ellps="WGS84")
    lat = 46.0 - (np.arange(240) + 0.5) * res
    cell = np.array([abs(geod.polygon_area_perimeter([50, 50 + res, 50 + res, 50],
                                                     [la - res / 2, la - res / 2, la + res / 2, la + res / 2])[0])
                     for la in lat])
    want = float(((data == 90) * cell[:, None]).sum())
    got = float(np.nansum(f) * 3600)
    assert abs(got - want) / want < 0.01
    assert np.array_equal(np.isnan(f3), np.isnan(f)) and np.allclose(np.nan_to_num(f3), np.nan_to_num(f))
    part = ex.raster_class_fraction([a, b], [90], tr, shape, block=64, bounds=(50.0, 45.9, 50.06, 46.0))
    assert np.nansum(part) < np.nansum(f)
    s = ex.fraction_samples(f, tr)
    assert s["weight"].sum() == pytest.approx(got) and s["fid"].nunique() == 2      # the patch and the half


def test_raster_class_fraction_samples_a_coarse_source(tmp_path):
    from caspian.config import CRS_EQUAL_AREA
    import pyproj
    data = np.array([[1, 2], [3, 255]], np.uint8)                   # 0.01 deg pixels, about 0.8-1.1 km
    p = tmp_path / "coarse.tif"
    _class_raster(p, data, 50.0, 46.0, 0.01, nodata=255)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    x0, y1 = fwd.transform(50.0, 46.0)
    tr = [60, 0, np.floor(x0 / 60) * 60, 0, -60, np.ceil(y1 / 60) * 60]
    f = ex.raster_class_fraction([p], [2, 3], tr, (40, 40))
    assert set(np.unique(f[~np.isnan(f)])) <= {0.0, 1.0}
    assert f[5, 5] == 0.0 and f[5, 25] == 1.0 and f[25, 5] == 1.0 and np.isnan(f[25, 25])
    with pytest.raises(ValueError):
        ex.raster_class_fraction([p], [2], tr, (40, 40), method="nearest")


def test_trend_plus_quadratic_curve():
    t, c = ex.trend_plus_quadratic_curve(-3.79, -0.057, 2020.0, 2100.0)
    assert t[0] == 2020 and t[-1] == 2100 and c[0] == 0
    assert c[-1] == pytest.approx(-3.79 - 0.057 * 80)
    assert c[np.flatnonzero(t == 2060)[0]] == pytest.approx(-0.057 * 40 - 3.79 / 4)
    assert np.all(np.diff(np.diff(c)) < 0)                         # the fall accelerates
    t2, _ = ex.trend_plus_quadratic_curve(-1.0, 0.0, 2020.0, 2025.5, step=2.0)
    assert t2[-1] == 2025.5
    d = ex.scenario_dates({"q": (t, c)}, [0.5], base_year=2026.0)
    # base anchoring: the fall after 2026 reaches 0.5 m; solve on the knots
    fall = np.interp(2026.0, t, c) - c
    k = np.flatnonzero((t >= 2026) & (fall >= 0.5))[0]
    assert t[k - 1] < d.iloc[0]["date"] <= t[k]
