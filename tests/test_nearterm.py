"""caspian.nearterm (A18 item 10) on synthetic grids."""

import numpy as np
import pytest

from caspian import exposure as ex
from caspian import hindcast_tiled as ht
from caspian import nearterm as nt


def _world(seed=0, H=40, W=50):
    """A seabed rising eastward with noise and a pit, open-sea seed in the west column,
    two regions (north, south)."""
    rng = np.random.default_rng(seed)
    z = (np.linspace(-6, 2, W)[None, :] + 0.3 * rng.standard_normal((H, W))).astype(np.float32)
    z[10:14, 30:34] = -8.0                                  # an isolated pit
    seed_m = np.zeros((H, W), bool)
    seed_m[:, 0] = True
    lab = np.ones((H, W), np.uint8)
    lab[H // 2:] = 2
    return z, seed_m, lab


def test_shifted_and_baseline_surfaces():
    z, seed, _ = _world()
    F = ht.flood_level_map(z, seed, tile=16)
    S = nt.ShiftedSurface(F, -1.4)
    Fz = ht.flood_level_map(z + np.float32(1.4), seed, tile=16)
    # a constant shift commutes with the minimax flood fill (float32 rounding aside)
    assert np.allclose(S[:, :], Fz.astype(float), atol=1e-5)
    Fb = F + 0.5                                           # a "baseline" map
    G = nt.BaselineSurface(Fb, F, level_2026=0.0)
    g = G[:, :]
    assert np.array_equal(g < 0.0, Fb < 0.0)
    for L in (-0.5, -2.0, -3.3):
        assert np.array_equal(g < L, (Fb < 0.0) & (F < L))


def test_curves_and_isolated_against_brute_force():
    z, seed, lab = _world(1)
    F = ht.flood_level_map(z, seed, tile=16)
    lv = ex.level_grid(-1.0, 4.0, above=1.0)
    cF = nt.region_counts(F, lab, 2, lv, tile=16)
    E = nt.curve_from_counts(cF, lv, -1.0, [0.5, 1.0, 2.0])
    for k in (1, 2):
        for j, D in enumerate([0.5, 1.0, 2.0]):
            want = ((lab == k) & (F < -1.0)).sum() - ((lab == k) & (F < -1.0 - D)).sum()
            assert E[k - 1, j] == want
    # common baseline: replace A(L26)
    base = np.array([100, 200])
    Eb = nt.curve_from_counts(cF, lv, -1.0, [1.0], baseline_counts=base)
    assert np.allclose(Eb[:, 0], base - cF[:, np.flatnonzero(np.isclose(lv, -2.0))[0]])
    # isolated water within the self-referenced baseline set F < -1
    inb = lambda r0, r1, c0, c1: F[r0:r1, c0:c1] < -1.0  # noqa: E731
    cz = nt.region_counts(z, lab, 2, lv, mask=inb, tile=16)
    cf = nt.region_counts(F, lab, 2, lv, mask=inb, tile=16)
    iso = nt.isolated_from_counts(cz, cf, lv, [-3.0])
    for k in (1, 2):
        want = ((lab == k) & (F < -1.0) & (z < -3.0) & ~(F < -3.0)).sum()
        assert iso[k - 1, 0] == want
    assert iso.sum() > 0                                     # the pit is isolated at -3 m


def test_interp_counts_outside_grid_raises():
    with pytest.raises(ValueError):
        nt.interp_counts(np.arange(5), np.arange(5.0), [7.0])


def test_pair_counts_match_exposure_and_brute_force():
    z, seed, lab = _world(2)
    F = ht.flood_level_map(z, seed, tile=16)
    H, W = z.shape
    rng = np.random.default_rng(3)
    code_top = np.where(rng.random((H, W)) < 0.1, -1, (z < -1.0).astype(np.int8)).astype(np.int8)
    code_u = np.where(rng.random((H, W)) < 0.1, -2, (z < -2.0).astype(np.int8)).astype(np.int8)
    outside = np.where(rng.random((H, W)) < 0.5, 1, -1).astype(np.int8)
    fin = np.isfinite(z)
    got = nt.pair_counts(F, -1.0, -2.0, code_top, code_u, outside, fin, lab, 2, shifts=(0.0, -0.7), tile=16)
    ref = ex.pair_region_counts(F, -1.0, -2.0, code_top, code_u, outside, fin, lab, 2, tile=16)
    for a, b in (("n_common", "n_common"), ("w_top", "obs_top"), ("w_u", "obs_u")):
        assert np.array_equal(got[a], ref[b])
    assert np.array_equal(got["p_top"][0], ref["pred_top"]) and np.array_equal(got["p_u"][0], ref["pred_u"])
    kt, wt = ex.known_masks(code_top, outside)
    ku, wu = ex.known_masks(code_u, outside)
    common = kt & ku & fin
    for k in (1, 2):
        m = common & (lab == k)
        assert got["w_both"][k - 1] == (m & wt & wu).sum()
        assert got["p_u"][1, k - 1] == (m & (F.astype(float) + 0.7 < -2.0)).sum()
        assert got["w_top_p_u"][1, k - 1] == (m & wt & (F.astype(float) + 0.7 < -2.0)).sum()


def test_scores_and_envelopes():
    s = nt.scale_scores([10.0, -5.0, 3.0, 0.0], [20.0, 5.0, 0.0, 0.0])
    assert s[0] == 0.5 and s[1] == 1.0 and np.isinf(s[2]) and np.isinf(s[3])
    lo, hi = nt.relative_envelope(100.0, 0.25)
    assert lo == pytest.approx(80.0) and hi == pytest.approx(100.0 / 0.75)
    # the envelope is exactly the set of E with |100 - E| <= 0.25 E
    for E in np.linspace(1, 300, 600):
        assert (abs(100.0 - E) <= 0.25 * E + 1e-12) == (lo - 1e-9 <= E <= hi + 1e-9)
    lo, hi = nt.relative_envelope(100.0, 2.0)
    assert lo == pytest.approx(100.0 / 3.0) and np.isinf(hi)
    lo, hi = nt.relative_envelope(100.0, float("inf"))
    assert lo == 0.0 and np.isinf(hi)
    assert all(np.isnan(nt.relative_envelope(-5.0, 0.5)))
    assert nt.decline_envelope(100.0, 30.0, 2.0) == (40.0, 160.0)
    assert nt.decline_envelope(10.0, 30.0, 2.0)[0] == 0.0


def test_grf_covariance():
    rng = np.random.default_rng(20261022)
    f, neg = nt.grf_exponential(24, 30, 2.0, 4.0, 30.0, rng, n=4000)
    assert f.shape == (4000, 24, 30)
    assert neg < 1e-3
    assert f.var() == pytest.approx(4.0, rel=0.05)
    assert abs(f.mean()) < 0.05
    for lag in (2, 5, 10):                                  # lag in cells of 2 units
        c = (f[:, :, :-lag] * f[:, :, lag:]).mean()
        assert c == pytest.approx(4.0 * np.exp(-3.0 * 2.0 * lag / 30.0), abs=0.15)


def test_bilinear_to_is_exact_on_planes():
    yy, xx = np.meshgrid(100.0 - 10.0 * np.arange(6), 10.0 * np.arange(8), indexing="ij")
    field = 2.0 * xx - 0.5 * yy
    ys, xs = np.array([95.0, 72.5, 55.0]), np.array([3.0, 41.0, 66.0])
    got = nt.bilinear_to(field, 100.0, 0.0, 10.0, ys, xs)
    want = 2.0 * xs[None, :] - 0.5 * ys[:, None]
    assert np.allclose(got, want)


def test_minimax_flood_matches_tiled():
    z, seed, _ = _world(4)
    z[5, 5] = np.nan
    a = nt.minimax_flood(z, seed)
    b = ht.flood_level_map(z, seed, tile=16)
    fa, fb = np.isfinite(a), np.isfinite(b)
    assert np.array_equal(fa, fb)
    assert np.array_equal(a[fa], b[fb].astype(float))


def test_block_reductions():
    a = np.arange(48, dtype=float).reshape(6, 8)
    a[0, 0] = np.nan
    m = nt.block_mean(a, 2, rows=4)
    assert m.shape == (3, 4)
    assert m[0, 0] == pytest.approx(np.nanmean([a[0, 1], a[1, 0], a[1, 1]]))
    assert m[2, 3] == pytest.approx(a[4:6, 6:8].mean())
    s = np.ones((6, 8), bool)
    s[3, 3] = False
    b = nt.block_all(s, 2, rows=4)
    assert not b[1, 1] and b.sum() == 11
    lab = np.zeros((7, 9), np.uint8)
    lab[:4, :] = 1
    lab[4:, :5] = 2
    c = nt.block_region_counts(lab, 2, 2, rows=4)
    assert c.shape == (2, 3, 4)
    assert c[0].sum() == 4 * 8 and c[1].sum() == 2 * 5   # rows 4-5, columns 0-4 (row 6, column 8 dropped)
    msk = np.zeros((7, 9), bool)
    msk[0, 0] = True
    assert nt.block_region_counts(lab, 2, 2, mask=msk)[0].sum() == 1


def test_agreement_ratio():
    b = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], bool)
    assert nt.agreement_ratio(b, [1, 1, 1, 1]) == pytest.approx(1 / 3)
    assert nt.agreement_ratio(b, [3, 1, 1, 1]) == pytest.approx(3 / 5)
    assert np.isnan(nt.agreement_ratio(np.zeros((2, 3), bool), [1, 1, 1]))


def test_distance_masks_match_whole_grid():
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(5)
    band = np.zeros((90, 110), bool)
    band[20:60, 30:80] = rng.random((40, 50)) < 0.9
    band[70:75, 5:100] = True
    dil, ero = nt.distance_masks(band, 250.0, 60.0, tile=32)
    want_dil = distance_transform_edt(~band) * 60.0 <= 250.0
    want_ero = (distance_transform_edt(np.pad(band, 1))[1:-1, 1:-1] * 60.0 > 250.0) & band
    assert np.array_equal(dil, want_dil)
    assert np.array_equal(ero, want_ero)
    assert ero.sum() < band.sum() < dil.sum()


def test_exposure_from_states_matches_exposure_at():
    import pandas as pd
    rng = np.random.default_rng(6)
    samples = pd.DataFrame({"fid": np.repeat(np.arange(30), 4), "row": 0, "col": 0,
                            "weight": rng.uniform(10, 100, 120)})
    fv = rng.uniform(-3, 1, 120)
    for rule in ("onshore", "length", "area"):
        ref = ex.exposure_at(rule, samples, fv, 0.0, -1.0)
        got = nt.exposure_from_states(rule, samples, fv < 0.0, fv < -1.0)
        pd.testing.assert_frame_equal(got, ref)
    with pytest.raises(ValueError):
        nt.exposure_from_states("access", samples, fv < 0, fv < -1)


def test_port_type():
    assert nt.port_type({"industrial": "port"}) == "port_harbour"
    assert nt.port_type('{"harbour": "yes", "name": "x"}') == "port_harbour"
    assert nt.port_type({"harbour": "no", "man_made": "pier"}) == "pier_landing"
    assert nt.port_type({"seamark:type": "harbour"}) == "port_harbour"
    assert nt.port_type({"amenity": "ferry_terminal"}) == "pier_landing"
    assert nt.port_type({}) == "pier_landing"


def test_two_way_shares():
    rows, cols = np.array([0.0, 10.0, 20.0]), np.array([1.0, 2.0, 4.0, 8.0])
    y = rows[:, None] + cols[None, :]
    s = nt.two_way_shares(y)
    assert s["interaction"] == pytest.approx(0.0, abs=1e-12)
    assert s["row"] + s["column"] == pytest.approx(1.0)
    y2 = y.copy()
    y2[0, 0] += 5.0
    s2 = nt.two_way_shares(y2)
    assert s2["row"] + s2["column"] + s2["interaction"] == pytest.approx(1.0)
    assert s2["interaction"] > 0
    assert np.isnan(nt.two_way_shares(np.ones((2, 2)))["row"])


def test_pathway_fall_and_rates():
    curve = (np.array([2020.0, 2100.0]), np.array([0.0, -8.0]))
    assert nt.pathway_fall(curve, 2035.0) == pytest.approx(0.9)
    assert nt.pathway_fall(curve, 2050.0) == pytest.approx(2.4)
    assert np.isnan(nt.pathway_fall(curve, 2110.0))
    t = np.linspace(2000, 2020, 50)
    r = nt.ols_rate(t, -26.0 - 0.07 * (t - 2000))
    assert r["rate_m_per_yr"] == pytest.approx(-0.07) and r["se_m_per_yr"] == pytest.approx(0.0, abs=1e-12)
    d = nt.decimal_year(["2021-01-01 00:00:00", "2021-07-02 12:00:00"])
    assert d[0] == pytest.approx(2021.0) and d[1] == pytest.approx(2021.5, abs=1e-3)
