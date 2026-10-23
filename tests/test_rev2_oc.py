"""caspian.rev2_oc: core functions of Amendment 19 items 5 and 9 (D-148)."""

import numpy as np
import pandas as pd
import pytest

from caspian import h1, hindcast_tiled as ht, nearterm as nt, rev2_oc as oc, rev_r


# --- fields -----------------------------------------------------------------------------------

def test_grf_variance_and_range_on_toy_grid():
    """The D-125 generator with sill sigma^2 and practical range R: the pointwise variance is
    sigma^2 and the correlation at lag R/3 is exp(-1) (exponential covariance)."""
    sigma, R, dx = 0.5, 12.0, 1.0
    rng = np.random.default_rng(20261023)
    f, neg = nt.grf_exponential(48, 48, dx, sigma ** 2, R, rng, n=600)
    assert neg < 1e-3
    assert f.var() == pytest.approx(sigma ** 2, rel=0.05)
    lag = 4                                           # = R / 3 cells
    c = np.mean(f[:, :, :-lag] * f[:, :, lag:]) / f.var()
    assert c == pytest.approx(np.exp(-1.0), abs=0.04)
    c2 = np.mean(f[:, :-lag * 2, :] * f[:, lag * 2:, :]) / f.var()
    assert c2 == pytest.approx(np.exp(-2.0), abs=0.04)


def test_unit_field_scales_by_sigma():
    """Common random numbers: the field of sigma is sigma times the unit field of the same stream."""
    a, _ = nt.grf_exponential(20, 30, 2.0, 1.0, 40.0, np.random.default_rng(5), n=1)
    b, _ = nt.grf_exponential(20, 30, 2.0, 0.25 ** 2, 40.0, np.random.default_rng(5), n=1)
    assert np.allclose(0.25 * a, b)


def test_bilinear_rows_equals_nearterm_bilinear_to():
    rng = np.random.default_rng(1)
    fld = rng.normal(size=(9, 13))
    y0, x0, dx = 1000.0, -500.0, 100.0
    ys = y0 - np.linspace(-150, 1000, 37)              # beyond the grid on both sides
    xs = x0 + np.linspace(-120, 1400, 41)
    ref = nt.bilinear_to(fld, y0, x0, dx, ys, xs)
    got = np.vstack([oc.bilinear_rows(fld, y0, x0, dx, ys, xs, r0, min(r0 + 10, ys.size))
                     for r0 in range(0, ys.size, 10)])
    assert np.allclose(got, ref, atol=1e-12)


# --- waterlines -------------------------------------------------------------------------------

def _toy_flood(seed=0, shape=(60, 70), dtype=np.float32):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    z = (0.05 * xx + 0.5 * np.sin(yy / 5.0) + rng.normal(0, 0.3, shape)).astype(dtype)
    z[10:14, 30:33] = np.nan                           # no height
    seed_m = np.zeros(shape, bool)
    seed_m[:, 0] = True
    F = nt.minimax_flood(z, seed_m).astype(dtype)
    F[45:50, 50:55] = np.inf                           # finite but unreachable (walled off)
    fin = np.isfinite(z)
    return F, fin


def test_neighbour_max_brute_force():
    F, fin = _toy_flood()
    H, W = F.shape
    G = oc.neighbour_max(F, fin, 0, H, 0, W)
    for r, c in [(0, 0), (5, 7), (12, 29), (59, 69), (30, 0), (47, 49)]:
        vals = [F[rr, cc] for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= rr < H and 0 <= cc < W and fin[rr, cc]]
        assert G[r, c] == (max(vals) if vals else -np.inf)
    # a tile equals the whole-grid result
    assert np.array_equal(oc.neighbour_max(F, fin, 7, 33, 11, 50), G[7:33, 11:50])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_waterline_candidates_equal_floodmap_waterline(dtype):
    F, fin = _toy_flood(dtype=dtype)
    fm = rev_r.FloodMap(F, fin, tile=16)
    finite_vals = F[fin & np.isfinite(F)]
    lo, hi = float(np.percentile(finite_vals, 20)), float(np.percentile(finite_vals, 70))
    cand = oc.waterline_candidates(F, fin, lo, hi, tile=17)
    sub = oc.restrict(cand, lo + 0.1, hi - 0.1)
    for T in np.r_[lo, np.linspace(lo, hi, 9), hi, float(F[fin][5])]:
        if not lo <= T <= hi:
            continue
        r, c = fm.waterline(float(T))
        got = oc.waterline_at(cand, float(T))
        assert sorted(zip(r.tolist(), c.tolist())) == sorted(zip(got[0].tolist(), got[1].tolist()))
        if lo + 0.1 <= T <= hi - 0.1:
            g2 = oc.waterline_at(sub, float(T))
            assert sorted(zip(g2[0].tolist(), g2[1].tolist())) == sorted(zip(r.tolist(), c.tolist()))
    with pytest.raises(ValueError):
        oc.waterline_at(cand, hi + 1.0)


def test_float32_thresholds_are_compared_in_float64():
    F = np.full((3, 3), np.float32(-27.3), np.float32)
    F[1, 1] = np.float32(-28.0)
    F[0, :] = np.float32(-26.0)
    fin = np.ones(F.shape, bool)
    T = float(np.float32(-27.3)) + 1e-9               # above the stored value in float64 only
    cand = oc.waterline_candidates(F, fin, T - 1, T + 1)
    r, c = oc.waterline_at(cand, T)
    ref = rev_r.FloodMap(F, fin).waterline(T)
    assert sorted(zip(r.tolist(), c.tolist())) == sorted(zip(ref[0].tolist(), ref[1].tolist()))
    assert (1, 0) in set(zip(r.tolist(), c.tolist()))


# --- area-constant counts ---------------------------------------------------------------------

def test_threshold_counter_equals_per_unit_searchsorted():
    rng = np.random.default_rng(3)
    v = np.r_[rng.normal(-27, 1, 5000), np.inf, -np.inf].astype(np.float32)
    cgrid = np.round(np.arange(-500, 501) / 100.0, 2)
    levels = [-28.75, -27.05, -25.65]
    sets = [L + cgrid for L in levels]
    tc = oc.ThresholdCounter(sets)
    q = tc.locate(v)
    for u, T in enumerate(sets):
        sel = rng.random(v.size) < 0.7
        ref = np.cumsum(np.bincount(np.searchsorted(T, np.asarray(v[sel], float), side="right"),
                                    minlength=T.size + 1))[:T.size]
        assert np.array_equal(tc.counts(q[sel], u), ref)
        # and directly: the number of values below each threshold, in float64
        assert tc.counts(q[sel], u)[123] == int((np.asarray(v[sel], float) < T[123]).sum())


# --- pooled displacement and the rule ---------------------------------------------------------

def test_codes_roundtrip_and_pooled_median_equals_hindcast_pooled():
    rng = np.random.default_rng(4)
    for n_parts in (1, 2, 3):
        parts_d = []
        for _ in range(n_parts):
            k = rng.integers(0, 4000, rng.integers(1, 40))
            parts_d.append(np.sqrt(k.astype(float)) * 60.0 * rng.choice([-1, 1], k.size))
        parts = [oc.codes(d) for d in parts_d]
        med, n = oc.pooled_median(parts)
        ref = ht.pooled([oc.decode(p) for p in parts])
        assert n == ref["n"] == sum(p.size for p in parts_d)
        assert med == ref["median"]
    inf = [oc.codes(np.array([60.0, np.inf])), oc.codes(np.array([120.0]))]
    assert oc.pooled_median(inf)[0] == np.inf
    assert np.isnan(oc.pooled_median([np.empty(0, np.int32)])[0])
    with pytest.raises(ValueError):
        oc.codes(np.array([np.nan]))


def test_displacement_codes_equal_hindcast_tiled():
    """Per sector, the measured |distances| equal caspian.hindcast_tiled.displacement_cells."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(6)
    F, fin = _toy_flood(seed=2)
    H, W = F.shape
    obs = np.where(rng.random((H, W)) < 0.02, -1, (F < np.float32(1.2)).astype(np.int8)).astype(np.int8)
    obs[:, 60:] = -2                                   # not valid: the outside state decides
    outside = np.full((H, W), -1, np.int8)
    outside[:20, 60:] = 1
    outside[20:40, 60:] = 0
    labels = np.zeros((H, W), np.uint8)
    labels[:, :25], labels[:, 25:50], labels[:, 50:] = 1, 2, 3
    T = 1.5
    cells = ht.displacement_cells(ht.flooded(F, T), obs, obs > -2, labels, 60.0, pred_valid=fin,
                                  outside_state=outside, shape=(H, W), tile=16)
    res = ht.summarise(cells, [1, 2, 3], return_distances=True)
    fm = rev_r.FloodMap(F, fin, tile=16)
    pred_rc = fm.waterline(T)
    obs_rc = rev_r.obs_waterline(obs, outside)
    kw, kl = rev_r.obs_states(obs, outside)
    ostate = np.where(kw, 1, np.where(kl, 0, -1)).astype(np.int8)
    tree = cKDTree(np.c_[obs_rc[0], obs_rc[1]].astype(float))
    got = oc.displacement_codes(pred_rc, obs_rc, ostate[pred_rc], fin[obs_rc], labels[pred_rc], labels[obs_rc],
                                [1, 3], obs_tree=tree)
    for s in (1, 3):
        ref = np.sort(np.abs(res[s]["distances"]))
        assert np.array_equal(np.sort(oc.decode(got[s])), np.sort(oc.decode(oc.codes(ref))))
        assert got[s].size == res[s]["n"]
    assert 2 not in got


def test_rule_flags_equal_h1_mispredicts():
    rng = np.random.default_rng(7)
    rows = []
    for p in ("a", "b"):
        for s in ("x", "y", "z"):
            for b in range(9):
                rows.append({"product": p, "variant": "shape", "sector": s, "bin": b,
                             "observed_level": bool(rng.random() < 0.85), "n": int(rng.integers(0, 3)),
                             "median": float(rng.choice([rng.uniform(0, 2000), np.inf, np.nan]))})
    prod = pd.DataFrame(rows)
    prod.loc[prod["n"] == 0, "median"] = np.nan
    noise = pd.DataFrame([{"sector": s, "bin": b, "n": int(rng.integers(0, 3)), "p95": rng.uniform(200, 1500)}
                          for s in ("x", "y", "z") for b in range(9)])
    summ, detail = h1.mispredicts(prod, noise)
    detail = detail.sort_values(["product", "sector", "bin"])
    D = detail["median"].to_numpy(float).reshape(2, 3, 9)
    Q = detail["p95_noise"].to_numpy(float).reshape(2, 3, 9)
    ev = detail["evaluable"].to_numpy(bool).reshape(2, 3, 9)
    flag, n_ex, n_ev = oc.rule_flags(D, Q, ev)
    s = summ.sort_values(["product", "sector"])
    assert np.array_equal(n_ex.ravel(), s["n_exceeding"].to_numpy())
    assert np.array_equal(n_ev.ravel(), s["n_evaluable"].to_numpy())
    ref = s["mispredicts"].fillna(False).astype(bool).to_numpy()
    assert np.array_equal(flag.ravel(), ref)


# --- summaries --------------------------------------------------------------------------------

def test_clopper_pearson_and_first_sigma():
    lo, hi = oc.clopper_pearson(0, 20)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 20), rel=1e-9)
    lo, hi = oc.clopper_pearson(20, 20)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 20), rel=1e-9)
    lo, hi = oc.clopper_pearson(10, 20)
    assert lo < 0.5 < hi
    assert oc.first_sigma_above([0.1, 0.25, 0.5, 1.0], [0.0, 0.5, 0.55, 1.0]) == 0.5
    assert oc.first_sigma_above([1.0, 0.1], [0.9, 0.6]) == 0.1
    assert oc.first_sigma_above([0.1, 0.25], [0.2, 0.5]) is None


def test_sector_epoch_sigma_median_over_epoch_bins():
    lv = pd.DataFrame({"product": ["p"] * 4 + ["q"] * 4, "sector": ["w"] * 8, "bin": [1, 2, 3, 4] * 2,
                       "sigma_budget_m": [0.1, 0.2, 0.3, 0.5] * 2})
    s = oc.sector_epoch_sigma(lv, [(1, "E1"), (2, "E1"), (3, "E1"), (3, "E2"), (4, "E2")])
    assert s[("w", "E1")] == pytest.approx(0.2)
    assert s[("w", "E2")] == pytest.approx(0.4)
