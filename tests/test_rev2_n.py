"""caspian.rev2_n (PREREG Amendment 19 items 1-3; D-146) on synthetic grids."""

import numpy as np
import pytest

from caspian import exposure as ex
from caspian import hindcast_tiled as ht
from caspian import nearterm as nt
from caspian import rev2_n as r2


def _world(seed=0, H=40, W=50):
    """A seabed rising eastward with noise, open-sea seed in the west column, two regions."""
    rng = np.random.default_rng(seed)
    z = (np.linspace(-6, 2, W)[None, :] + 0.3 * rng.standard_normal((H, W))).astype(np.float32)
    seed_m = np.zeros((H, W), bool)
    seed_m[:, 0] = True
    lab = np.ones((H, W), np.uint8)
    lab[H // 2:] = 2
    return z, seed_m, lab


def test_bisect_offset_matches_full_evaluation():
    z, seed, lab = _world(1)
    F = ht.flood_level_map(z, seed, tile=16).astype(np.float64)
    L26, c = -0.3, -0.7
    ks = np.arange(-300, 301)
    area = lambda k: float((F < L26 + c + k * 0.01).sum())  # noqa: E731
    lo, hi = area(-300), area(300)
    for target in np.linspace(lo, hi, 13).tolist() + [float((F < L26 + c).sum()) + 17, lo + 0.5]:
        full = r2.closest_offset([area(k) for k in ks], ks, target)
        b = r2.bisect_offset(area, target, -300, 300)
        assert b["k"] == full["k"] and b["area"] == full["area"]
        assert abs(b["offset_m"] - full["offset_m"]) < 1e-12


def test_bisect_offset_ties_and_bracket():
    steps = {k: (0.0 if k < 3 else 10.0) for k in range(-10, 11)}   # jump between k = 2 and 3
    area = steps.__getitem__
    # target halfway: every k <= 2 ties with k = 3; the smallest |k| (0, on the lower plateau) wins
    assert r2.bisect_offset(area, 5.0, -10, 10)["k"] == 0
    # nearer the upper step
    assert r2.bisect_offset(area, 6.0, -10, 10)["k"] == 3
    with pytest.raises(ValueError):
        r2.bisect_offset(area, 11.0, -10, 10)
    with pytest.raises(ValueError):
        r2.bisect_offset(lambda k: 5.0 + k, 4.0, 0, 10)
    # symmetric tie in sign: the smaller k
    assert r2.closest_offset([1.0, 0.0, 1.0], [-1, 0, 1], 1.0)["k"] == -1
    # plateaus: the tied k nearest 0 is taken, as by full evaluation
    for pl in ({k: (0.0 if k < -4 else 10.0) for k in range(-10, 11)},
               {k: (0.0 if k < 6 else 10.0) for k in range(-10, 11)},
               {k: float(min(max(k, -3), 4)) for k in range(-10, 11)}):
        ks = np.arange(-10, 11)
        for t in (0.0, 2.0, 7.0, 10.0, -3.0, 4.0, 0.5):
            if not (pl[-10] <= t <= pl[10]):
                continue
            assert r2.bisect_offset(pl.__getitem__, t, -10, 10)["k"] == \
                r2.closest_offset([pl[k] for k in ks], ks, t)["k"], (t,)


def test_weighted_area():
    F = np.array([[0.5, np.inf, -1.0], [np.nan, 2.0, 0.5]])
    w = np.array([[1, 5, 2], [7, 3, 0]])
    A = r2.WeightedArea(F, w)
    assert A(0.0) == 2.0            # only -1.0
    assert A(0.5) == 2.0            # strict: 0.5 not below 0.5
    assert A(0.51) == 3.0           # 0.5 (w 1); the other 0.5 has weight 0
    assert A(10.0) == 6.0           # inf and nan never count
    assert np.allclose(A(np.array([0.0, 10.0])), [2.0, 6.0])


def test_increment_from_counts_equals_brute_force():
    """The scripts' route (region counts of F on a level grid, restricted to S) gives the
    whole-array increment and misfit exactly, and the area-matched T leaves product sea
    outside S equal to the misfit up to one grid step."""
    z, seed, lab = _world(2)
    F = ht.flood_level_map(z, seed, tile=16)
    L26, c = 0.0, -0.5
    S = ht.flood_level_map(z + np.float32(0.4), seed, tile=16) < L26     # an "observed" sea
    lv = ex.level_grid(L26, 12.0, above=5.0)
    allc = nt.region_counts(F, lab, 2, lv, tile=16).sum(axis=0)
    sc = nt.region_counts(F, lab, 2, lv, mask=S, tile=16).sum(axis=0)
    i0 = int(np.flatnonzero(np.isclose(lv, L26 + c, atol=1e-9))[0])
    target = float(S.sum())
    fit = r2.bisect_offset(lambda k: allc[i0 + k], target, -500, 500)
    iT = i0 + fit["k"]
    T = float(lv[iT])
    deltas = (0.5, 1.0, 2.0, 3.0)
    E = [float(sc[iT] - sc[iT - int(round(100 * d))]) for d in deltas]
    M = target - float(sc[iT])
    bf = r2.increment_and_misfit(F, S, T, deltas)
    assert E == bf["E"] and M == bf["M"]
    assert bf["E"][1] > 0
    # area match: product sea outside S equals the misfit to within the cells of one step
    one_step = float(allc[iT + 1] - allc[iT - 1])
    assert abs(bf["outside"] - bf["M"]) <= one_step
    # the band's cells lie in S, are sea at T and dry 1 m lower
    band = r2.rereferenced_band(F, S, T, 1.0)
    f = np.asarray(F, float)
    assert (S[band]).all() and (f[band] < T).all() and (f[band] >= T - 1.0).all()


def test_level_shift_invariance_of_area_match():
    """Refitting c26 at L26 + 0.3 gives the same threshold T (L26 cancels)."""
    z, seed, lab = _world(3)
    F = ht.flood_level_map(z, seed, tile=16)
    c = -0.4
    lv = ex.level_grid(0.0, 12.0, above=5.0)
    allc = nt.region_counts(F, lab, 2, lv, tile=16).sum(axis=0)
    target = 700.0
    Ts = []
    for L in (0.0, 0.3, -0.3):
        i0 = int(np.flatnonzero(np.isclose(lv, L + c, atol=1e-9))[0])
        k = r2.bisect_offset(lambda k: allc[i0 + k], target, -400, 400)["k"]
        Ts.append(round(float(lv[i0 + k]), 6))
    assert Ts[0] == Ts[1] == Ts[2]


def test_reref_surface_states():
    z, seed, lab = _world(4)
    F = ht.flood_level_map(z, seed, tile=16)
    L26, shift = 0.0, -0.6
    B = np.where(ht.flood_level_map(z + np.float32(0.5), seed, tile=16) < L26, -1000.0, np.inf)
    S = B < L26
    T = L26 + shift
    G = r2.ReRefSurface(F, B, L26, shift)[:, :]
    f = np.asarray(F, float)
    assert np.array_equal(G < L26, S & (f - shift < L26))
    for d in (0.5, 1.0, 2.0):
        emerged = (G < L26) & ~(G < L26 - d)
        assert np.array_equal(emerged, S & (f - shift < L26) & ~(f - shift < L26 - d))
    Gs = r2.ReRefSurface(F, B, L26, shift, misfit="sea")[:, :]
    assert np.array_equal(Gs < L26, S)
    assert np.isfinite(Gs[S]).all()                  # distance profiles skip non-finite cells
    for d in (0.5, 1.0):
        assert np.array_equal((Gs < L26) & ~(Gs < L26 - d), (G < L26) & ~(G < L26 - d))
    # the scripts' form: no shift, sea level T, so the comparisons are F < T exactly
    H = r2.ReRefSurface(F, B, L26, 0.0, sea_level=T)[:, :]
    Hs = r2.ReRefSurface(F, B, L26, 0.0, misfit="sea", sea_level=T)[:, :]
    for d in (0.0, 0.5, 1.0):
        assert np.array_equal(H < T - d, S & (f < T - d))
    assert np.array_equal(Hs < T, S)
    assert np.array_equal((Hs < T) & ~(Hs < T - 1.0), r2.rereferenced_band(F, S, T, 1.0))
    with pytest.raises(ValueError):
        r2.ReRefSurface(F, B, L26, shift, misfit="x")


def test_lineage_table_and_ratio():
    prods = ["gebco_2026", "etopo_2022", "srtm15p_v2.7", "gebco_2022", "etopo1"]
    tab = np.array([[1.0, 2.0, 10.0, 3.0, 5.0], [4.0, 4.0, 1.0, 4.0, 0.0]])
    t, names = r2.lineage_table(tab, prods)
    assert names == ["GEBCO", "ETOPO1", "SRTM15+"]
    assert np.allclose(t, [[2.0, 5.0, 10.0], [4.0, 0.0, 1.0]])
    t2, _ = r2.lineage_table(tab, prods, representative={"GEBCO": "gebco_2022"})
    assert np.allclose(t2[:, 0], [3.0, 4.0])
    assert r2.max_min_ratio([2.0, 5.0, 10.0]) == 5.0
    assert r2.max_min_ratio([0.0, 1.0]) == np.inf
    assert np.isnan(r2.max_min_ratio([0.0, 0.0]))
