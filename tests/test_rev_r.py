"""caspian.rev_r: the revision workstream R core functions (D-116 to D-119)."""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from caspian import rev_r


# --- control construction -------------------------------------------------------------------

def test_native_cell_index_nearest_centre_and_outside():
    lon = np.array([10.0, 10.25, 10.5])
    lat = np.array([40.0, 40.25])
    i, j = rev_r.native_cell_index([10.0, 10.12, 10.13, 10.62, 10.63, 9.876], [40.0, 40.1, 40.13, 40.25, 40.0, 40.0],
                                   lon, lat)
    assert j.tolist() == [0, 0, 1, 2, -1, 0]
    assert i.tolist() == [0, 0, 1, 1, -1, 0]


def test_native_mean_blocks_equal_direct_mean_and_skip_nan():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(20, 30))
    v[3, 4] = np.nan
    i = rng.integers(0, 4, v.shape)
    j = rng.integers(0, 5, v.shape)
    j[0, 0] = -1                                    # outside: ignored
    acc = rev_r.NativeMean((4, 5))
    for r0 in range(0, 20, 7):
        acc.add(v[r0:r0 + 7], i[r0:r0 + 7], j[r0:r0 + 7])
    got = acc.mean()
    ok = np.isfinite(v) & (j >= 0)
    for a in range(4):
        for b in range(5):
            m = ok & (i == a) & (j == b)
            want = v[m].mean() if m.any() else np.nan
            assert got[a, b] == pytest.approx(want, nan_ok=True)


def test_native_mean_empty_cell_is_nan():
    acc = rev_r.NativeMean((2, 2))
    acc.add(np.array([1.0, 3.0]), np.array([0, 0]), np.array([0, 0]))
    m = acc.mean()
    assert m[0, 0] == 2.0 and np.isnan(m[1, 1])


def test_round_half_up_with_mask_and_nan():
    z = np.array([-27.5, -27.49, -27.51, 2.5, np.nan])
    assert rev_r.round_half_up(z)[:4].tolist() == [-27.0, -27.0, -28.0, 3.0]
    out = rev_r.round_half_up(z, np.array([True, False, True, False, True]))
    assert out[:4].tolist() == [-27.0, -27.49, -28.0, 2.5] and np.isnan(out[4])


def test_whole_metre_mask_tolerance():
    z = np.array([-27.0, -27.00014, -26.98, np.nan, -26.995])
    assert rev_r.whole_metre_mask(z, 0.01).tolist() == [True, True, False, False, True]


def test_fill_nearest_matches_whole_grid_transform_within_halo():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(40, 50)).astype(np.float32)
    v[rng.random(v.shape) < 0.7] = np.nan
    where = rng.random(v.shape) < 0.8
    got = rev_r.fill_nearest(v, where, tile=16, halo=60)
    dist, (ii, jj) = ndimage.distance_transform_edt(~np.isfinite(v), return_indices=True)
    want = np.where(where & ~np.isfinite(v), v[ii, jj], v)
    # ties may resolve to a different (equidistant) cell: compare the distances instead
    need = where & ~np.isfinite(v)
    assert np.isfinite(got[need]).all()
    for r, c in zip(*np.nonzero(need)):
        cand = np.argwhere(np.isfinite(v) & (v == got[r, c]))
        dmin = np.min(np.hypot(cand[:, 0] - r, cand[:, 1] - c))
        assert dmin == pytest.approx(dist[r, c])
    keep = ~need
    assert np.array_equal(np.isnan(got[keep]), np.isnan(want[keep]))
    assert np.allclose(got[keep][np.isfinite(want[keep])], want[keep][np.isfinite(want[keep])])


def test_fill_nearest_leaves_nan_beyond_halo():
    v = np.full((10, 40), np.nan, np.float32)
    v[5, 0] = 1.0
    got = rev_r.fill_nearest(v, np.ones(v.shape, bool), tile=8, halo=5)
    assert got[5, 4] == 1.0 and np.isnan(got[5, 30])


# --- excess and stop rules ------------------------------------------------------------------

def _levels(D, Q, Dc, ev, product="p", sector="s"):
    n = len(D)
    return pd.DataFrame({"product": product, "sector": sector, "bin": range(n), "D": D, "Q": Q,
                         "D_control": Dc, "evaluable": ev})


def test_excess_rule_majority_against_max_of_floor_and_control():
    d = _levels([5, 5, 5, 1, np.inf], [1, 1, 6, 1, 1], [2, 6, 1, 0, 1], [True] * 5)
    out = rev_r.excess_rule(d)
    # level 0: 5 > max(1,2); level 1: 5 < 6; level 2: 5 < 6; level 3: 1 = max(1,0) no; level 4: inf yes
    assert out.loc[0, "n_exceeding"] == 2 and out.loc[0, "n_evaluable"] == 5
    assert not bool(out.loc[0, "excess"])


def test_excess_rule_counts_only_evaluable_and_na_when_none():
    d = pd.concat([_levels([5, 5, 5], [1, 1, 1], [1, 1, 9], [True, True, False]),
                   _levels([5], [1], [1], [False], sector="t")])
    out = rev_r.excess_rule(d).set_index("sector")
    assert out.loc["s", "n_evaluable"] == 2 and out.loc["s", "n_exceeding"] == 2 and bool(out.loc["s", "excess"])
    assert pd.isna(out.loc["t", "excess"])


def test_stop_rule_counts_missing_and_na_as_failures():
    ex = pd.DataFrame({"product": ["a", "a", "b"], "sector": ["x", "y", "x"],
                       "excess": pd.array([True, pd.NA, False], dtype="boolean")})
    r = rev_r.stop_rule(ex, ["a", "b"], ["x", "y"])
    assert r["n_cases"] == 4 and r["n_excess_holds"] == 1 and r["n_excess_fails"] == 3 and r["fires"]
    r2 = rev_r.stop_rule(ex.assign(excess=pd.array([True, True, True], dtype="boolean")), ["a", "b"], ["x", "y"])
    assert r2["n_excess_fails"] == 1 and not r2["fires"]


# --- waterline engine against hindcast_tiled --------------------------------------------------

from caspian import hindcast_tiled as ht  # noqa: E402


def _synthetic(seed=0, shape=(70, 90)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    z = (0.05 * xx + 0.3 * np.sin(yy / 5.0) + rng.normal(0, 0.4, shape)).astype(np.float32)
    z[rng.random(shape) < 0.03] = np.nan
    seed_mask = np.zeros(shape, bool)
    seed_mask[:, :3] = True
    F = ht.flood_level_map(z, seed_mask, tile=32)
    finite = np.isfinite(z)
    code = rng.choice([-2, -1, 0, 1], size=shape, p=[0.2, 0.05, 0.4, 0.35]).astype(np.int8)
    code[:, :20][code[:, :20] >= 0] = 1                       # sea on the left
    code[:, 70:][code[:, 70:] >= 0] = 0                       # land on the right
    outside = rng.choice([-1, 0, 1], size=shape).astype(np.int8)
    labels = (xx // 30 + 1).astype(np.uint8)
    return F, finite, code, outside, labels


def _key(cells, part):
    c = cells[part]
    order = np.lexsort((c["cols"], c["rows"]))
    return {k: np.asarray(c[k])[order] for k in ("rows", "cols", "known", "wet", "label")}


def _same_cells(a, b):
    for part in ("pred", "obs"):
        ka, kb = _key(a, part), _key(b, part)
        for k in ka:
            assert np.array_equal(ka[k], kb[k]), (part, k)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_product_cells_equal_hindcast_tiled(seed):
    F, finite, code, outside, labels = _synthetic(seed)
    for T in (1.0, 2.0, 3.3):
        ref = ht.displacement_cells(ht.flooded(F, T), code, lambda r0, r1, c0, c1: code[r0:r1, c0:c1] > -2,
                                    labels, 60.0, pred_valid=finite, outside_state=outside, tile=32)
        fm = rev_r.FloodMap(F, finite, tile=16)
        got = rev_r.cells_from(fm.waterline(T), rev_r.obs_waterline(code, outside, tile=24),
                               lambda r, c: fm.at(r, c, T), rev_r.obs_at(code, outside), labels)
        _same_cells(ref, got)
        got = ht._distances(got, 60.0)
        sa = ht.summarise(ref, [1, 2, 3, None], return_distances=True)
        sb = ht.summarise(got, [1, 2, 3, None], return_distances=True)
        for k in sa:
            for f in ("median", "p95", "signed_median", "n", "n_pred", "n_obs", "observed_level"):
                assert sa[k][f] == sb[k][f] or (np.isnan(sa[k][f]) and np.isnan(sb[k][f])), (k, f)
            assert np.array_equal(np.sort(sa[k]["distances"]), np.sort(sb[k]["distances"]))


def test_engine_noise_floor_cells_equal_hindcast_tiled():
    F, finite, a, outside, labels = _synthetic(3)
    _, _, b, _, _ = _synthetic(4)
    ref = ht.noise_floor_cells(a, lambda r0, r1, c0, c1: a[r0:r1, c0:c1] > -2, b,
                               lambda r0, r1, c0, c1: b[r0:r1, c0:c1] > -2, labels, 60.0, outside_state=outside, tile=32)
    got = rev_r.cells_from(rev_r.obs_waterline(a, outside, tile=20), rev_r.obs_waterline(b, outside, tile=20),
                           rev_r.obs_at(a, outside), rev_r.obs_at(b, outside), labels)
    _same_cells(ref, got)


def test_floodmap_skip_is_exact_on_constant_tiles():
    F = np.full((40, 40), 5.0, np.float32)
    F[:, :10] = 0.0
    finite = np.ones(F.shape, bool)
    fm = rev_r.FloodMap(F, finite, tile=8)
    r, c = fm.waterline(1.0)
    assert set(c.tolist()) == {9} and len(r) == 40


def test_drop_cells():
    rc = (np.array([0, 1, 2], np.int32), np.array([0, 1, 2], np.int32))
    flag = np.zeros((3, 3), bool)
    flag[1, 1] = True
    r, c = rev_r.drop_cells(rc, flag)
    assert r.tolist() == [0, 2] and c.tolist() == [0, 2]


def test_majority_to_native_ties_water_and_invalid_kept():
    raw = np.array([[1, 0, -2, 0], [0, 1, 1, 0]], np.int8)
    nid = np.array([[0, 0, 1, 1], [0, 0, 1, -1]])
    out = rev_r.majority_to_native(raw, nid, 2)
    # cell 0: 2 water, 2 dry -> water; cell 1: valid (0,3)=0,(1,2)=1 -> tie -> water; (0,2) stays -2; nid -1 kept
    assert out.tolist() == [[1, 1, -2, 1], [1, 1, 1, 0]]


# --- gradients and budget -----------------------------------------------------------------------

def test_slope_of_a_plane():
    yy, xx = np.mgrid[0:20, 0:30].astype(float)
    z = 0.003 * xx * 30 + 0.004 * yy * 30          # 3 and 4 m per km on a 30 m grid
    s = rev_r.slope_m_per_km(z, 30.0)
    assert np.allclose(s, 5.0)


def test_budget_sigma_quadrature_and_absent_terms():
    assert rev_r.budget_sigma(0.03, 0.04) == pytest.approx(0.05)
    out = rev_r.budget_sigma(np.array([0.03, np.nan]), np.array([0.04, np.nan]), np.nan)
    assert out[0] == pytest.approx(0.05) and np.isnan(out[1])


def test_augmented_floor_units():
    # 0.1 m at 0.2 m per km is 0.5 km = 500 m; times 1.645 added to Q
    assert rev_r.augmented_floor(1000.0, 0.1, 0.2) == pytest.approx(1000.0 + 1.645 * 500.0)
