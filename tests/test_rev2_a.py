"""caspian.rev2_a: core functions of the second-round revision analysis A (D-147)."""

import numpy as np
import pandas as pd
import pytest

from caspian import hindcast_tiled as ht
from caspian import products, rev2_a, rev_r


# --- item 7(a) ---------------------------------------------------------------------------------

def test_total_threshold_is_level_minus_cref_as_flood_level():
    L = np.array([-28.75, -26.05, -25.65])
    got = rev2_a.total_threshold(L, 0.7098306999999977)
    assert np.array_equal(got, products.flood_level(L, 0.0, 0.7098306999999977, 0.0))
    assert np.allclose(got, L - 0.7098306999999977, atol=1e-12)


# --- item 8: unknown cells ----------------------------------------------------------------------

def test_disk_offsets_radius_two():
    off = rev2_a.disk_offsets(2)
    d2 = (off ** 2).sum(axis=1)
    assert len(off) == 12 and d2.min() == 1 and d2.max() == 4
    assert len({tuple(x) for x in off}) == 12


def test_unknown_at_states_and_grid_edge():
    code = np.array([[1, 0, -1, -2, -2]], np.int8)
    outside = np.array([[-1, -1, -1, 1, -1]], np.int8)
    got = rev2_a.unknown_at(code, outside, np.array([0, 0, 0, 0, 0, -1, 0]), np.array([0, 1, 2, 3, 4, 0, 5]))
    assert got.tolist() == [False, False, True, False, True, True, True]


def _brute_near(code, outside, radius=2):
    H, W = code.shape
    kw, kl = rev_r.obs_states(code, outside)
    U = np.ones((H + 2 * radius, W + 2 * radius), bool)     # beyond the edge: unknown
    U[radius:radius + H, radius:radius + W] = ~(kw | kl)
    out = np.zeros((H, W), bool)
    for r in range(H):
        for c in range(W):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if (dr or dc) and dr * dr + dc * dc <= radius * radius and U[r + radius + dr, c + radius + dc]:
                        out[r, c] = True
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_near_unknown_equals_brute_force(seed):
    rng = np.random.default_rng(seed)
    code = rng.choice([-2, -1, 0, 1], size=(17, 23), p=[0.15, 0.05, 0.4, 0.4]).astype(np.int8)
    outside = rng.choice([-1, 0, 1], size=code.shape).astype(np.int8)
    rr, cc = np.mgrid[0:17, 0:23]
    got = rev2_a.near_unknown(rr.ravel(), cc.ravel(), code, outside, 2).reshape(code.shape)
    assert np.array_equal(got, _brute_near(code, outside, 2))


def test_near_unknown_excludes_beyond_radius():
    code = np.ones((15, 15), np.int8)
    outside = np.full((15, 15), -1, np.int8)
    code[7, 7] = -2                                          # one unknown cell in the middle
    r = np.array([7, 7, 7, 9, 9, 10, 1])
    c = np.array([9, 10, 5, 9, 8, 7, 7])                      # d^2 = 4, 9, 4, 8, 5, 9; last: 1 from the edge
    assert rev2_a.near_unknown(r, c, code, outside, 2).tolist() == [True, False, True, False, False, False, True]


# --- item 8: recomputed displacement on the engine's cells ----------------------------------------

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
    code[:, :20][code[:, :20] >= 0] = 1
    code[:, 70:][code[:, 70:] >= 0] = 0
    outside = rng.choice([-1, 0, 1], size=shape).astype(np.int8)
    labels = (xx // 30 + 1).astype(np.uint8)
    return F, finite, code, outside, labels


def _cells(seed=0, T=2.0):
    F, finite, code, outside, labels = _synthetic(seed)
    fm = rev_r.FloodMap(F, finite, tile=16)
    cells = rev_r.cells_from(fm.waterline(T), rev_r.obs_waterline(code, outside, tile=24),
                             lambda r, c: fm.at(r, c, T), rev_r.obs_at(code, outside), labels)
    return ht._distances(cells, 60.0), code, outside


@pytest.mark.parametrize("seed", [0, 2])
def test_measure_all_counts_every_predicted_cell(seed):
    cells, _, _ = _cells(seed)
    ids = [1, 2, 3, None]
    reg = ht.summarise(cells, ids, return_distances=True)
    allp = ht.summarise(rev2_a.measure_all(cells), ids, return_distances=True)
    assert cells["pred"]["known"].sum() < cells["pred"]["known"].size      # some cells were unknown
    for k in ids:
        assert allp[k]["n_pred"] == reg[k]["n_pred"] + reg[k]["n_pred_unknown"]
        assert allp[k]["n_pred_unknown"] == 0 and allp[k]["n_obs"] == reg[k]["n_obs"]
        assert allp[k]["observed_level"] == reg[k]["observed_level"]
    # the unknown cells enter with their distance to the nearest observed waterline cell, sign +
    p = cells["pred"]
    unk = ~p["known"]
    assert np.all(cells["signed_p"][unk] >= 0)
    d_all = np.sort(np.abs(allp[None]["distances"]))
    want = np.sort(np.abs(np.r_[cells["signed_p"], cells["signed_o"][cells["obs"]["known"]]]))
    assert np.array_equal(d_all, want)
    assert reg is not allp and cells["pred"]["known"].sum() < cells["pred"]["known"].size   # input unchanged


def test_drop_measured_near_unknown_keeps_targets():
    cells, code, outside = _cells(1)
    o = cells["obs"]
    flag = rev2_a.near_unknown(o["rows"], o["cols"], code, outside, 2)
    assert flag.any() and (~flag).any()
    got = rev2_a.drop_measured(cells, "obs", flag)
    reg = ht.summarise(cells, [None], return_distances=True)[None]
    new = ht.summarise(got, [None], return_distances=True)[None]
    assert new["n_obs"] == int((o["known"] & ~flag).sum())
    assert new["n_pred"] == reg["n_pred"]
    # predicted distances are unchanged: the dropped observed cells are still targets
    keep = np.r_[cells["signed_p"][cells["pred"]["known"]], cells["signed_o"][o["known"] & ~flag]]
    assert np.array_equal(np.sort(new["distances"]), np.sort(keep))
    with pytest.raises(ValueError):
        rev2_a.drop_measured(cells, "obs", flag[:-1])


def test_unknown_share_pooled_and_unit_shares():
    disp = pd.DataFrame({
        "product": ["a"] * 5, "variant": ["shape"] * 5, "sector": ["west"] * 4 + ["all"],
        "epoch": ["E1", "E2", "E3", "pooled", "E1"],
        "n_pred": [30, 10, 0, 999, 5], "n_pred_unknown": [10, 10, 0, 999, 5]})
    t = rev2_a.unknown_share(disp)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["n_units"] == 3 and r["n_units_with_pred"] == 2
    assert r["pooled_share"] == pytest.approx(20 / 60)
    assert r["unit_share_median"] == pytest.approx((0.25 + 0.5) / 2)


# --- tables -------------------------------------------------------------------------------------

def test_sector_counts_and_compare_handle_na():
    base = pd.DataFrame({"product": ["p"] * 3, "sector": ["west", "volga", "komsomolets"],
                         "n_evaluable": [10, 10, 5], "n_exceeding": [9, 2, 4],
                         "mispredicts": pd.array([True, False, True], dtype="boolean")})
    new = base.copy()
    new["mispredicts"] = pd.array([True, pd.NA, False], dtype="boolean")
    c = rev2_a.sector_counts(new)["p"]
    assert c == {"all7": 1, "validated4": 1, "six_without_komsomolets": 1, "sectors": ["west"],
                 "not_evaluable": ["volga"]}
    cmp = rev2_a.compare_verdicts(base, new)
    assert cmp["sector"].tolist() == ["west", "volga", "komsomolets"]
    assert cmp["changed"].tolist() == [False, True, True]
    assert cmp["validated"].tolist() == [True, False, False]
