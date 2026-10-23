"""caspian.rev2_d: core functions of the second-review analyses D (PREREG Amendment 19)."""

import numpy as np
import pandas as pd
import pytest

from caspian import rev2_d, validation


# --- level bands ------------------------------------------------------------------------------

def test_level_band_right_closed_half_metre_bands():
    lo, hi = rev2_d.level_band([-28.75, -28.55, -28.45, -26.55, -26.45, -25.65, -27.0])
    assert hi.tolist() == [-28.5, -28.5, -28.0, -26.5, -26.0, -25.5, -27.0]
    assert lo.tolist() == [-29.0, -29.0, -28.5, -27.0, -26.5, -26.0, -27.5]


# --- single changepoint -----------------------------------------------------------------------

def test_changepoint_finds_the_planted_step_and_its_means():
    lev = np.repeat(np.round(np.arange(-28.75, -25.6, 0.1), 2), 3)
    y = np.where(lev < -27.7, 1.0, -0.5) + np.tile([-0.01, 0.0, 0.01], lev.size // 3)
    edges = np.round(np.arange(-28.7, -25.6, 0.1), 2)
    r = rev2_d.single_changepoint(lev, y, edges, min_levels=3)
    assert r["edge"] == pytest.approx(-27.7)
    assert r["mean_below"] == pytest.approx(1.0)
    assert r["mean_above"] == pytest.approx(-0.5)
    assert r["r2"] > 0.99
    assert r["n_below"] + r["n_above"] == r["n"] == lev.size


def test_changepoint_respects_the_minimum_segment_and_brute_force():
    rng = np.random.default_rng(1)
    lev = np.repeat(np.round(np.arange(-28.75, -26.0, 0.1), 2), 2)
    y = rng.normal(size=lev.size)
    edges = np.round(np.arange(-28.7, -26.0, 0.1), 2)
    r = rev2_d.single_changepoint(lev, y, edges, min_levels=4)
    adm = [p["edge"] for p in r["profile"]]
    assert min(adm) == pytest.approx(-28.4) and max(adm) == pytest.approx(-26.4)   # 4 levels each side
    best, best_rss = None, np.inf
    for t in adm:
        a, b = y[lev < t], y[lev > t]
        rss = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if rss < best_rss - 1e-12:
            best, best_rss = t, rss
    assert r["edge"] == pytest.approx(best)
    assert r["rss"] == pytest.approx(best_rss)


def test_changepoint_drops_nonfinite_and_rejects_edge_on_a_level():
    lev = np.array([1.0, 2, 3, 4, 5, 6, np.nan])
    y = np.array([0, 0, 0, 1, 1, np.inf, 5.0])
    r = rev2_d.single_changepoint(lev, y, [1.5, 2.5, 3.5, 4.5], min_levels=2)
    assert r["n"] == 5 and r["edge"] == 3.5
    with pytest.raises(ValueError):
        rev2_d.single_changepoint(lev, y, [3.0], min_levels=1)


# --- weighted block bootstrap -----------------------------------------------------------------

def _concat_stats(e, blocks, n_boot, seed):
    def stat(ix):
        x = e[ix]
        return [validation.bias(x), float(np.median(x)), validation.nmad(x), validation.rmse(x)]
    return validation.block_bootstrap(blocks, stat, n_boot, seed)


@pytest.mark.parametrize("seed", [0, 20261023])
def test_block_bootstrap_stats_equal_the_concatenated_draws(seed):
    rng = np.random.default_rng(seed)
    nb = 7
    sizes = rng.integers(1, 40, nb)
    blocks = np.repeat([f"{i}_{-i}" for i in range(nb)], sizes)
    e = np.round(rng.normal(0.2, 1.0, blocks.size), 2)           # rounded: many ties
    e[:5] = 0.0
    got = rev2_d.block_bootstrap_stats(e, blocks, 300, seed)
    want = _concat_stats(e, blocks, 300, seed)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_block_bootstrap_stats_single_block_and_even_odd_sizes():
    e = np.array([3.0, -1.0, 0.5, 0.5])
    got = rev2_d.block_bootstrap_stats(e, ["a"] * 4, 5, 3)
    want = _concat_stats(e, np.array(["a"] * 4), 5, 3)
    np.testing.assert_allclose(got, want, atol=1e-12)
    m, med, nm, r = rev2_d.weighted_stats(np.sort(e), np.ones(4, int))
    assert med == np.median(e) and nm == pytest.approx(validation.nmad(e)) and r == pytest.approx(validation.rmse(e))
    m, med, nm, r = rev2_d.weighted_stats(np.array([-2.0, 0.0, 7.0]), np.array([0, 3, 2]))
    x = np.array([0.0, 0.0, 0.0, 7.0, 7.0])
    assert med == np.median(x) and nm == pytest.approx(validation.nmad(x)) and m == pytest.approx(x.mean())


def test_block_bootstrap_stats_rejects_bad_input():
    with pytest.raises(ValueError):
        rev2_d.block_bootstrap_stats([], [], 10, 0)
    with pytest.raises(ValueError):
        rev2_d.block_bootstrap_stats([1.0, np.nan], ["a", "b"], 10, 0)


def test_percentile_interval_ignores_nan():
    lo, hi = rev2_d.percentile_interval(np.r_[np.arange(101.0), np.nan])
    assert lo == pytest.approx(2.5) and hi == pytest.approx(97.5)


# --- correlations -----------------------------------------------------------------------------

def test_pairwise_correlations_pairs_and_pairwise_dropping():
    w = pd.DataFrame({"p": [1.0, 2, 3, 4, 5], "q": [2.0, 4, 6, 8, np.nan], "r": [5.0, 4, 3, 2, 1]})
    c = rev2_d.pairwise_correlations(w).set_index(["a", "b"])
    assert len(c) == 3
    assert c.loc[("p", "q"), "n"] == 4 and c.loc[("p", "q"), "pearson_r"] == pytest.approx(1.0)
    assert c.loc[("p", "r"), "spearman_rho"] == pytest.approx(-1.0)
    w2 = pd.DataFrame({"p": [1.0, 2], "q": [3.0, 4]})
    assert np.isnan(rev2_d.pairwise_correlations(w2)["pearson_r"].iloc[0])


# --- the subset definitions of scripts/rev2_d_constants.py ------------------------------------

def test_constant_subsets_partition_the_units():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rev2_d_constants", root / "scripts" / "rev2_d_constants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bins = [12, 13, 31, 31, 32, 32, 34, 34, 35, 35, 43]
    eps = ["E3", "E3", "E2", "E3", "E2", "E3", "E1", "E2", "E1", "E2", "E1"]
    lev = [-30.0 + 0.1 * b + 0.05 for b in bins]
    sub = mod.subsets(lev, eps, bins)
    kinds = {}
    for kind, label, m in sub:
        kinds.setdefault(kind, []).append((label, m))
    for kind in ("epoch", "band"):              # each a partition of the units
        tot = np.sum([m.astype(int) for _, m in kinds[kind]], axis=0)
        assert tot.tolist() == [1] * len(bins)
    shared = dict(kinds["epoch_shared_levels"])
    assert set(shared) == {"E2 at E2&E3 bins 31-32", "E3 at E2&E3 bins 31-32",
                           "E1 at E1&E2 bins 34-35", "E2 at E1&E2 bins 34-35"}
    assert shared["E2 at E2&E3 bins 31-32"].sum() == 2
