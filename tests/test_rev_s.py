"""Revision statistics (caspian.rev_s; PREREG Amendment 18)."""

import numpy as np
import pandas as pd
import pytest

from caspian import h1, rev_s
from caspian import hindcast_tiled as ht


# --- moving-block bootstrap -------------------------------------------------------------------

def test_block_indices_are_contiguous_blocks_cut_to_n():
    idx = rev_s.block_indices(7, [0.0, 0.0, 0.999])
    assert idx.tolist() == [0, 1, 2, 0, 1, 2, 4]          # start floor(0.999 * 5) = 4
    idx = rev_s.block_indices(9, [0.5, 0.2, 0.9])
    assert idx.size == 9
    for b in idx.reshape(3, 3):
        assert np.all(np.diff(b) == 1) and b[0] >= 0 and b[-1] <= 8


def test_block_indices_short_sequence_and_errors():
    assert rev_s.block_indices(2, [0.7]).tolist() == [0, 1]
    assert rev_s.block_indices(0, []).size == 0
    with pytest.raises(ValueError):
        rev_s.block_indices(10, [0.1, 0.2])


def test_mbb_extremes_and_reproducibility():
    cells = {"all": np.ones(25, bool), "none": np.zeros(25, bool), "half": np.r_[np.ones(13), np.zeros(12)].astype(bool)}
    s1, f1, k1 = rev_s.mbb_mispredict(cells, 300, 20261022)
    s2, f2, _ = rev_s.mbb_mispredict(cells, 300, 20261022)
    assert np.array_equal(f1, f2)
    p = s1.set_index("key")["p_mispredict"]
    assert p["all"] == 1.0 and p["none"] == 0.0 and 0.0 < p["half"] < 1.0
    assert bool(s1.set_index("key").loc["half", "observed_mispredicts"]) is True
    assert k1 == list(cells)


def test_mbb_marginal_does_not_depend_on_the_other_cells():
    rng = np.random.default_rng(1)
    a = rng.random(25) < 0.55
    alone, fa, _ = rev_s.mbb_mispredict({"a": a}, 200, 7)
    joint, fj, _ = rev_s.mbb_mispredict({"a": a, "b": rng.random(24) < 0.5}, 200, 7)
    assert np.array_equal(fa[:, 0], fj[:, 0])
    dist = rev_s.count_distribution(fj, 2)
    assert dist.sum() == pytest.approx(1.0)


def test_ratio_summary_keeps_infinite_ratios():
    r = rev_s.ratio_summary([1.0, 2.0, np.inf, 4.0], [1.0, 1.0, 1.0, 2.0])
    assert r["n"] == 4 and r["n_inf"] == 1
    assert r["median"] == pytest.approx(2.0)
    assert r["share_above_1"] == pytest.approx(0.75)
    r2 = rev_s.ratio_summary([3.0, 2.0, np.inf], [np.inf, 1.0, np.inf])
    assert r2["n"] == 2 and r2["median"] == pytest.approx(1.0)       # 3/inf = 0 kept; inf/inf left out


# --- permutation tests and Holm ---------------------------------------------------------------

def test_exact_group_test_enumerates_35_labellings():
    vals = {s: float(v) for s, v in zip("abcdefg", [7, 6, 5, 4, 3, 2, 1])}
    r = rev_s.exact_group_test(vals, {"a", "b", "c"})
    assert r["n_labellings"] == 35 and r["min_p"] == pytest.approx(1 / 35)
    # the group holds the 3 largest, yet {a, b, d} gives the same medians (6 and 2.5):
    # with medians of 3 and 4 units the top labelling always ties with one other, so the
    # smallest attainable p is 2/35, not 1/35
    assert r["p_value"] == pytest.approx(2 / 35)
    assert r["statistic"] == pytest.approx(6 - 2.5)
    flat = rev_s.exact_group_test({s: 1.0 for s in "abcdefg"}, {"a", "b", "c"})
    assert flat["p_value"] == 1.0


def test_exact_rank_sum_reaches_one_in_35_and_matches_a_hand_count():
    vals = {s: float(v) for s, v in zip("abcdefg", [7, 6, 5, 4, 3, 2, 1])}
    top = rev_s.exact_rank_sum_test(vals, {"a", "b", "c"})
    assert top["statistic"] == 18 and top["p_value"] == pytest.approx(1 / 35)
    # ranks 2, 4 and 6 of 7 counted from the top (a GEBCO_2026 illustration): rank sum
    # 12 = the null expectation, and 20 of the 35 labellings reach it
    mid = rev_s.exact_rank_sum_test(vals, {"b", "d", "f"})
    assert mid["statistic"] == 12 == mid["expected"]
    assert mid["p_value"] == pytest.approx(20 / 35)


def test_block_permutation_null_reproduces_registered_test():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"sector": rng.choice(list("abcdefg"), 80), "value": rng.gamma(2.0, 1000.0, 80)})
    reg = h1.permutation_test(df, {"a", "b"}, n_perm=500, seed=20261020)
    mine = rev_s.block_permutation_null(df["value"], df["sector"].isin(["a", "b"]), 500, 20261020)
    assert mine["p_value"] == reg["p_value"]
    assert mine["statistic"] == pytest.approx(reg["statistic"])
    assert mine["null"].size == 500


def test_holm_step_down():
    adj = rev_s.holm([0.01, 0.04, 0.03, np.nan])
    assert adj[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert np.isnan(adj[3])
    assert rev_s.holm([0.9, 0.8]).tolist() == [1.0, 1.0]
    assert rev_s.holm([0.01, 0.5]).tolist() == pytest.approx([0.02, 0.5])


# --- summaries --------------------------------------------------------------------------------

def test_theil_sen_and_quantiles():
    x = np.arange(10.0)
    t = rev_s.theil_sen(x, 3 * x + 1)
    assert t["slope"] == pytest.approx(3.0) and t["intercept"] == pytest.approx(1.0)
    q = rev_s.quantile_summary([1, 2, 3, 4, np.nan])
    assert q["n"] == 4 and q["median"] == 2.5 and q["iqr"] == pytest.approx(1.5)


def test_majority_votes_drop_ties():
    df = pd.DataFrame({"s": ["a", "a", "b", "b", "c", "c", "c"], "v": [True, True, True, False, False, False, True]})
    v = rev_s.majority_votes(df, "s", "v")
    assert v.to_dict() == {"a": True, "c": False}


# --- waterline cells in a mask ----------------------------------------------------------------

def test_mask_labels_counts_only_kept_cells_against_every_waterline():
    shape = (40, 40)
    obs = np.zeros(shape, bool)
    obs[:, :20] = True                       # observed water: columns 0-19
    pred = np.zeros(shape, bool)
    pred[:, :23] = True                      # predicted water 3 columns too far
    valid = np.ones(shape, bool)
    labels = np.ones(shape, np.int8)
    cells = ht.displacement_cells(pred, obs, valid, labels, 60.0, shape=shape, tile=16)
    full = ht.summarise(cells, [1])[1]
    top = rev_s.mask_labels(cells, lambda r, c: r < 20)
    half = ht.summarise(top, [1])[1]
    assert half["n"] < full["n"] and half["n"] > 0
    assert half["median"] == pytest.approx(full["median"])     # every row has the same 3-cell offset
    none = ht.summarise(rev_s.mask_labels(cells, lambda r, c: r < 0), [1])[1]
    assert none["n_pred"] == 0 and none["n_obs"] == 0
