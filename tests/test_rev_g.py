"""Tests of caspian.rev_g (revision analysis G), on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from caspian import rev_g, validation


def test_paired_diffs_uses_common_finite_rows():
    e_dem = np.array([0.0, 1.0, -1.0, np.nan, 2.0])
    e_p = {"a": np.array([1.0, 2.0, np.nan, 5.0, 0.0])}
    d = rev_g.paired_diffs(e_dem, e_p, ("rmse",))
    ok = np.array([True, True, False, False, True])
    assert d[("a", "rmse")] == pytest.approx(validation.rmse(e_dem[ok]) - validation.rmse(e_p["a"][ok]))


def test_leave_one_out_and_influence():
    rng = np.random.default_rng(0)
    g = np.repeat(["b1", "b2", "b3"], 100)
    e_dem = rng.normal(0, 0.3, 300)
    e_p = {"p": rng.normal(0, 1.0, 300)}
    e_p["p"][g == "b3"] *= 10            # one block dominates the product error
    loo = rev_g.leave_one_out(e_dem, e_p, g, ("rmse",))
    assert set(loo["left_out"]) == {"b1", "b2", "b3"}
    keep = g != "b3"
    want = validation.rmse(e_dem[keep]) - validation.rmse(e_p["p"][keep])
    assert loo.loc[loo.left_out == "b3", "diff"].iloc[0] == pytest.approx(want)
    s = rev_g.summarise_influence(loo)
    assert s["most_influential"].iloc[0] == "b3"
    assert s["n_groups"].iloc[0] == 3


def test_jackknife_se_matches_formula_and_mean_case():
    # for the mean, the jackknife SE equals the usual SE of the mean
    x = np.array([1.0, 2.0, 4.0, 7.0])
    loo = np.array([(x.sum() - v) / 3 for v in x])
    assert rev_g.jackknife_se(loo) == pytest.approx(x.std(ddof=1) / np.sqrt(4))
    assert np.isnan(rev_g.jackknife_se([1.0]))


def test_mark_not_estimable():
    t = pd.DataFrame({"scope": ["overall", "west", "ural"], "ci_low": [0.1, 0.2, np.nan],
                      "ci_high": [0.3, 0.2, np.nan], "n_blocks": [14, 1, 0]})
    m = rev_g.mark_not_estimable(t, ["ci_low", "ci_high"])
    assert m.loc[0, "interval_status"] == "estimated" and m.loc[0, "ci_low"] == 0.1
    assert m.loc[1, "interval_status"].startswith("not estimable") and np.isnan(m.loc[1, "ci_high"])
    assert m.loc[2, "interval_status"] == "no block"
    assert t.loc[1, "ci_low"] == 0.2          # the input is not changed


def test_cell_states_and_independent_dry():
    code = np.array([[1, 0], [-2, 0]], np.int8)
    tr = [60.0, 0, 1000.0, 0, -60.0, 2000.0]
    x = np.array([1010.0, 1070.0, 1010.0, 1100.0, 5000.0])
    y = np.array([1990.0, 1990.0, 1930.0, 1900.0, 1990.0])
    s = rev_g.cell_states(x, y, code, tr)
    assert s.tolist() == [1, 0, -2, 0, -3]
    assert rev_g.independent_dry(s).tolist() == [False, True, False, True, False]


def _cells():
    # predicted waterline: two cells; observed waterline: three cells
    return {"pred": {"rows": np.array([0, 0]), "cols": np.array([0, 10]), "known": np.array([True, True]),
                     "label": np.array([1, 1])},
            "obs": {"rows": np.array([0, 0, 0]), "cols": np.array([1, 9, 5]), "known": np.array([True, True, False]),
                    "label": np.array([1, 1, 1])},
            "signed_p": np.array([60.0, -60.0]), "signed_o": np.array([60.0, np.nan, 240.0])}


def test_attribute_sources():
    a = rev_g.attribute_sources(_cells(), np.array([10, 11]))
    p = a[a.kind == "pred"]
    assert p["source"].tolist() == [10, 11] and p["abs_m"].tolist() == [60.0, 60.0]
    o = a[a.kind == "obs"]
    # the second observed cell has an undefined distance and the third is unknown: one left,
    # nearest predicted cell (0, 0), source 10
    assert len(o) == 1 and o["source"].iloc[0] == 10 and o["col"].iloc[0] == 1
    with pytest.raises(ValueError):
        rev_g.attribute_sources(_cells(), np.array([10]))


def test_matched_levels():
    pl = pd.DataFrame({"sector": ["s"] * 4, "level_m": [-27.0, -27.0, -27.1, -27.1],
                       "source": ["fab", "lake", "fab", "lake"], "median_m": [100.0, 900.0, 200.0, 800.0],
                       "n": [25, 30, 25, 5]})
    m = rev_g.matched_levels(pl, ["fab", "lake"], min_n=20)
    m = m.set_index("level_m")
    assert bool(m.loc[-27.0, "matched"]) and not bool(m.loc[-27.1, "matched"])
    assert m.loc[-27.0, "median_lake"] == 900.0 and m.loc[-27.1, "n_lake"] == 5


def test_group_nmad_bootstrap_and_within_block():
    rng = np.random.default_rng(1)
    n = 4000
    blocks = np.repeat([f"b{i}" for i in range(8)], n // 8)
    group = np.where(rng.random(n) < 0.5, "fab", "lake")
    e_et = np.where(group == "fab", rng.normal(0, 0.5, n), rng.normal(0, 1.0, n))
    e_dem = rng.normal(0, 0.6, n)
    t = rev_g.group_nmad_bootstrap({"dem": e_dem, "etopo": e_et}, group, blocks, ("fab", "lake"), n_boot=200, seed=3)
    r = t[(t.series == "etopo") & (t.statistic == "diff_lake_minus_fab")].iloc[0]
    assert r["value"] > 0.3 and r["ci_low"] > 0
    dd = t[t.series == "etopo minus dem"].iloc[0]
    assert dd["value"] == pytest.approx(r["value"] - t[(t.series == "dem") &
                                                       (t.statistic == "diff_lake_minus_fab")]["value"].iloc[0])
    w = rev_g.within_block_contrast({"dem": e_dem, "etopo": e_et}, group, blocks, ("fab", "lake"), min_n=50)
    assert len(w) == 8 and (w["etopo_diff"] > 0).all()
    w2 = rev_g.within_block_contrast({"etopo": e_et}, group, blocks, ("fab", "lake"), min_n=10_000)
    assert w2.empty
