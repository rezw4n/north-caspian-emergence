"""Conformal calibration: the pre-registered Mondrian fallback (PREREG section 8) and
one inclusion rule for calibration and evaluation (review findings 11 and 31)."""

import numpy as np
import pytest

from caspian import conformal


# --- finding 11: Mondrian with the pooled fallback ---------------------------------------

def test_small_sector_uses_pooled_quantile():
    s = np.r_[np.arange(1.0, 101.0), np.arange(1.0, 11.0)]
    g = np.r_[np.zeros(100), np.ones(10)]
    pooled = conformal.conformal_quantile(s, 0.1)
    q = conformal.mondrian_quantiles(s, g, 0.1)
    assert q[0.0] == conformal.conformal_quantile(s[:100], 0.1)
    assert q[1.0] == pooled                    # 10 < 50 points: pooled, not its own 10.0
    cal = conformal.mondrian_calibration(s, g, 0.1)
    assert cal["min_n"] == 50
    assert cal["pooled"] == {"q": pooled, "n": 110}
    assert cal["groups"][1.0] == {"q": pooled, "n": 10, "used_pooled": True}
    assert cal["groups"][0.0]["used_pooled"] is False and cal["groups"][0.0]["n"] == 100


def test_tiny_sector_does_not_get_infinite_interval():
    s = np.r_[np.arange(1.0, 101.0), np.arange(1.0, 6.0)]
    g = np.r_[np.zeros(100), np.ones(5)]
    q = conformal.mondrian_quantiles(s, g, 0.1)
    assert np.isfinite(q[1.0])


def test_fallback_threshold_is_inclusive_at_50():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 200)
    g = np.r_[np.zeros(50), np.ones(49), np.full(101, 2.0)]
    cal = conformal.mondrian_calibration(s, g, 0.1)
    assert cal["groups"][0.0]["used_pooled"] is False   # exactly 50: own q̂
    assert cal["groups"][1.0]["used_pooled"] is True    # 49: pooled
    assert cal["groups"][0.0]["q"] == conformal.conformal_quantile(s[:50], 0.1)


def test_group_size_counts_usable_scores_only():
    s = np.r_[np.ones(52), np.arange(1.0, 101.0)]
    s[:3] = np.nan                             # 49 usable points in group 0
    g = np.r_[np.zeros(52), np.ones(100)]
    cal = conformal.mondrian_calibration(s, g, 0.1)
    assert cal["groups"][0.0]["n"] == 49 and cal["groups"][0.0]["used_pooled"]
    assert cal["pooled"]["n"] == 149


def test_mondrian_q_for_test_points():
    s = np.r_[np.full(60, 1.0), np.full(60, 3.0)]
    g = np.r_[np.zeros(60), np.ones(60)]
    cal = conformal.mondrian_calibration(s, g, 0.1)
    q = conformal.mondrian_q([0.0, 1.0], cal)
    assert q[0] == 1.0 and q[1] == 3.0


def test_test_sector_without_calibration_points_is_recorded_not_silent():
    """PREREG section 8: a sector that uses the pooled q̂ is reported. A sector with
    no calibration point at all falls back too, so it must appear in the record."""
    s = np.r_[np.full(60, 1.0), np.full(60, 3.0)]
    g = np.r_[np.zeros(60), np.ones(60)]
    cal = conformal.mondrian_calibration(s, g, 0.1)
    with pytest.raises(KeyError):
        conformal.mondrian_q([0.0, 7.0], cal)  # unrecorded fallback is refused
    cal = conformal.mondrian_calibration(s, g, 0.1, all_groups=[0, 1, 7])
    assert cal["groups"][7] == {"q": cal["pooled"]["q"], "n": 0, "used_pooled": True}
    assert cal["groups"][0.0]["used_pooled"] is False      # existing groups unchanged
    q = conformal.mondrian_q([0.0, 1.0, 7.0], cal)
    assert q[0] == 1.0 and q[1] == 3.0 and q[2] == cal["pooled"]["q"]
    names = np.r_[np.full(60, "volga"), np.full(60, "ural")]
    cal_s = conformal.mondrian_calibration(s, names, 0.1, all_groups=["volga", "ural", "west"])
    assert cal_s["groups"]["west"]["used_pooled"] and cal_s["groups"]["west"]["n"] == 0
    assert conformal.mondrian_q(np.array(["west"]), cal_s)[0] == cal_s["pooled"]["q"]


def test_mondrian_min_n_must_be_positive():
    with pytest.raises(ValueError):
        conformal.mondrian_calibration([1.0, 2.0], [0, 0], 0.1, min_n=0)


# --- finding 31: one inclusion rule -------------------------------------------------------

def test_coverage_excludes_points_without_prediction():
    y = np.zeros(10)
    lo, hi = np.full(10, -1.0), np.full(10, 1.0)
    lo[:3] = np.nan                            # no DEM value (outside the TIN hull)
    c = conformal.coverage(y, lo, hi)
    assert c["k"] == 7 and c["n"] == 7 and c["coverage"] == 1.0
    assert c["n_excluded"] == 3
    y2 = y.copy()
    y2[5] = np.nan                             # missing reference height
    assert conformal.coverage(y2, np.full(10, -1.0), np.full(10, 1.0))["n"] == 9


def test_infinite_interval_counts_as_covered():
    c = conformal.coverage([0.0, 5.0], [-np.inf, -1.0], [np.inf, 1.0])
    assert c["n"] == 2 and c["k"] == 1 and c["n_excluded"] == 0


def test_conformal_quantile_keeps_infinite_scores():
    # NaN (no prediction) is dropped: n = 19, k = ceil(20 * 0.9) = 18
    assert conformal.conformal_quantile(np.r_[np.arange(1.0, 20.0), np.nan], 0.1) == 18
    s = np.r_[np.arange(1.0, 10.0), np.inf]    # n = 10, k = ceil(11 * 0.9) = 10 -> the inf score
    assert conformal.conformal_quantile(s, 0.1) == np.inf


def test_calibration_and_evaluation_use_the_same_points():
    rng = np.random.default_rng(7)
    n = 400
    y = rng.normal(0, 1, n)
    yhat = np.zeros(n)
    scale = rng.uniform(0.5, 2.0, n)
    yhat[rng.choice(n, 40, replace=False)] = np.nan   # no prediction
    scale[rng.choice(n, 20, replace=False)] = np.nan  # no scale covariate
    use = conformal.usable(y, yhat, scale)
    sc = conformal.scores(y, yhat, scale)
    assert np.array_equal(~np.isnan(sc), use)
    cal, test = np.arange(n) < 200, np.arange(n) >= 200
    q = conformal.conformal_quantile(sc[cal], 0.1)
    lo, hi = conformal.intervals(yhat[test], q, scale[test])
    c = conformal.coverage(y[test], lo, hi)
    assert c["n"] == int(use[test].sum())
    assert c["n_excluded"] == int((~use[test]).sum())


def test_scores_reject_nonpositive_scale_but_pass_missing():
    with pytest.raises(ValueError):
        conformal.scores([1.0], [0.0], [0.0])
    out = conformal.scores([1.0, 2.0], [0.0, 0.0], [np.nan, 1.0])
    assert np.isnan(out[0]) and out[1] == 2.0
