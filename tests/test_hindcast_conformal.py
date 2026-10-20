"""Hindcast metrics and conformal calibration (synthetic)."""

import numpy as np
import pytest

from caspian import conformal, hindcast


def test_flood_fill_respects_connectivity():
    z = np.array([[0, 0, 5, 0],
                  [0, 0, 5, 0],
                  [5, 5, 5, 0]], float)
    seed = np.zeros_like(z, bool)
    seed[0, 0] = True
    wet = hindcast.flood_fill(z, level=1.0, seed=seed)
    assert wet[:2, :2].all()
    assert not wet[:, 3].any()  # the low basin on the right is not connected


def test_flood_fill_diagonal_connectivity():
    z = np.array([[0, 5], [5, 0]], float)
    seed = np.array([[True, False], [False, False]])
    assert hindcast.flood_fill(z, 1.0, seed, connectivity=8)[1, 1]
    assert not hindcast.flood_fill(z, 1.0, seed, connectivity=4)[1, 1]


def test_extent_metrics_known_case():
    obs = np.zeros((10, 10), bool)
    obs[:, :5] = True
    pred = np.zeros((10, 10), bool)
    pred[:, :6] = True
    m = hindcast.extent_metrics(pred, obs, np.ones_like(obs), cell_area=2.0)
    assert m["area_bias"] == pytest.approx(20.0)
    assert m["area_bias_rel"] == pytest.approx(0.2)
    assert m["iou"] == pytest.approx(50 / 60)
    assert m["f1"] == pytest.approx(100 / 110)


def test_shoreline_displacement_sign_and_size():
    obs = np.zeros((20, 40), bool)
    obs[:, :10] = True
    pred = np.zeros_like(obs)
    pred[:, :13] = True  # predicted water 3 cells too far landward
    d = hindcast.shoreline_displacement(pred, obs, np.ones_like(obs), pixel_size=30.0)
    assert d["median"] == pytest.approx(90.0)
    assert d["signed_median"] == pytest.approx(90.0)
    d2 = hindcast.shoreline_displacement(obs, pred, np.ones_like(obs), pixel_size=30.0)
    assert d2["signed_median"] == pytest.approx(-90.0)


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 20)  # n = 19, alpha 0.1 -> k = ceil(20 * 0.9) = 18
    assert conformal.conformal_quantile(s, 0.1) == 18
    assert conformal.conformal_quantile([1, 2, 3], 0.1) == float("inf")


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(11)
    covs = []
    for _ in range(200):
        x = rng.uniform(1, 5, 600)
        y = rng.normal(0, x)  # heteroscedastic, scale known up to a constant
        cal, test = slice(0, 300), slice(300, 600)
        q = conformal.conformal_quantile(conformal.scores(y[cal], 0, x[cal]), 0.1)
        lo, hi = conformal.intervals(np.zeros(300), q, x[test])
        covs.append(conformal.coverage(y[test], lo, hi)["coverage"])
    assert np.mean(covs) == pytest.approx(0.90, abs=0.01)


def test_mondrian_groups():
    s = np.r_[np.full(50, 1.0), np.full(50, 3.0)]
    g = np.r_[np.zeros(50), np.ones(50)]
    q = conformal.mondrian_quantiles(s, g, 0.1)
    assert q[0.0] == 1.0 and q[1.0] == 3.0


def test_clopper_pearson_and_consistency():
    lo, hi = conformal.clopper_pearson(90, 100)
    assert lo < 0.90 < hi
    assert conformal.consistent_with_nominal(90, 100)
    assert not conformal.consistent_with_nominal(700, 1000)
    assert conformal.clopper_pearson(0, 10)[0] == 0.0 and conformal.clopper_pearson(10, 10)[1] == 1.0
