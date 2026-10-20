"""Level interpolation with Brownian-bridge uncertainty (synthetic series)."""

import numpy as np
import pandas as pd
import pytest

from caspian import levels


def _series():
    t = pd.date_range("2000-01-01", periods=5, freq="10D")
    return t, np.array([0.0, 0.1, 0.2, 0.3, 0.4]), np.full(5, 0.02)


def test_exact_epoch_returns_observation():
    t, h, s = _series()
    out = levels.interpolate(t, h, s, [t[2]], q=1e-4)
    assert out["level_m"].iloc[0] == pytest.approx(0.2)
    assert out["sigma_m"].iloc[0] == pytest.approx(0.02)
    assert out["gap_days"].iloc[0] == 0


def test_midpoint_variance_formula():
    t, h, s = _series()
    q = 1e-3
    out = levels.interpolate(t, h, s, [t[1] + pd.Timedelta(days=5)], q=q)
    expected_var = 0.25 * 0.02**2 + 0.25 * 0.02**2 + q * 10 * 0.25
    assert out["level_m"].iloc[0] == pytest.approx(0.15)
    assert out["sigma_m"].iloc[0] ** 2 == pytest.approx(expected_var)


def test_outside_and_long_gap_are_nan():
    t = pd.to_datetime(["2000-01-01", "2000-01-11", "2000-06-01"])
    out = levels.interpolate(t, [0, 0.1, 0.5], [0.01] * 3,
                             pd.to_datetime(["1999-12-01", "2000-03-01"]), q=1e-4, max_gap_days=60)
    assert out["in_gap"].all() and out["level_m"].isna().all()


def test_diffusion_recovers_random_walk_rate():
    rng = np.random.default_rng(7)
    q_true, dt, n, sig = 4e-4, 10.0, 4000, 0.01
    walk = np.cumsum(rng.normal(0, np.sqrt(q_true * dt), n))
    obs = walk + rng.normal(0, sig, n)
    t = pd.Timestamp("1990-01-01") + pd.to_timedelta(np.arange(n) * dt, unit="D")
    q_hat = levels.estimate_diffusion(t, obs, np.full(n, sig))
    assert q_hat == pytest.approx(q_true, rel=0.15)


def test_bins():
    k = levels.bin_index([-27.04, -27.0, -26.95], width=0.1, origin=-30.0)
    assert list(k) == [29, 30, 30]
    assert levels.bin_center(30, 0.1, -30.0) == pytest.approx(-26.95)


def test_empirical_sigma_recovers_white_noise():
    # Unbiased on average over replicates (one replicate's NMAD has ~5% sampling error).
    days = np.arange(0, 3650, 10)
    t = pd.Timestamp("2000-01-01") + pd.to_timedelta(days, unit="D")
    signal = 0.3 * np.sin(2 * np.pi * days / 365.25) - 0.0002 * days
    est = [levels.empirical_sigma(t, signal + np.random.default_rng(k).normal(0, 0.05, days.size))
           for k in range(20)]
    assert np.mean(est) == pytest.approx(0.05, rel=0.04)
