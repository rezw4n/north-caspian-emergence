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


# --- joint noise and diffusion ---------------------------------------------------------

SIG_TRUE, Q_TRUE = 0.05, 3e-4


def _bm_series(seed, n=2000, sig=SIG_TRUE, q=Q_TRUE, drop=0.1):
    """Brownian motion (rate q, m^2/d) plus white noise sig, on a ~10-day grid with
    0.5-day jitter and ~10% of records missing (irregular gaps of 10-30 days)."""
    rng = np.random.default_rng(seed)
    days = np.arange(n) * 9.9156 + rng.uniform(-0.5, 0.5, n)
    days = days[rng.random(n) > drop]
    inc = rng.normal(0, np.sqrt(q * np.diff(days, prepend=days[0])))
    obs = np.cumsum(inc) + rng.normal(0, sig, days.size)
    t = pd.Timestamp("1992-09-28") + pd.to_timedelta(days, unit="D")
    return t, obs


def test_joint_noise_diffusion_recovers_sigma_and_q():
    est = np.array([levels.estimate_noise_and_diffusion(*_bm_series(k))[:2] for k in range(10)])
    assert est[:, 0].mean() == pytest.approx(SIG_TRUE, rel=0.06)
    assert est[:, 1].mean() == pytest.approx(Q_TRUE, rel=0.20)


def test_two_step_floor_then_q_is_biased():
    # The superseded pipeline (empirical_sigma, then estimate_diffusion with that floor):
    # the floor absorbs bridge variance, so sigma comes out high and q low. This is the
    # defect the joint estimate removes; it fails the tolerances of the test above.
    fl, qs = [], []
    for k in range(10):
        t, obs = _bm_series(k)
        f = levels.empirical_sigma(t, obs)
        fl.append(f)
        qs.append(levels.estimate_diffusion(t, obs, np.full(obs.size, f)))
    assert np.mean(fl) > 1.10 * SIG_TRUE
    assert np.mean(qs) < 0.80 * Q_TRUE


def test_joint_noise_with_known_q_removes_bridge_term():
    est = [levels.estimate_noise_and_diffusion(*_bm_series(k), q=Q_TRUE).sigma for k in range(10)]
    assert np.mean(est) == pytest.approx(SIG_TRUE, rel=0.04)


def test_joint_estimate_without_diffusion_returns_small_q():
    est = np.array([levels.estimate_noise_and_diffusion(*_bm_series(k, q=0.0))[:2] for k in range(6)])
    assert est[:, 0].mean() == pytest.approx(SIG_TRUE, rel=0.05)
    assert est[:, 1].mean() < 0.15 * Q_TRUE


def test_joint_bootstrap_se_matches_replicate_spread():
    runs = [levels.estimate_noise_and_diffusion(*_bm_series(k, n=900), n_boot=100, seed=k) for k in range(16)]
    q_hat = np.array([r.q for r in runs])
    q_se = np.array([r.q_se for r in runs])
    s_hat = np.array([r.sigma for r in runs])
    s_se = np.array([r.sigma_se for r in runs])
    assert 0.5 < q_se.mean() / q_hat.std(ddof=1) < 2.0
    assert 0.5 < s_se.mean() / s_hat.std(ddof=1) < 2.0


def test_neighbour_zscores_are_standard_under_the_model():
    t, obs = _bm_series(3, n=6000)
    z = levels.neighbour_zscores(t, obs, SIG_TRUE, Q_TRUE)
    z = z[np.isfinite(z)]
    assert 1.4826 * np.median(np.abs(z)) == pytest.approx(1.0, abs=0.05)
    assert np.mean(np.abs(z) > 2.5) == pytest.approx(0.0124, abs=0.006)


# --- fractional gaps ------------------------------------------------------------------

def test_gap_days_keeps_fractional_days():
    t = pd.to_datetime(["2024-01-17 01:43:48", "2024-05-24 23:24:38", "2024-06-03 21:23:00"])
    g = levels.gap_days(t)
    assert g[0] == pytest.approx(128.9034, abs=1e-4)      # .dt.days would give 128
    assert g[1] == pytest.approx(9.9155, abs=1e-3)       # .dt.days would give 9


# --- robust extremes ------------------------------------------------------------------

def test_smoothed_extremes_ignore_a_single_spike():
    days = np.arange(300, 800, 10.0)
    level = 0.2 * np.cos(2 * np.pi * (days - 547.5) / 365.25)   # one peak, on day 547.5, +0.2 m
    k_spike = int(np.argmin(np.abs(days - 480)))                 # one record 2.7 sigma (0.063 m) high,
    level[k_spike] += 0.17                                       # which becomes the raw maximum
    trough = int(np.argmin(level))
    level[trough + 1] -= 0.15                                    # and one low record after the trough
    t = pd.Timestamp("1994-01-01") + pd.to_timedelta(days, unit="D")
    raw_max_day = days[int(np.argmax(level))]
    ex = levels.smoothed_extremes(t, level, k=3)
    assert raw_max_day == days[k_spike]
    max_day = (ex["time_of_max"] - pd.Timestamp("1994-01-01")) / pd.Timedelta(days=1)
    # A median's top is flat over neighbouring records, so the date is good to about two
    # sampling intervals; the raw maximum is 67 days early.
    assert abs(max_day - 547.5) <= 20
    assert ex["max_m"] == pytest.approx(0.2, abs=0.01)
    assert ex["min_m"] == pytest.approx(-0.2, abs=0.01)
    assert ex["range_m"] == pytest.approx(0.4, abs=0.02)


def test_running_median_requires_odd_window():
    with pytest.raises(ValueError):
        levels.running_median([1.0, 2.0, 3.0], k=2)
    assert list(levels.running_median([0.0, 5.0, 0.0, 0.0], k=3)) == [2.5, 0.0, 0.0, 0.0]


# --- gap-fill offset ------------------------------------------------------------------

def test_pair_series_interpolates_bracketing_records():
    tr = pd.to_datetime(["2020-01-05", "2020-01-20", "2020-03-01"])
    to = pd.to_datetime(["2020-01-01", "2020-01-09", "2020-01-30", "2020-02-02"])
    p = levels.pair_series(tr, [1.0, 2.0, 3.0], to, [0.0, 0.8, 1.0, 1.2], max_half_gap_days=10)
    # 01-05: between 01-01 (0.0) and 01-09 (0.8) -> 0.4; 01-20: 01-09..01-30 both within
    # 11 and 10 days -> rejected (11 > 10); 03-01: after the last record -> no pair.
    assert len(p) == 1
    assert p["other"].iloc[0] == pytest.approx(0.4)
    assert p["diff"].iloc[0] == pytest.approx(0.6)


def _drifting_pairs(seed=0, tau=0.0):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(0, 34 * 365.25, 600))
    t = pd.Timestamp("1992-09-28") + pd.to_timedelta(days, unit="D")
    true = 0.14 - 0.06 * days / days.max()                 # 0.14 -> 0.08 m
    yr_eff = rng.normal(0, tau, 40)[t.year.to_numpy() - 1992] if tau else 0.0
    return t, true + yr_eff + rng.normal(0, 0.07, days.size), true


def test_moving_median_offset_tracks_a_drifting_offset():
    # Over 10 replicates, the moving median is unbiased at the end of the record, where
    # the gap-fill is used; the pooled median of all pairs is ~0.025 m too high there.
    bias_mov, bias_pool, se_scaled = [], [], []
    for k in range(10):
        t, diff, true = _drifting_pairs(k)
        late = t >= pd.Timestamp("2023-01-01")
        mo = levels.moving_median_offset(t, diff, t[late], half_window_days=730.5)
        bias_mov.append(np.mean(mo["offset_m"].to_numpy() - true[late]))
        bias_pool.append(np.mean(np.median(diff) - true[late]))
        se_scaled.append(np.median(mo["se_m"] * np.sqrt(mo["n_pairs"])))
    assert abs(np.mean(bias_mov)) < 0.006
    assert np.mean(bias_pool) > 0.02
    assert np.mean(se_scaled) == pytest.approx(np.sqrt(np.pi / 2) * 0.07, rel=0.15)


def test_moving_median_offset_falls_back_to_nearest_pairs():
    t, diff, _ = _drifting_pairs()
    mo = levels.moving_median_offset(t, diff, [pd.Timestamp("1960-01-01")], half_window_days=730.5,
                                     min_pairs=25)
    assert mo["n_pairs"].iloc[0] == 25
    assert mo["offset_m"].iloc[0] == pytest.approx(np.median(np.asarray(diff)[:25]))


def test_offset_excess_sd_detects_year_to_year_variability():
    taus_none = [levels.offset_excess_sd(*_drifting_pairs(k)[:2], 730.5)["tau_m"] for k in range(5)]
    taus_03 = [levels.offset_excess_sd(*_drifting_pairs(k, tau=0.03)[:2], 730.5)["tau_m"] for k in range(5)]
    assert np.mean(taus_none) < 0.012
    assert np.mean(taus_03) == pytest.approx(0.03, abs=0.012)


def test_interpolate_shifted_offset_error_is_common_not_shrunk():
    t = pd.to_datetime(["2024-01-01", "2024-01-11"])
    mid = [pd.Timestamp("2024-01-06")]
    out = levels.interpolate_shifted(t, [0.0, 0.2], [0.0, 0.0], mid, offset=0.08, offset_sigma=0.03, q=0.0)
    assert out["level_m"].iloc[0] == pytest.approx(0.18)
    assert out["sigma_m"].iloc[0] == pytest.approx(0.03)      # a common error keeps its full size
    # Folding the same error into each record (the superseded approach) shrinks it by sqrt(2):
    per_rec = levels.interpolate(t, [0.08, 0.28], [0.03, 0.03], mid, q=0.0)
    assert per_rec["sigma_m"].iloc[0] == pytest.approx(0.03 / np.sqrt(2))


# --- era-local offset error and exported sigma parts ------------------------------------

def _era_pairs(seed, tau_late=0.04, from_year=2016, n=1200):
    """Drifting offset (0.14 -> 0.08 m) with year-to-year effects of sd ``tau_late`` only
    from ``from_year`` on (stable before), plus 0.07 m pair noise."""
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(0, 34 * 365.25, n))
    t = pd.Timestamp("1992-09-28") + pd.to_timedelta(days, unit="D")
    yr = t.year.to_numpy()
    yr_eff = np.where(yr >= from_year, rng.normal(0, tau_late, 40)[yr - 1992], 0.0)
    return t, 0.14 - 0.06 * days / days.max() + yr_eff + rng.normal(0, 0.07, days.size)


def test_local_excess_sd_follows_the_variable_era():
    # The record-wide tau averages the stable early years with the variable late ones and
    # comes out at about half the late-era value; the window-local tau recovers it.
    late = pd.to_datetime(["2022-06-01", "2024-03-01"])
    early = pd.to_datetime(["1998-06-01", "2005-06-01"])
    loc_late, loc_early, pooled = [], [], []
    for k in range(20):
        ex = levels.offset_excess_sd(*_era_pairs(k), 730.5)
        pooled.append(ex["tau_m"])
        loc_late.append(levels.local_excess_sd(ex["by_year"], late, 730.5)["tau_m"].mean())
        loc_early.append(levels.local_excess_sd(ex["by_year"], early, 730.5)["tau_m"].mean())
    assert np.mean(loc_late) == pytest.approx(0.04, abs=0.008)
    assert np.mean(loc_early) < 0.01
    assert np.mean(pooled) < 0.6 * np.mean(loc_late)


def test_local_excess_sd_uses_nearest_years_at_the_record_end():
    rows = [{"time_mid": f"{y}-07-01", "term_m2": (0.0 if y < 2020 else 0.0016)} for y in range(2000, 2027)]
    out = levels.local_excess_sd(rows, pd.to_datetime(["2026-09-01", "2010-07-01"]), 730.5, min_years=5)
    # 2026-09: only the 2025 and 2026 year-mids lie within +/-2 y (2024-07 is 2.2 y away),
    # so the 5 nearest years (2022-2026) are used
    assert out["n_years"].iloc[0] == 5
    assert out["tau_m"].iloc[0] == pytest.approx(0.04)
    assert out["n_years"].iloc[1] == 5 and out["tau_m"].iloc[1] == 0.0
    with pytest.raises(ValueError):
        levels.local_excess_sd(rows[:3], pd.to_datetime(["2001-01-01"]), 730.5, min_years=5)


def test_interpolate_shifted_exports_random_and_common_parts():
    t = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-06-01"])
    q_t = pd.to_datetime(["2024-01-06", "2024-03-01"])          # mid-gap, and inside a 142-day gap
    out = levels.interpolate_shifted(t, [0.0, 0.2, 0.4], [0.02, 0.02, 0.02], q_t, offset=[0.08, 0.08],
                                     offset_sigma=[0.04, 0.04], q=1e-4, max_gap_days=60)
    rand = np.sqrt(0.25 * 0.02 ** 2 * 2 + 1e-4 * 10 * 0.25)
    assert out["sigma_random_m"].iloc[0] == pytest.approx(rand)
    assert out["sigma_offset_m"].iloc[0] == pytest.approx(0.04)
    assert out["sigma_m"].iloc[0] ** 2 == pytest.approx(rand ** 2 + 0.04 ** 2)
    assert out["in_gap"].iloc[1] and np.isnan(out["sigma_offset_m"].iloc[1])
