"""Wind-setup features and fit (synthetic forcing with known coefficients)."""

import numpy as np
import pandas as pd
import pytest

from caspian import surge


def _era5(hours=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=hours, freq="h")
    u = 6 * np.sin(np.arange(hours) / 17.0) + rng.normal(0, 1, hours)
    v = 4 * np.cos(np.arange(hours) / 29.0) + rng.normal(0, 1, hours)
    msl = 101300 + 800 * np.sin(np.arange(hours) / 90.0)
    cols = pd.MultiIndex.from_tuples([("northeast", "msl"), ("northeast", "u"), ("northeast", "v")])
    return pd.DataFrame(np.column_stack([msl, u, v]), index=idx, columns=cols)


def test_window_mean_matches_direct_computation():
    era = _era5()
    t = pd.to_datetime(["2020-01-10 12:30", "2020-02-01 00:00"])
    f = surge.forcing_features(era, "northeast", t)
    s = era["northeast"]
    su = np.hypot(s["u"], s["v"]) * s["u"]
    t0 = pd.Timestamp("2020-01-10 12:00")
    direct = su[(su.index > t0 - pd.Timedelta(hours=6)) & (su.index <= t0)].mean()
    assert f.loc[0, "su_0_6"] == pytest.approx(direct)
    direct_lag = su[(su.index > t0 - pd.Timedelta(hours=48)) & (su.index <= t0 - pd.Timedelta(hours=24))].mean()
    assert f.loc[0, "su_24_48"] == pytest.approx(direct_lag)


def test_features_nan_before_record_start():
    era = _era5()
    f = surge.forcing_features(era, "northeast", pd.to_datetime(["2020-01-01 03:00"]))
    assert np.isnan(f.loc[0, "su_24_48"])


def test_fit_recovers_known_coefficients():
    era = _era5(hours=24 * 400, seed=3)
    t = pd.date_range("2020-01-05", periods=600, freq="13h")
    f = surge.forcing_features(era, "northeast", t)
    rng = np.random.default_rng(1)
    y = 0.02 + 0.004 * f["su_0_6"] - 0.002 * f["sv_6_12"] - 0.01 * f["dp_0_6"] + rng.normal(0, 0.01, len(t))
    m = surge.fit_setup(f, y)
    assert m["coef"]["su_0_6"] == pytest.approx(0.004, abs=5e-4)
    assert m["coef"]["sv_6_12"] == pytest.approx(-0.002, abs=5e-4)
    assert m["coef"]["dp_0_6"] == pytest.approx(-0.01, abs=2e-3)
    assert m["resid_sd"] == pytest.approx(0.01, rel=0.15)
    pred = surge.predict_setup(m, f)
    assert np.corrcoef(pred, y)[0, 1] > 0.9
