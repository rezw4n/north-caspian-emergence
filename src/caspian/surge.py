"""A5: wind-setup screening.

The water level at a North Caspian waterline differs from the basin-representative
altimetry level by wind setup and the inverse-barometer effect. For each sector s
and acquisition time t we build lagged forcing features from hourly ERA5
sector-mean fields:

    su_k = mean over [t - b_k, t - a_k] h of |U| u      (quadratic wind-stress proxy, m2 s-2)
    sv_k = mean over the same window of |U| v
    dp   = mean sea-level pressure anomaly over [t - 6, t] h, relative to the
           sector's long-term mean (hPa); inverse barometer is about -1 cm per hPa

and fit setup_s(t) = c_s + sum_k (a_sk su_k + b_sk sv_k) + d_s dp by least squares
against observed anomalies (SWOT sector-mean water-surface elevation minus the
basin level). Coefficients are fitted on training data only (PREREG section 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS_H = ((0, 6), (6, 12), (12, 24), (24, 48))
BANDS = {"u_component_of_wind_10m": "u", "v_component_of_wind_10m": "v", "mean_sea_level_pressure": "msl"}


def load_era5(paths) -> pd.DataFrame:
    """Hourly sector-mean ERA5 table: index = UTC hour, columns = MultiIndex (sector, var)."""
    frames = []
    for p in sorted(Path(x) for x in paths):
        d = pd.read_csv(p)
        d["time"] = pd.to_datetime(d["t"].round(-5).astype("int64"), unit="ms").dt.round("h")
        d = d.rename(columns=BANDS)[["time", "sector", "u", "v", "msl"]]
        frames.append(d)
    d = pd.concat(frames).drop_duplicates(["time", "sector"]).sort_values("time")
    wide = d.pivot(index="time", columns="sector", values=["u", "v", "msl"]).swaplevel(axis=1).sort_index(axis=1)
    return wide


def forcing_features(era5: pd.DataFrame, sector: str, times, windows=WINDOWS_H) -> pd.DataFrame:
    """Lagged wind-stress and pressure features for one sector at arbitrary times."""
    s = era5[sector]
    speed = np.hypot(s["u"], s["v"])
    su, sv = speed * s["u"], speed * s["v"]
    msl_anom = (s["msl"] - s["msl"].mean()) / 100.0  # Pa -> hPa
    cum = pd.DataFrame({"su": su, "sv": sv, "dp": msl_anom}).cumsum()
    hours = cum.index
    t = pd.to_datetime(pd.Series(times)).dt.floor("h").to_numpy()
    out = {}

    def window_mean(col, a, b):
        # mean over hours (t-b, t-a]; cumulative sums give O(1) per query
        hi = np.searchsorted(hours.values, t - np.timedelta64(a, "h"), side="right") - 1
        lo = np.searchsorted(hours.values, t - np.timedelta64(b, "h"), side="right") - 1
        valid = (lo >= 0) & (hi > lo)
        c = cum[col].to_numpy()
        res = np.full(t.shape, np.nan)
        res[valid] = (c[hi[valid]] - c[lo[valid]]) / (hi[valid] - lo[valid])
        return res

    for a, b in windows:
        out[f"su_{a}_{b}"] = window_mean("su", a, b)
        out[f"sv_{a}_{b}"] = window_mean("sv", a, b)
    out["dp_0_6"] = window_mean("dp", 0, 6)
    return pd.DataFrame(out, index=pd.RangeIndex(len(t)))


def fit_setup(features: pd.DataFrame, anomaly, ridge: float = 0.0) -> dict:
    """Least-squares (optionally ridge) fit of anomaly on features plus intercept.
    Returns coefficients, residual SD and n."""
    X = features.to_numpy(float)
    y = np.asarray(anomaly, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    Xc = np.column_stack([np.ones(len(X)), X])
    reg = ridge * np.eye(Xc.shape[1])
    reg[0, 0] = 0.0
    beta = np.linalg.solve(Xc.T @ Xc + reg, Xc.T @ y)
    resid = y - Xc @ beta
    dof = max(len(y) - Xc.shape[1], 1)
    return {"intercept": float(beta[0]), "coef": dict(zip(features.columns, map(float, beta[1:]))),
            "resid_sd": float(np.sqrt(resid @ resid / dof)), "n": int(len(y))}


def predict_setup(model: dict, features: pd.DataFrame) -> np.ndarray:
    pred = np.full(len(features), model["intercept"])
    for k, c in model["coef"].items():
        pred = pred + c * features[k].to_numpy(float)
    return pred
