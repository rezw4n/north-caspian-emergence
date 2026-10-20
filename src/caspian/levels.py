"""A4: water level at each acquisition time, with uncertainty.

The level between two altimetry epochs is modelled as a Brownian bridge plus
measurement error. For a query time between epochs i and j (gap Δt, fraction
u = (t - t_i)/Δt):

    h(t)   = (1 - u) h_i + u h_j
    var(t) = (1 - u)^2 σ_i^2 + u^2 σ_j^2 + q Δt u (1 - u)

where q (m² per day) is the diffusion rate of the level process, estimated from
the series' own increments: E[(h_j - h_i)^2] = q Δt + σ_i^2 + σ_j^2 for
neighbouring epochs (after removing the mean trend over the increment). The
seasonal cycle is kept in h (it adds contours, brief section 8 A4); q absorbs
the sub-sampling variability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_EPOCH = pd.Timestamp("1970-01-01")


def _days(t) -> np.ndarray:
    """Days since 1970-01-01, independent of the datetime resolution (pandas 3
    infers s/ms/us/ns, so integer casts are not safe)."""
    return ((pd.to_datetime(pd.Series(t)) - _EPOCH) / pd.Timedelta(days=1)).to_numpy(float)


def estimate_diffusion(times, levels, sigmas, max_gap_days: float = 60.0) -> float:
    """Least-squares estimate of q from neighbour increments (m² per day), >= 0."""
    d = _days(times)
    h = np.asarray(levels, float)
    s = np.nan_to_num(np.asarray(sigmas, float), nan=0.0)
    order = np.argsort(d)
    d, h, s = d[order], h[order], s[order]
    dt = np.diff(d)
    dh = np.diff(h)
    keep = (dt > 0) & (dt <= max_gap_days)
    if keep.sum() < 5:
        raise ValueError("too few neighbour increments to estimate q")
    dt, dh = dt[keep], dh[keep]
    noise = (s[:-1] ** 2 + s[1:] ** 2)[keep]
    rate = np.sum(dh) / np.sum(dt)            # mean trend over the increments
    resid2 = (dh - rate * dt) ** 2 - noise
    q = float(np.sum(resid2 * dt) / np.sum(dt ** 2))
    return max(q, 0.0)


def interpolate(times, levels, sigmas, query_times, q: float | None = None,
                max_gap_days: float = 60.0) -> pd.DataFrame:
    """Level and 1σ uncertainty at ``query_times``. Queries outside the series or in
    gaps longer than ``max_gap_days`` are returned as NaN with ``in_gap`` True."""
    d = _days(times)
    h = np.asarray(levels, float)
    s = np.nan_to_num(np.asarray(sigmas, float), nan=0.0)
    order = np.argsort(d)
    d, h, s = d[order], h[order], s[order]
    if q is None:
        q = estimate_diffusion(times, levels, sigmas, max_gap_days)
    tq = _days(query_times)
    j = np.searchsorted(d, tq, side="left")
    exact = (j < d.size) & (d[np.clip(j, 0, d.size - 1)] == tq)
    i = np.clip(j - 1, 0, d.size - 1)
    j = np.clip(j, 0, d.size - 1)
    gap = d[j] - d[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(gap > 0, (tq - d[i]) / gap, 0.0)
    level = (1 - u) * h[i] + u * h[j]
    var = (1 - u) ** 2 * s[i] ** 2 + u ** 2 * s[j] ** 2 + q * gap * u * (1 - u)
    level = np.where(exact, h[j], level)
    var = np.where(exact, s[j] ** 2, var)
    outside = (tq < d[0]) | (tq > d[-1])
    in_gap = outside | ((gap > max_gap_days) & ~exact)
    level = np.where(in_gap, np.nan, level)
    sigma = np.where(in_gap, np.nan, np.sqrt(var))
    return pd.DataFrame({"level_m": level, "sigma_m": sigma, "gap_days": np.where(exact, 0.0, gap),
                         "in_gap": in_gap})


def bin_index(levels, width: float, origin: float = 0.0) -> np.ndarray:
    """Integer bin for each level; bin k covers [origin + k w, origin + (k + 1) w)."""
    return np.floor((np.asarray(levels, float) - origin) / width).astype(int)


def bin_center(k, width: float, origin: float = 0.0):
    return origin + (np.asarray(k) + 0.5) * width


def empirical_sigma(times, levels, max_gap_days: float = 30.0) -> float:
    """Robust measurement-noise estimate of a level series (D-018).

    Each record is predicted by linear interpolation from its two neighbours; with
    w = (t_i - t_{i-1}) / (t_{i+1} - t_{i-1}) and i.i.d. noise sigma, the residual
    has variance sigma^2 (1 + w^2 + (1 - w)^2). sigma is the NMAD of the scaled
    residuals over triplets whose gaps are both <= ``max_gap_days``. Real
    variability on time scales shorter than the gaps (e.g. wind setup) is included,
    so the estimate is conservative."""
    d = _days(times)
    h = np.asarray(levels, float)
    order = np.argsort(d)
    d, h = d[order], h[order]
    g1, g2 = d[1:-1] - d[:-2], d[2:] - d[1:-1]
    keep = (g1 > 0) & (g2 > 0) & (g1 <= max_gap_days) & (g2 <= max_gap_days)
    w = g1 / (g1 + g2)
    pred = h[:-2] + (h[2:] - h[:-2]) * w
    e = (h[1:-1] - pred) / np.sqrt(1 + w ** 2 + (1 - w) ** 2)
    e = e[keep]
    if e.size < 10:
        raise ValueError("too few neighbour triplets")
    return float(1.4826 * np.median(np.abs(e - np.median(e))))
