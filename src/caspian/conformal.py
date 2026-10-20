"""Split conformal prediction intervals (brief section 9).

Locally normalised split conformal (Lei et al. 2018): with calibration residuals
r_i = |y_i - ŷ_i| / s_i for a scale model s (interpolation distance, contour
density, slope), the interval for a new point is ŷ ± q̂ s, where q̂ is the
ceil((n + 1)(1 - α))-th smallest calibration score. Coverage ≥ 1 - α holds
under exchangeability of calibration and test points; spatial blocking is
used to make that assumption plausible (calibration blocks disjoint from
training and test blocks). Mondrian calibration computes q̂ separately per
group (e.g. sector) for group-conditional coverage.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample split-conformal quantile. Returns inf if n is too small for
    the requested level (ceil((n + 1)(1 - α)) > n)."""
    s = np.sort(np.asarray(scores, float)[np.isfinite(scores)])
    n = s.size
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def scores(y, yhat, scale=None):
    r = np.abs(np.asarray(y, float) - np.asarray(yhat, float))
    if scale is None:
        return r
    sc = np.asarray(scale, float)
    if np.any(sc <= 0):
        raise ValueError("scale must be positive")
    return r / sc


def intervals(yhat, q: float, scale=None):
    yhat = np.asarray(yhat, float)
    half = q * (np.ones_like(yhat) if scale is None else np.asarray(scale, float))
    return yhat - half, yhat + half


def mondrian_quantiles(score_values, groups, alpha: float) -> dict:
    sv = np.asarray(score_values, float)
    g = np.asarray(groups)
    return {k: conformal_quantile(sv[g == k], alpha) for k in np.unique(g)}


def coverage(y, lo, hi, conf: float = 0.95) -> dict:
    """Empirical coverage with an exact (Clopper-Pearson) binomial interval."""
    y = np.asarray(y, float)
    inside = (y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))
    k, n = int(inside.sum()), int(inside.size)
    lo_ci, hi_ci = clopper_pearson(k, n, conf)
    return {"k": k, "n": n, "coverage": k / n if n else float("nan"), "ci_low": lo_ci, "ci_high": hi_ci}


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    a = 1 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def consistent_with_nominal(k: int, n: int, nominal: float = 0.90, conf: float = 0.95) -> bool:
    lo, hi = clopper_pearson(k, n, conf)
    return lo <= nominal <= hi
