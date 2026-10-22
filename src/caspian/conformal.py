"""Split conformal prediction intervals (PREREG section 8).

Locally normalised split conformal (Lei et al. 2018): with calibration residuals
r_i = |y_i - ŷ_i| / s_i for a scale model s (interpolation distance, contour
density, slope), the interval for a new point is ŷ ± q̂ s, where q̂ is the
ceil((n + 1)(1 - α))-th smallest calibration score. Coverage ≥ 1 - α holds
under exchangeability of calibration and test points; spatial blocking is
used to make that assumption plausible (calibration blocks disjoint from
training and test blocks).

Mondrian calibration computes q̂ per group (sector) when the group has at least
``min_n`` = 50 usable calibration points, and otherwise uses the pooled q̂ over
all calibration points; which groups fell back is returned (PREREG section 8),
including evaluated groups with no calibration point (``all_groups``).

Inclusion rule, the same for calibration and evaluation: NaN means missing. A
point enters iff none of its reference height y, prediction ŷ and (if used)
scale s is NaN (``usable``). A point without a prediction (e.g. outside the TIN
hull, where the DEM is NaN) produces a NaN score and NaN interval bounds; it is
dropped from calibration and excluded from coverage, and the number excluded is
reported. Infinite scores and bounds are kept, which is conservative (an
infinite score raises q̂; an infinite interval covers).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def usable(y, yhat, scale=None):
    """Boolean mask of points that enter calibration and evaluation: none of y, ŷ
    and (if given) the scale is NaN. Equals ``~np.isnan(scores(y, yhat, scale))``
    for finite inputs."""
    m = ~np.isnan(np.asarray(y, float)) & ~np.isnan(np.asarray(yhat, float))
    if scale is not None:
        m &= ~np.isnan(np.asarray(scale, float))
    return m


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample split-conformal quantile over the non-NaN scores (NaN = point
    without a prediction). Infinite scores are kept. Returns inf if n is too small
    for the requested level (ceil((n + 1)(1 - α)) > n)."""
    s = np.asarray(scores, float)
    s = np.sort(s[~np.isnan(s)])
    n = s.size
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def scores(y, yhat, scale=None):
    """Nonconformity scores |y - ŷ| / s. NaN where y, ŷ or s is missing; a finite
    scale must be positive."""
    r = np.abs(np.asarray(y, float) - np.asarray(yhat, float))
    if scale is None:
        return r
    sc = np.asarray(scale, float)
    if np.any(sc[~np.isnan(sc)] <= 0):
        raise ValueError("scale must be positive")
    return r / sc


def intervals(yhat, q: float, scale=None):
    yhat = np.asarray(yhat, float)
    half = q * (np.ones_like(yhat) if scale is None else np.asarray(scale, float))
    return yhat - half, yhat + half


def mondrian_calibration(score_values, groups, alpha: float, min_n: int = 50, *,
                         all_groups=None) -> dict:
    """Mondrian (group-conditional) calibration with the pre-registered pooled
    fallback (PREREG section 8).

    A group with at least ``min_n`` usable (non-NaN) calibration scores gets its
    own q̂; every other group uses the pooled q̂ over all usable calibration
    scores. ``all_groups`` lists every group that will be evaluated (e.g. all
    sectors), so that a group with no calibration point at all is recorded too
    (n = 0, pooled). Returns {"alpha", "min_n", "pooled": {"q", "n"},
    "groups": {g: {"q", "n", "used_pooled"}}}."""
    if min_n < 1:
        raise ValueError("min_n must be at least 1")
    sv = np.asarray(score_values, float)
    g = np.asarray(groups)
    if sv.shape != g.shape:
        raise ValueError("score_values and groups must have the same shape")
    ok = ~np.isnan(sv)
    pooled_q = conformal_quantile(sv, alpha)
    out_groups = {}
    for k in np.unique(g):
        sel = g == k
        n_k = int(np.sum(sel & ok))
        if n_k >= min_n:
            out_groups[k] = {"q": conformal_quantile(sv[sel], alpha), "n": n_k, "used_pooled": False}
        else:
            out_groups[k] = {"q": pooled_q, "n": n_k, "used_pooled": True}
    for k in ([] if all_groups is None else all_groups):
        if k not in out_groups:
            out_groups[k] = {"q": pooled_q, "n": 0, "used_pooled": True}
    return {"alpha": alpha, "min_n": min_n, "pooled": {"q": pooled_q, "n": int(ok.sum())},
            "groups": out_groups}


def mondrian_quantiles(score_values, groups, alpha: float, min_n: int = 50) -> dict:
    """{group: q̂} with the pooled fallback for groups below ``min_n`` usable points.
    Use ``mondrian_calibration`` to get which groups fell back (to be reported)."""
    cal = mondrian_calibration(score_values, groups, alpha, min_n)
    return {k: v["q"] for k, v in cal["groups"].items()}


def mondrian_q(groups, calibration: dict):
    """Per-point q̂ for test points from a ``mondrian_calibration`` result.

    Every test group must appear in the calibration record, so that each fallback
    to the pooled q̂ is reported (PREREG section 8). A group with no calibration
    points is added with ``mondrian_calibration(..., all_groups=...)``; a group
    missing from the record raises KeyError instead of falling back silently."""
    g = np.asarray(groups)
    table = calibration["groups"]
    missing = sorted({k for k in np.unique(g) if k not in table}, key=str)
    if missing:
        raise KeyError(f"test groups {missing} are not in the calibration record; pass "
                       "every evaluated group as all_groups to mondrian_calibration")
    return np.array([table[k]["q"] for k in g.ravel()], float).reshape(g.shape)


def coverage(y, lo, hi, conf: float = 0.95) -> dict:
    """Empirical coverage with an exact (Clopper-Pearson) binomial interval.

    Points with a NaN y or a NaN bound (no prediction or no scale) are excluded
    from k and n, matching calibration, and counted in ``n_excluded``. Infinite
    bounds are kept (an interval from q̂ = inf covers)."""
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    keep = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    inside = (y[keep] >= lo[keep]) & (y[keep] <= hi[keep])
    k, n = int(inside.sum()), int(inside.size)
    lo_ci, hi_ci = clopper_pearson(k, n, conf)
    return {"k": k, "n": n, "coverage": k / n if n else float("nan"), "ci_low": lo_ci,
            "ci_high": hi_ci, "n_excluded": int((~keep).sum())}


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
