"""Revision statistics for the H1 audit (PREREG Amendment 18 items 1, 2, 4, 5 and 8).

Pure functions on the frozen hindcast tables (and, for ``filter_cells``, on the
waterline-cell records of caspian.hindcast_tiled). Nothing here reads files.

- ``block_indices`` and ``mbb_mispredict``: the moving-block bootstrap over levels
  (blocks of consecutive evaluable levels, one set of uniform draws shared by
  every cell of a resample).
- ``ratio_summary``: median and quartiles of D/Q, infinite ratios kept and counted.
- ``exact_group_test``: the Eq. S12 statistic (median of the group minus median of the
  rest) over every labelling of k of n units (the sector-level deltaic test).
- ``block_permutation_null``: the registered block permutation test with its null
  statistics returned (same random sequence as caspian.h1.permutation_test).
- ``holm``: Holm's step-down adjusted p values.
- ``theil_sen``: Theil-Sen slope with its interval (scipy).
- ``quantile_summary``: minimum, quartiles and maximum.
- ``majority_votes``: one vote per group, the majority of a boolean column, ties dropped.
- ``mask_labels``: the waterline cells of hindcast_tiled.displacement_cells relabelled
  outside a mask on their grid positions (e.g. test blocks), so that the per-region
  summaries count only the kept cells.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


# --- item 2: moving-block bootstrap over levels ----------------------------------------------

def block_indices(n: int, u, block: int = 3) -> np.ndarray:
    """Indices 0..n-1 of one moving-block resample of a sequence of length ``n``.

    ``u`` holds uniform numbers in [0, 1); the k-th block starts at
    floor(u[k] * (n - block + 1)) and covers ``block`` consecutive positions. The first
    ceil(n / block) blocks are joined and cut to n. With n < block the whole sequence is
    one block (start 0). ``u`` must hold at least ceil(n / block) numbers."""
    if n <= 0:
        return np.empty(0, int)
    b = min(block, n)
    k = math.ceil(n / b)
    u = np.asarray(u, float)
    if u.size < k:
        raise ValueError(f"need {k} uniform numbers, got {u.size}")
    starts = np.floor(u[:k] * (n - b + 1)).astype(int)
    starts = np.clip(starts, 0, n - b)
    idx = (starts[:, None] + np.arange(b)[None, :]).ravel()
    return idx[:n]


def mbb_mispredict(cells: dict, n_boot: int, seed: int, block: int = 3) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Moving-block bootstrap of the majority rule.

    ``cells``: {key: boolean array of the "exceeds" flags of the cell's evaluable
    levels, in level order}. In each resample one vector of uniform numbers serves every
    cell (``block_indices``), so each cell's marginal is the moving-block bootstrap of its
    own sequence and all cells draw blocks at the same relative position.

    Returns (summary, flags, keys): summary has one row per key with n_levels,
    n_exceeding, the observed verdict and p_mispredict (share of resamples in which the
    exceeding levels are more than half); flags is the boolean [n_boot, n_cells] matrix
    of resampled verdicts (columns in the order of ``keys``)."""
    keys = list(cells)
    seqs = [np.asarray(cells[k], bool) for k in keys]
    n_max = max((s.size for s in seqs), default=0)
    k_max = max(1, math.ceil(n_max / block))
    rng = np.random.default_rng(seed)
    flags = np.zeros((n_boot, len(keys)), bool)
    for b in range(n_boot):
        u = rng.random(k_max)
        for j, s in enumerate(seqs):
            if s.size == 0:
                continue
            r = s[block_indices(s.size, u, block)]
            flags[b, j] = r.sum() > s.size / 2
    rows = []
    for j, (k, s) in enumerate(zip(keys, seqs)):
        rows.append({"key": k, "n_levels": int(s.size), "n_exceeding": int(s.sum()),
                     "observed_mispredicts": bool(s.sum() > s.size / 2) if s.size else None,
                     "p_mispredict": float(flags[:, j].mean()) if s.size else float("nan")})
    return pd.DataFrame(rows), flags, keys


def count_distribution(flags: np.ndarray, max_count: int) -> np.ndarray:
    """Share of resamples with 0..max_count mispredicted cells (``flags`` [B, cells])."""
    c = flags.sum(axis=1)
    return np.bincount(c, minlength=max_count + 1)[: max_count + 1] / flags.shape[0]


# --- item 2: D/Q ------------------------------------------------------------------------------

def ratio_summary(d, q) -> dict:
    """Median and quartiles (numpy linear percentiles) of d / q over levels; a level with
    d = +inf (no predicted waterline) gives +inf and is counted, not dropped; a level
    with q = +inf (a floor distance with no counterpart waterline) gives 0 for finite d.
    Levels with q <= 0 or a NaN, or with both infinite, are left out."""
    d, q = np.asarray(d, float), np.asarray(q, float)
    ok = ~np.isnan(q) & (q > 0) & ~np.isnan(d) & ~(np.isinf(d) & np.isinf(q))
    r = d[ok] / q[ok]
    if r.size == 0:
        return {"n": 0, "n_inf": 0, "median": float("nan"), "p25": float("nan"), "p75": float("nan"),
                "share_above_1": float("nan")}
    return {"n": int(r.size), "n_inf": int(np.isinf(r).sum()), "median": float(np.percentile(r, 50)),
            "p25": float(np.percentile(r, 25)), "p75": float(np.percentile(r, 75)),
            "share_above_1": float((r > 1).mean())}


# --- item 5: sector-level and block permutation tests -----------------------------------------

def group_statistic(values, in_group) -> float:
    """Eq. S12: median of the group minus median of the rest."""
    v, g = np.asarray(values, float), np.asarray(in_group, bool)
    return float(np.median(v[g]) - np.median(v[~g]))


def exact_group_test(values: dict, group, tol: float = 1e-9) -> dict:
    """Exact one-sided permutation test of ``group_statistic`` over every labelling of
    len(group) of the units in ``values`` ({unit: value}). p = share of labellings with a
    statistic >= the observed one - tol (the observed labelling included)."""
    units = list(values)
    v = np.array([values[u] for u in units], float)
    if np.isnan(v).any():
        raise ValueError("a unit has no value")
    group = set(group)
    if not group <= set(units):
        raise ValueError(f"group units not in values: {sorted(group - set(units))}")
    k = len(group)
    obs = group_statistic(v, [u in group for u in units])
    null, labels = [], []
    for comb in itertools.combinations(range(len(units)), k):
        g = np.zeros(len(units), bool)
        g[list(comb)] = True
        null.append(group_statistic(v, g))
        labels.append("+".join(units[i] for i in comb))
    null = np.array(null)
    return {"statistic": obs, "n_labellings": int(null.size), "n_at_least": int((null >= obs - tol).sum()),
            "p_value": float((null >= obs - tol).mean()), "min_p": 1.0 / null.size,
            "null": null, "labellings": labels, "group_size": k, "rest_size": len(units) - k}


def exact_rank_sum_test(values: dict, group) -> dict:
    """Exact one-sided rank-sum test over every labelling of len(group) of the units in
    ``values`` (a sensitivity): the statistic is the sum of the group's ranks (rank 1
    = smallest value, average ranks for ties); p = share of labellings with a rank sum at
    least the observed one."""
    from scipy.stats import rankdata
    units = list(values)
    v = np.array([values[u] for u in units], float)
    rk = rankdata(v)
    group = set(group)
    k = len(group)
    obs = float(sum(rk[i] for i, u in enumerate(units) if u in group))
    null = np.array([rk[list(c)].sum() for c in itertools.combinations(range(len(units)), k)])
    return {"statistic": obs, "expected": float(k * (len(units) + 1) / 2), "n_labellings": int(null.size),
            "p_value": float((null >= obs - 1e-9).mean()), "null": null}


def block_permutation_null(values, in_group, n_perm: int, seed: int) -> dict:
    """caspian.h1.permutation_test with its null statistics returned: the same random
    sequence (``default_rng(seed)``, one permutation per draw), so the p value equals the
    registered one for the same inputs."""
    v, in_g = np.asarray(values, float), np.asarray(in_group, bool)
    keep = ~np.isnan(v)
    v, in_g = v[keep], in_g[keep]
    t = float(np.median(v[in_g]) - np.median(v[~in_g]))
    rng = np.random.default_rng(seed)
    k = int(in_g.sum())
    perm = np.empty(n_perm)
    for i in range(n_perm):
        g = np.zeros(v.size, bool)
        g[rng.permutation(v.size)[:k]] = True
        perm[i] = np.median(v[g]) - np.median(v[~g])
    p = (1 + int(np.sum(perm >= t))) / (1 + n_perm)
    return {"statistic": t, "p_value": p, "null": perm, "n_group": k, "n_rest": int(v.size - k)}


def holm(pvals) -> np.ndarray:
    """Holm's step-down adjusted p values (monotone, capped at 1), in input order; NaN
    stays NaN and does not count towards the number of tests."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = ok.size
    order = ok[np.argsort(p[ok], kind="stable")]
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[j]))
        out[j] = running
    return out


# --- item 4 and generic summaries -------------------------------------------------------------

def theil_sen(x, y) -> dict:
    """Theil-Sen slope of y on x with its 95% interval (scipy.stats.theilslopes)."""
    from scipy.stats import theilslopes
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "slope": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "intercept": float("nan")}
    r = theilslopes(y[ok], x[ok], alpha=0.95)
    return {"n": int(ok.sum()), "slope": float(r.slope), "lo": float(r.low_slope), "hi": float(r.high_slope),
            "intercept": float(r.intercept)}


def quantile_summary(x) -> dict:
    """n, minimum, quartiles (numpy linear percentiles), maximum and IQR of finite values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, **{k: float("nan") for k in ("min", "p25", "median", "p75", "max", "iqr")}}
    p25, p50, p75 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "min": float(x.min()), "p25": float(p25), "median": float(p50),
            "p75": float(p75), "max": float(x.max()), "iqr": float(p75 - p25)}


def majority_votes(df: pd.DataFrame, by, col: str) -> pd.Series:
    """One vote per group of ``by``: True if most of the group's ``col`` values are True,
    False if most are False; ties dropped."""
    g = df.groupby(by)[col].agg(["sum", "size"])
    t, n = g["sum"].astype(int), g["size"].astype(int)
    vote = pd.Series(pd.NA, index=g.index, dtype="boolean")
    vote[2 * t > n] = True
    vote[2 * t < n] = False
    return vote.dropna().astype(bool)


# --- waterline cells kept by a mask -----------------------------------------------------------

def mask_labels(cells: dict, keep, off: int = 0) -> dict:
    """The waterline-cell record of hindcast_tiled.displacement_cells (or
    noise_floor_cells) with the region label set to ``off`` wherever ``keep(rows, cols)``
    is False. hindcast_tiled.summarise then counts, per region id, only the kept cells,
    while each distance is still the one measured against every waterline cell of the
    other extent (and "the other extent has a waterline" still refers to the whole grid)."""
    out = dict(cells)
    for side in ("pred", "obs"):
        rec = dict(cells[side])
        m = np.asarray(keep(rec["rows"], rec["cols"]), bool)
        lab = np.array(rec["label"], copy=True)
        lab[~m] = off
        rec["label"] = lab
        out[side] = rec
    return out
