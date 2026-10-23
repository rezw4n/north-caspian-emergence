"""Second-review analyses D (PREREG Amendment 19 items 4, 6, 10, 12 and 14): core functions.

Pure functions (arrays in, arrays out), tested in tests/test_rev2_d.py. The drivers are
scripts/rev2_d_*.py.

- ``level_band``: the 0.5 m level band (lo, hi] of a level (A19 item 6a).
- ``single_changepoint``: one mean shift at a bin edge, by least squares (A19 item 12).
- ``block_bootstrap_stats``: mean, median, NMAD and RMSE of a block bootstrap, exactly as
  caspian.validation.block_bootstrap would give them from the concatenated draws, but
  computed from per-point weights so millions of points stay cheap (A19 item 10).
- ``pairwise_correlations``: Pearson and Spearman over levels for every pair of columns
  (A19 item 6b).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

NMAD_SCALE = 1.4826          # as caspian.validation.NMAD_SCALE


# --- level bands (A19 item 6a) -----------------------------------------------------------------

def level_band(level, width: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) of the band (lo, hi] that holds each level, with edges at multiples of
    ``width``. A level on an edge belongs to the band below it (right-closed)."""
    x = np.asarray(level, float)
    hi = np.ceil(np.round(x / width, 9)) * width
    return np.round(hi - width, 9), np.round(hi, 9)


# --- single changepoint (A19 item 12) ----------------------------------------------------------

def single_changepoint(x, y, edges, min_levels: int = 3) -> dict:
    """One mean shift in ``y`` at a threshold on ``x``: for every candidate edge t (with at
    least ``min_levels`` distinct x values on each side), the residual sum of squares of the
    two segment means (x < t and x > t); the edge with the smallest RSS wins, ties to the
    lowest edge. Rows with non-finite x or y are dropped. Returns the edge, both segments'
    n, mean and the RSS profile, and the RSS without a break."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        raise ValueError("no finite rows")
    rss0 = float(np.sum((y - y.mean()) ** 2))
    prof = []
    for t in sorted(float(e) for e in edges):
        lo, hi = x < t, x > t
        if np.any(x == t):
            raise ValueError(f"edge {t} coincides with a level")
        if np.unique(x[lo]).size < min_levels or np.unique(x[hi]).size < min_levels:
            continue
        rss = float(np.sum((y[lo] - y[lo].mean()) ** 2) + np.sum((y[hi] - y[hi].mean()) ** 2))
        prof.append({"edge": t, "rss": rss, "n_below": int(lo.sum()), "n_above": int(hi.sum()),
                     "mean_below": float(y[lo].mean()), "mean_above": float(y[hi].mean())})
    if not prof:
        raise ValueError("no admissible edge")
    best = min(prof, key=lambda r: (r["rss"], r["edge"]))
    return {**best, "rss_no_break": rss0, "r2": 1.0 - best["rss"] / rss0 if rss0 > 0 else float("nan"),
            "n": int(x.size), "min_levels": min_levels, "profile": prof}


# --- weighted block bootstrap (A19 item 10) ----------------------------------------------------

def _kth(v_sorted, cw, k: int) -> float:
    """The k-th (0-based) element of the sample in which sorted value i is repeated
    w_i times (cw = cumsum w)."""
    return float(v_sorted[np.searchsorted(cw, k, side="right")])


def _kth_abs_dev(v_sorted, w_sorted, m: float, k: int) -> float:
    """The k-th (0-based) smallest |v - m| of the weighted sample (exact; v sorted
    ascending, integer weights)."""
    i = int(np.searchsorted(v_sorted, m, side="right"))          # v[:i] <= m < v[i:]
    dl = (m - v_sorted[:i])[::-1]                                 # ascending
    cl = np.cumsum(w_sorted[:i][::-1])
    dr = v_sorted[i:] - m                                         # ascending
    cr = np.cumsum(w_sorted[i:])

    def count_le(t):
        a = np.searchsorted(dl, t, side="right")
        b = np.searchsorted(dr, t, side="right")
        return (int(cl[a - 1]) if a else 0) + (int(cr[b - 1]) if b else 0)

    best = math.inf
    for d in (dl, dr):
        lo, hi = 0, d.size - 1
        if d.size == 0 or count_le(d[hi]) < k + 1:
            continue
        while lo < hi:                                            # smallest index with count >= k + 1
            mid = (lo + hi) // 2
            if count_le(d[mid]) >= k + 1:
                hi = mid
            else:
                lo = mid + 1
        best = min(best, float(d[lo]))
    return best


def weighted_stats(v_sorted, w_sorted) -> tuple[float, float, float, float]:
    """(mean, median, NMAD, RMSE) of the sample in which sorted value i is repeated w_i
    (integer) times, with numpy's median convention (the two middle values averaged)."""
    w = np.asarray(w_sorted, np.int64)
    n = int(w.sum())
    if n == 0:
        return (float("nan"),) * 4
    cw = np.cumsum(w)
    mean = float(np.dot(w, v_sorted) / n)
    rmse = float(np.sqrt(np.dot(w, v_sorted * v_sorted) / n))
    k1, k2 = (n - 1) // 2, n // 2
    med = (_kth(v_sorted, cw, k1) + _kth(v_sorted, cw, k2)) / 2
    mad = (_kth_abs_dev(v_sorted, w, med, k1) + _kth_abs_dev(v_sorted, w, med, k2)) / 2
    return mean, med, NMAD_SCALE * mad, rmse


def block_bootstrap_stats(e, block_labels, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, 4]: mean, median, NMAD and RMSE of each block-bootstrap replicate. The
    draws are those of caspian.validation.block_bootstrap (blocks in sorted label order,
    ``rng.integers(0, n_blocks, n_blocks)`` per replicate from default_rng(seed)), so each
    row equals the statistics of the concatenated draw; here each replicate is evaluated
    from per-block weights instead of building the concatenation."""
    e = np.asarray(e, float)
    lab = np.asarray(block_labels).astype(str)
    if e.size == 0 or e.size != lab.size:
        raise ValueError("need one block label per error, and at least one error")
    if not np.all(np.isfinite(e)):
        raise ValueError("errors must be finite")
    uniq, inv = np.unique(lab, return_inverse=True)
    nb = uniq.size
    order = np.argsort(e, kind="stable")
    v, bs = e[order], inv[order]
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 4))
    for r in range(n_boot):
        w_block = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        out[r] = weighted_stats(v, w_block[bs])
    return out


def percentile_interval(boot, conf: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates (as caspian.validation)."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan"), float("nan")
    a = (1 - conf) / 2 * 100
    lo, hi = np.percentile(b, [a, 100 - a])
    return float(lo), float(hi)


# --- cross-product correlations (A19 item 6b) --------------------------------------------------

def pairwise_correlations(wide: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """Pearson and Spearman correlation over rows (levels) for every pair of columns
    (products); rows where either value is non-finite are dropped pairwise. Pairs with
    fewer than ``min_n`` rows get NaN."""
    rows = []
    for a, b in itertools.combinations(list(wide.columns), 2):
        x, y = wide[a].to_numpy(float), wide[b].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        n = int(ok.sum())
        if n >= min_n and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
            pr = stats.pearsonr(x[ok], y[ok])
            sr = stats.spearmanr(x[ok], y[ok])
            r, p_r, rho, p_rho = float(pr[0]), float(pr[1]), float(sr[0]), float(sr[1])
        else:
            r = p_r = rho = p_rho = float("nan")
        rows.append({"a": a, "b": b, "n": n, "pearson_r": r, "pearson_p": p_r,
                     "spearman_rho": rho, "spearman_p": p_rho})
    return pd.DataFrame(rows)
