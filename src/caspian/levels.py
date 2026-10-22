"""Water level at each acquisition time, with uncertainty.

The level between two altimetry epochs is modelled as a Brownian bridge plus
measurement error. For a query time between epochs i and j (gap Δt, fraction
u = (t - t_i)/Δt):

    h(t)   = (1 - u) h_i + u h_j
    var(t) = (1 - u)^2 σ_i^2 + u^2 σ_j^2 + q Δt u (1 - u)

where q (m² per day) is the effective diffusion rate of the interpolation error.
The record noise σ and q are estimated jointly (``estimate_noise_and_diffusion``)
from the cross-validated residuals of this same linear interpolation, so the two
are consistent with each other and with ``interpolate``. The seasonal cycle is
kept in h (it adds contours); q absorbs the sub-sampling
variability (including seasonal curvature between records), so it describes the
interpolation error of this series, not the sea alone: a denser or smoother product
has a different q (Hydroweb's own joint fit gives a much smaller q than DAHITI's;
see the diagnostics of scripts/a4_scene_levels.py).

A second series is brought onto the reference of the primary one with a moving
median of their paired differences (``pair_series``, ``moving_median_offset``).
The uncertainty of that offset (the SE of the moving median and the excess
variability of the offset in the years around each query, ``offset_excess_sd``,
``local_excess_sd``) is an error common to the bracketing records and is added
after interpolation (``interpolate_shifted``).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


_EPOCH = pd.Timestamp("1970-01-01")
_NMAD = 1.4826
_SE_MEDIAN = float(np.sqrt(np.pi / 2))  # SE of a Gaussian sample median = 1.2533 sd / sqrt(n)


def _days(t) -> np.ndarray:
    """Days since 1970-01-01, independent of the datetime resolution (pandas 3
    infers s/ms/us/ns, so integer casts are not safe)."""
    return ((pd.to_datetime(pd.Series(t)) - _EPOCH) / pd.Timedelta(days=1)).to_numpy(float)


def _nmad(x) -> float:
    x = np.asarray(x, float)
    return float(_NMAD * np.median(np.abs(x - np.median(x))))


def gap_days(times) -> np.ndarray:
    """Gaps between consecutive records in fractional days (time-sorted; length n - 1).
    Unlike ``Series.diff().dt.days`` this does not floor each gap to whole days."""
    return np.diff(np.sort(_days(times)))


def estimate_diffusion(times, levels, sigmas, max_gap_days: float = 60.0) -> float:
    """Least-squares estimate of q from neighbour increments (m² per day), >= 0, given
    the per-record noise ``sigmas``. The pipeline uses ``estimate_noise_and_diffusion``
    instead: feeding this function a noise floor that already
    contains bridge variance biases q low."""
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


def interpolate_shifted(times, levels, sigmas, query_times, offset, offset_sigma, q: float,
                        max_gap_days: float = 60.0) -> pd.DataFrame:
    """``interpolate`` a secondary series and shift it onto the primary reference.

    ``sigmas`` are the per-record random errors (independent between records, so the
    bridge weights them by (1-u)^2 and u^2). ``offset`` and ``offset_sigma`` are given
    at ``query_times`` (scalars broadcast). The offset error is common to both
    bracketing records, so it is added in quadrature after interpolation and never
    shrinks between records.

    Besides the total ``sigma_m`` the result carries its two parts, ``sigma_random_m``
    (records and bridge) and ``sigma_offset_m`` (the offset error, shared by every
    query whose offset comes from the same pairs), so that an aggregation over many
    queries can treat the second as common rather than averaging it down."""
    out = interpolate(times, levels, sigmas, query_times, q=q, max_gap_days=max_gap_days)
    off = np.broadcast_to(np.asarray(offset, float), (len(out),))
    off_s = np.broadcast_to(np.asarray(offset_sigma, float), (len(out),))
    rand = out["sigma_m"].to_numpy()
    out["level_m"] = out["level_m"].to_numpy() + off
    out["sigma_m"] = np.sqrt(rand ** 2 + off_s ** 2)
    out["sigma_random_m"] = rand
    out["sigma_offset_m"] = np.where(np.isfinite(rand), off_s, np.nan)
    return out


def bin_index(levels, width: float, origin: float = 0.0) -> np.ndarray:
    """Integer bin for each level; bin k covers [origin + k w, origin + (k + 1) w)."""
    return np.floor((np.asarray(levels, float) - origin) / width).astype(int)


def bin_center(k, width: float, origin: float = 0.0):
    return origin + (np.asarray(k) + 0.5) * width


def empirical_sigma(times, levels, max_gap_days: float = 30.0) -> float:
    """Neighbour-interpolation scatter of a level series (superseded in the pipeline
    by ``estimate_noise_and_diffusion``).

    Each record is predicted by linear interpolation from its two neighbours; with
    w = (t_i - t_{i-1}) / (t_{i+1} - t_{i-1}) the residual has variance
    sigma^2 (1 + w^2 + (1 - w)^2) + q g1 g2 / (g1 + g2) under the Brownian-bridge
    model. This function returns the NMAD of the residuals scaled by the first
    factor only, so it equals sigma only when the bridge term is negligible (q = 0
    or very short gaps); otherwise it is biased high."""
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
    return _nmad(e)


# --- joint estimate of record noise and diffusion --------------------------------------

class NoiseDiffusion(NamedTuple):
    sigma: float          # record noise, m
    q: float              # effective diffusion of the interpolation error, m² per day
    sigma_se: float       # moving-block bootstrap SE (NaN if n_boot = 0)
    q_se: float
    n_residuals: int


def _skip_residuals(d: np.ndarray, h: np.ndarray, max_span_days: float, max_skip: int):
    """Leave-one-out residuals of record i predicted by linear interpolation from
    records i-a and i+b (1 <= a, b <= max_skip) with total span <= max_span_days.
    Returns (centre index, residual, f = 1 + w^2 + (1-w)^2, B = g1 g2 / (g1 + g2))."""
    n = d.size
    ci, rr, ff, bb = [], [], [], []
    for a in range(1, max_skip + 1):
        for b in range(1, max_skip + 1):
            i = np.arange(a, n - b)
            if i.size == 0:
                continue
            g1 = d[i] - d[i - a]
            g2 = d[i + b] - d[i]
            ok = (g1 > 0) & (g2 > 0) & (g1 + g2 <= max_span_days)
            i, g1, g2 = i[ok], g1[ok], g2[ok]
            w = g1 / (g1 + g2)
            ci.append(i)
            rr.append(h[i] - ((1 - w) * h[i - a] + w * h[i + b]))
            ff.append(1 + w ** 2 + (1 - w) ** 2)
            bb.append(g1 * g2 / (g1 + g2))
    if not ci:
        return np.empty(0, int), np.empty(0), np.empty(0), np.empty(0)
    return np.concatenate(ci), np.concatenate(rr), np.concatenate(ff), np.concatenate(bb)


def _fit_noise_diffusion(z: np.ndarray, x: np.ndarray, n_bins: int, q_fixed: float | None,
                         n_iter: int = 20):
    """Iteratively reweighted LS of binned robust variances s_b^2 = sigma^2 + q x_b
    (equal-count bins of x). The weights n_b / v_b^2 are the inverse sampling variance
    of a variance, with v_b the *fitted* variance of the previous iteration (weighting
    by each bin's own estimate would favour bins whose variance came out low).
    Returns (sigma^2, q), both >= 0."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    xb, sb, nb = [], [], []
    for j in range(n_bins):
        m = k == j
        if m.sum() >= 10:
            xb.append(x[m].mean())
            sb.append(_nmad(z[m]) ** 2)
            nb.append(m.sum())
    xb, sb, nb = np.array(xb), np.array(sb), np.array(nb, float)
    if xb.size < 2:
        raise ValueError("too few residual bins")

    def solve(wts):
        if q_fixed is not None:
            return max(float(np.average(sb - q_fixed * xb, weights=wts)), 0.0), float(q_fixed)
        a = np.vstack([np.ones_like(xb), xb]).T * np.sqrt(wts)[:, None]
        s2, q = np.linalg.lstsq(a, sb * np.sqrt(wts), rcond=None)[0]
        if q < 0:                              # no detectable bridge term
            return max(float(np.average(sb, weights=wts)), 0.0), 0.0
        if s2 < 0:                             # no detectable white noise
            return 0.0, max(float(np.sum(wts * xb * sb) / np.sum(wts * xb ** 2)), 0.0)
        return float(s2), float(q)

    s2, q = solve(nb)                          # equal-count bins: start unweighted
    for _ in range(n_iter):
        fitted = np.maximum(s2 + q * xb, 1e-12)
        s2_new, q_new = solve(nb / fitted ** 2)
        done = abs(s2_new - s2) <= 1e-9 * max(s2, 1e-12) and abs(q_new - q) <= 1e-9 * max(q, 1e-15)
        s2, q = s2_new, q_new
        if done:
            break
    return s2, q


def estimate_noise_and_diffusion(times, levels, max_span_days: float = 60.0, max_skip: int = 6,
                                 n_bins: int = 10, q: float | None = None, n_boot: int = 0,
                                 block: int = 50, seed: int = 0) -> NoiseDiffusion:
    """Joint estimate of the record noise sigma and the diffusion q of ``interpolate``.

    Every record i is predicted by linear interpolation from records i-a and i+b
    (1 <= a, b <= ``max_skip``, total span g1 + g2 <= ``max_span_days``, the longest
    gap ``interpolate`` bridges). Under the model (Brownian motion with rate q plus
    i.i.d. noise sigma) the residual has variance sigma^2 f + q B, with
    w = g1 / (g1 + g2), f = 1 + w^2 + (1 - w)^2 and B = g1 g2 / (g1 + g2). The
    residuals scaled by sqrt(f) have variance sigma^2 + q x, x = B / f. They are grouped
    into ``n_bins`` equal-count bins of x; the robust variance (NMAD^2) of each bin is
    fitted by iteratively reweighted least squares, giving sigma^2 (intercept) and q
    (slope), both constrained >= 0. With ``q`` given, only sigma is estimated (bridge
    term removed).

    Standard errors (``n_boot`` > 0) come from a moving-block bootstrap over the
    records (blocks of ``block`` consecutive centre records, fixed ``seed``), which
    keeps the correlation between residuals that share records."""
    d = _days(times)
    h = np.asarray(levels, float)
    order = np.argsort(d)
    d, h = d[order], h[order]
    ci, r, f, b = _skip_residuals(d, h, max_span_days, max_skip)
    if r.size < 10 * n_bins:
        raise ValueError("too few interpolation residuals to estimate noise and diffusion")
    z, x = r / np.sqrt(f), b / f
    s2, q_hat = _fit_noise_diffusion(z, x, n_bins, q)
    sig_se = q_se = float("nan")
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        o = np.argsort(ci, kind="stable")
        ci_s, z_s, x_s = ci[o], z[o], x[o]
        n_rec = d.size
        n_blocks = int(np.ceil(n_rec / block))
        first = np.searchsorted(ci_s, np.arange(n_rec + 1))   # residuals of centre c: first[c]:first[c+1]
        bs = []
        for _ in range(n_boot):
            st = rng.integers(0, max(n_rec - block, 1), n_blocks)
            idx = np.concatenate([np.arange(first[s], first[min(s + block, n_rec)]) for s in st])
            try:
                bs.append(_fit_noise_diffusion(z_s[idx], x_s[idx], n_bins, q))
            except ValueError:
                continue
        bs = np.array(bs)
        sig_se = float(np.std(np.sqrt(bs[:, 0]), ddof=1))
        q_se = float(np.std(bs[:, 1], ddof=1)) if q is None else 0.0
    return NoiseDiffusion(float(np.sqrt(s2)), float(q_hat), sig_se, q_se, int(r.size))


def neighbour_zscores(times, levels, sigma: float, q: float, max_gap_days: float = 30.0) -> np.ndarray:
    """Standardised residual of each record against its two neighbours under the
    fitted model, z = r / sqrt(sigma^2 f + q B), in the input order (NaN for the end
    records and where a gap exceeds ``max_gap_days``). Under the model |z| > 2.5
    occurs for about 1.2% of records."""
    d = _days(times)
    h = np.asarray(levels, float)
    order = np.argsort(d, kind="stable")
    d, h = d[order], h[order]
    g1, g2 = d[1:-1] - d[:-2], d[2:] - d[1:-1]
    keep = (g1 > 0) & (g2 > 0) & (g1 <= max_gap_days) & (g2 <= max_gap_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = g1 / (g1 + g2)
        r = h[1:-1] - ((1 - w) * h[:-2] + w * h[2:])
        v = sigma ** 2 * (1 + w ** 2 + (1 - w) ** 2) + q * g1 * g2 / (g1 + g2)
        zz = np.where(keep, r / np.sqrt(v), np.nan)
    z_sorted = np.full(d.size, np.nan)
    z_sorted[1:-1] = zz
    out = np.full(d.size, np.nan)
    out[order] = z_sorted
    return out


# --- robust extremes -------------------------------------------------------------------

def running_median(values, k: int = 3) -> np.ndarray:
    """Centred running median over ``k`` consecutive records (k odd); the window is
    truncated at the ends of the series."""
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be a positive odd integer")
    v = np.asarray(values, float)
    return pd.Series(v).rolling(k, center=True, min_periods=1).median().to_numpy()


def smoothed_extremes(times, levels, k: int = 3) -> dict:
    """Maximum and minimum of the k-point running median of a level series (records in
    time order), with their dates. A single record that lies several sigma above or
    below its neighbours cannot set either extreme. Ties go to the earliest date."""
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    order = np.argsort(_days(t), kind="stable")
    t = t.iloc[order].reset_index(drop=True)
    s = running_median(np.asarray(levels, float)[order], k)
    i, j = int(np.nanargmax(s)), int(np.nanargmin(s))
    return {"k": int(k), "max_m": float(s[i]), "time_of_max": t.iloc[i], "min_m": float(s[j]),
            "time_of_min": t.iloc[j], "range_m": float(s[i] - s[j])}


# --- secondary series onto the primary reference -----------------------------------------

def pair_series(t_ref, h_ref, t_other, h_other, max_half_gap_days: float) -> pd.DataFrame:
    """For each reference epoch, the other series linearly interpolated between its two
    bracketing records when both lie within ``max_half_gap_days``. Returns time, ref,
    other and diff = ref - other."""
    d_ref = _days(t_ref)
    h_ref = np.asarray(h_ref, float)
    d_o = _days(t_other)
    h_o = np.asarray(h_other, float)
    oo = np.argsort(d_o, kind="stable")
    d_o, h_o = d_o[oo], h_o[oo]
    j = np.searchsorted(d_o, d_ref, side="left")
    ok = (j > 0) & (j < d_o.size)
    jj = np.clip(j, 1, max(d_o.size - 1, 1))
    t0, t1 = d_o[jj - 1], d_o[jj]
    ok &= ((d_ref - t0) <= max_half_gap_days) & ((t1 - d_ref) <= max_half_gap_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(t1 > t0, (d_ref - t0) / (t1 - t0), 0.0)
    hv = (1 - u) * h_o[jj - 1] + u * h_o[jj]
    times = pd.to_datetime(pd.Series(t_ref)).reset_index(drop=True)
    out = pd.DataFrame({"time": times[ok].to_numpy(), "ref": h_ref[ok], "other": hv[ok]})
    out["diff"] = out["ref"] - out["other"]
    return out


def moving_median_offset(pair_times, diffs, query_times, half_window_days: float,
                         min_pairs: int = 20) -> pd.DataFrame:
    """Time-varying offset between two series: the median of the paired differences
    within +/- ``half_window_days`` of each query time. Where fewer than ``min_pairs``
    fall in the window, the ``min_pairs`` pairs nearest in time are used instead.
    ``se_m`` is the standard error of that median, sqrt(pi/2) NMAD / sqrt(n)."""
    dp = _days(pair_times)
    x = np.asarray(diffs, float)
    o = np.argsort(dp, kind="stable")
    dp, x = dp[o], x[o]
    if dp.size < min_pairs:
        raise ValueError("fewer pairs than min_pairs")
    tq = _days(query_times)
    uq, inv = np.unique(tq, return_inverse=True)
    off, se, nn = np.empty(uq.size), np.empty(uq.size), np.empty(uq.size, int)
    lo = np.searchsorted(dp, uq - half_window_days, side="left")
    hi = np.searchsorted(dp, uq + half_window_days, side="right")
    for k, t in enumerate(uq):
        if hi[k] - lo[k] >= min_pairs:
            sel = x[lo[k]:hi[k]]
        else:
            sel = x[np.argsort(np.abs(dp - t), kind="stable")[:min_pairs]]
        off[k] = np.median(sel)
        se[k] = _SE_MEDIAN * _nmad(sel) / np.sqrt(sel.size)
        nn[k] = sel.size
    inv = np.asarray(inv).ravel()
    return pd.DataFrame({"offset_m": off[inv], "se_m": se[inv], "n_pairs": nn[inv]})


def offset_excess_sd(pair_times, diffs, half_window_days: float, min_pairs: int = 20,
                     min_per_year: int = 5) -> dict:
    """Variability of the offset on time scales the moving median cannot follow.

    For each calendar year with at least ``min_per_year`` pairs, the year's median is
    compared with the moving median of all other years' pairs at that year's middle
    pair time. tau^2 = max(0, mean[(m_y - M_-y)^2 - SE_y^2 - SE_M^2]) is the excess of
    that scatter over the sampling variance (method of moments), pooled over all years.

    The pooled tau averages eras of different stability; where the offset varies more
    in some years than others, use ``local_excess_sd`` on the returned ``by_year`` rows
    (each carries its middle pair time ``time_mid`` and its moment term ``term_m2``)."""
    tp = pd.to_datetime(pd.Series(pair_times)).reset_index(drop=True)
    x = np.asarray(diffs, float)
    yrs = tp.dt.year.to_numpy()
    terms, rows = [], []
    for y in np.unique(yrs):
        m = yrs == y
        if m.sum() < min_per_year:
            continue
        t_mid = tp[m].sort_values().iloc[int(m.sum() // 2)]
        others = moving_median_offset(tp[~m], x[~m], [t_mid], half_window_days, min_pairs).iloc[0]
        my = float(np.median(x[m]))
        se_y = _SE_MEDIAN * _nmad(x[m]) / np.sqrt(m.sum())
        term = float((my - others["offset_m"]) ** 2 - se_y ** 2 - others["se_m"] ** 2)
        terms.append(term)
        rows.append({"year": int(y), "n": int(m.sum()), "time_mid": t_mid.isoformat(), "median_m": my,
                     "se_m": float(se_y), "moving_median_others_m": float(others["offset_m"]),
                     "moving_median_others_se_m": float(others["se_m"]), "term_m2": term})
    tau2 = max(float(np.mean(terms)), 0.0) if terms else 0.0
    return {"tau_m": float(np.sqrt(tau2)), "n_years": len(rows), "by_year": rows}


def local_excess_sd(by_year, query_times, half_window_days: float, min_years: int = 5) -> pd.DataFrame:
    """tau at each query time from the years near it (the ``by_year`` rows of
    ``offset_excess_sd``).

    tau(t)^2 = max(0, mean of term_m2) over the years whose middle pair time lies within
    +/- ``half_window_days`` of t (the window of the moving-median offset); where fewer
    than ``min_years`` do, the ``min_years`` years nearest in time are used instead.
    Returns tau_m and n_years per query."""
    if len(by_year) < min_years:
        raise ValueError("fewer years than min_years")
    ty = _days([r["time_mid"] for r in by_year])
    term = np.array([r["term_m2"] for r in by_year], float)
    tq = _days(query_times)
    uq, inv = np.unique(tq, return_inverse=True)
    tau, ny = np.empty(uq.size), np.empty(uq.size, int)
    for k, t in enumerate(uq):
        dist = np.abs(ty - t)
        sel = dist <= half_window_days
        if sel.sum() < min_years:
            sel = np.zeros(ty.size, bool)
            sel[np.argsort(dist, kind="stable")[:min_years]] = True
        tau[k] = np.sqrt(max(float(term[sel].mean()), 0.0))
        ny[k] = int(sel.sum())
    inv = np.asarray(inv).ravel()
    return pd.DataFrame({"tau_m": tau[inv], "n_years": ny[inv]})
