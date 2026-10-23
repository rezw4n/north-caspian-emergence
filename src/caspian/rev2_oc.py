"""Operating characteristic of the H1 decision rule and the systematic-budget Monte Carlo
(PREREG Amendment 19 items 5 and 9; D-148). Pure functions, tested in tests/test_rev2_oc.py;
the drivers are scripts/rev2_oc_*.py.

1. Fields (item 5). Unit-variance Gaussian random fields with the D-125 covariance
   (exponential, practical range R: C(h) = exp(-3 h / R)) come from
   caspian.nearterm.grf_exponential on a coarse grid; ``bilinear_rows`` samples such a field
   at the cell centres of the 60 m grid in row blocks (separable bilinear interpolation,
   the same values as caspian.nearterm.bilinear_to without its whole-grid index arrays).
2. Waterlines at many thresholds. A predicted waterline cell at threshold T is a cell with a
   finite height, F < T (flooded) and a finite 4-neighbour with F >= T (known land), as
   caspian.rev_r.FloodMap.waterline defines it. With G the largest F among a cell's finite
   4-neighbours, that is F < T <= G. ``waterline_candidates`` keeps the cells whose interval
   [F, G) meets a range of thresholds, once; ``waterline_at`` then reads the waterline at any
   T in the range from those arrays. Comparisons are in float64, as a7_hindcast compares.
3. Area-constant counts (item 5). ``ThresholdCounter`` gives, for many sets of thresholds
   (one per unit), the number of cells with F below each threshold, from one sorted union of
   all thresholds: the counts of a7_hindcast.unit_counts without a search per unit.
4. Pooled displacement. Distances on the 60 m lattice are coded as their squared length in
   cells (``codes``; INF for an infinite distance). ``pooled_median`` is np.median of the
   decoded absolute distances, +inf if any is infinite (caspian.hindcast's summary).
5. The H1 rule, vectorised (``rule_flags``): a level exceeds when it is evaluable and D > Q
   or D is infinite; a sector is flagged when the exceeding levels are more than half of the
   evaluable ones (caspian.h1.mispredicts).
6. Summaries: ``clopper_pearson``, ``first_sigma_above`` and ``sector_epoch_sigma`` (the
   Amendment 18 item 7 budget reduced to a sector-epoch median).
"""

from __future__ import annotations

import math

import numpy as np

INF_CODE = np.iinfo(np.int32).max


# --- 1. fields ------------------------------------------------------------------------------

def _axis_weights(coord, origin: float, dx: float, n: int, descending: bool):
    """Indices i0 and weights w of linear interpolation along one axis of a grid whose node k
    lies at origin -/+ k dx; points beyond the grid take the nearest node (mode "nearest")."""
    f = (origin - np.asarray(coord, float)) / dx if descending else (np.asarray(coord, float) - origin) / dx
    f = np.clip(f, 0.0, n - 1.0)
    i0 = np.minimum(np.floor(f).astype(np.int64), n - 2) if n > 1 else np.zeros(f.shape, np.int64)
    w = f - i0 if n > 1 else np.zeros(f.shape)
    return i0, w


def bilinear_rows(field, y0: float, x0: float, dx: float, ys, xs, r0: int, r1: int) -> np.ndarray:
    """Rows r0:r1 of the bilinear interpolation of ``field`` (row i at y0 - i dx, column j at
    x0 + j dx; north-up) at the points ys[r0:r1] x xs, as float64. Equal to
    caspian.nearterm.bilinear_to(field, y0, x0, dx, ys, xs)[r0:r1]."""
    fld = np.asarray(field, float)
    ny, nx = fld.shape
    i0, wi = _axis_weights(np.asarray(ys)[r0:r1], y0, dx, ny, True)
    j0, wj = _axis_weights(xs, x0, dx, nx, False)
    i1 = np.minimum(i0 + 1, ny - 1)
    j1 = np.minimum(j0 + 1, nx - 1)
    rows = fld[i0] * (1.0 - wi)[:, None] + fld[i1] * wi[:, None]          # [r, nx]
    return rows[:, j0] * (1.0 - wj)[None, :] + rows[:, j1] * wj[None, :]


# --- 2. waterlines at many thresholds -------------------------------------------------------

def _tiles(shape, tile):
    return [(r0, min(r0 + tile, shape[0]), c0, min(c0 + tile, shape[1]))
            for r0 in range(0, shape[0], tile) for c0 in range(0, shape[1], tile)]


def neighbour_max(F, finite, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    """G on rows r0:r1, cols c0:c1: the largest F among the finite 4-neighbours of each cell
    (-inf if none; cells beyond the grid are not neighbours), in F's float dtype (a maximum
    of stored values, so exact)."""
    H, W = np.shape(F)
    w0, w1, v0, v1 = max(r0 - 1, 0), min(r1 + 1, H), max(c0 - 1, 0), min(c1 + 1, W)
    z = np.asarray(F[w0:w1, v0:v1])
    z = z.astype(np.result_type(z.dtype, np.float32), copy=False)
    fin = np.asarray(finite[w0:w1, v0:v1], bool)
    z = np.where(fin, z, z.dtype.type(-np.inf))
    pad = np.full((z.shape[0] + 2, z.shape[1] + 2), -np.inf, z.dtype)
    pad[1:-1, 1:-1] = z
    g = np.maximum(np.maximum(pad[:-2, 1:-1], pad[2:, 1:-1]), np.maximum(pad[1:-1, :-2], pad[1:-1, 2:]))
    return g[r0 - w0:r1 - w0, c0 - v0:c1 - v0]


def waterline_candidates(F, finite, t_lo: float, t_hi: float, *, tile: int = 1024) -> dict:
    """The cells that are predicted waterline cells at some threshold T in [t_lo, t_hi]: finite,
    F < t_hi, G >= t_lo and F < G (compared in float64). Returns {"idx" (flat, int64), "F",
    "G" (F's float dtype), "shape", "t_lo", "t_hi"}; ``waterline_at`` reads any T in the range
    from it."""
    shape = tuple(np.shape(F))
    lo, hi = np.float64(t_lo), np.float64(t_hi)
    I, Fs, Gs = [], [], []
    for r0, r1, c0, c1 in _tiles(shape, tile):
        z = np.asarray(F[r0:r1, c0:c1])
        z = z.astype(np.result_type(z.dtype, np.float32), copy=False)
        fin = np.asarray(finite[r0:r1, c0:c1], bool)
        pre = fin & (z < hi)
        if not pre.any():
            continue
        g = neighbour_max(F, finite, r0, r1, c0, c1)
        m = pre & (g >= lo) & (z < g)
        r, c = np.nonzero(m)
        I.append((r + r0).astype(np.int64) * shape[1] + (c + c0))
        Fs.append(z[r, c])
        Gs.append(g[r, c])
    dt = np.result_type(np.asarray(F[:1, :1]).dtype, np.float32)
    cat = (lambda a, d: np.concatenate(a) if a else np.empty(0, d))
    return {"idx": cat(I, np.int64), "F": cat(Fs, dt), "G": cat(Gs, dt), "shape": shape,
            "t_lo": float(t_lo), "t_hi": float(t_hi)}


def restrict(cand: dict, t_lo: float, t_hi: float) -> dict:
    """The candidates of a narrower threshold range."""
    if t_lo < cand["t_lo"] or t_hi > cand["t_hi"]:
        raise ValueError("range outside the candidates' range")
    m = (cand["F"] < np.float64(t_hi)) & (cand["G"] >= np.float64(t_lo))
    return {"idx": cand["idx"][m], "F": cand["F"][m], "G": cand["G"][m], "shape": cand["shape"],
            "t_lo": float(t_lo), "t_hi": float(t_hi)}


def waterline_at(cand: dict, T: float):
    """Rows and cols (int32) of the predicted waterline at T (t_lo <= T <= t_hi)."""
    if not cand["t_lo"] <= T <= cand["t_hi"]:
        raise ValueError(f"threshold {T} outside [{cand['t_lo']}, {cand['t_hi']}]")
    T = np.float64(T)
    idx = cand["idx"][(cand["F"] < T) & (cand["G"] >= T)]
    r, c = np.divmod(idx, cand["shape"][1])
    return r.astype(np.int32), c.astype(np.int32)


# --- 3. area-constant counts ----------------------------------------------------------------

class ThresholdCounter:
    """For values v (float64) and threshold sets T_u (each strictly increasing): the number of
    selected values below each threshold of T_u, exactly as a7_hindcast.unit_counts counts
    them (searchsorted side "right" on T_u, cumulated). All thresholds are merged into one
    sorted array; each value is located once."""

    def __init__(self, threshold_sets):
        self.sets = [np.asarray(t, float) for t in threshold_sets]
        for t in self.sets:
            if np.any(np.diff(t) <= 0):
                raise ValueError("thresholds must increase strictly")
        self.all = np.unique(np.concatenate(self.sets))
        self.pos = [np.searchsorted(self.all, t) for t in self.sets]

    def locate(self, v) -> np.ndarray:
        """Number of thresholds <= each value (int32)."""
        return np.searchsorted(self.all, np.asarray(v, float), side="right").astype(np.int32)

    def counts(self, q, u: int) -> np.ndarray:
        """Counts below each threshold of set u, from located values ``q`` (already selected)."""
        h = np.cumsum(np.bincount(q, minlength=self.all.size + 1))
        return h[self.pos[u]]


# --- 4. pooled displacement -----------------------------------------------------------------

def codes(d, pixel: float = 60.0) -> np.ndarray:
    """|d| as int32 squared lattice lengths (cells^2); INF_CODE for an infinite distance.
    NaN distances must be removed first."""
    d = np.abs(np.asarray(d, float))
    if np.isnan(d).any():
        raise ValueError("NaN distance")
    inf = np.isinf(d)
    k = np.rint((np.where(inf, 0.0, d) / pixel) ** 2)
    if np.any(k >= INF_CODE):
        raise ValueError("distance too large for the code")
    out = k.astype(np.int32)
    out[inf] = INF_CODE
    return out


def decode(k, pixel: float = 60.0) -> np.ndarray:
    k = np.asarray(k, np.int64)
    return np.where(k == INF_CODE, np.inf, np.sqrt(k.astype(float)) * pixel)


def pooled_median(parts, pixel: float = 60.0) -> tuple[float, int]:
    """(median |d|, n) of the union of code arrays: np.median of the decoded distances, +inf
    when any distance is infinite, NaN when there is none."""
    parts = [np.asarray(p) for p in parts if len(p)]
    n = int(sum(p.size for p in parts))
    if n == 0:
        return float("nan"), 0
    a = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()
    if (a == INF_CODE).any():
        return float("inf"), n
    lo, hi = (n - 1) // 2, n // 2
    a = np.partition(a, [lo, hi]) if hi != lo else np.partition(a, lo)
    v = decode([a[lo], a[hi]], pixel)
    return float((v[0] + v[1]) / 2.0), n


def displacement_codes(pred_rc, obs_rc, obs_state_at_pred, pred_known_at_obs, lab_pred, lab_obs, sector_ids,
                       *, obs_tree=None, pixel: float = 60.0, workers: int = 1) -> dict:
    """Absolute displacement distances (``codes``) per sector id, as caspian.hindcast_tiled
    measures them: a predicted waterline cell whose observed state is known (``obs_state_at_pred``
    >= 0) is measured to the nearest observed waterline cell of the whole grid (NaN, and so
    left out, when there is none); an observed waterline cell where the prediction is known
    (``pred_known_at_obs``) to the nearest predicted waterline cell (+inf when there is none).
    Only cells labelled with one of ``sector_ids`` are measured. ``obs_tree``: a cKDTree of the
    observed waterline (rows, cols), reused across thresholds."""
    from scipy.spatial import cKDTree
    pr, pc = (np.asarray(x) for x in pred_rc)
    orr, oc = (np.asarray(x) for x in obs_rc)
    sids = np.asarray(list(sector_ids))
    lp, lo = np.asarray(lab_pred), np.asarray(lab_obs)
    mp = np.isin(lp, sids) & (np.asarray(obs_state_at_pred) >= 0)
    mo = np.isin(lo, sids) & np.asarray(pred_known_at_obs, bool)
    if mp.any() and orr.size:
        tree = obs_tree if obs_tree is not None else cKDTree(np.c_[orr, oc].astype(float))
        dp = tree.query(np.c_[pr[mp], pc[mp]].astype(float), workers=workers)[0] * pixel
    else:
        dp = np.full(int(mp.sum()), np.nan)
    if mo.any() and pr.size:
        dpo = cKDTree(np.c_[pr, pc].astype(float)).query(np.c_[orr[mo], oc[mo]].astype(float),
                                                          workers=workers)[0] * pixel
    else:
        dpo = np.full(int(mo.sum()), np.inf)
    lab_p, lab_o = lp[mp], lo[mo]
    out = {}
    for s in sids.tolist():
        d = np.r_[dp[lab_p == s], dpo[lab_o == s]]
        out[s] = codes(d[~np.isnan(d)], pixel)
    return out


# --- 5. the H1 rule, vectorised -------------------------------------------------------------

def rule_flags(D, Q, evaluable):
    """Arrays [..., level]: D (median displacement; inf allowed, NaN when not measured), Q (the
    floor) and ``evaluable``. Returns (flagged, n_exceeding, n_evaluable) over the last axis;
    flagged is False where no level is evaluable (``n_evaluable`` 0 says so)."""
    D = np.asarray(D, float)
    Q = np.asarray(Q, float)
    ev = np.asarray(evaluable, bool)
    with np.errstate(invalid="ignore"):
        ex = ev & ((D == np.inf) | (D > Q))
    n_ex = ex.sum(axis=-1)
    n_ev = ev.sum(axis=-1)
    return (n_ev > 0) & (n_ex > n_ev / 2.0), n_ex, n_ev


# --- 6. summaries ---------------------------------------------------------------------------

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial (Clopper-Pearson) interval for k successes in n trials."""
    from scipy.stats import beta
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def first_sigma_above(sigmas, p, threshold: float = 0.5):
    """The smallest sigma (in increasing order) whose P exceeds ``threshold``; None if none."""
    order = np.argsort(np.asarray(sigmas, float))
    for i in order:
        if float(p[i]) > threshold:
            return float(sigmas[i])
    return None


def sector_epoch_sigma(levels, units) -> dict:
    """{(sector, epoch): median sigma_budget_m over the bins at which the epoch has a unit}.
    ``levels``: rows with sector, bin and sigma_budget_m (one value per sector and bin; the
    Amendment 18 item 7 budget); ``units``: (bin, epoch) pairs."""
    s = levels.drop_duplicates(["sector", "bin"]).set_index(["sector", "bin"])["sigma_budget_m"]
    by_ep: dict = {}
    for b, ep in units:
        by_ep.setdefault(ep, set()).add(int(b))
    out = {}
    for sec in sorted(set(levels["sector"])):
        for ep, bins in sorted(by_ep.items()):
            v = [float(s[(sec, b)]) for b in sorted(bins) if (sec, b) in s.index and math.isfinite(float(s[(sec, b)]))]
            out[(sec, ep)] = float(np.median(v)) if v else float("nan")
    return out
