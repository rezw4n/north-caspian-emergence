"""Second-round revision analyses of near-term emergence (PREREG Amendment 19 items 1-3;
D-146). Additive helpers for scripts/rev2_n_*.py; nothing here changes a registered or
Amendment 18 result.

Conventions as caspian.nearterm: F is a flood-level map (a cell is sea at level L when
F < L, compared in float64). The re-referenced product is F - c - c26 (c: the D-070 shape
constant; c26: the constant that matches the product's 2026 sea area to the observed
2026 sea S_obs), so a cell is product sea at L2026 when F < T, T = L2026 + c + c26.

Contents
- ``bisect_offset``: c26 on a 0.01 m grid by integer bisection on a non-decreasing area;
- ``closest_offset``: the same choice by full evaluation (D-070's rule; a check);
- ``WeightedArea``: area below a level for weighted cells (coarse grids);
- ``rereferenced_band``, ``increment_and_misfit``: the A19 item 1 quantities on whole
  arrays (the scripts compute them from counts);
- ``ReRefSurface``: the window-readable surface G for exposure;
- ``lineage_table``, ``max_min_ratio``: item 2.
"""

from __future__ import annotations

import numpy as np

LINEAGES = {"GEBCO": ("gebco_2026", "gebco_2022", "etopo_2022"), "ETOPO1": ("etopo1",),
            "SRTM15+": ("srtm15p_v2.7",)}


# --- c26 ----------------------------------------------------------------------------------------

def _choose(cands, area_at, target, step):
    """Among grid integers ``cands``: the smallest |area - target|; ties to the smaller
    |k|, then the smaller k (D-070)."""
    best = None
    for k in cands:
        a = float(area_at(k))
        key = (abs(a - target), abs(k), k)
        if best is None or key < best[0]:
            best = (key, k, a)
    _, k, a = best
    return {"k": int(k), "offset_m": round(k * step, 10), "area": a, "error": a - target}


def bisect_offset(area_at, target: float, k_lo: int = -500, k_hi: int = 500, step: float = 0.01) -> dict:
    """The grid offset k * ``step`` (k integer in [k_lo, k_hi]) whose area is closest to
    ``target``, by integer bisection. ``area_at(k)``: the area at offset k * step, non-
    decreasing in k. Bisection finds the smallest k* with area(k*) >= target; of k* and
    k* - 1 the one with the smaller |area - target| is taken, ties to the smaller |k|, then
    the smaller k. ValueError when the target lies outside [area(k_lo), area(k_hi)].
    Returns {k, offset_m, area, error, n_eval}."""
    n_eval = 2
    a_lo, a_hi = float(area_at(k_lo)), float(area_at(k_hi))
    if a_lo > target or a_hi < target:
        raise ValueError(f"target {target} not bracketed by [{a_lo}, {a_hi}] on k in [{k_lo}, {k_hi}]")
    if a_lo >= target:
        ks = k_lo
    else:
        lo, hi = k_lo, k_hi                 # area(lo) < target <= area(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            n_eval += 1
            if float(area_at(mid)) >= target:
                hi = mid
            else:
                lo = mid
        ks = hi
    # every k with the minimal error lies on the plateau of k* - 1 (k <= k* - 1) or of k*
    # (k >= k*); on each, the k nearest 0 is found by walking toward 0 (D-070's tie rule)
    cands = []
    for k0, direction in ((ks - 1, -1), (ks, +1)):
        if k0 < k_lo:
            continue
        a0, k = float(area_at(k0)), k0
        if (direction < 0 and k0 > 0) or (direction > 0 and k0 < 0):
            toward = -1 if k0 > 0 else 1
            while k != 0 and k_lo <= k + toward <= k_hi and float(area_at(k + toward)) == a0:
                k += toward
                n_eval += 1
        cands.append(k)
    out = _choose(cands, area_at, target, step)
    out["n_eval"] = n_eval + len(cands)
    return out


def closest_offset(areas, ks, target: float, step: float = 0.01) -> dict:
    """D-070's rule by full evaluation: the k of ``ks`` minimising |area - target|, ties to
    the smaller |k|, then the smaller k; with the number of tied minima."""
    a = np.asarray(areas, float)
    k = np.asarray(ks, np.int64)
    err = np.abs(a - target)
    cand = np.flatnonzero(err == err.min())
    j = cand[np.lexsort((k[cand], np.abs(k[cand])))[0]]
    return {"k": int(k[j]), "offset_m": round(int(k[j]) * step, 10), "area": float(a[j]),
            "error": float(a[j] - target), "n_tied": int(cand.size)}


class WeightedArea:
    """Area of the cells below a level, Σ w [F < L], for a flood-level grid with cell
    weights (e.g. 60 m domain cells per coarse block). Non-finite F never counts."""

    def __init__(self, F, w):
        f = np.asarray(F, np.float64).ravel()
        w = np.asarray(w, np.float64).ravel()
        ok = np.isfinite(f) & (w > 0)
        o = np.argsort(f[ok], kind="stable")
        self.f = f[ok][o]
        self.cw = np.concatenate([[0.0], np.cumsum(w[ok][o])])

    def __call__(self, level):
        idx = np.searchsorted(self.f, np.asarray(level, np.float64), side="left")   # values < level
        return self.cw[idx]


# --- item 1 quantities ----------------------------------------------------------------------------

def rereferenced_band(F, S, T: float, delta: float) -> np.ndarray:
    """S ∩ {T - delta <= F < T}: observed 2026 sea that the re-referenced product holds as
    sea at the 2026 level (F < T) and dry ``delta`` lower (not F < T - delta)."""
    f = np.asarray(F, np.float64)
    return np.asarray(S, bool) & (f < T) & ~(f < T - delta)


def increment_and_misfit(F, S, T: float, deltas, weights=None) -> dict:
    """Whole-array E(delta) = |S ∩ {T - delta <= F < T}| for each delta and the misfit
    M = |S ∩ {F >= T}| (not F < T; non-finite F counts as dry), with optional cell weights;
    also the product's sea outside S, |{F < T} \\ S|."""
    f = np.asarray(F, np.float64)
    s = np.asarray(S, bool)
    w = np.ones(f.shape) if weights is None else np.asarray(weights, float)
    sea = f < T
    return {"E": [float((rereferenced_band(f, s, T, d) * w).sum()) for d in deltas],
            "M": float(((s & ~sea) * w).sum()),
            "outside": float(((~s & sea) * w).sum())}


MISFIT_SEA_VALUE = -1000.0      # finite: distance profiles skip non-finite cells


class ReRefSurface:
    """Window-readable surface for the exposure rules: on the observed sea S_obs
    (``base_map`` < ``base_level``) G = F - shift; elsewhere +inf. With the sea level
    ``sea_level`` (default ``base_level``), G < sea_level is the re-referenced product's
    sea within S_obs and, for L <= sea_level, the emerged set between the two levels is
    the increment. ``misfit="sea"`` sets G = MISFIT_SEA_VALUE on S_obs cells the product
    holds dry at sea_level (they stay sea in both states, so the 2026 sea is S_obs
    itself); ``misfit="land"`` (default) keeps F - shift there (land in both states).
    With shift = 0 and sea_level = T the comparisons are F < T exactly as in the counts."""

    def __init__(self, F, base_map, base_level: float, shift: float = 0.0, misfit: str = "land",
                 sea_level: float | None = None):
        if misfit not in ("land", "sea"):
            raise ValueError("misfit must be 'land' or 'sea'")
        self.F, self.B, self.L26, self.shift, self.misfit = F, base_map, float(base_level), float(shift), misfit
        self.sea_level = self.L26 if sea_level is None else float(sea_level)
        self.shape, self.dtype = tuple(np.shape(F)), np.dtype(np.float64)

    def __getitem__(self, key):
        base = np.asarray(self.B[key], np.float64) < self.L26
        g = np.asarray(self.F[key], np.float64) - self.shift
        if self.misfit == "sea":
            g = np.where(base & ~(g < self.sea_level), MISFIT_SEA_VALUE, g)
        return np.where(base, g, np.inf)


# --- item 2 -------------------------------------------------------------------------------------

def lineage_table(tab, products, lineages=None, representative=None) -> tuple:
    """Columns of ``tab`` [rows, products] combined per lineage: the mean of the members'
    values, or with ``representative`` {lineage: product} that product's column alone.
    Returns (table [rows, lineages], lineage names)."""
    lineages = lineages or LINEAGES
    t = np.asarray(tab, float)
    idx = {p: i for i, p in enumerate(products)}
    cols, names = [], []
    for name, members in lineages.items():
        if representative and name in representative:
            cols.append(t[:, idx[representative[name]]])
        else:
            cols.append(t[:, [idx[m] for m in members]].mean(axis=1))
        names.append(name)
    return np.stack(cols, axis=1), names


def max_min_ratio(values) -> float:
    """Largest over smallest value; inf if the smallest is 0 and the largest positive,
    nan if all are 0 or any is not finite."""
    v = np.asarray(values, float)
    if not np.isfinite(v).all() or v.max() <= 0:
        return float("nan")
    return float(v.max() / v.min()) if v.min() > 0 else float("inf")
