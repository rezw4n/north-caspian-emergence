"""A8 (H3): near-term seabed emergence and exposure (PREREG section 10; brief section 8 A8).

The functions here are applied by scripts/a8_emergence.py; the H3 decision rule is in
caspian.h3 and is never called by the driver. Everything is on the DAHITI working
datum of the hindcast (D-070): levels, product heights with the per-cell geoid
correction (the a7 "total" variant) and the waterline DEM. Grids are the 60 m
equal-area hindcast grid; inputs may be arrays, memory maps or window callables
(caspian.hindcast_tiled conventions), so the whole grid is never held in float64.

Definitions (the choices the pre-registration leaves open are marked "proposed"; they
are fixed before any hindcast or projection metric exists):

1. Vertical error learned in section 9 (proposed). For product p, region r (a sector
   inside the domain, or the domain) and hindcast unit u (epoch, level bin, level
   L_u), the area-matching shift ``delta`` is the constant c of ``C_GRID`` (0.01 m over
   +-5 m, as D-070) that minimises |area bias| of the flood fill at L_u + c over the
   unit's known cells in r. The bias is the section 9 total area bias, counted as in
   scripts/a7_hindcast.unit_counts (known: inside the valid mask with a known
   observed state, or outside it with a known outside state; finite product height).
   Ties go to the smallest |c|, then the smaller c (D-072). Positive delta: the
   product is too high (too shallow). The shift is "matched" when the bias changes
   sign or reaches zero on the grid (otherwise it lies at a grid end), and "usable"
   when it is matched and the region holds both known water and known land at the
   unit; only usable shifts enter the fit (a region that is all sea matches
   trivially).
2. Depth below the lowest observed level, measured on the product (proposed):
   d = L_min - z, with z the product height (working datum, before the correction) and
   L_min the lowest level-bin centre among the units the model is fitted on. For a
   unit, the product contour that matches the observed one lies at z_u = L_u +
   delta_u, so d_u = L_min - L_u - delta_u. Measured on the product, d exists below the
   observed range, where only the product does.
3. Error model (per sector, fitted on E2 and E3; PREREG section 10; form proposed):
   delta(d) = a + b * clip(d, d_lo, d_hi) ("linear_hold", primary): the Theil-Sen line
   through the usable units, held constant beyond the range of fitted depths
   [d_lo, d_hi], so it never extrapolates a trend. Sensitivities: "constant" (a =
   median delta, b = 0) and "linear" (no hold). Corrected height: z' = z - delta(L_min
   - z), which increases with z when 1 + b > 0; a fit with 1 + b < ``MIN_MONOTONE``
   falls back to "constant". A sector with fewer than ``MIN_UNITS`` usable units uses
   the model of the whole domain. Both fallbacks are recorded. Cells deeper than d_hi
   are flagged as extrapolation. The form is fixed here, before any result; the
   leave-one-epoch-out test reports its transfer and does not select it.
4. Fused surface (PREREG section 10): the waterline DEM inside the observed range,
   the corrected product below it. Inside the observed range means: a finite DEM
   value on a cell that is not known sea at the lowest observed level. Where the
   corrected product contradicts that lowest observed extent it is clamped: known sea
   to just below L_min, known land to L_min (the float32 values on either side of it;
   counted; proposed), so that the fused surface floods at L_min as observed.
5. Emerged area of a scenario (proposed): E_S(D) = A_S(L26) - A_S(L26 - D), the
   region's sea area (cells flooded from the open-sea seed) on surface S at the 2026
   level minus that at the scenario level, so each surface is referenced to its own
   2026 shoreline, as published projections are, and a constant datum offset of a
   product largely cancels (the conservative reading, as D-072 for H1). Sensitivity
   ("sea_area"): the same with the fused surface's A(L26) as the baseline for every
   surface.
6. Area intervals (PREREG section 10): split conformal on the absolute hindcast area
   errors at held-out observed levels (leave-one-epoch-out for the corrected
   product; every unit for the uncorrected product, which is not fitted). For the
   "emergence" functional the hindcast error of unit u is the error of the emerged
   area between the held-out epoch's highest level and L_u, over the cells known in
   both units; for "sea_area" it is the total area bias at L_u. One calibration per
   product, surface, region and functional; never summed over regions or pixels. The
   fused surface takes the corrected product's scores: below the observed range,
   where the scenarios emerge, it is the corrected product.
7. Exposure intervals (proposed): the bounds of the area interval are converted to
   the levels at which the surface's own emergence curve reaches them
   (``equivalent_levels``, rounded outward), and exposure is counted at those levels.
   Every exposure count here is monotone in the level, so the interval follows from
   the area interval without any per-pixel interval.
8. The 2026 level (proposed): the mean of the DAHITI records dated 2026
   (``reference_level``); sensitivity: the last value of the 3-record running median
   (the "present level" of D-040).
9. Scenario dates (proposed): each published pathway is linear between the points the
   paper states (its reference year at zero change, then each stated year), and the
   date of a scenario is when the pathway's own fall after 2026 reaches the scenario
   depth (``scenario_dates``, anchor "base"); never extrapolated beyond the last stated
   year. Sensitivity: the pathway anchored at the observed level of its reference year.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import conformal, design
from . import hindcast_tiled as ht

C_GRID = np.round(np.arange(-500, 501) / 100.0, 2)    # D-070 search grid (scripts/a7_hindcast.C_GRID)
FORMS = ("linear_hold", "constant", "linear")
PRIMARY_FORM = "linear_hold"
MIN_UNITS = 5             # proposed: matched units a sector needs for its own model
MIN_MONOTONE = 0.2        # proposed: 1 + b must reach this, else the constant model
ALPHA = round(1.0 - design.CONFORMAL_LEVEL, 10)
FUNCTIONALS = ("emergence", "sea_area")

# source codes of the fused surface
SRC_NONE, SRC_DEM, SRC_PRODUCT, SRC_EXTRAPOLATED = 0, 1, 2, 3


# --- counting on the grid -------------------------------------------------------------------

def _window(a, r0, r1, c0, c1):
    return np.asarray(ht._get(a, r0, r1, c0, c1))


def known_masks(code, outside):
    """(known, water) for a window of an observed unit: ``code`` as
    scripts/a7_hindcast.classify (1 sea, 0 land, -1 unknown, -2 outside the valid mask)
    and ``outside`` the outside states (1 water, 0 land, -1 unknown)."""
    c = np.asarray(code)
    st = np.asarray(outside)
    k_in, k_out = c >= 0, (c == -2) & (st >= 0)
    return k_in | k_out, ((c == 1) & k_in) | ((st == 1) & k_out)


def below_counts(F, labels, n_regions: int, thresholds, *, mask=None, shape=None, tile: int = ht.TILE):
    """Per region k = 1..n_regions (cells with ``labels`` == k, and ``mask`` if given):
    the number of cells with F below each threshold (F compared in float64, as the flood
    fill compares heights), an int64 array [n_regions, n_thresholds], and the number of
    cells in the region. NaN and +inf never count as below."""
    T = np.asarray(thresholds, float)
    if T.ndim != 1 or T.size == 0 or np.any(np.diff(T) <= 0):
        raise ValueError("thresholds must be a non-empty, strictly increasing 1-D array")
    shape = shape or ht._shape(F, labels, mask)
    nT, nR = T.size, n_regions + 1
    hist = np.zeros(nR * (nT + 1), np.int64)
    total = np.zeros(nR, np.int64)
    for r0, r1, c0, c1 in ht.tiles(shape, tile):
        lab = _window(labels, r0, r1, c0, c1).astype(np.int64)
        ok = (lab >= 1) & (lab <= n_regions)
        if mask is not None:
            ok &= _window(mask, r0, r1, c0, c1).astype(bool)
        lk = lab[ok]
        total += np.bincount(lk, minlength=nR)
        v = np.asarray(_window(F, r0, r1, c0, c1), float)[ok]
        idx = np.searchsorted(T, v, side="right")
        idx[~np.isfinite(v)] = nT
        hist += np.bincount(lk * (nT + 1) + idx, minlength=nR * (nT + 1))
    below = np.cumsum(hist.reshape(nR, nT + 1), axis=1)[:, :nT]
    return below[1:], total[1:]


def unit_region_counts(F, code, outside, finite, labels, n_regions: int, thresholds, *, mask=None, shape=None,
                       tile: int = ht.TILE) -> dict:
    """For one unit and product, per region: known cells, known water cells, and known
    cells with F below each threshold (the counts of a7_hindcast.unit_counts, for every
    region in one pass). ``below - water`` is the total area bias (cells) at each
    threshold. ``mask`` (optional) restricts the cells counted (e.g. training blocks)."""
    shape = shape or ht._shape(F, code, outside, labels)

    def base(r0, r1, c0, c1):
        m = _window(finite, r0, r1, c0, c1).astype(bool)
        if mask is not None:
            m &= _window(mask, r0, r1, c0, c1).astype(bool)
        return m

    def known(r0, r1, c0, c1):
        k, _ = known_masks(_window(code, r0, r1, c0, c1), _window(outside, r0, r1, c0, c1))
        return k & base(r0, r1, c0, c1)

    def water(r0, r1, c0, c1):
        _, w = known_masks(_window(code, r0, r1, c0, c1), _window(outside, r0, r1, c0, c1))
        return w & base(r0, r1, c0, c1)

    below, n_known = below_counts(F, labels, n_regions, thresholds, mask=known, shape=shape, tile=tile)
    _, n_water = below_counts(F, labels, n_regions, [0.0], mask=water, shape=shape, tile=tile)
    return {"below": below, "known": n_known, "water": n_water}


def pair_region_counts(F, level_top: float, level_u: float, code_top, code_u, outside, finite, labels,
                       n_regions: int, *, shape=None, tile: int = ht.TILE) -> dict:
    """Counts for the hindcast error of the emerged area between an anchor unit (the
    held-out epoch's highest level) and unit u, over the cells known in both units (and
    with a finite surface), per region: ``n_common``; observed water at the anchor and
    at u (``obs_top``, ``obs_u``); surface cells flooded at each level (``pred_top``,
    ``pred_u``; F < level). Emergence error (cells) = (pred_top - pred_u) - (obs_top -
    obs_u)."""
    shape = shape or ht._shape(F, code_top, code_u, outside, labels)
    keys = ("n_common", "obs_top", "obs_u", "pred_top", "pred_u")
    out = {k: np.zeros(n_regions + 1, np.int64) for k in keys}
    for r0, r1, c0, c1 in ht.tiles(shape, tile):
        st = _window(outside, r0, r1, c0, c1)
        kt, wt = known_masks(_window(code_top, r0, r1, c0, c1), st)
        ku, wu = known_masks(_window(code_u, r0, r1, c0, c1), st)
        lab = _window(labels, r0, r1, c0, c1).astype(np.int64)
        common = kt & ku & _window(finite, r0, r1, c0, c1).astype(bool) & (lab >= 1) & (lab <= n_regions)
        v = np.asarray(_window(F, r0, r1, c0, c1), float)
        for k, m in (("n_common", common), ("obs_top", common & wt), ("obs_u", common & wu),
                     ("pred_top", common & (v < level_top)), ("pred_u", common & (v < level_u))):
            out[k] += np.bincount(lab[m], minlength=n_regions + 1)
    return {k: v[1:] for k, v in out.items()}


# --- 1. area-matching shift ------------------------------------------------------------------

def matching_shift(bias, c_grid=C_GRID) -> dict:
    """The c of ``c_grid`` minimising |bias| (cells, non-decreasing in c); ties to the
    smallest |c|, then the smaller c (D-070, D-072). ``matched``: the bias reaches zero or
    changes sign on the grid."""
    b = np.asarray(bias, float)
    c = np.asarray(c_grid, float)
    if b.shape != c.shape:
        raise ValueError("bias and c_grid must have the same shape")
    a = np.abs(b)
    cand = np.flatnonzero(a == a.min())
    k = cand[np.lexsort((c[cand], np.abs(c[cand])))[0]]
    return {"shift_m": float(c[k]), "matched": bool(b.min() <= 0 <= b.max()),
            "residual_cells": float(b[k]), "n_tied": int(cand.size)}


# --- 2-3. error model ------------------------------------------------------------------------

def depth_below_lowest(z, lowest_level: float):
    """d = L_min - z (m): positive below the lowest observed level."""
    return float(lowest_level) - np.asarray(z, float)


def unit_depths(level, shift, lowest_level: float):
    """Depth below L_min, on the product, of the product contour that matches the
    observed one at ``level``: d = L_min - (level + shift)."""
    return float(lowest_level) - (np.asarray(level, float) + np.asarray(shift, float))


def fit_model(d, shift, form: str = PRIMARY_FORM, *, min_units: int = MIN_UNITS,
              min_monotone: float = MIN_MONOTONE) -> dict:
    """The error model delta(d) of the module docstring from matched units (depths
    ``d``, shifts ``shift``; NaN rows are dropped). Returns {form, form_requested, a, b,
    d_lo, d_hi, n, fallback} or None when no unit is left. ``fallback`` names why a
    linear form became constant ("monotone", "few_units", "no_depth_spread")."""
    if form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}")
    d = np.asarray(d, float)
    y = np.asarray(shift, float)
    ok = np.isfinite(d) & np.isfinite(y)
    d, y = d[ok], y[ok]
    if d.size == 0:
        return None
    rec = {"form_requested": form, "n": int(d.size), "d_lo": float(d.min()), "d_hi": float(d.max()),
           "fallback": None}
    if form != "constant":
        if d.size < max(min_units, 3):
            rec["fallback"] = "few_units"
        elif np.unique(d).size < 2:
            rec["fallback"] = "no_depth_spread"
        else:
            with warnings.catch_warnings():       # the slope CI (unused) can warn for small n
                warnings.simplefilter("ignore", RuntimeWarning)
                b, a, _, _ = stats.theilslopes(y, d, method="joint")
            if 1.0 + b < min_monotone:
                rec["fallback"] = "monotone"
            else:
                return {**rec, "form": form, "a": float(a), "b": float(b)}
    return {**rec, "form": "constant", "a": float(np.median(y)), "b": 0.0}


def predict_shift(model: dict, d):
    """delta(d) of a fitted model (m)."""
    d = np.asarray(d, float)
    if model["form"] == "constant":
        return np.full(d.shape, model["a"]) if d.ndim else np.float64(model["a"])
    if model["form"] == "linear_hold":
        d = np.clip(d, model["d_lo"], model["d_hi"])
    return model["a"] + model["b"] * d


def fit_models(shifts: pd.DataFrame, regions, *, epochs, form: str = PRIMARY_FORM,
               domain_region: str = "domain", min_units: int = MIN_UNITS) -> dict:
    """Per region, the error model fitted on the matched units of ``epochs``.

    ``shifts``: one row per unit and region with region, epoch, level_m, shift_m and
    usable (bool; else ``matched``), e.g. the shifts.csv table of scripts/a8_emergence.py
    for one product. L_min is the lowest level among the fit units (every region). A
    region with fewer than ``min_units`` usable units takes the model of
    ``domain_region`` (``used_domain`` True). Returns {"lowest_level_m", "epochs",
    "form", "models": {region: model}}."""
    s = shifts[shifts["epoch"].isin(list(epochs))]
    if s.empty:
        raise ValueError(f"no units of epochs {list(epochs)}")
    lowest = float(s["level_m"].min())
    m = s[s["usable" if "usable" in s.columns else "matched"].astype(bool)]

    def one(reg):
        r = m[m["region"] == reg]
        d = unit_depths(r["level_m"].to_numpy(float), r["shift_m"].to_numpy(float), lowest)
        return fit_model(d, r["shift_m"].to_numpy(float), form, min_units=min_units), int(len(r))

    dom, _ = one(domain_region)
    models = {}
    for reg in regions:
        mod, n = one(reg)
        if reg != domain_region and (mod is None or n < min_units):
            if dom is None:
                models[reg] = None
                continue
            models[reg] = {**dom, "used_domain": True, "n_own": n}
        else:
            models[reg] = None if mod is None else {**mod, "used_domain": False, "n_own": n}
    if domain_region not in models:
        models[domain_region] = None if dom is None else {**dom, "used_domain": False,
                                                          "n_own": int((m["region"] == domain_region).sum())}
    return {"lowest_level_m": lowest, "epochs": list(epochs), "form": form, "models": models}


class CorrectedSurface:
    """Product heights corrected by the per-sector error models, read window by window:
    z' = z - delta_s(L_min - z) for a cell of sector s (``labels`` == code s). Cells
    of a sector without a model, and NaN heights, are left as they are. ``extrapolated``
    gives, for a window, the cells deeper than the fitted range of their model (d >
    d_hi)."""

    def __init__(self, heights, labels, models: dict, lowest_level: float):
        if tuple(np.shape(heights)) != tuple(np.shape(labels)):
            raise ValueError("heights and labels must be on the same grid")
        self.h, self.lab, self.models, self.lowest = heights, labels, models, float(lowest_level)
        self.shape, self.dtype = tuple(np.shape(heights)), np.dtype(np.float32)

    def _parts(self, key):
        z = np.asarray(self.h[key], np.float32)
        lab = np.asarray(self.lab[key])
        return z, lab

    def __getitem__(self, key):
        z, lab = self._parts(key)
        out = z.copy()
        for k, m in self.models.items():
            if m is None:
                continue
            sel = (lab == k) & np.isfinite(z)
            if sel.any():
                d = depth_below_lowest(z[sel], self.lowest)
                out[sel] = (z[sel].astype(np.float64) - predict_shift(m, d)).astype(np.float32)
        return out

    def extrapolated(self, key):
        z, lab = self._parts(key)
        ex = np.zeros(z.shape, bool)
        for k, m in self.models.items():
            if m is None:
                continue
            sel = (lab == k) & np.isfinite(z)
            ex[sel] = depth_below_lowest(z[sel], self.lowest) > m["d_hi"]
        return ex


# --- 4. fused surface -------------------------------------------------------------------------

def float32_below(x: float) -> np.float32:
    """The largest float32 strictly below x (compared in float64)."""
    v = np.float32(x)
    while float(v) >= x:
        v = np.nextafter(v, np.float32(-np.inf))
    return v


def float32_at_or_above(x: float) -> np.float32:
    """The smallest float32 at or above x (compared in float64)."""
    v = np.float32(x)
    while float(v) < x:
        v = np.nextafter(v, np.float32(np.inf))
    return v


class FusedSurface:
    """PREREG section 10 fused surface, window by window (module docstring, item 4).

    ``dem``: the waterline DEM on the grid (NaN outside it); ``corrected``: a
    CorrectedSurface (or any window-readable surface); ``low_code`` and ``outside``: the
    observed unit at the lowest observed level (a7 classify codes) and the outside
    states, which give known sea and known land at that level. ``source`` codes each
    cell: SRC_DEM, SRC_PRODUCT, SRC_EXTRAPOLATED (corrected product deeper than its
    fitted range) or SRC_NONE (no value)."""

    def __init__(self, dem, corrected, low_code, outside, lowest_level: float):
        self.dem, self.corr, self.low, self.out = dem, corrected, low_code, outside
        self.lowest = float(lowest_level)
        self.below, self.at = float32_below(self.lowest), float32_at_or_above(self.lowest)
        self.shape, self.dtype = tuple(np.shape(dem)), np.dtype(np.float32)

    def _compute(self, key):
        dem = np.asarray(self.dem[key], np.float32)
        z = np.asarray(self.corr[key], np.float32)
        known, water = known_masks(np.asarray(self.low[key]), np.asarray(self.out[key]))
        sea, land = known & water, known & ~water
        use_dem = np.isfinite(dem) & ~sea
        out = np.where(use_dem, dem, z).astype(np.float32)
        prod = ~use_dem & np.isfinite(z)
        clamp_sea = prod & sea & (out.astype(np.float64) >= self.lowest)
        clamp_land = prod & land & (out.astype(np.float64) < self.lowest)
        out[clamp_sea] = self.below
        out[clamp_land] = self.at
        src = np.full(dem.shape, SRC_NONE, np.uint8)
        src[use_dem] = SRC_DEM
        src[prod] = SRC_PRODUCT
        if hasattr(self.corr, "extrapolated"):
            src[prod & self.corr.extrapolated(key)] = SRC_EXTRAPOLATED
        return out, src, clamp_sea, clamp_land

    def __getitem__(self, key):
        return self._compute(key)[0]

    def source(self, key):
        return self._compute(key)[1]

    def summary(self, labels, n_regions: int, *, tile: int = ht.TILE) -> pd.DataFrame:
        """Cells per region (labels 1..n) by source, and the clamped cells."""
        rows = np.zeros((n_regions + 1, 6), np.int64)
        for r0, r1, c0, c1 in ht.tiles(self.shape, tile):
            key = (slice(r0, r1), slice(c0, c1))
            _, src, cs, cl = self._compute(key)
            lab = np.asarray(ht._get(labels, r0, r1, c0, c1)).astype(np.int64)
            ok = (lab >= 1) & (lab <= n_regions)
            for j, m in enumerate((src == SRC_NONE, src == SRC_DEM, src == SRC_PRODUCT,
                                   src == SRC_EXTRAPOLATED, cs, cl)):
                rows[:, j] += np.bincount(lab[ok & m], minlength=n_regions + 1)
        return pd.DataFrame(rows[1:], columns=["n_none", "n_dem", "n_product", "n_extrapolated",
                                               "n_clamped_sea", "n_clamped_land"])


# --- 5-6. emerged area and its interval -------------------------------------------------------

def level_grid(level_2026: float, max_drop: float, *, above: float = 0.5, step: float = 0.01) -> np.ndarray:
    """Increasing levels from L26 - max_drop to L26 + above on a ``step`` grid that
    contains L26 exactly (rounded to 1e-6 m)."""
    n_lo, n_hi = int(round(max_drop / step)), int(round(above / step))
    return np.round(level_2026 + np.arange(-n_lo, n_hi + 1) * step, 6)


def emerged_curve(sea_cells, levels, level_2026: float, baseline_cells=None) -> np.ndarray:
    """Emerged cells E(L) = A(L26) - A(L) for every level of ``levels`` (increasing),
    from the sea cells A at those levels (one region). ``baseline_cells`` replaces
    A(L26) (the "sea_area" functional)."""
    lv = np.asarray(levels, float)
    k = np.flatnonzero(np.isclose(lv, level_2026, rtol=0, atol=1e-6))
    if k.size != 1:
        raise ValueError("the level grid must contain the 2026 level once")
    a = np.asarray(sea_cells, float)
    base = a[k[0]] if baseline_cells is None else float(baseline_cells)
    return base - a


def interval(estimate: float, scores, *, alpha: float = ALPHA, lower: float | None = 0.0,
             upper: float | None = None) -> dict:
    """Split-conformal interval estimate +- q, q the finite-sample quantile of the
    absolute scores (conformal.conformal_quantile; inf when there are too few), clipped
    to [lower, upper] when given (an emerged area lies between zero and the region's
    2026 sea area, so clipping keeps the coverage)."""
    s = np.abs(np.asarray(scores, float))
    s = s[~np.isnan(s)]
    n = int(s.size)
    q = conformal.conformal_quantile(s, alpha) if n else float("inf")
    lo, hi = estimate - q, estimate + q
    if lower is not None:
        lo = max(lo, lower)
    if upper is not None:
        hi = min(hi, upper)
    return {"estimate": float(estimate), "lo": float(lo), "hi": float(hi), "q": float(q), "n_scores": n,
            "alpha": float(alpha)}


def equivalent_levels(levels, emerged, e_lo: float, e_hi: float) -> dict:
    """Levels at which an emergence curve reaches the bounds of an area interval, rounded
    outward: ``level_lo_bound`` = the lowest grid level with E <= e_lo (no more emergence
    than the lower bound), ``level_hi_bound`` = the highest grid level with E >= e_hi.
    ``levels`` increasing, ``emerged`` non-increasing along them. A bound beyond the
    curve takes the grid end and is flagged."""
    lv = np.asarray(levels, float)
    e = np.asarray(emerged, float)
    if np.any(np.diff(lv) <= 0):
        raise ValueError("levels must increase")
    if np.any(np.diff(e) > 1e-9):
        raise ValueError("an emergence curve cannot increase with the level")
    lo_ok = np.flatnonzero(e <= e_lo)
    hi_ok = np.flatnonzero(e >= e_hi)
    out = {"level_lo_bound": float(lv[lo_ok[0]]) if lo_ok.size else float(lv[-1]),
           "lo_beyond_grid": not lo_ok.size,
           "level_hi_bound": float(lv[hi_ok[-1]]) if hi_ok.size else float(lv[0]),
           "hi_beyond_grid": not hi_ok.size}
    return out


# --- 7. exposure ------------------------------------------------------------------------------

def _cell(x, y, transform):
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    return np.floor((np.asarray(y, float) - f) / e).astype(np.int64), np.floor(
        (np.asarray(x, float) - c) / a).astype(np.int64)


def feature_samples(geoms, transform, shape, *, spacing: float | None = None) -> pd.DataFrame:
    """Sample points of features (shapely geometries in the grid CRS), one row per
    sample: fid (position in ``geoms``), row, col and weight. Points: their cell, weight 1.
    Lines: points every ``spacing`` m (default half a cell) at the middle of equal
    pieces, weight = piece length (m). Polygons: the cells whose centre lies inside,
    weight = cell area (m2); a polygon too small to contain a centre is its
    representative point, weighted by its area. Samples off the grid are dropped."""
    import shapely
    from affine import Affine
    from rasterio.features import geometry_mask
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    rows_n, cols_n = shape
    spacing = spacing or abs(a) / 2
    cell_area = abs(a * e)
    out = []
    for fid, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        kind = g.geom_type
        if kind in ("Point", "MultiPoint"):
            pts = shapely.get_coordinates(g)
            w = np.ones(len(pts))
        elif kind in ("LineString", "MultiLineString", "LinearRing"):
            parts = list(getattr(g, "geoms", [g]))
            pts, w = [], []
            for p in parts:
                L = p.length
                if L <= 0:
                    continue
                n = max(1, int(math.ceil(L / spacing)))
                s = (np.arange(n) + 0.5) * (L / n)
                pts.append(shapely.get_coordinates(shapely.line_interpolate_point(p, s)))
                w.append(np.full(n, L / n))
            if not pts:
                continue
            pts, w = np.vstack(pts), np.concatenate(w)
        elif kind in ("Polygon", "MultiPolygon"):
            x0, y0, x1, y1 = g.bounds
            j0, j1 = max(int(math.floor((x0 - c) / a)), 0), min(int(math.ceil((x1 - c) / a)), cols_n)
            i0, i1 = max(int(math.floor((y1 - f) / e)), 0), min(int(math.ceil((y0 - f) / e)), rows_n)
            ii = np.empty(0, np.int64)
            if j1 > j0 and i1 > i0:
                win = Affine(a, 0, c + a * j0, 0, e, f + e * i0)
                inside = geometry_mask([g], out_shape=(i1 - i0, j1 - j0), transform=win, invert=True)
                ii, jj = np.nonzero(inside)
            if ii.size:
                out.append(pd.DataFrame({"fid": fid, "row": ii + i0, "col": jj + j0, "weight": cell_area}))
                continue
            p = g.representative_point()
            pts, w = np.array([[p.x, p.y]]), np.array([g.area])
        else:
            raise ValueError(f"unsupported geometry type {kind}")
        r, cc = _cell(pts[:, 0], pts[:, 1], transform)
        out.append(pd.DataFrame({"fid": fid, "row": r, "col": cc, "weight": w}))
    if not out:
        return pd.DataFrame(columns=["fid", "row", "col", "weight"])
    s = pd.concat(out, ignore_index=True)
    on = (s["row"] >= 0) & (s["row"] < rows_n) & (s["col"] >= 0) & (s["col"] < cols_n)
    return s[on].reset_index(drop=True)


def _source_windows(r, bounds, block):
    """Blocks (rasterio Windows) of a source raster, limited to ``bounds`` (west, south,
    east, north in its CRS) when given."""
    from rasterio.windows import Window
    R0, C0, R1, C1 = 0, 0, r.height, r.width
    if bounds is not None:
        inv = ~r.transform
        cs, rs = zip(*(inv * (x, y) for x in (bounds[0], bounds[2]) for y in (bounds[1], bounds[3])))
        R0, R1 = max(int(np.floor(min(rs))), 0), min(int(np.ceil(max(rs))), r.height)
        C0, C1 = max(int(np.floor(min(cs))), 0), min(int(np.ceil(max(cs))), r.width)
    for r0 in range(R0, R1, block):
        for c0 in range(C0, C1, block):
            yield Window(c0, r0, min(block, C1 - c0), min(block, R1 - r0))


def raster_class_fraction(paths, classes, transform, shape, *, crs: str | None = None, factor: int = 1,
                          block: int = 4096, bounds=None, method: str = "auto") -> np.ndarray:
    """Share of each grid cell covered by ``classes`` in categorical rasters (e.g. a
    land-cover or wetland map), float32 on the grid; NaN where no valid source pixel.

    "bin" (sources finer than the grid): each source pixel, or each ``factor`` x
    ``factor`` group of them counted together, goes to the grid cell that contains its
    centre; the share is the class count over the valid count. Counts add across blocks
    and tiles, so the result does not depend on the blocking (``block`` is rounded down to
    a multiple of ``factor``). "sample" (sources coarser
    than the grid): each grid cell takes the class of the source pixel under its centre
    (share 0 or 1; the first valid source wins). "auto" picks "bin" when the source
    pixel, measured at the centre of the read area, is no larger than a grid cell.
    Source nodata pixels are not valid. ``crs``: the grid's CRS (default
    config.CRS_EQUAL_AREA). ``bounds`` (west, south, east, north in the source CRS)
    limits the source pixels read, e.g. to the domain of a global raster."""
    from pathlib import Path

    import pyproj
    import rasterio

    from .config import CRS_EQUAL_AREA
    if method not in ("auto", "bin", "sample"):
        raise ValueError("method must be auto, bin or sample")
    if factor < 1 or block < factor:
        raise ValueError("factor must be at least 1 and at most block")
    block -= block % factor                                  # whole groups in every block
    dst_crs = crs or CRS_EQUAL_AREA
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    rows_n, cols_n = shape
    cls = np.asarray(list(classes))
    num = np.zeros(rows_n * cols_n, np.float32)            # counts: exact in float32 up to 2**24
    den = np.zeros(rows_n * cols_n, np.float32)
    sampled = np.full(shape, np.nan, np.float32)
    for p in paths:
        with rasterio.open(p) as r:
            nod = r.nodata
            fwd = pyproj.Transformer.from_crs(r.crs, dst_crs, always_xy=True)
            w, s_, e_, n_ = (r.bounds if bounds is None else bounds)
            xc, yc = (w + e_) / 2, (s_ + n_) / 2
            x0, y0 = fwd.transform(xc, yc)
            x1, y1 = fwd.transform(xc + r.transform.a, yc)
            px = float(np.hypot(x1 - x0, y1 - y0))          # source pixel width (m) at the centre
            m = method if method != "auto" else ("bin" if px <= abs(a) else "sample")
            if m == "bin" and px * factor > abs(a):
                raise ValueError(f"{Path(p).name}: {factor} x {px:.0f} m groups are larger than a grid cell")
            if m == "sample":
                inv = pyproj.Transformer.from_crs(dst_crs, r.crs, always_xy=True)
                xs = c + a * (np.arange(cols_n) + 0.5)
                ys = f + e * (np.arange(rows_n) + 0.5)
                edge_x = np.r_[xs, xs, np.full(rows_n, xs[0]), np.full(rows_n, xs[-1])]
                edge_y = np.r_[np.full(cols_n, ys[0]), np.full(cols_n, ys[-1]), ys, ys]
                ex_, ey_ = inv.transform(edge_x, edge_y)
                gb = (np.min(ex_), np.min(ey_), np.max(ex_), np.max(ey_))
                if bounds is not None:
                    gb = (max(gb[0], bounds[0]), max(gb[1], bounds[1]), min(gb[2], bounds[2]), min(gb[3], bounds[3]))
                wins = list(_source_windows(r, gb, max(r.width, r.height)))
                if not wins:
                    continue
                win = wins[0]
                band = r.read(1, window=win)
                rinv = ~r.window_transform(win)
                for i in range(rows_n):
                    todo = np.isnan(sampled[i])
                    if not todo.any():
                        continue
                    lon, lat = inv.transform(xs[todo], np.full(todo.sum(), f + e * (i + 0.5)))
                    cc, rr = rinv * (np.asarray(lon), np.asarray(lat))
                    cc, rr = np.floor(cc).astype(np.int64), np.floor(rr).astype(np.int64)
                    ok = (rr >= 0) & (rr < band.shape[0]) & (cc >= 0) & (cc < band.shape[1])
                    v = np.full(todo.sum(), np.nan, np.float32)
                    val = band[rr[ok], cc[ok]]
                    good = val != nod if nod is not None else np.ones(val.size, bool)
                    vv = np.where(np.isin(val, cls), 1.0, 0.0).astype(np.float32)
                    vv[~good] = np.nan
                    v[ok] = vv
                    row = sampled[i]
                    row[todo] = v
                continue
            for win in _source_windows(r, bounds, block):
                arr = r.read(1, window=win)
                valid = np.ones(arr.shape, bool) if nod is None else arr != nod
                hit = np.isin(arr, cls) & valid
                h, w_ = arr.shape
                hh, ww = -(-h // factor) * factor, -(-w_ // factor) * factor
                pads = ((0, hh - h), (0, ww - w_))
                nh, nv = (np.pad(mm, pads).reshape(hh // factor, factor, ww // factor, factor).sum(axis=(1, 3))
                          for mm in (hit, valid))
                keep = nv > 0
                if not keep.any():
                    continue
                ii, jj = np.nonzero(keep)
                st = r.window_transform(win)
                lon = st.c + st.a * (jj * factor + np.minimum(factor, w_ - jj * factor) / 2.0)
                lat = st.f + st.e * (ii * factor + np.minimum(factor, h - ii * factor) / 2.0)
                x, y = fwd.transform(lon, lat)
                col = np.floor((np.asarray(x) - c) / a).astype(np.int64)
                row = np.floor((np.asarray(y) - f) / e).astype(np.int64)
                on = (row >= 0) & (row < rows_n) & (col >= 0) & (col < cols_n)
                u, inv_k = np.unique(row[on] * cols_n + col[on], return_inverse=True)
                num[u] += np.bincount(inv_k, weights=nh[ii[on], jj[on]])
                den[u] += np.bincount(inv_k, weights=nv[ii[on], jj[on]])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(den > 0, num / den, np.nan).reshape(shape).astype(np.float32)
    fill = np.isnan(frac) & ~np.isnan(sampled)
    frac[fill] = sampled[fill]
    return frac


def fraction_samples(fraction, transform, *, min_fraction: float = 0.0) -> pd.DataFrame:
    """Samples of a habitat share raster for exposure_at's "area" rule: one sample per
    cell with share above ``min_fraction``, weight = share x cell area (m2), and features
    = the 8-connected patches of such cells (fid = patch label - 1)."""
    from scipy import ndimage
    f = np.nan_to_num(np.asarray(fraction, float), nan=0.0)
    m = f > min_fraction
    lab, _ = ndimage.label(m, structure=np.ones((3, 3), bool))
    rr, cc = np.nonzero(m)
    a, e = float(transform[0]), float(transform[4])
    return pd.DataFrame({"fid": lab[rr, cc] - 1, "row": rr, "col": cc, "weight": f[rr, cc] * abs(a * e)})


def sample_values(F, rows, cols) -> np.ndarray:
    """F at sample cells (float64), reading each sample's row of the grid once."""
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    out = np.empty(rows.size)
    for r in np.unique(rows):
        k = rows == r
        out[k] = np.asarray(F[int(r):int(r) + 1, :], float)[0, cols[k]]
    return out


def distance_profiles(F, rows, cols, pixel: float, radius_m: float) -> list:
    """For each sample cell, the distance to the sea as a step function of the level,
    within ``radius_m``: arrays (f, dist) with f increasing and dist decreasing such that
    the nearest cell with F < L lies dist[k] away for the largest k with f[k] < L
    (beyond ``radius_m`` when no such k). Distances are between cell centres, in m."""
    rows_n, cols_n = np.shape(F)
    R = int(math.ceil(radius_m / pixel))
    out = []
    for r, c in zip(np.asarray(rows, np.int64), np.asarray(cols, np.int64)):
        r0, r1, c0, c1 = max(r - R, 0), min(r + R + 1, rows_n), max(c - R, 0), min(c + R + 1, cols_n)
        v = np.asarray(F[r0:r1, c0:c1], float)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        d = np.hypot(rr - r, cc - c) * pixel
        ok = np.isfinite(v) & (d <= radius_m)
        v, d = v[ok], d[ok]
        if v.size == 0:
            out.append((np.empty(0), np.empty(0)))
            continue
        o = np.lexsort((d, v))
        v, d = v[o], d[o]
        cm = np.minimum.accumulate(d)
        keep = np.r_[True, cm[1:] < cm[:-1]]
        out.append((v[keep], cm[keep]))
    return out


def distance_at(profiles, level: float) -> np.ndarray:
    """Distance to the sea at ``level`` for each profile of ``distance_profiles``; inf
    beyond the profile's radius."""
    res = np.full(len(profiles), np.inf)
    for i, (f, d) in enumerate(profiles):
        k = np.searchsorted(f, level, side="left") - 1      # largest k with f[k] < level
        if k >= 0:
            res[i] = d[k]
    return res


RULES = {
    # layer kind -> how a feature changes status between L26 and a scenario level
    "onshore": "point features in the sea at L26 that are no longer in the sea (count)",
    "length": "line length in the sea at L26 that is land at the level (km), and features left fully dry",
    "area": "polygon area in the sea at L26 that is land at the level (km2), and features left fully dry",
    "access": "features with the sea within the threshold at L26 and beyond it at the level (count)",
}


def exposure_at(rule: str, samples: pd.DataFrame, f_values, level_2026: float, level: float, *,
                profiles=None, threshold_m: float | None = None) -> pd.DataFrame:
    """Per feature (fid), its status change between L26 and ``level`` under ``rule``
    (``RULES``). ``f_values``: the surface's flood level at each sample (F < L: sea at
    L). For "access", ``profiles`` (distance_profiles of the samples) and
    ``threshold_m``. Columns: fid, considered (bool: wet or within the threshold at
    L26), changed (bool), amount (km or km2 for length/area; 1 for a change otherwise),
    and for access the distances at both levels (m)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {sorted(RULES)}")
    if level > level_2026:
        raise ValueError("the scenario level must not exceed the 2026 level")
    fv = np.asarray(f_values, float)
    s = samples.assign(wet_ref=fv < level_2026, wet=fv < level)
    if rule == "access":
        if profiles is None or threshold_m is None:
            raise ValueError("access needs distance profiles and a threshold")
        s = s.assign(d_ref=distance_at(profiles, level_2026), d=distance_at(profiles, level))
        g = s.groupby("fid").agg(d_ref=("d_ref", "min"), d=("d", "min"))
        considered = g["d_ref"] <= threshold_m
        changed = considered & (g["d"] > threshold_m)
        return pd.DataFrame({"fid": g.index, "considered": considered.to_numpy(), "changed": changed.to_numpy(),
                             "amount": changed.astype(float).to_numpy(), "d_ref_m": g["d_ref"].to_numpy(),
                             "d_m": g["d"].to_numpy()}).reset_index(drop=True)
    g = s.groupby("fid")
    wet_ref, wet = g["wet_ref"].any(), g["wet"].any()
    changed = wet_ref & ~wet
    if rule == "onshore":
        amount = changed.astype(float)
    else:
        em = s["wet_ref"] & ~s["wet"]
        scale = 1e-3 if rule == "length" else 1e-6
        amount = (s["weight"] * em).groupby(s["fid"]).sum() * scale
    return pd.DataFrame({"fid": wet_ref.index, "considered": wet_ref.to_numpy(), "changed": changed.to_numpy(),
                         "amount": amount.reindex(wet_ref.index).to_numpy()}).reset_index(drop=True)


def summarise_exposure(per_feature: pd.DataFrame, rule: str) -> dict:
    """Region totals of exposure_at: n_considered, n_changed and, for length and area,
    the emerged amount (km or km2) and the features with any emerged part; for access,
    the median increase of the distance to the sea among the considered features
    (those still within the profile radius) and how many are beyond the radius."""
    out = {"n_considered": int(per_feature["considered"].sum()), "n_changed": int(per_feature["changed"].sum())}
    if rule in ("length", "area"):
        out["amount"] = float(per_feature["amount"].sum())
        out["n_any"] = int((per_feature["amount"] > 0).sum())
    else:
        out["amount"] = float(out["n_changed"])
    if rule == "access":
        c = per_feature[per_feature["considered"]]
        inc = (c["d_m"] - c["d_ref_m"]).to_numpy(float)
        fin = np.isfinite(inc)
        out["median_distance_increase_m"] = float(np.median(inc[fin])) if fin.any() else float("nan")
        out["n_beyond_radius"] = int((~np.isfinite(c["d_m"].to_numpy(float))).sum())
    return out


def feature_regions(samples: pd.DataFrame, labels) -> pd.Series:
    """Region code of each feature: the label at its samples, the most common by weight
    (ties to the smaller code)."""
    lab = sample_values(labels, samples["row"], samples["col"]).astype(np.int64)
    s = samples.assign(region=lab)
    w = s.groupby(["fid", "region"])["weight"].sum().reset_index()
    w = w.sort_values(["fid", "weight", "region"], ascending=[True, False, True])
    return w.drop_duplicates("fid").set_index("fid")["region"]


# --- scenario dates -------------------------------------------------------------------------

def pathway_curves(pathways: pd.DataFrame) -> dict:
    """{pathway_id: (years, change)} with change (m, negative = fall) relative to the
    pathway's reference year, as piecewise-linear knots: (ref_year, 0) and each row's
    (year, value_m). ``pathways``: rows with pathway_id, kind ("projection" or
    "observed_trend": value_m = change by ``year``; "rate": value_m in m per year from
    ref_year to ``year``), ref_year and year (numbers). Rows of one pathway must share
    the reference year."""
    out = {}
    for pid, g in pathways.groupby("pathway_id", sort=True):
        ref = g["ref_year"].astype(float).unique()
        if ref.size != 1:
            raise ValueError(f"pathway {pid}: rows have different reference years")
        ref = float(ref[0])
        yrs, ch = [ref], [0.0]
        for _, r in g.sort_values("year").iterrows():
            y = float(r["year"])
            if y <= ref:
                raise ValueError(f"pathway {pid}: year {y} not after the reference year {ref}")
            v = float(r["value_m"]) * (y - ref) if r["kind"] == "rate" else float(r["value_m"])
            yrs.append(y)
            ch.append(v)
        yrs, ch = np.asarray(yrs), np.asarray(ch)
        if np.any(np.diff(yrs) <= 0):
            raise ValueError(f"pathway {pid}: duplicate years")
        out[pid] = (yrs, ch)
    return out


def _first_reach(t, g, target: float, t_start: float):
    """First t >= t_start at which the piecewise-linear g (knots t increasing, values g)
    reaches ``target`` from below (g(t) >= target); None if it never does by t[-1]."""
    t, g = np.asarray(t, float), np.asarray(g, float)
    if t_start > t[-1]:
        return None
    gs = float(np.interp(t_start, t, g))
    if gs >= target:
        return float(t_start)
    k = np.flatnonzero(t > t_start)
    tk, gk = np.r_[t_start, t[k]], np.r_[gs, g[k]]
    for i in range(tk.size - 1):
        if gk[i + 1] >= target:
            return float(tk[i] + (target - gk[i]) / (gk[i + 1] - gk[i]) * (tk[i + 1] - tk[i]))
    return None


def trend_plus_quadratic_curve(climate_part_m: float, trend_m_per_yr: float, ref_year: float, end_year: float,
                               step: float = 1.0):
    """(years, change) knots of a fall made of a linear trend and a climate part that grows
    with the square of the elapsed time and reaches ``climate_part_m`` at ``end_year``:
    c(t) = trend (t - ref) + climate_part ((t - ref) / (end - ref))^2. This is the time path
    implied by integrating a P - E anomaly that grows linearly from zero (Samant & Prange
    2023, Eq. 1, where the change over the window is half the trend times the window),
    with the observed rate they add on top; knots every ``step`` years, linear between."""
    t = np.arange(ref_year, end_year + 1e-9, step)
    if t[-1] < end_year:
        t = np.r_[t, end_year]
    x = (t - ref_year) / (end_year - ref_year)
    return t, trend_m_per_yr * (t - ref_year) + climate_part_m * x ** 2


def scenario_dates(curves: dict, deltas, *, base_year: float = 2026.0, anchor: str = "base",
                   level_2026: float | None = None, ref_levels: dict | None = None) -> pd.DataFrame:
    """The date (decimal year) at which each pathway brings the sea D metres below the
    2026 level, by linear interpolation between its knots; NaN with ``reached`` False if
    not within the pathway's horizon (never extrapolated beyond its last year).

    anchor "base" (proposed default): the pathway's own change after ``base_year``, i.e.
    the fall c(base_year) - c(t) measured from the pathway's value at the base year (the
    base year must lie inside the pathway), so the observed level of 2026 is the start
    whatever the pathway assumed for 2020-2026. anchor "reference": the level path
    ref_levels[pid] + c(t) (the observed level in the pathway's reference year, on the
    datum of ``level_2026``) reaching level_2026 - D; a date before base_year is reported
    as "already_passed"."""
    if anchor not in ("base", "reference"):
        raise ValueError('anchor must be "base" or "reference"')
    rows = []
    for pid, (yrs, ch) in curves.items():
        yrs, ch = np.asarray(yrs, float), np.asarray(ch, float)
        for D in deltas:
            rec = {"pathway_id": pid, "delta_m": float(D), "horizon_year": float(yrs[-1]), "anchor": anchor}
            if anchor == "base":
                if not yrs[0] <= base_year <= yrs[-1]:
                    rows.append({**rec, "date": np.nan, "reached": False, "note": "base year outside the pathway"})
                    continue
                c0 = float(np.interp(base_year, yrs, ch))
                t = _first_reach(yrs, c0 - ch, float(D), base_year)     # fall since base_year
            else:
                if level_2026 is None or ref_levels is None or pid not in ref_levels:
                    raise ValueError("anchor 'reference' needs level_2026 and ref_levels for every pathway")
                t = _first_reach(yrs, -ch, float(ref_levels[pid]) - (level_2026 - float(D)), float(yrs[0]))
            if t is None:
                rows.append({**rec, "date": np.nan, "reached": False, "note": f"not reached by {yrs[-1]:.0f}"})
                continue
            note = "already_passed" if anchor == "reference" and t < base_year else ""
            rows.append({**rec, "date": round(t, 3), "reached": True, "note": note})
    return pd.DataFrame(rows, columns=["pathway_id", "delta_m", "horizon_year", "anchor", "date", "reached", "note"])


def reference_level(times, levels_m, year: int = 2026, method: str = "annual_mean") -> dict:
    """The "2026 level" (proposed): the mean of the level records dated in ``year``
    ("annual_mean"), or the last value of the 3-record running median, as D-040 took
    the present level ("latest_running_median")."""
    t = pd.to_datetime(pd.Series(times))
    v = np.asarray(levels_m, float)
    if method == "annual_mean":
        k = (t.dt.year == year).to_numpy()
        if not k.any():
            raise ValueError(f"no level record in {year}")
        return {"level_m": float(np.mean(v[k])), "method": method, "n_records": int(k.sum()),
                "first": str(t[k].min().date()), "last": str(t[k].max().date())}
    if method == "latest_running_median":
        from .levels import running_median
        o = np.argsort(t.to_numpy())
        rm = running_median(v[o], 3)
        return {"level_m": float(rm[-1]), "method": method, "n_records": int(v.size),
                "date": str(t.iloc[o[-1]].date())}
    raise ValueError("method must be annual_mean or latest_running_median")


# --- helpers for the driver -------------------------------------------------------------------

def sea_cells_by_region(F, labels, n_regions: int, levels, *, mask=None, shape=None, tile: int = ht.TILE):
    """Sea cells (F < L) per region at each level: [n_regions, n_levels]."""
    below, _ = below_counts(F, labels, n_regions, levels, mask=mask, shape=shape, tile=tile)
    return below


def shift_table(counts: dict, level: float, region_names, thresholds_c=C_GRID) -> pd.DataFrame:
    """Per region (and the sum over regions, "domain"): the matching shift of a unit from
    unit_region_counts (thresholds level + C_GRID), the area bias at c = 0 (cells), the
    counts, and ``usable`` (matched, and the region holds both known water and known
    land, so the shift is informative; a region that is all water or all land matches
    trivially)."""
    below, water, known = counts["below"], counts["water"], counts["known"]
    rows = []
    allr = [(n, below[i], water[i], known[i]) for i, n in enumerate(region_names)]
    allr.append(("domain", below.sum(axis=0), water.sum(), known.sum()))
    k0 = int(np.flatnonzero(np.asarray(thresholds_c) == 0)[0])
    for name, b, w, n in allr:
        ms = matching_shift(b - w, thresholds_c)
        rows.append({"region": name, "level_m": level, "n_known": int(n), "n_water": int(w),
                     "bias0_cells": int(b[k0] - w), **ms,
                     # informative: matched, with known water and known land in the region
                     "usable": bool(ms["matched"] and 0 < w < n)})
    return pd.DataFrame(rows)

