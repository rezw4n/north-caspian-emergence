"""Revision analyses of near-term emergence (PREREG Amendment 18 item 10; D-123 to D-127).

Additive helpers for scripts/rev_n_*.py. Nothing here changes a registered result. The
conventions are those of caspian.exposure and caspian.hindcast_tiled: F is a flood-level
map (a cell is sea at level L when F < L, compared in float64); grids may be arrays,
memory maps or window-readable objects.

Contents
- surfaces: ``ShiftedSurface`` (F - c, the shifted products of D-123 (b)) and
  ``BaselineSurface`` (the common baseline of D-123 (a), location version);
- counts and curves: ``region_counts``, ``curve_from_counts``, ``isolated_from_counts``;
- envelopes (D-124): ``pair_counts``, ``scale_scores``, ``relative_envelope``,
  ``decline_envelope``;
- simulation (D-125): ``grf_exponential``, ``bilinear_to``, ``minimax_flood``,
  ``block_mean``, ``block_all``, ``block_region_counts``, ``agreement_ratio``;
- exposure (D-126): ``distance_masks``, ``port_type``;
- pathways (D-127): ``two_way_shares``, ``pathway_fall``, ``ols_rate``.
"""

from __future__ import annotations

import json
import math

import numpy as np

from . import hindcast_tiled as ht
from .exposure import below_counts, float32_below, known_masks


# --- surfaces ----------------------------------------------------------------------------------

class ShiftedSurface:
    """F - c, window by window (float64): the flood level of the product's heights less a
    constant c. A constant shift commutes with the minimax flood fill, so this is the
    flood-level map of the shifted heights (D-123 (b))."""

    def __init__(self, F, c: float):
        self.F, self.c = F, float(c)
        self.shape, self.dtype = tuple(np.shape(F)), np.dtype(np.float64)

    def __getitem__(self, key):
        return np.asarray(self.F[key], np.float64) - self.c


class BaselineSurface:
    """The common-baseline surface G of D-126: on the baseline set (``F_base < level_2026``)
    G = min(F, the largest float32 below level_2026), elsewhere +inf. So G < level_2026 is
    the baseline set, and for L <= level_2026, G < L is the baseline set intersected with
    F < L. ``F`` may be a ShiftedSurface."""

    def __init__(self, F_base, F, level_2026: float):
        self.Fb, self.F, self.L26 = F_base, F, float(level_2026)
        self.cap = float(float32_below(self.L26))
        self.shape, self.dtype = tuple(np.shape(F)), np.dtype(np.float64)

    def __getitem__(self, key):
        base = np.asarray(self.Fb[key], np.float64) < self.L26
        f = np.minimum(np.asarray(self.F[key], np.float64), self.cap)
        return np.where(base, f, np.inf)


# --- counts and curves ---------------------------------------------------------------------------

def region_counts(F, labels, n_regions: int, levels, *, mask=None, shape=None, tile: int = ht.TILE):
    """Cells with F < L per region (labels 1..n_regions) at every level of ``levels``
    (strictly increasing): int64 [n_regions, n_levels] (exposure.below_counts)."""
    below, _ = below_counts(F, labels, n_regions, levels, mask=mask, shape=shape, tile=tile)
    return below


def interp_counts(counts, levels, at) -> np.ndarray:
    """Counts (last axis over ``levels``) at the levels ``at``, by linear interpolation
    between grid levels (exact on grid levels)."""
    lv = np.asarray(levels, float)
    c = np.asarray(counts, float)
    at = np.atleast_1d(np.asarray(at, float))
    if at.min() < lv[0] - 1e-9 or at.max() > lv[-1] + 1e-9:
        raise ValueError("a requested level lies outside the level grid")
    flat = c.reshape(-1, c.shape[-1])
    out = np.stack([np.interp(at, lv, row) for row in flat])
    return out.reshape(c.shape[:-1] + (at.size,))


def curve_from_counts(counts, levels, baseline_level: float, deltas, *, baseline_counts=None) -> np.ndarray:
    """Emerged cells E(D) = A(baseline) - A(baseline_level - D) for each D of ``deltas``,
    from counts of sea cells on a level grid (last axis). ``baseline_counts`` (same leading
    shape) replaces A(baseline_level), e.g. the area of a common baseline (D-123 (a))."""
    d = np.asarray(deltas, float)
    a = interp_counts(counts, levels, baseline_level - d)
    base = (interp_counts(counts, levels, [baseline_level])[..., 0] if baseline_counts is None
            else np.asarray(baseline_counts, float))
    return base[..., None] - a


def isolated_from_counts(z_counts, f_counts, levels, at) -> np.ndarray:
    """Isolated-water cells at level L (z < L <= F) within a set: N(z < L) - N(F < L), from
    counts of z and of F below each grid level over the same set (z <= F cell by cell)."""
    return interp_counts(z_counts, levels, at) - interp_counts(f_counts, levels, at)


# --- envelopes (D-124) -----------------------------------------------------------------------------

def pair_counts(F, level_top: float, level_u: float, code_top, code_u, outside, finite, labels,
                n_regions: int, shifts=(0.0,), *, shape=None, tile: int = ht.TILE) -> dict:
    """Counts for one hindcast pair (anchor ``top``, unit u) over the cells known at both
    units with a finite surface and a region label 1..n_regions, per shift c (the surface
    F - c) and region: ``n_common``; observed water at the anchor and at u (``w_top``,
    ``w_u``) and at both (``w_both``); surface sea at each level (``p_top``: F - c <
    level_top; ``p_u``); and observed anchor water that is surface sea at u
    (``w_top_p_u``). Observed counts are [n_regions]; surface counts [n_shifts, n_regions].
    exposure.pair_region_counts gives n_common, w_top, w_u, p_top and p_u for c = 0."""
    shape = shape or ht._shape(F, code_top, code_u, outside, labels)
    shifts = [float(c) for c in shifts]
    nS, nR = len(shifts), n_regions + 1
    obs = {k: np.zeros(nR, np.int64) for k in ("n_common", "w_top", "w_u", "w_both")}
    sur = {k: np.zeros((nS, nR), np.int64) for k in ("p_top", "p_u", "w_top_p_u")}
    for r0, r1, c0, c1 in ht.tiles(shape, tile):
        st = np.asarray(ht._get(outside, r0, r1, c0, c1))
        kt, wt = known_masks(np.asarray(ht._get(code_top, r0, r1, c0, c1)), st)
        ku, wu = known_masks(np.asarray(ht._get(code_u, r0, r1, c0, c1)), st)
        lab = np.asarray(ht._get(labels, r0, r1, c0, c1)).astype(np.int64)
        common = kt & ku & np.asarray(ht._get(finite, r0, r1, c0, c1)).astype(bool) & (lab >= 1) & (lab <= n_regions)
        lc = lab[common]
        for k, m in (("n_common", None), ("w_top", wt), ("w_u", wu), ("w_both", wt & wu)):
            obs[k] += np.bincount(lc if m is None else lc[m[common]], minlength=nR)
        v = np.asarray(ht._get(F, r0, r1, c0, c1), np.float64)[common]
        wtc = wt[common]
        for i, c in enumerate(shifts):
            pt, pu = (v - c) < level_top, (v - c) < level_u
            sur["p_top"][i] += np.bincount(lc[pt], minlength=nR)
            sur["p_u"][i] += np.bincount(lc[pu], minlength=nR)
            sur["w_top_p_u"][i] += np.bincount(lc[pu & wtc], minlength=nR)
    out = {k: v[1:] for k, v in obs.items()}
    out.update({k: v[:, 1:] for k, v in sur.items()})
    return out


def scale_scores(errors, observed_change) -> np.ndarray:
    """|error| / observed change (D-124): +inf where the observed change is not positive
    (a relative error is undefined there; infinite scores are kept by the quantile)."""
    e = np.abs(np.asarray(errors, float))
    o = np.asarray(observed_change, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(o > 0, e / np.where(o > 0, o, 1.0), np.inf)
    return s


def relative_envelope(estimate, q: float) -> tuple:
    """The values E > 0 consistent with |estimate - E| <= q E: [estimate / (1 + q),
    estimate / (1 - q)] for q < 1; the upper bound is +inf for q >= 1, and q = +inf gives
    [0, +inf]. (nan, nan) for an estimate <= 0, where the inversion is not defined."""
    est = np.asarray(estimate, float)
    if not np.isfinite(q):
        lo, hi = np.zeros_like(est), np.full_like(est, np.inf)
    else:
        lo = est / (1.0 + q)
        hi = est / (1.0 - q) if q < 1 else np.full_like(est, np.inf)
    bad = ~(est > 0)
    lo, hi = np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)
    return (float(lo), float(hi)) if np.ndim(estimate) == 0 else (lo, hi)


def decline_envelope(estimate, q: float, delta) -> tuple:
    """estimate +- q * delta, the lower bound clipped at 0 (scores in km2 per m of decline)."""
    est, d = np.asarray(estimate, float), np.asarray(delta, float)
    lo, hi = np.maximum(est - q * d, 0.0), est + q * d
    return (float(lo), float(hi)) if np.ndim(estimate) == 0 and np.ndim(delta) == 0 else (lo, hi)


# --- simulation (D-125) ----------------------------------------------------------------------------

def _embedding_eigenvalues(ny: int, nx: int, dx: float, sill: float, practical_range: float):
    from scipy import fft
    my, mx = fft.next_fast_len(2 * ny), fft.next_fast_len(2 * nx)
    iy = np.minimum(np.arange(my), my - np.arange(my)) * dx
    ix = np.minimum(np.arange(mx), mx - np.arange(mx)) * dx
    h = np.hypot(iy[:, None], ix[None, :])
    c = sill * np.exp(-3.0 * h / practical_range)
    lam = np.real(fft.fft2(c))
    neg = float(-lam[lam < 0].sum() / np.abs(lam).sum())
    return np.clip(lam, 0.0, None), neg


def grf_exponential(ny: int, nx: int, dx: float, sill: float, practical_range: float, rng, n: int = 1):
    """``n`` stationary Gaussian fields on an ny x nx grid of spacing ``dx``, mean 0 and
    covariance sill * exp(-3 h / practical_range), by circulant embedding on a torus of at
    least twice the grid (negative eigenvalues set to 0). Returns (fields [n, ny, nx],
    share of the negative eigenvalue mass). Each complex draw gives two fields."""
    from scipy import fft
    lam, neg = _embedding_eigenvalues(ny, nx, dx, sill, practical_range)
    my, mx = lam.shape
    s = np.sqrt(lam / (my * mx))
    out = []
    while len(out) < n:
        eps = rng.standard_normal((my, mx)) + 1j * rng.standard_normal((my, mx))
        y = fft.fft2(s * eps)
        out.append(np.real(y)[:ny, :nx])
        if len(out) < n:
            out.append(np.imag(y)[:ny, :nx])
    return np.stack(out[:n]), neg


def bilinear_to(field, y0: float, x0: float, dx: float, ys, xs) -> np.ndarray:
    """Bilinear interpolation of a grid field (row i at y0 - i dx, column j at x0 + j dx;
    north-up) at the points (ys rows) x (xs columns): a [len(ys), len(xs)] array. Points
    beyond the grid take the nearest edge value."""
    from scipy.ndimage import map_coordinates
    fi = (y0 - np.asarray(ys, float)) / dx
    fj = (np.asarray(xs, float) - x0) / dx
    ii, jj = np.meshgrid(fi, fj, indexing="ij")
    return map_coordinates(np.asarray(field, float), [ii, jj], order=1, mode="nearest")


def minimax_flood(z, seed) -> np.ndarray:
    """Flood level of every cell of a whole (in-memory) grid: the minimax path height from
    the seed cells (8-connectivity), +inf where unreachable; NaN heights are walls. Same
    definition as hindcast_tiled.flood_level_map, by one grayscale reconstruction."""
    from skimage.morphology import reconstruction
    z = np.asarray(z, np.float64)
    mask = np.where(np.isnan(z), np.inf, z)
    s = np.asarray(seed, bool) & np.isfinite(z)
    marker = np.where(s, mask, np.inf)
    if not s.any():
        return np.full(z.shape, np.inf)
    return reconstruction(marker, mask, method="erosion", footprint=np.ones((3, 3), bool))


def _blocks(a, k: int, r0: int, r1: int):
    a = np.asarray(a[r0:r1])
    R, C = (r1 - r0) // k, a.shape[1] // k
    return a[:R * k, :C * k].reshape(R, k, C, k)


def block_mean(a, k: int, *, rows: int = 2048) -> np.ndarray:
    """Mean of the finite values in each k x k block (NaN if none); trailing rows and
    columns that do not fill a block are dropped."""
    H, W = np.shape(a)
    R, C = H // k, W // k
    out = np.full((R, C), np.nan, np.float64)
    step = max(k, rows - rows % k)
    for r0 in range(0, R * k, step):
        r1 = min(r0 + step, R * k)
        b = _blocks(a, k, r0, r1).astype(np.float64)
        n = np.isfinite(b).sum(axis=(1, 3))
        s = np.nansum(b, axis=(1, 3))
        out[r0 // k:r1 // k] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def block_all(a, k: int, *, rows: int = 2048) -> np.ndarray:
    """True where every cell of a k x k block is true."""
    H, W = np.shape(a)
    R, C = H // k, W // k
    out = np.zeros((R, C), bool)
    step = max(k, rows - rows % k)
    for r0 in range(0, R * k, step):
        r1 = min(r0 + step, R * k)
        out[r0 // k:r1 // k] = _blocks(a, k, r0, r1).astype(bool).all(axis=(1, 3))
    return out


def block_region_counts(labels, k: int, n_regions: int, *, mask=None, rows: int = 2048) -> np.ndarray:
    """Per k x k block, the number of cells of each region 1..n_regions (and of ``mask``
    if given): int32 [n_regions, R, C]."""
    H, W = np.shape(labels)
    R, C = H // k, W // k
    out = np.zeros((n_regions, R, C), np.int32)
    step = max(k, rows - rows % k)
    for r0 in range(0, R * k, step):
        r1 = min(r0 + step, R * k)
        lab = _blocks(labels, k, r0, r1)
        m = None if mask is None else _blocks(mask, k, r0, r1).astype(bool)
        for g in range(n_regions):
            sel = lab == g + 1
            if m is not None:
                sel = sel & m
            out[g, r0 // k:r1 // k] = sel.sum(axis=(1, 3))
    return out


def agreement_ratio(bands, weights) -> float:
    """Area where every band holds the cell over the area where any does (weights = cell
    areas or counts); nan if no band holds any cell."""
    b = np.asarray(bands, bool)
    w = np.asarray(weights, float)
    any_ = (b.any(axis=0) * w).sum()
    all_ = (b.all(axis=0) * w).sum()
    return float(all_ / any_) if any_ > 0 else float("nan")


# --- exposure (D-126) ------------------------------------------------------------------------------

def distance_masks(band, r_m: float, pixel: float, *, tile: int = 2048):
    """(dilated, eroded) copies of a boolean grid by a Euclidean distance r_m (m): dilated
    = cells within r_m of a band cell (centre to centre), eroded = band cells farther than
    r_m from every non-band cell. Tiled with a halo of ceil(r_m / pixel) + 1 cells, exact
    for that radius. Cells beyond the grid edge count as non-band."""
    from scipy.ndimage import distance_transform_edt
    band = np.asarray(band, bool)
    H, W = band.shape
    h = int(math.ceil(r_m / pixel)) + 1
    dil, ero = np.zeros((H, W), bool), np.zeros((H, W), bool)
    for r0, r1, c0, c1 in ht.tiles((H, W), tile):
        w0, w1, v0, v1 = max(r0 - h, 0), min(r1 + h, H), max(c0 - h, 0), min(c1 + h, W)
        b = band[w0:w1, v0:v1]
        core = (slice(r0 - w0, r1 - w0), slice(c0 - v0, c1 - v0))
        if b.any():
            d_out = distance_transform_edt(~b) * pixel
            dil[r0:r1, c0:c1] = (d_out <= r_m)[core]
        if b.any():
            pad = np.pad(b, 1, constant_values=False)        # beyond the grid: non-band
            d_in = distance_transform_edt(pad)[1:-1, 1:-1] * pixel
            ero[r0:r1, c0:c1] = (d_in > r_m)[core] & b[core]
    return dil, ero


def exposure_from_states(rule: str, samples, wet_ref, wet) -> "pd.DataFrame":
    """exposure.exposure_at for the "onshore", "length" and "area" rules from boolean sea
    states of each sample at the 2026 level (``wet_ref``) and at the scenario level
    (``wet``) instead of flood levels, e.g. for bands dilated or eroded by a distance
    (D-126). Same columns: fid, considered, changed, amount (count, km or km2)."""
    import pandas as pd
    if rule not in ("onshore", "length", "area"):
        raise ValueError("rule must be onshore, length or area")
    s = samples.assign(wet_ref=np.asarray(wet_ref, bool), wet=np.asarray(wet, bool))
    g = s.groupby("fid")
    wr, w = g["wet_ref"].any(), g["wet"].any()
    changed = wr & ~w
    if rule == "onshore":
        amount = changed.astype(float)
    else:
        em = s["wet_ref"] & ~s["wet"]
        amount = (s["weight"] * em).groupby(s["fid"]).sum() * (1e-3 if rule == "length" else 1e-6)
    return pd.DataFrame({"fid": wr.index, "considered": wr.to_numpy(), "changed": changed.to_numpy(),
                         "amount": amount.reindex(wr.index).to_numpy()}).reset_index(drop=True)


PORT_TAGS = (("industrial", "port"), ("landuse", "port"), ("seamark:type", "harbour"))


def port_type(tags) -> str:
    """"port_harbour" for OSM features tagged as a port or harbour (industrial=port,
    landuse=port, harbour=* other than "no", seamark:type=harbour), else "pier_landing"
    (piers, ferry terminals and the rest). ``tags``: a dict or its JSON text."""
    t = json.loads(tags) if isinstance(tags, str) else dict(tags or {})
    if any(t.get(k) == v for k, v in PORT_TAGS):
        return "port_harbour"
    if "harbour" in t and str(t["harbour"]).lower() != "no":
        return "port_harbour"
    return "pier_landing"


# --- pathways (D-127) ------------------------------------------------------------------------------

def two_way_shares(y) -> dict:
    """Shares of the total sum of squares of a two-way table without replication (rows:
    pathways, columns: products): row effect, column effect and the remainder
    (interaction). All zero shares (nan) when the table is constant."""
    y = np.asarray(y, float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("y must be a finite 2-D table")
    m = y.mean()
    ss_t = float(((y - m) ** 2).sum())
    ss_r = float(y.shape[1] * ((y.mean(axis=1) - m) ** 2).sum())
    ss_c = float(y.shape[0] * ((y.mean(axis=0) - m) ** 2).sum())
    ss_i = ss_t - ss_r - ss_c
    if ss_t <= 0:
        return {"ss_total": 0.0, "row": float("nan"), "column": float("nan"), "interaction": float("nan")}
    return {"ss_total": ss_t, "row": ss_r / ss_t, "column": ss_c / ss_t, "interaction": max(ss_i, 0.0) / ss_t}


def pathway_fall(curve, year: float, base_year: float = 2026.0) -> float:
    """Fall (m, positive down) of a piecewise-linear pathway (years, change) between
    ``base_year`` and ``year``: c(base) - c(year) (exposure.scenario_dates, anchor "base").
    nan outside the pathway's years."""
    yrs, ch = (np.asarray(v, float) for v in curve)
    if not (yrs[0] <= base_year <= yrs[-1] and yrs[0] <= year <= yrs[-1]):
        return float("nan")
    return float(np.interp(base_year, yrs, ch) - np.interp(year, yrs, ch))


def ols_rate(t_years, levels) -> dict:
    """Ordinary least-squares line of level (m) on decimal time (years): rate (m per year),
    its standard error, intercept at the mean time and n."""
    t = np.asarray(t_years, float)
    v = np.asarray(levels, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    n = int(t.size)
    if n < 3:
        raise ValueError("at least three records are needed")
    tm = t.mean()
    sxx = float(((t - tm) ** 2).sum())
    b = float(((t - tm) * (v - v.mean())).sum() / sxx)
    a = float(v.mean())
    resid = v - (a + b * (t - tm))
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))
    return {"rate_m_per_yr": b, "se_m_per_yr": se, "level_at_mean_time_m": a, "mean_time": float(tm), "n": n}


def decimal_year(times) -> np.ndarray:
    """Decimal years of datetimes (pandas-parsable)."""
    import pandas as pd
    t = pd.to_datetime(pd.Series(times))
    start = pd.to_datetime(t.dt.year.astype(str) + "-01-01")
    end = pd.to_datetime((t.dt.year + 1).astype(str) + "-01-01")
    return (t.dt.year + (t - start).dt.total_seconds() / (end - start).dt.total_seconds()).to_numpy(float)
