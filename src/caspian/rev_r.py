"""Revision workstream R (PREREG Amendment 18 items 3, 6 and 7; D-116 to D-119): the
raster-side analyses of the H1 hindcast after the internal review.

Functions here are pure (arrays in, arrays out) and tested in tests/test_rev_r.py.
The drivers are scripts/rev_r_*.py.

1. "Perfect but degraded" control (A18 item 6; D-116). The waterline DEM is degraded the
   way a product stores the seabed: put into the product's vertical frame, averaged to
   the product's native cell, rounded to whole metres where the product stores whole
   metres, and sampled back to the 60 m grid as caspian.products samples the products.
   - ``native_cell_index``: the native cell whose centre is nearest a point (the cell
     that contains it), in the product's own lon/lat index space.
   - ``NativeMean``: running sums and counts per native cell, so the 60 m grid can be
     streamed in row blocks.
   - ``round_half_up``: whole metres where a mask says so.
   - ``fill_nearest``: nearest finite value within a halo, tile by tile.
2. Excess rule and stop rule (A18 item 6): ``excess_rule``, ``stop_rule``.

Later sections (fast waterline displacement, majority aggregation, gradients, the error
budget) are added below with their own docstrings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

# --- 1. degrading the waterline DEM to a product's grid --------------------------------------


def native_cell_index(lon, lat, native_lon, native_lat):
    """(i, j) of the native cell that contains each point: the nearest native cell centre
    along each axis (centres ``native_lat`` and ``native_lon``, regular and ascending), as
    caspian.products' nearest-neighbour rule. -1 where the point lies more than half a cell
    beyond the outermost centres."""
    lon0, dlon = float(native_lon[0]), float((native_lon[-1] - native_lon[0]) / (len(native_lon) - 1))
    lat0, dlat = float(native_lat[0]), float((native_lat[-1] - native_lat[0]) / (len(native_lat) - 1))
    j = np.floor((np.asarray(lon, float) - lon0) / dlon + 0.5).astype(np.int64)
    i = np.floor((np.asarray(lat, float) - lat0) / dlat + 0.5).astype(np.int64)
    bad = (i < 0) | (i >= len(native_lat)) | (j < 0) | (j >= len(native_lon))
    i[bad] = -1
    j[bad] = -1
    return i, j


class NativeMean:
    """Mean of fine-grid values per native cell, accumulated block by block. Only finite
    values count; a native cell with none is NaN."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.sum = np.zeros(self.shape, np.float64)
        self.n = np.zeros(self.shape, np.int64)

    def add(self, values, i, j):
        v = np.asarray(values, float).ravel()
        i, j = np.asarray(i).ravel(), np.asarray(j).ravel()
        ok = np.isfinite(v) & (i >= 0) & (j >= 0)
        flat = i[ok] * self.shape[1] + j[ok]
        size = self.shape[0] * self.shape[1]
        self.sum += np.bincount(flat, weights=v[ok], minlength=size).reshape(self.shape)
        self.n += np.bincount(flat, minlength=size).reshape(self.shape)

    def mean(self) -> np.ndarray:
        out = np.full(self.shape, np.nan)
        m = self.n > 0
        out[m] = self.sum[m] / self.n[m]
        return out


def round_half_up(z, mask=None) -> np.ndarray:
    """Whole metres (halves rounded up, floor(z + 0.5)) where ``mask`` is True (everywhere
    if None); NaN stays NaN."""
    z = np.asarray(z, float)
    r = np.floor(z + 0.5)
    if mask is None:
        return r
    return np.where(np.asarray(mask, bool), r, z)


def whole_metre_mask(z, tol: float = 0.01) -> np.ndarray:
    """Cells whose stored value lies within ``tol`` m of a whole metre (NaN: False)."""
    z = np.asarray(z, float)
    return np.isfinite(z) & (np.abs(z - np.round(z)) <= tol)


def fill_nearest(values, where, *, tile: int = 2048, halo: int = 256) -> np.ndarray:
    """Copy of ``values`` (2-D) in which each cell of ``where`` that is not finite takes
    the value of the nearest finite cell within ``halo`` cells (Euclidean, on a window of
    the tile plus the halo; ties as scipy's distance transform resolves them). Cells with
    no finite value within the halo stay NaN."""
    v = np.asarray(values)
    out = np.array(v, dtype=np.float32, copy=True)
    W = np.asarray(where, bool)
    H, Wd = v.shape
    for r0 in range(0, H, tile):
        for c0 in range(0, Wd, tile):
            r1, c1 = min(r0 + tile, H), min(c0 + tile, Wd)
            need = W[r0:r1, c0:c1] & ~np.isfinite(v[r0:r1, c0:c1])
            if not need.any():
                continue
            w0, w1, v0, v1 = max(r0 - halo, 0), min(r1 + halo, H), max(c0 - halo, 0), min(c1 + halo, Wd)
            win = np.asarray(v[w0:w1, v0:v1], float)
            fin = np.isfinite(win)
            if not fin.any():
                continue
            dist, (ii, jj) = ndimage.distance_transform_edt(~fin, return_indices=True)
            core = (slice(r0 - w0, r1 - w0), slice(c0 - v0, c1 - v0))
            near = win[ii[core], jj[core]]
            ok = need & (dist[core] <= halo)
            blk = out[r0:r1, c0:c1]
            blk[ok] = near[ok]
    return out


# --- 2. excess rule and stop rule (A18 item 6) ------------------------------------------------


def excess_rule(levels: pd.DataFrame) -> pd.DataFrame:
    """A18 item 6 excess rule, per product and sector.

    ``levels``: one row per product, sector and level with ``D`` (the product's median
    displacement, m; inf allowed), ``Q`` (the floor's 95th percentile), ``D_control`` (the
    control's median displacement) and ``evaluable`` (True where the level is evaluable
    for both the product and the control, and any further restriction applies).

    A level exceeds when D > max(Q, D_control). The product's error in the sector is
    attributed to more than quantisation and resolution (``excess`` True) when the
    exceeding levels are more than half of the evaluable levels; NA when none is
    evaluable."""
    d = levels.copy()
    ev = d["evaluable"].astype(bool)
    ref = np.maximum(d["Q"].astype(float), d["D_control"].astype(float))
    d["exceeds"] = ev & (d["D"].astype(float) > ref)
    g = d.groupby(["product", "sector"], sort=True)
    out = pd.DataFrame({"n_evaluable": g["evaluable"].apply(lambda x: int(x.astype(bool).sum())),
                        "n_exceeding": g["exceeds"].sum().astype(int)}).reset_index()
    out["excess"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    m = out["n_evaluable"] > 0
    out.loc[m, "excess"] = out.loc[m, "n_exceeding"] > out.loc[m, "n_evaluable"] / 2
    return out


def stop_rule(excess: pd.DataFrame, products, sectors) -> dict:
    """A18 item 6 stop rule: over the product x sector cases given, the number in which
    the excess rule fails (False or not evaluable); the rule fires when that number is
    more than half of the cases."""
    sub = excess[excess["product"].isin(list(products)) & excess["sector"].isin(list(sectors))]
    n_cases = len(list(products)) * len(list(sectors))
    holds = sub["excess"].fillna(False).astype(bool)
    n_fail = n_cases - int(holds.sum())
    return {"n_cases": n_cases, "n_rows": int(len(sub)), "n_excess_holds": int(holds.sum()),
            "n_excess_fails": int(n_fail), "fires": bool(n_fail > n_cases / 2),
            "fails": sorted(f"{r.product}|{r.sector}" for r in sub[~holds].itertuples())}


# --- 3. waterline cells built once and reused (A18 items 3 and 6) -----------------------------
#
# caspian.hindcast_tiled.displacement_cells extracts both waterlines and their flags in one
# pass over the grid. The revision runs need the same displacement for many thresholds of
# one flood-level map (the constant search), for observed waterlines with some cells removed
# (the static-contour filter), and for aggregated observed extents. The functions below
# build the same ``cells`` dictionary in separate steps:
#   - ``obs_waterline`` and ``pred_waterline``: the waterline cells of an observed extent
#     (codes as scripts/a7_hindcast.classify returns them, with the outside states) and of a
#     flood-level map at a threshold, by the rules of caspian.hindcast.known_states and
#     waterline (4-neighbourhood, cells beyond the grid unknown);
#   - ``cells_from``: the other extent's known state and water at each waterline cell, and
#     the sector labels;
# and then hindcast_tiled._distances and hindcast_tiled.summarise do the rest, unchanged.
# tests/test_rev_r.py checks the result against hindcast_tiled.displacement_cells and
# noise_floor_cells.

_FOUR = ndimage.generate_binary_structure(2, 1)


def _win(shape, r0, r1, c0, c1, halo=1):
    rows, cols = shape
    w0, w1, v0, v1 = max(r0 - halo, 0), min(r1 + halo, rows), max(c0 - halo, 0), min(c1 + halo, cols)
    return (w0, w1, v0, v1), (slice(r0 - w0, r1 - w0), slice(c0 - v0, c1 - v0))


def _tiles(shape, tile):
    return [(r0, min(r0 + tile, shape[0]), c0, min(c0 + tile, shape[1]))
            for r0 in range(0, shape[0], tile) for c0 in range(0, shape[1], tile)]


def _waterline_rc(states, shape, tile, skip=None):
    """Rows and cols (int32) of known-water cells with a known-land 4-neighbour.
    ``states(w0, w1, v0, v1)`` -> (known_water, known_land) on that window; ``skip(r0, r1,
    c0, c1)`` may say that a tile has no waterline."""
    R, C = [], []
    for r0, r1, c0, c1 in _tiles(shape, tile):
        if skip is not None and skip(r0, r1, c0, c1):
            continue
        (w0, w1, v0, v1), core = _win(shape, r0, r1, c0, c1, 1)
        kw, kl = states(w0, w1, v0, v1)
        sh = (kw & ndimage.binary_dilation(kl, structure=_FOUR))[core]
        r, c = np.nonzero(sh)
        R.append((r + r0).astype(np.int32))
        C.append((c + c0).astype(np.int32))
    if not R:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    return np.concatenate(R), np.concatenate(C)


def obs_states(code, outside):
    """(known_water, known_land) of an observed extent: inside the valid mask (code > -2)
    the code (1 water, 0 land, -1 unknown); elsewhere, and where the code is unknown, the
    outside state (1 water, 0 land, other unknown), as hindcast.known_states reads
    a7_hindcast's codes."""
    code = np.asarray(code)
    o = np.asarray(outside)
    out = code < 0
    return (code == 1) | (out & (o == 1)), (code == 0) | (out & (o == 0))


def obs_waterline(code, outside, *, tile: int = 1024):
    """Waterline cells (rows, cols) of an observed extent (``obs_states``)."""
    shape = np.shape(code)
    return _waterline_rc(lambda w0, w1, v0, v1: obs_states(code[w0:w1, v0:v1], outside[w0:w1, v0:v1]),
                         shape, tile)


class FloodMap:
    """A flood-level map F with its finite-height mask, for waterlines at many thresholds.
    ``F`` and ``finite`` are 2-D arrays (or memory maps); per tile (with a one-cell halo)
    the least F of finite cells and the largest are kept, so that a tile without both
    sides of a threshold is skipped (exact: such a tile has no waterline)."""

    def __init__(self, F, finite, *, tile: int = 256):
        self.F, self.finite, self.tile = F, finite, tile
        self.shape = tuple(np.shape(F))
        self.lo, self.hi = {}, {}
        for r0, r1, c0, c1 in _tiles(self.shape, tile):
            (w0, w1, v0, v1), _ = _win(self.shape, r0, r1, c0, c1, 1)
            z = np.asarray(F[w0:w1, v0:v1], float)
            fin = np.asarray(finite[w0:w1, v0:v1], bool)
            zf = z[fin]
            self.lo[(r0, c0)] = float(zf.min()) if zf.size else np.inf
            self.hi[(r0, c0)] = float(zf.max()) if zf.size else -np.inf

    def states(self, T: float):
        def f(w0, w1, v0, v1):
            fin = np.asarray(self.finite[w0:w1, v0:v1], bool)
            p = np.asarray(self.F[w0:w1, v0:v1], float) < T
            return p & fin, ~p & fin
        return f

    def waterline(self, T: float):
        """Rows and cols of the predicted waterline at threshold T (flood fill F < T)."""
        def skip(r0, r1, c0, c1):
            return self.lo[(r0, c0)] >= T or self.hi[(r0, c0)] < T
        return _waterline_rc(self.states(T), self.shape, self.tile, skip)

    def at(self, rows, cols, T: float):
        """(known, wet) of the prediction at cells: finite height, and F < T there."""
        fin = np.asarray(self.finite[rows, cols], bool) if len(rows) else np.empty(0, bool)
        wet = (np.asarray(self.F[rows, cols], float) < T) & fin if len(rows) else np.empty(0, bool)
        return fin, wet


def cells_from(pred_rc, obs_rc, pred_at, obs_at, labels) -> dict:
    """The ``cells`` dictionary of hindcast_tiled: for the predicted (or first) extent's
    waterline cells, the observed (second) extent's state there (``obs_at(rows, cols)`` ->
    (known, wet)), and the reverse; labels from the 2-D ``labels`` array."""
    pr, pc = pred_rc
    orr, oc = obs_rc
    pk, pw = obs_at(pr, pc)
    ok_, ow = pred_at(orr, oc)
    lab = np.asarray(labels)
    return {"pred": {"rows": pr, "cols": pc, "known": np.asarray(pk, bool), "wet": np.asarray(pw, bool),
                     "label": lab[pr, pc] if len(pr) else np.empty(0, lab.dtype)},
            "obs": {"rows": orr, "cols": oc, "known": np.asarray(ok_, bool), "wet": np.asarray(ow, bool),
                    "label": lab[orr, oc] if len(orr) else np.empty(0, lab.dtype)}}


def obs_at(code, outside):
    """(known, wet) lookup of an observed extent at cells."""
    def f(rows, cols):
        if not len(rows):
            return np.empty(0, bool), np.empty(0, bool)
        kw, kl = obs_states(np.asarray(code)[rows, cols], np.asarray(outside)[rows, cols])
        return kw | kl, kw
    return f


def drop_cells(rc, flag):
    """Waterline cells without those where the 2-D boolean ``flag`` is True."""
    r, c = rc
    if not len(r):
        return rc
    keep = ~np.asarray(flag)[r, c]
    return r[keep], c[keep]


# --- 4. majority aggregation of observed extents to a native grid (A18 item 6) ---------------


def majority_to_native(raw, native_id, n_native: int):
    """Observed extent (a7_hindcast.observed_raw codes: -2 not valid, 0 dry, 1 water)
    aggregated by majority to native cells and broadcast back to the cells that were
    valid. ``native_id``: the native cell of each grid cell (-1: none, the cell is left
    as it was). A native cell is water when its valid cells hold at least as many water
    cells as dry ones (the >= 0.5 rule of the frequency contour). Cells that were not
    valid stay not valid."""
    raw = np.asarray(raw)
    nid = np.asarray(native_id)
    ok = (raw > -2) & (nid >= 0)
    n_w = np.bincount(nid[ok & (raw == 1)], minlength=n_native)
    n_d = np.bincount(nid[ok & (raw == 0)], minlength=n_native)
    state = (n_w >= n_d).astype(np.int8)
    out = raw.copy()
    out[ok] = state[nid[ok]]
    return out


# --- 5. gradients and the vertical error budget (A18 item 7) ---------------------------------


def slope_m_per_km(z, res_m: float) -> np.ndarray:
    """|grad z| in m per km on a regular grid of spacing ``res_m`` (central differences
    inside, one-sided at the edges and next to NaN, as numpy.gradient; NaN where z is NaN
    or a neighbour needed is NaN)."""
    z = np.asarray(z, float)
    gy, gx = np.gradient(z, res_m)
    return np.hypot(gx, gy) * 1000.0


def budget_sigma(*terms) -> np.ndarray:
    """1-sigma vertical budget: the terms (m) combined in quadrature; a NaN term is left
    out (absent), and all-NaN gives NaN."""
    t = np.vstack([np.broadcast_to(np.asarray(x, float), np.broadcast(*terms).shape) for x in terms])
    have = np.isfinite(t)
    s = np.sqrt(np.where(have, t ** 2, 0.0).sum(axis=0))
    return np.where(have.any(axis=0), s, np.nan)


def augmented_floor(Q_m, sigma_m, g_m_per_km, z: float = 1.645) -> np.ndarray:
    """A18 item 7: Q + z * sigma / g (horizontal, m), the vertical budget converted with
    the shelf gradient and added to the floor rather than combined in quadrature."""
    return np.asarray(Q_m, float) + z * np.asarray(sigma_m, float) / np.asarray(g_m_per_km, float) * 1000.0
