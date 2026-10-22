"""Spatial structure of the water surface and spatially varying product offsets
(PREREG Amendment 18 item 9; D-120, D-121).

1. SWOT water-surface elevation on the working datum. SWOT L2_HR_Raster ``wse`` is
   the height above the EGM2008 geoid "with a correction to refer the value to the
   mean tide system" (JPL D-56416 Rev C, the ``geoid`` variable), with the periodic
   solid-Earth tide removed and the permanent part kept, so it is a mean-tide height.
   The working datum is DAHITI: normal heights above EIGEN-6C4, taken as
   conventional tide-free (IERS 2010 solid tide; an inference, docs/DATUMS.md).
   With N_s the tide-free SWOT geoid, N_e the tide-free EIGEN-6C4 geoid, g(phi) the
   tide-free -> mean-tide geoid change and e(phi) the tide-free -> mean-tide crust
   (surface) change (the ICESat-2 parameters ``geoid_free2mean`` and
   ``tide_earth_free2mean``):

       tide-free working datum:  H = wse + N_s - N_e + g(phi) - e(phi)
       mean-tide working datum:  H = wse + N_s - N_e

   (a C20 change moves both geoids alike, to 3e-7 m over the box; results/provenance/
   geoid_difference.json).
2. Sector departures: per pass and sector the mean over the sector's blocks of
   (block-mean height - basin level at the pass time), then the median over passes.
3. Per-sector level shifts applied to a flood-level map: a cell of sector k floods at
   level L when F < L + d_k, i.e. (F - d_k) < L (``ShiftedFloodMap``).
4. Planar offset fields c(x, y) = c0 + a x + b y (x, y in km on the equal-area grid)
   and the Eq. S11 area criterion counted from quantised flood levels
   (``plane_values``, ``quantised_counts``, ``profile_c0``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ICESat-2 Release 007 parameters (ATL06 data dictionary; docs/DATUMS.md), m:
# value = a + b sin^2(latitude)
GEOID_FREE2MEAN = (0.1287, -0.3848)
EARTH_FREE2MEAN = (0.06029, -0.180873)


def free2mean(lat_deg):
    """(g, e): the tide-free -> mean-tide change of the geoid and of the surface height
    at latitude ``lat_deg`` (m)."""
    s2 = np.sin(np.radians(np.asarray(lat_deg, float))) ** 2
    return GEOID_FREE2MEAN[0] + GEOID_FREE2MEAN[1] * s2, EARTH_FREE2MEAN[0] + EARTH_FREE2MEAN[1] * s2


def swot_to_working(wse, n_swot_tidefree, n_eigen_tidefree, lat_deg, working_tide: str = "tide_free"):
    """SWOT ``wse`` (above SWOT's mean-tide geoid) as a height on the working datum
    (EIGEN-6C4 normal heights) in the stated tide system of the working datum."""
    wse, ns, ne = (np.asarray(v, float) for v in (wse, n_swot_tidefree, n_eigen_tidefree))
    if working_tide == "mean_tide":
        return wse + ns - ne
    if working_tide != "tide_free":
        raise ValueError("working_tide must be 'tide_free' or 'mean_tide'")
    g, e = free2mean(lat_deg)
    return wse + ns - ne + g - e


def sector_departures(df: pd.DataFrame, value: str = "dep_m", *, min_blocks: int = 2,
                      min_passes: int = 20) -> pd.DataFrame:
    """Per sector: the median over passes of the pass's mean over the sector's blocks of
    ``value``; with the IQR over passes, the number of passes and of distinct blocks,
    and the number of block-passes. ``df`` needs pass_id, block_i, block_j, sector and
    ``value``. A sector with fewer than ``min_blocks`` blocks or ``min_passes`` passes
    gets NaN and the note "too thin"."""
    need = {"pass_id", "block_i", "block_j", "sector", value}
    if not need <= set(df.columns):
        raise ValueError(f"missing columns: {sorted(need - set(df.columns))}")
    d = df[np.isfinite(df[value].to_numpy(float))]
    per_pass = d.groupby(["sector", "pass_id"], as_index=False)[value].mean()
    rows = []
    for s, g in per_pass.groupby("sector", sort=True):
        blk = d.loc[d["sector"] == s, ["block_i", "block_j"]].drop_duplicates()
        v = g[value].to_numpy(float)
        n_p, n_b = int(v.size), int(len(blk))
        thin = n_b < min_blocks or n_p < min_passes
        q25, q50, q75 = np.percentile(v, [25, 50, 75]) if v.size else (np.nan,) * 3
        rows.append({"sector": s, "departure_m": np.nan if thin else float(q50), "n_passes": n_p, "n_blocks": n_b,
                     "iqr_m": np.nan if thin else float(q75 - q25),
                     "n_block_passes": int((d["sector"] == s).sum()),
                     "note": (f"too thin: {n_b} blocks, {n_p} passes (needs {min_blocks} and {min_passes})"
                              if thin else "")})
    return pd.DataFrame(rows, columns=["sector", "departure_m", "n_passes", "n_blocks", "iqr_m", "n_block_passes",
                                       "note"])


def departure_lut(departures: dict, sector_names, n_codes: int | None = None, default: float = 0.0) -> np.ndarray:
    """Lookup table from sector code (1 = first of ``sector_names``; 0 = none) to the
    departure (m); ``default`` for code 0 and for any sector missing from
    ``departures``."""
    n = max(len(sector_names) + 1, n_codes or 0)
    lut = np.full(n, float(default))
    for k, s in enumerate(sector_names):
        if s in departures and np.isfinite(departures[s]):
            lut[k + 1] = float(departures[s])
    return lut


class ShiftedFloodMap:
    """A flood-level map F less a per-sector level shift, read window by window:
    ``ShiftedFloodMap(F, sector, lut)[w] = F[w] - lut[sector[w]]`` in float64. A cell
    of sector k is flooded at level L when F < L + lut[k], so the observed level of
    that sector is L + lut[k] (A18 item 9). Where F is +inf (unreachable) it stays +inf."""

    def __init__(self, F, sector, lut):
        if tuple(np.shape(F)) != tuple(np.shape(sector)):
            raise ValueError("F and sector must be on the same grid")
        self.F, self.sector, self.lut = F, sector, np.asarray(lut, float)
        self.shape, self.dtype = tuple(np.shape(F)), np.dtype(np.float64)

    def __getitem__(self, key):
        return np.asarray(self.F[key], np.float64) - self.lut[np.asarray(self.sector[key]).astype(np.int64)]


# --- ellipsoids and zero-degree terms (c_ref decomposition, D-122) ---------------------------

WGS84 = (6378137.0, 1 / 298.257223563)        # a (m), f; ICESat-2 Comparison Guide R007, Table 4.1
TOPEX = (6378136.3, 1 / 298.257)              # T/P ellipsoid, same table


def ellipsoid_height(lat_deg, lon_deg, h, src=WGS84, dst=TOPEX):
    """(lat, h) of a point given on ellipsoid ``src`` re-expressed on ``dst`` (same centre
    and axes): geodetic -> Cartesian on src, then Cartesian -> geodetic on dst by
    Bowring's iteration (the two-step method of the Comparison Guide, section 3.1)."""
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    h = np.asarray(h, float)
    a, f = src
    e2 = f * (2 - f)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    X, Y = (N + h) * np.cos(lat) * np.cos(lon), (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - e2) + h) * np.sin(lat)
    a2, f2 = dst
    e22 = f2 * (2 - f2)
    p = np.hypot(X, Y)
    phi = np.arctan2(Z, p * (1 - e22))
    for _ in range(10):
        N2 = a2 / np.sqrt(1 - e22 * np.sin(phi) ** 2)
        h2 = p / np.cos(phi) - N2
        phi = np.arctan2(Z, p * (1 - e22 * N2 / (N2 + h2)))
    N2 = a2 / np.sqrt(1 - e22 * np.sin(phi) ** 2)
    return np.degrees(phi), p / np.cos(phi) - N2


def zero_degree_term(gm_model: float, gm_ref: float, w0: float | None = None, u0: float | None = None,
                     gamma: float = 9.80665, radius: float = 6371000.0) -> float:
    """ICGEM FAQ Q16, Eq. 24: N0 = (GM_model - GM_ref) / (gamma R) - (W0 - U0) / gamma (m);
    the second part only when W0 and U0 are given."""
    n0 = (gm_model - gm_ref) / (gamma * radius)
    if w0 is not None and u0 is not None:
        n0 -= (w0 - u0) / gamma
    return float(n0)


# --- planar offset field (A18 item 9, D-121) -------------------------------------------

def plane_values(c0: float, a: float, b: float, x_km, y_km):
    """c(x, y) = c0 + a x + b y (m), x and y in km, a and b in m per km."""
    return c0 + a * np.asarray(x_km, float) + b * np.asarray(y_km, float)


def quantised_counts(q, weights_known, n_bins: int) -> np.ndarray:
    """Cumulative counts of known cells by quantised flood level: out[m] = number of
    known cells with q < m, for m = 0..n_bins (q are integer bin indices, clipped to
    [0, n_bins] by the caller; cells at n_bins never count)."""
    h = np.bincount(np.asarray(q, np.int64), weights=None if weights_known is None else weights_known,
                    minlength=n_bins + 1)[:n_bins + 1]
    return np.concatenate([[0], np.cumsum(h)[:n_bins]]).astype(np.int64)


def profile_c0(bias: np.ndarray, c_grid) -> dict:
    """The D-070 rule for the constant part: the c0 of ``c_grid`` minimising the sum
    over units of |bias| (``bias`` [unit, c]); ties to the smallest |c0|, then the
    smaller c0."""
    c_grid = np.asarray(c_grid, float)
    obj = np.abs(bias).sum(axis=0)
    cand = np.flatnonzero(obj == obj.min())
    k = cand[np.lexsort((c_grid[cand], np.abs(c_grid[cand])))[0]]
    return {"c0_m": float(c_grid[k]), "objective": float(obj[k]), "k": int(k), "n_tied": int(cand.size)}
