"""Great Salt Lake lake-bed control (PREREG Amendment 19 item 13; D-150).

Small helpers for auditing the HydroShare lake-bed elevation model (Tarboton and Merck;
BIO-WEST; USGS soundings of Baskin 2005, 2006) as an additional control, exactly as the
3DEP substitute is audited by scripts/g1_hindcast.py:

- the model's vertical datum (NGVD 29, m) to NAVD 88 with VERTCON 3.0 differences;
- the area-weighted mean of the native cells in each cell of the audit grid;
- level bins against the model's source boundaries (soundings, map contours, NED);
- survey timing per bin from the arm's daily gauge series.

Bins follow the Caspian machinery (caspian.design): bin b spans
[origin + b w, origin + (b + 1) w) on the working datum minus ``level_offset_m``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from caspian import design

FT_M = 0.3048


# --- datum ---------------------------------------------------------------------------------

def ft_ngvd29_to_navd88_m(ft: float, delta_m: float) -> float:
    """A contour in feet above NGVD 29 as metres above NAVD 88 (delta = NAVD 88 - NGVD 29)."""
    return float(ft) * FT_M + float(delta_m)


def to_navd88(z_ngvd29_m, delta_m) -> np.ndarray:
    """Heights in metres above NGVD 29 to NAVD 88: z + delta, where delta is a constant (the
    arm's gauge difference) or an array of the same shape (per-cell VERTCON). NaN stays NaN."""
    z = np.asarray(z_ngvd29_m, dtype=np.float32)
    return (z + np.asarray(delta_m, dtype=np.float32)).astype(np.float32)


def lattice_interpolator(lon, lat, values):
    """Bilinear interpolator of a regular lon/lat lattice (``values[i, j]`` at ``lat[i]``,
    ``lon[j]``); points beyond the lattice are extrapolated linearly from the edge cells."""
    from scipy.interpolate import RegularGridInterpolator
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    v = np.asarray(values, float)
    if v.shape != (lat.size, lon.size):
        raise ValueError(f"values shape {v.shape} != ({lat.size}, {lon.size})")
    if np.isnan(v).any():
        raise ValueError("lattice has missing values")
    f = RegularGridInterpolator((lat, lon), v, bounds_error=False, fill_value=None)
    return lambda qlon, qlat: f(np.c_[np.ravel(qlat), np.ravel(qlon)]).reshape(np.shape(qlon))


# --- resampling -----------------------------------------------------------------------------

def average_to_grid(src, transform, shape, crs: str, num_threads: int = 2) -> np.ndarray:
    """Area-weighted mean of the source raster's valid cells in each cell of the target grid
    (GDAL "average"; no-data cells excluded), float32, NaN where no valid source cell
    falls. ``src``: an open rasterio dataset (band 1 is read through GDAL's warper, in
    chunks); ``transform``: [a, 0, c, 0, e, f] of the target grid."""
    from affine import Affine
    import rasterio
    from rasterio.warp import Resampling, reproject

    out = np.full(tuple(shape), np.nan, np.float32)
    reproject(source=rasterio.band(src, 1), destination=out, src_transform=src.transform, src_crs=src.crs,
              src_nodata=src.nodata, dst_transform=Affine(*[float(v) for v in transform[:6]]), dst_crs=crs,
              dst_nodata=np.nan, resampling=Resampling.average, num_threads=num_threads,
              warp_mem_limit=512)
    return out


# --- bins -----------------------------------------------------------------------------------

def bin_edges(b, level_offset_m: float = 0.0, width: float = design.BIN_WIDTH_M,
              origin: float = design.BIN_ORIGIN_M) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edges of bin(s) ``b`` on the working datum (bin scale + offset)."""
    b = np.asarray(b, float)
    lo = origin + b * width + level_offset_m
    return np.round(lo, 6), np.round(lo + width, 6)


def max_bin_at_or_below(boundary_m: float, level_offset_m: float = 0.0, width: float = design.BIN_WIDTH_M,
                        origin: float = design.BIN_ORIGIN_M) -> int:
    """The highest bin whose upper edge lies at or below ``boundary_m`` (working datum)."""
    return int(math.floor(round((boundary_m - level_offset_m - origin) / width, 6))) - 1


def min_bin_at_or_above(boundary_m: float, level_offset_m: float = 0.0, width: float = design.BIN_WIDTH_M,
                        origin: float = design.BIN_ORIGIN_M) -> int:
    """The lowest bin whose lower edge lies at or above ``boundary_m`` (the 3DEP rule of
    scripts/g1_hindcast.py)."""
    return int(math.ceil(round((boundary_m - level_offset_m - origin) / width, 6)))


def source_class(b, sounded_top_m: float, contours_top_m: float, level_offset_m: float = 0.0) -> np.ndarray:
    """Which part of the model a bin's levels lie in: "sounded" (upper edge at or below the
    top of the soundings), "map_contours" (between that and the top of the bathymetric
    model), "ned" (lower edge at or above the top of the bathymetric model), else "mixed"."""
    lo, hi = bin_edges(b, level_offset_m)
    lo, hi = np.atleast_1d(lo), np.atleast_1d(hi)
    out = np.full(lo.shape, "mixed", dtype=object)
    out[hi <= sounded_top_m + 1e-9] = "sounded"
    out[(lo >= sounded_top_m - 1e-9) & (hi <= contours_top_m + 1e-9)] = "map_contours"
    out[lo >= contours_top_m - 1e-9] = "ned"
    return out


# --- survey timing --------------------------------------------------------------------------

def survey_timing(levels: pd.Series, bins, window: tuple[str, str], level_offset_m: float = 0.0,
                  depth_limit_m: float = 1.0, since: str = "1984-01-01") -> pd.DataFrame:
    """Per bin, how the survey relates to the arm's water level (D-150 item 6).

    ``levels``: daily gauge levels on the working datum (DatetimeIndex). ``window``: the
    survey's first and last day. Columns: the bin's edges; the window's lowest and highest
    level; ``under_water_in_survey`` (upper edge below the window's lowest level, so the
    bed at the bin was under water on every survey day); ``within_sounding_depth`` (upper
    edge more than ``depth_limit_m`` below it); ``first_reached`` (first day since
    ``since`` with the level at or below the upper edge); ``first_reached_after_survey``
    (that day is after the window; NaT counts as not reached, False)."""
    s = levels.dropna().sort_index()
    s = s[s.index >= pd.Timestamp(since)]
    w = s.loc[window[0]:window[1]]
    if w.empty:
        raise ValueError("no gauge values in the survey window")
    wmin, wmax = float(w.min()), float(w.max())
    end = pd.Timestamp(window[1])
    rows = []
    for b in np.atleast_1d(bins):
        lo, hi = bin_edges(int(b), level_offset_m)
        hit = s.index[s.to_numpy() <= float(hi)]
        first = hit.min() if len(hit) else pd.NaT
        rows.append({"bin": int(b), "lower_m": float(lo), "upper_m": float(hi), "survey_min_m": wmin,
                     "survey_max_m": wmax, "under_water_in_survey": bool(hi < wmin),
                     "within_sounding_depth": bool(hi < wmin - depth_limit_m), "first_reached": first,
                     "first_reached_after_survey": bool(pd.notna(first) and first > end)})
    return pd.DataFrame(rows)
