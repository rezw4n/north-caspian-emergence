"""A6: interpolate level-attributed contour points to a grid.

TIN (Delaunay, linear) interpolation, plus the interpolation distance (distance
from each grid cell to the nearest contour point), which is stored for every
pixel and used as a scale variable for the conformal intervals. Gaussian-process
or kriging interpolation is compared in the robustness checks (brief section 10,
check 7).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def grid_centers(x0: float, y0: float, nx: int, ny: int, res: float):
    """Cell-centre coordinates of a north-up grid whose top-left corner is (x0, y0)."""
    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = y0 - (np.arange(ny) + 0.5) * res
    return np.meshgrid(xs, ys)


def tin(points_xy, z, gx, gy):
    """Linear interpolation on the Delaunay triangulation of the points; NaN outside
    the convex hull."""
    f = LinearNDInterpolator(np.asarray(points_xy, float), np.asarray(z, float))
    return f(gx, gy)


def interpolation_distance(points_xy, gx, gy):
    tree = cKDTree(np.asarray(points_xy, float))
    d, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]))
    return d.reshape(gx.shape)


def emergence_days(z, t_days, level):
    """Emergence date of seabed at height ``z`` (PREREG section 2; D-069): the first
    date of the level series after the last date on which the level was at or above
    z, so the pixel stayed dry from then on. ``t_days`` (ascending, days since
    1970-01-01) and ``level`` are the basin-level series on the DEM's reference.
    NaN where z is above every level of the series (never sea in the record) or at or
    below its last level (not yet emerged), and where z is NaN."""
    t = np.asarray(t_days, float)
    lv = np.asarray(level, float)
    if t.ndim != 1 or t.shape != lv.shape or not np.all(np.diff(t) > 0):
        raise ValueError("t_days must be strictly ascending and match level")
    if not np.isfinite(lv).all():
        raise ValueError("the level series must be finite")
    z = np.asarray(z, float)
    m = np.maximum.accumulate(lv[::-1])[::-1]          # max of the level from each date on
    k = np.searchsorted(-m, -np.where(np.isfinite(z), z, np.inf), side="right") - 1
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (k >= 0) & (z > lv[-1])
    out[ok] = t[k[ok] + 1]
    return out
