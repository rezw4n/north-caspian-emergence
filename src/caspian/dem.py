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
