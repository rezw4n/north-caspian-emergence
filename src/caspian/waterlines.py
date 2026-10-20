"""A6: sub-pixel waterlines from water-frequency rasters (marching squares).

A frequency raster f (fraction of valid observations classed as water within
one level bin) is contoured at f = 0.5 with scikit-image's marching squares.
Each contour is attributed with the bin's level. Pixel (row, col) refers to the
pixel centre; map coordinates follow the raster's affine transform.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString
from skimage import measure


def pixel_to_map(rows, cols, transform):
    """Pixel-centre (row, col) -> map (x, y) for an affine transform
    (a, b, c, d, e, f) as in rasterio: x = a*col + b*row + c, y = d*col + e*row + f,
    evaluated at col + 0.5, row + 0.5."""
    a, b, c, d, e, f = transform[:6]
    cc = np.asarray(cols) + 0.5
    rr = np.asarray(rows) + 0.5
    return a * cc + b * rr + c, d * cc + e * rr + f


def extract(freq, transform, level: float = 0.5, min_vertices: int = 5,
            valid=None) -> list[LineString]:
    """Contours of ``freq`` at ``level`` as map-coordinate LineStrings.

    ``valid`` (bool array) marks pixels with enough observations; contours are not
    traced through invalid pixels. Contours with fewer than ``min_vertices``
    vertices (sub-pixel speckle) are dropped."""
    f = np.asarray(freq, float)
    mask = np.isfinite(f) if valid is None else (np.asarray(valid, bool) & np.isfinite(f))
    f = np.where(mask, f, 0.0)
    lines = []
    for c in measure.find_contours(f, level, mask=mask):
        if len(c) < min_vertices:
            continue
        x, y = pixel_to_map(c[:, 0], c[:, 1], transform)
        lines.append(LineString(np.column_stack([x, y])))
    return lines


def contour_points(lines, spacing: float):
    """Resample contour lines to points at roughly ``spacing`` map units."""
    pts = []
    for ln in lines:
        n = max(int(np.ceil(ln.length / spacing)), 1)
        for t in np.linspace(0, ln.length, n + 1):
            p = ln.interpolate(t)
            pts.append((p.x, p.y))
    return np.array(pts).reshape(-1, 2)


STATIC_MIN_ETA, STATIC_MAX_VALLEY = 0.70, 0.50   # validity of the split, as for scene thresholds (D-061)
STATIC_RHO_RANGE = (0.2, 0.8)
STATIC_RHO_FALLBACK = 0.5


def static_threshold(rho) -> dict:
    """Threshold rho* of the static-contour rule (D-034, D-075) from training-block
    rho_level values: Otsu's split of their histogram (256 bins on [-1, 1]) when the
    split is valid (eta >= 0.70, valley ratio <= 0.50) and lies in [0.2, 0.8],
    otherwise 0.5. Contour points with rho_level < rho* are static."""
    from . import thresholds
    r = np.asarray(rho, float)
    r = r[np.isfinite(r)]
    hist, _ = np.histogram(np.clip(r, -1, 1), bins=thresholds.EDGES)
    d = thresholds.diagnostics(hist)
    ok = (np.isfinite(d["threshold"]) and d["eta"] >= STATIC_MIN_ETA and d["valley_ratio"] <= STATIC_MAX_VALLEY
          and STATIC_RHO_RANGE[0] <= d["threshold"] <= STATIC_RHO_RANGE[1])
    return {"rho_star": float(d["threshold"]) if ok else STATIC_RHO_FALLBACK, "source": "otsu" if ok else "fallback",
            "diagnostics": d, "n_train_points": int(r.size), "hist": hist.tolist()}
