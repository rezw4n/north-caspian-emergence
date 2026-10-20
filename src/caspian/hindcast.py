"""A7: hindcast audit of bathymetry products.

For an observed water level, a product's predicted water extent is the set of
cells below that level that are hydrologically connected to the open sea (seed
mask). Predicted and observed extents are compared within a valid mask
(observed and inside the analysis region) by area bias, IoU, F1 and shoreline
displacement.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_FOUR = ndimage.generate_binary_structure(2, 1)
_EIGHT = ndimage.generate_binary_structure(2, 2)


def flood_fill(elevation, level: float, seed, connectivity: int = 8):
    """Cells with elevation < level connected (4 or 8) to any seed cell that is also
    below the level. NaN elevation counts as not flooded."""
    z = np.asarray(elevation, float)
    below = np.isfinite(z) & (z < level)
    labels, _ = ndimage.label(below, structure=_EIGHT if connectivity == 8 else _FOUR)
    seeded = np.unique(labels[np.asarray(seed, bool) & below])
    seeded = seeded[seeded > 0]
    return np.isin(labels, seeded)


def extent_metrics(pred, obs, valid, cell_area: float = 1.0) -> dict:
    """Area bias (pred - obs), relative bias, IoU and F1 of water extents within ``valid``."""
    p = np.asarray(pred, bool) & np.asarray(valid, bool)
    o = np.asarray(obs, bool) & np.asarray(valid, bool)
    tp = np.sum(p & o)
    fp = np.sum(p & ~o)
    fn = np.sum(~p & o)
    a_pred, a_obs = p.sum() * cell_area, o.sum() * cell_area
    union = tp + fp + fn
    return {
        "area_pred": float(a_pred),
        "area_obs": float(a_obs),
        "area_bias": float(a_pred - a_obs),
        "area_bias_rel": float((a_pred - a_obs) / a_obs) if a_obs > 0 else float("nan"),
        "iou": float(tp / union) if union else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
    }


def shoreline(mask):
    """Water cells that touch a non-water cell (4-neighbourhood)."""
    m = np.asarray(mask, bool)
    return m & ~ndimage.binary_erosion(m, structure=_FOUR, border_value=1)


def shoreline_displacement(pred, obs, valid, pixel_size: float = 1.0) -> dict:
    """Distances from each predicted shoreline cell to the nearest observed
    shoreline cell (inside ``valid``): median, 95th percentile, and the signed
    median (positive = predicted shoreline lies on the observed land side, i.e.
    the product predicts too much water)."""
    p_sh = shoreline(np.asarray(pred, bool)) & np.asarray(valid, bool)
    o_sh = shoreline(np.asarray(obs, bool))
    if not p_sh.any() or not o_sh.any():
        return {"median": float("nan"), "p95": float("nan"), "signed_median": float("nan"), "n": 0}
    dist = ndimage.distance_transform_edt(~o_sh) * pixel_size
    d = dist[p_sh]
    sign = np.where(np.asarray(obs, bool)[p_sh], -1.0, 1.0)  # inside observed water -> too little water
    return {"median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
            "signed_median": float(np.median(sign * d)), "n": int(d.size)}
