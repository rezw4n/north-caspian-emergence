"""A6: water frequency from stage-2 counts.

Stage 2 exports, per group g (epoch x level bin x acquisition parity), the number of
acquisitions that saw a pixel as water (w_g) and as water or dry (v_g; emergent
vegetation and masked pixels are not valid). The water frequency of a set of groups
is sum(w) / sum(v), defined only where sum(v) >= MIN_VALID_OBS (PREREG section 4).
Groups combine by summing counts, never by averaging frequencies: odd + even gives
the full sample, E2 + E3 gives the DEM sample.
"""

from __future__ import annotations

import numpy as np

from . import design


def combine(counts: dict, groups) -> tuple[np.ndarray, np.ndarray]:
    """Summed (w, v) over ``groups``; ``counts`` maps group -> (w, v) arrays."""
    groups = list(groups)
    if not groups:
        raise ValueError("no groups to combine")
    w = np.zeros_like(np.asarray(counts[groups[0]][0]), dtype=np.int64)
    v = np.zeros_like(w)
    for g in groups:
        gw, gv = counts[g]
        gw, gv = np.asarray(gw, np.int64), np.asarray(gv, np.int64)
        if (gw > gv).any():
            raise ValueError(f"group {g}: water count exceeds valid count")
        w += gw
        v += gv
    return w, v


def water_frequency(w, v, min_valid: int = design.MIN_VALID_OBS) -> np.ndarray:
    """sum(w) / sum(v) where v >= min_valid, NaN elsewhere."""
    w, v = np.asarray(w, float), np.asarray(v, float)
    f = np.full(w.shape, np.nan)
    ok = v >= min_valid
    f[ok] = w[ok] / v[ok]
    return f
