"""Per-acquisition MNDWI thresholds from stage-1 histograms (local, NumPy).

Stage 1 (run 2) stores, for every scene, the 256-bucket histogram of clear in-zone
MNDWI at 240 m (water.OTSU_BINS buckets on water.MNDWI_RANGE). Scenes of one
acquisition (same overpass) share illumination and atmosphere, so their histograms
are summed and one Otsu threshold is computed per acquisition from the larger sample.

Diagnostics per histogram, used by the validity rule:
- the Otsu threshold (water.otsu_from_histogram, the tested reference). When the
  valley between two clean modes is empty, the criterion is flat across it and the
  first bucket of the gap is returned, which is still a valid split;
- the class shares below and above it;
- eta, the share of the histogram's variance explained by the two-class split
  (between-class over total variance). One normal mode gives 2/pi = 0.64; two
  well-separated modes give values near 1;
- valley_ratio: on the histogram smoothed over SMOOTH buckets, the lowest density
  between the two class peaks divided by the lower peak. One mode gives a ratio near
  1, two modes with a clear valley a ratio near 0.
(A separation index of the two halves, sqrt(2)|mu1 - mu0| / sqrt(s0^2 + s1^2), is not
used: a single normal mode split at its centre scores about 2.65 by construction.)
"""

from __future__ import annotations

import numpy as np

from . import water

EDGES = np.linspace(water.MNDWI_RANGE[0], water.MNDWI_RANGE[1], water.OTSU_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def parse_hist(s) -> np.ndarray | None:
    """A stored histogram ("c0 c1 ... c255") as an int array, or None."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return None
    h = np.array([int(x) for x in str(s).split()], dtype=np.int64)
    if h.size != water.OTSU_BINS:
        raise ValueError(f"histogram has {h.size} buckets, expected {water.OTSU_BINS}")
    return h


SMOOTH = 5


def diagnostics(hist) -> dict:
    """Otsu threshold, class shares, eta and valley ratio of one histogram."""
    h = np.asarray(hist, float)
    n = h.sum()
    empty = {"n": int(n), "threshold": np.nan, "share_low": np.nan, "share_high": np.nan,
             "eta": np.nan, "valley_ratio": np.nan}
    if n <= 0:
        return empty
    t = water.otsu_from_histogram(h, CENTERS)
    lo = CENTERS < t
    w0, w1 = h[lo].sum() / n, h[~lo].sum() / n
    if w0 == 0 or w1 == 0:
        return {**empty, "threshold": float(t), "share_low": float(w0), "share_high": float(w1)}
    mu = (h * CENTERS).sum() / n
    mu0 = (h[lo] * CENTERS[lo]).sum() / h[lo].sum()
    mu1 = (h[~lo] * CENTERS[~lo]).sum() / h[~lo].sum()
    var_t = (h * (CENTERS - mu) ** 2).sum() / n
    eta = (w0 * (mu0 - mu) ** 2 + w1 * (mu1 - mu) ** 2) / var_t if var_t > 0 else np.nan
    sm = np.convolve(h, np.ones(SMOOTH) / SMOOTH, mode="same")
    i0 = int(np.argmax(np.where(lo, sm, -1)))
    i1 = int(np.argmax(np.where(~lo, sm, -1)))
    valley = sm[i0:i1 + 1].min() if i1 > i0 else np.nan
    ratio = valley / min(sm[i0], sm[i1]) if min(sm[i0], sm[i1]) > 0 else np.nan
    return {"n": int(n), "threshold": float(t), "share_low": float(w0), "share_high": float(w1),
            "eta": float(eta), "valley_ratio": float(ratio)}


def pooled(hists) -> np.ndarray:
    """Sum of several histograms (None entries skipped)."""
    hs = [np.asarray(h, np.int64) for h in hists if h is not None]
    if not hs:
        return np.zeros(water.OTSU_BINS, np.int64)
    return np.sum(hs, axis=0)
