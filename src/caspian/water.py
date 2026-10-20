"""A3 water classification.

Per scene: MNDWI (Xu 2006) and AWEI_sh (Feyisa et al. 2014) from harmonised
reflectance; a per-scene threshold by Otsu's method (Otsu 1979), computed from an
MNDWI histogram sampled near water edges (edge-guided Otsu: Canny edges on MNDWI,
dilated, intersected with the JRC GSW transition zone); emergent vegetation
(NDVI above a threshold while not water) is returned as its own class.

Output classes: 1 water, 0 dry, 2 emergent vegetation; masked = not observed.

The NumPy functions are the reference implementation and are unit-tested; the
Earth Engine functions mirror them server-side.
"""

from __future__ import annotations

import numpy as np

WATER, DRY, VEG = 1, 0, 2
NDVI_VEG = 0.30          # emergent-vegetation NDVI threshold (sensitivity in brief section 10, check 9)
OTSU_BINS = 256
MNDWI_RANGE = (-1.0, 1.0)


# ----------------------------------------------------------------------------- NumPy reference

def mndwi(green, swir1):
    green, swir1 = np.asarray(green, float), np.asarray(swir1, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (green - swir1) / (green + swir1)


def awei_sh(blue, green, nir, swir1, swir2):
    return (np.asarray(blue) + 2.5 * np.asarray(green) - 1.5 * (np.asarray(nir) + np.asarray(swir1))
            - 0.25 * np.asarray(swir2))


def ndvi(red, nir):
    red, nir = np.asarray(red, float), np.asarray(nir, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - red) / (nir + red)


def otsu_threshold(values, bins: int = OTSU_BINS, value_range=MNDWI_RANGE) -> float:
    """Otsu's threshold maximising between-class variance of a 1-D sample."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError("need at least two finite values")
    counts, edges = np.histogram(v, bins=bins, range=value_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return otsu_from_histogram(counts, centers)


def otsu_from_histogram(counts, centers) -> float:
    counts = np.asarray(counts, float)
    centers = np.asarray(centers, float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("empty histogram")
    w0 = np.cumsum(counts)
    w1 = total - w0
    s0 = np.cumsum(counts * centers)
    mu_t = s0[-1] / total
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = s0 / w0
        mu1 = (s0[-1] - s0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1
    k = int(np.argmax(between[:-1]))
    # threshold halfway between the last centre of class 0 and the first of class 1
    return float(0.5 * (centers[k] + centers[k + 1])) if np.isfinite(mu_t) else float(centers[k])


def classify(mndwi_img, ndvi_img, threshold: float, ndvi_veg: float = NDVI_VEG):
    """Water / dry / emergent-vegetation classes; NaN where unobserved."""
    m = np.asarray(mndwi_img, float)
    n = np.asarray(ndvi_img, float)
    out = np.full(m.shape, np.nan)
    obs = np.isfinite(m) & np.isfinite(n)
    water = obs & (m > threshold)
    veg = obs & ~water & (n > ndvi_veg)
    out[obs] = DRY
    out[veg] = VEG
    out[water] = WATER
    return out


# ----------------------------------------------------------------------------- Earth Engine

def ee_indices(img):
    """Add mndwi, awei_sh and ndvi bands to a harmonised image (caspian.masks)."""
    m = img.normalizedDifference(["green", "swir1"]).rename("mndwi")
    n = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    a = img.expression("b + 2.5*g - 1.5*(n + s1) - 0.25*s2", {
        "b": img.select("blue"), "g": img.select("green"), "n": img.select("nir"),
        "s1": img.select("swir1"), "s2": img.select("swir2")}).rename("awei_sh")
    return img.addBands([m, n, a])


def ee_otsu(histogram):
    """Otsu threshold from an ee.Reducer.histogram() output dictionary (server side)."""
    import ee

    counts = ee.Array(ee.Dictionary(histogram).get("histogram"))
    means = ee.Array(ee.Dictionary(histogram).get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum_all = counts.multiply(means).reduce(ee.Reducer.sum(), [0]).get([0])
    mean_all = ee.Number(sum_all).divide(total)
    indices = ee.List.sequence(1, ee.Number(size).subtract(1))

    def bss(i):
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean = a_means.multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count)
        b_count = ee.Number(total).subtract(a_count)
        b_mean = ee.Number(sum_all).subtract(ee.Number(a_count).multiply(a_mean)).divide(b_count)
        return (ee.Number(a_count).multiply(ee.Number(a_mean).subtract(mean_all).pow(2))
                .add(b_count.multiply(b_mean.subtract(mean_all).pow(2))))

    variances = ee.Array(indices.map(bss))
    # Split i = j + 1 puts buckets [0, i) in class 0. As in otsu_from_histogram, the
    # threshold is halfway between the last bucket of class 0 and the first of class 1.
    j = ee.Number(variances.argmax().get(0))
    return ee.Number(means.get([j])).add(ee.Number(means.get([j.add(1)]))).divide(2)


def ee_histogram_reducer():
    """Fixed-bin MNDWI histogram with exactly the binning of otsu_threshold():
    OTSU_BINS equal buckets over MNDWI_RANGE."""
    import ee

    return ee.Reducer.fixedHistogram(MNDWI_RANGE[0], MNDWI_RANGE[1], OTSU_BINS)


def ee_otsu_fixed(hist_array):
    """Otsu threshold from an ee_histogram_reducer() output, an N x 2 array of
    [bucket lower edge, count] (server side). Matches otsu_threshold() on the same
    sample, because the buckets are the same."""
    import ee

    a = ee.Array(hist_array)
    width = (MNDWI_RANGE[1] - MNDWI_RANGE[0]) / OTSU_BINS
    centers = a.slice(1, 0, 1).project([0]).add(width / 2)
    counts = a.slice(1, 1, 2).project([0])
    return ee_otsu(ee.Dictionary({"histogram": counts, "bucketMeans": centers}))


def ee_edge_zone(mndwi, gsw_zone, canny_threshold: float = 0.5, sigma: float = 1.0,
                 dilate_m: float = 90):
    """Pixels near MNDWI edges and inside the GSW transition zone."""
    import ee

    edges = ee.Algorithms.CannyEdgeDetector(mndwi, threshold=canny_threshold, sigma=sigma)
    near = edges.gt(0).focalMax(dilate_m, "circle", "meters")
    return near.And(gsw_zone)


def ee_classify(img, region, gsw_zone, scale: float = 30, ndvi_veg: float = NDVI_VEG,
                fallback_threshold: float = 0.0, min_samples: int = 500):
    """Per-scene edge-guided Otsu water classification (server side).

    Returns the image with bands: class (uint8: 0 dry, 1 water, 2 veg; masked if
    unobserved) and properties: otsu_threshold, otsu_n, threshold_source.
    """
    import ee

    img = ee_indices(img)
    m = img.select("mndwi")
    zone = ee_edge_zone(m, gsw_zone)
    hist = m.updateMask(zone).reduceRegion(
        reducer=ee_histogram_reducer().combine(ee.Reducer.count(), "", True),
        geometry=region, scale=scale, maxPixels=1e9, bestEffort=True, tileScale=4)
    n = ee.Number(ee.Algorithms.If(hist.get("mndwi_count"), hist.get("mndwi_count"), 0))
    ok = n.gte(min_samples)
    thr = ee.Number(ee.Algorithms.If(ok, ee_otsu_fixed(hist.get("mndwi_histogram")), fallback_threshold))
    water = m.gt(thr)
    veg = water.Not().And(img.select("ndvi").gt(ndvi_veg))
    cls = ee.Image(DRY).where(veg, VEG).where(water, WATER).updateMask(m.mask()).uint8().rename("class")
    return (img.addBands(cls).set("otsu_threshold", thr).set("otsu_n", n)
            .set("threshold_source", ee.Algorithms.If(ok, "otsu", "fallback")))
