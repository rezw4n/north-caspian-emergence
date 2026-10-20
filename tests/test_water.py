"""Water-classification reference functions (synthetic inputs)."""

import numpy as np
import pytest

from caspian import water


def test_mndwi_known_values():
    assert water.mndwi(0.3, 0.1) == pytest.approx(0.5)
    assert water.mndwi(0.1, 0.3) == pytest.approx(-0.5)
    assert np.isnan(water.mndwi(0.0, 0.0))


def test_awei_sh_formula():
    # Feyisa et al. 2014: blue + 2.5 green - 1.5 (nir + swir1) - 0.25 swir2
    assert water.awei_sh(0.1, 0.2, 0.05, 0.02, 0.01) == pytest.approx(0.1 + 0.5 - 0.105 - 0.0025)


def test_otsu_separates_bimodal_sample():
    rng = np.random.default_rng(0)
    land = rng.normal(-0.4, 0.08, 4000)
    sea = rng.normal(0.45, 0.08, 3000)
    t = water.otsu_threshold(np.concatenate([land, sea]))
    assert -0.15 < t < 0.20  # lies between the modes
    # The threshold separates nearly all of both classes.
    assert (land < t).mean() > 0.99 and (sea > t).mean() > 0.99


def test_otsu_shifts_with_modes():
    rng = np.random.default_rng(1)
    a = water.otsu_threshold(np.concatenate([rng.normal(-0.5, 0.05, 2000), rng.normal(0.1, 0.05, 2000)]))
    b = water.otsu_threshold(np.concatenate([rng.normal(-0.2, 0.05, 2000), rng.normal(0.4, 0.05, 2000)]))
    assert b - a == pytest.approx(0.3, abs=0.03)


def test_otsu_rejects_empty():
    with pytest.raises(ValueError):
        water.otsu_threshold([np.nan])


def test_classify_classes_and_unobserved():
    m = np.array([[0.4, -0.3, -0.3, np.nan]])
    n = np.array([[-0.1, 0.1, 0.6, 0.2]])
    c = water.classify(m, n, threshold=0.0)
    assert c[0, 0] == water.WATER
    assert c[0, 1] == water.DRY
    assert c[0, 2] == water.VEG
    assert np.isnan(c[0, 3])


def test_water_wins_over_vegetation():
    # Floating or submerged vegetation with high MNDWI stays water.
    c = water.classify(np.array([0.5]), np.array([0.5]), threshold=0.0)
    assert c[0] == water.WATER


def test_nd_treats_negative_reflectance_as_zero():
    # clear water: slightly negative SWIR1 and NIR (Earth Engine's normalizedDifference masks these)
    assert water.mndwi(0.11, -0.006) == pytest.approx(1.0)
    assert water.ndvi(0.043, -0.006) == pytest.approx(-1.0)
    assert water.mndwi(-0.01, 0.2) == pytest.approx(-1.0)
    assert np.isnan(water.mndwi(-0.01, -0.02))           # both at most 0: undefined
    v = water.nd(np.array([0.3, 0.1, -1.0]), np.array([0.1, 0.3, 0.5]))
    assert np.all((v >= -1) & (v <= 1))


def test_classify_water_needs_only_mndwi():
    m = np.array([0.6, -0.4, -0.4])
    n = np.array([np.nan, np.nan, 0.1])
    c = water.classify(m, n, threshold=0.0)
    assert c[0] == water.WATER          # water with unknown NDVI stays water
    assert np.isnan(c[1])               # not water, NDVI unknown: unobserved
    assert c[2] == water.DRY
