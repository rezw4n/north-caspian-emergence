"""Server-side (Earth Engine) parity tests. Run with CASPIAN_EE_TESTS=1; they need
network access and an authenticated Earth Engine account, so they are skipped by
default."""

import os

import numpy as np
import pytest

from caspian import water

pytestmark = pytest.mark.skipif(os.environ.get("CASPIAN_EE_TESTS") != "1",
                                reason="Earth Engine integration tests disabled")


@pytest.fixture(scope="module")
def ee_mod():
    import ee

    from caspian.config import EE_PROJECT

    ee.Initialize(project=EE_PROJECT)
    return ee


def test_ee_otsu_matches_numpy(ee_mod):
    ee = ee_mod
    rng = np.random.default_rng(3)
    sample = np.concatenate([rng.normal(-0.35, 0.1, 5000), rng.normal(0.3, 0.12, 3000)])
    counts, edges = np.histogram(sample, bins=128, range=(-1, 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ref = water.otsu_from_histogram(counts, centers)
    hist = ee.Dictionary({"histogram": counts.tolist(), "bucketMeans": centers.tolist()})
    got = water.ee_otsu(hist).getInfo()
    assert got == pytest.approx(ref, abs=1e-9)


def test_ee_otsu_fixed_matches_numpy_reference(ee_mod):
    # same 256 buckets over [-1, 1] as water.otsu_threshold (review finding 39)
    rng = np.random.default_rng(5)
    sample = np.concatenate([rng.normal(-0.3, 0.12, 6000), rng.normal(0.25, 0.1, 2500)])
    ref = water.otsu_threshold(sample)
    counts, edges = np.histogram(sample, bins=water.OTSU_BINS, range=water.MNDWI_RANGE)
    arr = np.column_stack([edges[:-1], counts]).tolist()
    got = water.ee_otsu_fixed(arr).getInfo()
    assert got == pytest.approx(ref, abs=1e-9)


def test_fixed_histogram_reducer_buckets(ee_mod):
    ee = ee_mod
    img = ee.Image.constant(0.123).rename("mndwi")
    h = img.reduceRegion(water.ee_histogram_reducer(), ee.Geometry.Point([50, 45]).buffer(100), 30).get("mndwi")
    rows = ee.Array(h).getInfo()
    assert len(rows) == water.OTSU_BINS
    width = 2 / water.OTSU_BINS
    k = int((0.123 + 1) // width)
    assert rows[k][0] == pytest.approx(-1 + k * width)
    assert rows[k][1] > 0
