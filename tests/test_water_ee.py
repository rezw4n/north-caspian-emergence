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
