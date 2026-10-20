import numpy as np
import pytest

from caspian import thresholds, water


def _hist(sample):
    return np.histogram(sample, bins=thresholds.EDGES)[0]


def test_bimodal_histogram_is_separated():
    rng = np.random.default_rng(0)
    h = _hist(np.concatenate([rng.normal(-0.3, 0.08, 6000), rng.normal(0.5, 0.08, 4000)]))
    d = thresholds.diagnostics(h)
    assert -0.1 < d["threshold"] < 0.3          # anywhere in the gap between the modes
    assert d["eta"] > 0.9 and d["valley_ratio"] < 0.1
    assert d["share_low"] == pytest.approx(0.6, abs=0.03)


def test_unimodal_histogram_has_low_separation():
    rng = np.random.default_rng(1)
    h = _hist(rng.normal(-0.2, 0.15, 10000))
    d = thresholds.diagnostics(h)
    assert d["eta"] == pytest.approx(2 / np.pi, abs=0.03)   # one normal mode: eta = 2/pi
    assert d["valley_ratio"] > 0.8


def test_threshold_matches_reference_otsu():
    rng = np.random.default_rng(2)
    sample = np.concatenate([rng.normal(-0.25, 0.1, 5000), rng.normal(0.35, 0.1, 3000)])
    assert thresholds.diagnostics(_hist(sample))["threshold"] == pytest.approx(water.otsu_threshold(sample))


def test_parse_and_pool():
    h = thresholds.parse_hist(" ".join(["1"] * water.OTSU_BINS))
    assert h.sum() == water.OTSU_BINS
    assert thresholds.parse_hist(None) is None and thresholds.parse_hist(float("nan")) is None
    assert thresholds.pooled([h, None, h]).sum() == 2 * water.OTSU_BINS
    with pytest.raises(ValueError):
        thresholds.parse_hist("1 2 3")


def test_empty_histogram():
    d = thresholds.diagnostics(np.zeros(water.OTSU_BINS))
    assert d["n"] == 0 and np.isnan(d["threshold"])


def test_overlapping_bimodal_has_intermediate_diagnostics():
    rng = np.random.default_rng(4)
    h = _hist(np.concatenate([rng.normal(-0.2, 0.12, 5000), rng.normal(0.3, 0.12, 5000)]))
    d = thresholds.diagnostics(h)
    assert 0.75 < d["eta"] < 0.95 and d["valley_ratio"] < 0.6
