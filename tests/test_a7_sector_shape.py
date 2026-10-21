"""scripts/a7_sector_shape.py: per-sector D-070 counts and the smoothed offset map."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a7_sector_shape", ROOT / "scripts" / "a7_sector_shape.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)
a7 = ss.a7


def _case(seed=0, shape=(37, 53)):
    rng = np.random.default_rng(seed)
    F = rng.normal(0.0, 1.0, shape).astype(np.float32)
    code = rng.choice([-2, -1, 0, 1], size=shape)
    outside = rng.choice([-1, 0, 1], size=shape)
    region = rng.random(shape) < 0.8
    sector = rng.integers(0, 4, shape)
    fin = np.isfinite(F) & (rng.random(shape) < 0.95)
    finite = lambda r0, r1, c0, c1: fin[r0:r1, c0:c1]  # noqa: E731
    return F, code, outside, finite, region, sector


def test_sector_counts_sum_to_unit_counts_and_match_per_sector_regions():
    F, code, outside, finite, region, sector = _case()
    T = np.linspace(-2, 2, 41)
    cnt, water = ss.unit_counts_by_sector(F, code, outside, finite, region, sector, 3, T, tile=16)
    c_all, w_all = a7.unit_counts(F, code, outside, finite, region, T, tile=16)
    assert np.array_equal(cnt.sum(axis=0), c_all) and water.sum() == w_all
    for k in range(4):
        ck, wk = a7.unit_counts(F, code, outside, finite, region & (sector == k), T, tile=16)
        assert np.array_equal(cnt[k], ck) and water[k] == wk


def test_sector_ids_outside_range_count_as_no_sector():
    F, code, outside, finite, region, sector = _case(1)
    sector = sector.copy()
    sector[:5] = 9
    T = np.linspace(-1, 1, 5)
    cnt, water = ss.unit_counts_by_sector(F, code, outside, finite, region, sector, 3, T, tile=16)
    c_all, w_all = a7.unit_counts(F, code, outside, finite, region, T, tile=16)
    assert np.array_equal(cnt.sum(axis=0), c_all) and water.sum() == w_all


def test_offset_map_is_constant_for_equal_offsets_and_smooth_across_an_edge():
    sector = np.zeros((96, 160), int)
    sector[:, 80:] = 1
    out = np.zeros(sector.shape, np.float32)
    ss.offset_map(sector, {1: -1.5}, -1.5, out, coarse=4, sigma_m=480.0, pixel=60.0, tile=32)
    assert np.allclose(out, -1.5, atol=1e-6)
    ss.offset_map(sector, {1: 1.0}, 0.0, out, coarse=4, sigma_m=480.0, pixel=60.0, tile=32)
    row = out[48]
    assert abs(row[0]) < 1e-3 and abs(row[-1] - 1.0) < 1e-3          # far from the edge: the constants
    assert np.all(np.diff(row) >= -1e-6)                               # monotone across the edge
    assert np.max(np.diff(row)) < 0.2                                  # no step at the edge
