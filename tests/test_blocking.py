import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from caspian import blocking, design


def test_block_index_floor_division():
    i, j = blocking.block_index([0, 24_999, 25_000, -1], [0, 0, 50_000, -25_001], 25_000)
    assert i.tolist() == [0, 0, 1, -1]
    assert j.tolist() == [0, 0, 2, -2]


def test_block_grid_covers_bounds():
    g = blocking.block_grid((-30_000, 10_000, 40_000, 60_000), 25_000)
    assert g["i"].min() == -2 and g["i"].max() == 1
    assert g["j"].min() == 0 and g["j"].max() == 2
    assert np.allclose(g["xc"] - g["x0"], 12_500)


def test_assign_sectors_by_centre():
    g = blocking.block_grid((0, 0, 100_000, 50_000), 25_000)
    sec = {"a": box(0, 0, 50_000, 50_000), "b": box(50_000, 0, 100_000, 50_000)}
    out = blocking.assign_sectors(g, sec)
    assert len(out) == 8
    assert (out.loc[out["xc"] < 50_000, "sector"] == "a").all()
    assert (out.loc[out["xc"] > 50_000, "sector"] == "b").all()


def _blocks(n_a=37, n_b=12):
    rows = [{"i": k, "j": 0, "sector": "a"} for k in range(n_a)] + [{"i": k, "j": 1, "sector": "b"} for k in range(n_b)]
    return pd.DataFrame(rows)


def test_assign_splits_stratified_proportions():
    b = _blocks()
    s = blocking.assign_splits(b)
    assert s.notna().all()
    for sector, n in (("a", 37), ("b", 12)):
        counts = s[b["sector"] == sector].value_counts()
        for name, p in design.SPLIT.items():
            assert abs(counts.get(name, 0) - p * n) < 1.0


def test_assign_splits_deterministic_and_seed_dependent():
    b = _blocks()
    s1 = blocking.assign_splits(b)
    s2 = blocking.assign_splits(b.sample(frac=1, random_state=1))  # row order must not matter
    assert (s1 == s2.loc[s1.index]).all()
    s3 = blocking.assign_splits(b, seed=1)
    assert not (s1 == s3).all()


def test_split_of_points_lookup():
    b = pd.DataFrame({"i": [0, 1], "j": [0, 0]})
    s = pd.Series(["train", "test"])
    out = blocking.split_of_points([10, 30_000, 80_000], [10, 10, 10], b, s, 25_000)
    assert out[0] == "train" and out[1] == "test" and pd.isna(out[2])


def test_block_size_rule():
    assert blocking.block_size_for_range(20_000) == 25_000
    assert blocking.block_size_for_range(25_000) == 25_000
    assert blocking.block_size_for_range(30_000) == 50_000
    assert blocking.block_size_for_range(51_000) == 75_000


def test_semivariogram_recovers_exponential_range():
    # Gaussian field with exponential covariance, a = 4 km (practical range 12 km), no nugget
    rng = np.random.default_rng(0)
    n = 1200
    x, y = rng.uniform(0, 60_000, n), rng.uniform(0, 60_000, n)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    cov = np.exp(-d / 4_000) + 1e-9 * np.eye(n)
    r = np.linalg.cholesky(cov) @ rng.standard_normal(n)
    v = blocking.empirical_semivariogram(x, y, r, bin_width=1_000, max_lag=30_000)
    fit = blocking.fit_exponential(v)
    assert fit["practical_range"] == pytest.approx(12_000, rel=0.35)
    assert fit["nugget"] < 0.2


def test_spatial_structure_test_white_noise_and_smooth_field():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 200_000, 3000), rng.uniform(0, 200_000, 3000)
    noise = rng.normal(0, 0.3, x.size)
    t0 = blocking.spatial_structure_test(x, y, noise, short_lag=7_500, max_lag=150_000, n_perm=199)
    assert not t0["detected"] and t0["p"] > 0.05
    field = np.sin(x / 15_000) + np.cos(y / 20_000)          # structure at tens of km
    t1 = blocking.spatial_structure_test(x, y, field + noise, short_lag=7_500, max_lag=150_000, n_perm=199)
    assert t1["detected"] and t1["T"] < t0["T"]
