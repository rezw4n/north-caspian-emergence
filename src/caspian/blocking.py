"""Spatial blocking for validation and calibration (PREREG section 6).

Blocks are squares on the equal-area grid (D-002), aligned to multiples of the block
size. Each block takes the sector that contains its centre. Blocks are randomly
assigned to training, calibration and test, stratified by sector, with the
pre-registered seed. The block size is checked before any test-block evaluation
against the practical range of an exponential semivariogram fitted to residuals in
training blocks; if the range is longer, the size grows to the next multiple of the
base size at or above it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import design


def block_index(x, y, size_m: float = design.BLOCK_SIZE_M):
    """Integer block indices (i, j) of points (x, y) in equal-area metres."""
    return (np.floor(np.asarray(x, float) / size_m).astype(int),
            np.floor(np.asarray(y, float) / size_m).astype(int))


def block_grid(bounds_xy, size_m: float = design.BLOCK_SIZE_M) -> pd.DataFrame:
    """Every block whose square intersects the bounds (xmin, ymin, xmax, ymax), with
    its lower-left corner and centre."""
    xmin, ymin, xmax, ymax = bounds_xy
    i0, j0 = math.floor(xmin / size_m), math.floor(ymin / size_m)
    i1, j1 = math.ceil(xmax / size_m) - 1, math.ceil(ymax / size_m) - 1
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    df = pd.DataFrame({"i": ii.ravel(), "j": jj.ravel()})
    df["x0"], df["y0"] = df["i"] * size_m, df["j"] * size_m
    df["xc"], df["yc"] = df["x0"] + size_m / 2, df["y0"] + size_m / 2
    return df


def assign_sectors(blocks: pd.DataFrame, sector_polys: dict) -> pd.DataFrame:
    """Sector of each block centre (``sector_polys``: name -> shapely polygon in the
    same equal-area CRS). Blocks whose centre lies in no sector are dropped."""
    names = []
    for xc, yc in zip(blocks["xc"], blocks["yc"]):
        p = Point(xc, yc)
        names.append(next((n for n, poly in sector_polys.items() if poly.contains(p)), None))
    out = blocks.assign(sector=names)
    return out[out["sector"].notna()].reset_index(drop=True)


def _largest_remainder(n: int, proportions) -> list[int]:
    raw = [p * n for p in proportions]
    counts = [math.floor(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda k: (-(raw[k] - counts[k]), k))
    for k in order[: n - sum(counts)]:
        counts[k] += 1
    return counts


def assign_splits(blocks: pd.DataFrame, proportions: dict = design.SPLIT, seed: int = design.SEED,
                  by: str = "sector") -> pd.Series:
    """Random split label per block, stratified by ``by``. Within each stratum (in
    sorted order) the blocks are sorted by (i, j), permuted with one generator
    seeded once, and split by largest-remainder counts, so the result depends only
    on the block set, the proportions and the seed."""
    rng = np.random.default_rng(seed)
    labels = pd.Series(index=blocks.index, dtype=object)
    names, props = list(proportions), list(proportions.values())
    for _, grp in sorted(blocks.groupby(by), key=lambda kv: str(kv[0])):
        idx = grp.sort_values(["i", "j"]).index.to_numpy()
        idx = idx[rng.permutation(len(idx))]
        start = 0
        for name, c in zip(names, _largest_remainder(len(idx), props)):
            labels.loc[idx[start:start + c]] = name
            start += c
    return labels


def split_of_points(x, y, blocks: pd.DataFrame, split: pd.Series, size_m: float = design.BLOCK_SIZE_M):
    """Split label of each point (NaN if its block is not in ``blocks``)."""
    i, j = block_index(x, y, size_m)
    lut = {(a, b): s for a, b, s in zip(blocks["i"], blocks["j"], split)}
    return np.array([lut.get((a, b), np.nan) for a, b in zip(i, j)], dtype=object)


def empirical_semivariogram(x, y, r, bin_width: float, max_lag: float, max_pairs: int = 400_000,
                            seed: int = design.SEED) -> pd.DataFrame:
    """Binned semivariogram gamma(h) = mean of (r_i - r_j)^2 / 2 over point pairs.
    With more than ``max_pairs`` pairs, a random subset of pairs is used."""
    x, y, r = (np.asarray(v, float) for v in (x, y, r))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    x, y, r = x[ok], y[ok], r[ok]
    n = len(r)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        a, b = np.triu_indices(n, k=1)
    else:
        a = rng.integers(0, n, max_pairs)
        b = rng.integers(0, n, max_pairs)
        keep = a != b
        a, b = a[keep], b[keep]
    h = np.hypot(x[a] - x[b], y[a] - y[b])
    g = 0.5 * (r[a] - r[b]) ** 2
    edges = np.arange(0, max_lag + bin_width, bin_width)
    k = np.digitize(h, edges) - 1
    rows = []
    for m in range(len(edges) - 1):
        sel = k == m
        if sel.any():
            rows.append({"lag": h[sel].mean(), "gamma": g[sel].mean(), "n_pairs": int(sel.sum())})
    return pd.DataFrame(rows)


def fit_exponential(vario: pd.DataFrame) -> dict:
    """Weighted (by pair count) least-squares fit of gamma(h) = c0 + c1 (1 - exp(-h/a)).
    Returns nugget c0, partial sill c1, a, and the practical range 3a (95% of the sill)."""
    from scipy.optimize import curve_fit

    h, g, w = vario["lag"].to_numpy(), vario["gamma"].to_numpy(), vario["n_pairs"].to_numpy(float)

    def model(hh, c0, c1, a):
        return c0 + c1 * (1 - np.exp(-hh / a))

    p0 = [max(g.min(), 0.0), max(g.max() - g.min(), 1e-12), max(h.max() / 3, 1.0)]
    (c0, c1, a), _ = curve_fit(model, h, g, p0=p0, sigma=1 / np.sqrt(w),
                               bounds=([0, 0, 1e-6], [np.inf, np.inf, np.inf]), maxfev=20000)
    return {"nugget": float(c0), "partial_sill": float(c1), "a": float(a), "practical_range": float(3 * a)}


def block_size_for_range(practical_range: float, base: float = design.BLOCK_SIZE_M) -> float:
    """PREREG section 6: keep the base size if the practical range does not exceed
    it, otherwise the next multiple of the base at or above the range."""
    if practical_range <= base:
        return float(base)
    return float(math.ceil(practical_range / base) * base)
