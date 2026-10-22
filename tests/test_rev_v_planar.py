"""scripts/rev_v_planar.py: the quantised (approximate-search) counts equal the exact
D-070 counts of scripts/a7_hindcast.unit_counts, and the planar correction map."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rev_v_planar", ROOT / "scripts" / "rev_v_planar.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
a7 = rp.a7


def _fitdata(tmp_path, seed=0, shape=(30, 41), n_units=4):
    rng = np.random.default_rng(seed)
    region = rng.random(shape) < 0.7
    idx = np.flatnonzero(region.ravel()).astype(np.int64)
    known = np.stack([np.packbits(rng.random(idx.size) < 0.8) for _ in range(n_units)])
    water = np.stack([np.packbits(rng.random(idx.size) < 0.4) for _ in range(n_units)])
    bins = np.arange(20, 20 + n_units)
    levels = np.array([a7._bin_level(int(b)) for b in bins])
    transform = np.array([60.0, 0.0, -1200.0, 0.0, -60.0, 900.0])
    path = tmp_path / "m.npz"
    np.savez(path, idx=idx, known=known, water=water, n=np.int64(idx.size), bins=bins,
             epochs=np.array(["E3"] * n_units), levels=levels, shape=np.array(shape), transform=transform)
    fd = rp.FitData(path)
    F = rng.normal(-27.5, 1.5, shape)
    F[rng.random(shape) < 0.02] = np.inf
    fin = np.isfinite(F) | (rng.random(shape) < 0.5)
    return fd, F, fin.ravel()[idx]


def _brute(fd, Fv, finite, c_grid):
    out = np.zeros((len(fd.levels), len(c_grid)), np.int64)
    for u, L in enumerate(fd.levels):
        k = fd.unit(u) & finite
        w = int((fd.unit(u, "water") & finite).sum())
        for j, c in enumerate(c_grid):
            out[u, j] = int((np.asarray(Fv[k], float) < float(L) + c).sum()) - w
    return out


def test_exact_bias_equals_brute_force(tmp_path):
    fd, F, finite = _fitdata(tmp_path)
    Fv = F.ravel()[fd.idx]
    c = np.round(np.arange(-300, 301, 7) / 100.0, 2)
    assert np.array_equal(rp.exact_bias(fd, Fv, finite, c), _brute(fd, Fv, finite, c))


def test_approx_counts_equal_exact_counts_of_the_shifted_map(tmp_path):
    fd, F, finite = _fitdata(tmp_path, seed=3)
    Fv = F.ravel()[fd.idx]
    c = np.round(np.arange(-300, 301) / 100.0, 2)
    ap = rp.Approx(fd, Fv, finite, 1, c)
    for a, b in [(0.0, 0.0), (0.004, -0.003), (-0.01, 0.002)]:
        want = rp.exact_bias(fd, Fv - (a * fd.x + b * fd.y), finite, c)
        assert np.array_equal(ap.bias(a, b), want)
    # the profile follows the D-070 rule (a7_hindcast.fit_offset on its own grid)
    ap_full = rp.Approx(fd, Fv, finite, 1)
    assert ap_full.profile(0.0, 0.0)["c0_m"] == a7.fit_offset(rp.exact_bias(fd, Fv, finite), fd.epochs)["offset_m"]


def test_subsample_and_xy(tmp_path):
    fd, F, finite = _fitdata(tmp_path, seed=5)
    ap = rp.Approx(fd, F.ravel()[fd.idx], finite, 2)
    assert ap.x.size == int(((fd.rows % 2 == 0) & (fd.cols % 2 == 0)).sum()) and ap.scale == 4
    # cell centres in km: column 0 is at x0 + 30 m
    j = np.flatnonzero(fd.cols == 0)[0]
    assert np.isclose(fd.x[j], (-1200.0 + 30.0) / 1000.0)


def test_plane_corr_subtracts_the_plane(tmp_path):
    shape, tile = (20, 33), 8
    transform = [60.0, 0.0, -600.0, 0.0, -60.0, 600.0]
    corr = np.random.default_rng(0).normal(0, 0.1, shape).astype(np.float32)
    out = rp.plane_corr(corr, 0.01, -0.02, transform, shape, tile, tmp_path / "pc.npy")
    x = (transform[2] + 60.0 * (np.arange(shape[1]) + 0.5)) / 1000.0
    y = (transform[5] - 60.0 * (np.arange(shape[0]) + 0.5)) / 1000.0
    want = corr - (0.01 * x[None, :] - 0.02 * y[:, None])
    assert np.allclose(out, want, atol=1e-6)


def test_sector_counts_sum_to_domain_counts_and_approx_matches(tmp_path):
    fd, F, finite = _fitdata(tmp_path, seed=7)
    Fv = F.ravel()[fd.idx]
    sec = np.random.default_rng(1).integers(1, 5, fd.n)
    c = np.round(np.arange(-300, 301) / 100.0, 2)
    bs = rp.exact_bias_sector(fd, Fv, finite, sec, c)
    assert np.array_equal(bs.sum(axis=1), rp.exact_bias(fd, Fv, finite, c))
    # brute force for one sector
    s = 3
    u = 1
    k = fd.unit(u) & finite & (sec == s)
    w = int((fd.unit(u, "water") & finite & (sec == s)).sum())
    want = np.array([int((np.asarray(Fv[k], float) < float(fd.levels[u]) + cc).sum()) - w for cc in c])
    assert np.array_equal(bs[u, s], want)
    # the quantised approximation per sector equals the exact counts of the shifted map
    ap = rp.Approx(fd, Fv, finite, 1, c, sector=sec)
    a, b = 0.003, -0.002
    exact = rp.exact_bias_sector(fd, Fv - (a * fd.x.astype(float) + b * fd.y.astype(float)), finite, sec, c)
    assert np.array_equal(ap.bias(a, b), exact.reshape(-1, c.size))
