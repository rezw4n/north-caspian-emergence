"""Journal-revision analyses (PREREG Amendment 20, items 1-8; D-161).

Pure functions (arrays and tables in, arrays and tables out), tested in tests/test_rev3.py.
The drivers are scripts/rev3_*.py. Nothing here changes a registered result or Eq. (rule).

Contents
- item 1: ``marine_codes`` (sector codes of the marine domain), ``area_bias_rel`` (the
  article's relative area bias from summed cell counts), ``band_display_parts`` (the
  display rule of item 1e on the 250 m extents; reads two files);
- item 2: ``lattice_code``, ``quantile_from_counts`` and ``BlockDist`` (per block, the
  pooled absolute distances as counts of squared lattice lengths, so that an exact
  median and 95th percentile need no list of distances), ``boundary_blocks``;
- item 4: ``surface_stratum``, ``height_band``, ``HistBoot`` (block bootstrap of the mean,
  median and NMAD from per-block histograms), ``interval``;
- item 5: ``distance_to_blocks`` (distance from points to the nearest block of a set);
- item 6: ``native_cell_m``, ``half_metre_distance_m``, ``tier_levels``, ``tier_sector``,
  ``mbb_median_interval``;
- item 7: ``icc_oneway`` (intra-cluster correlation), ``kish_effective_n``;
- item 8: ``abs_summary``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .rev_s import block_indices

INF_K = np.iinfo(np.int64).max          # code of an infinite distance


# --- item 1: the marine domain ---------------------------------------------------------------

def marine_codes(sector_order, deltaic) -> list[int]:
    """1-based sector codes (the order of ``sector_order``) of the sectors that are not
    in ``deltaic``. Every deltaic name must be a sector."""
    missing = set(deltaic) - set(sector_order)
    if missing:
        raise ValueError(f"deltaic sectors not in the sector list: {sorted(missing)}")
    return [k + 1 for k, s in enumerate(sector_order) if s not in set(deltaic)]


def area_bias_rel(counts: dict) -> float:
    """The relative total area bias of caspian.hindcast_tiled.metrics_from_counts from
    cell counts summed over a region: ((tp + fp) - (tp + fn) + fp_out - fn_out) / (tp + fn).
    NaN when no observed water was counted."""
    obs = counts["tp"] + counts["fn"]
    if obs <= 0:
        return float("nan")
    return float((counts["fp"] - counts["fn"] + counts["fp_out"] - counts["fn_out"]) / obs)


def lake_mask_from_polygon(polygon_lonlat, transform, shape, crs) -> np.ndarray:
    """True for the cells of a north-up grid (``transform`` = (a, 0, c, 0, e, f) in ``crs``,
    ``shape`` = (rows, columns)) whose centre lies inside ``polygon_lonlat``, a shapely
    geometry in longitude and latitude (holes are outside). The polygon is projected
    vertex by vertex, so its edges must be short against the projection's curvature, as
    those of a surveyed shoreline are."""
    import pyproj
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    tf = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS(crs), always_xy=True)
    g = shapely.transform(polygon_lonlat, lambda xy: np.column_stack(tf.transform(xy[:, 0], xy[:, 1])))
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    return rasterize([(g, 1)], out_shape=tuple(shape), transform=Affine(a, 0.0, c, 0.0, e, f), fill=0,
                     all_touched=False, dtype="uint8").astype(bool)


def band_display_parts(extents_npz, glwd_tif, sector_polygons: dict, deltaic, *, saline_lake: int = 2,
                       outline_npz=None) -> dict:
    """Item 1e, the display rule, on the 250 m grid of the study-area extents (with
    ``lake_mask_from_polygon``, the only functions here that read files or project).

    ``extents_npz``: results/study_area/extents_v6_250m.npz (``high``, ``low``,
    ``transform``, ``crs``: the equal-area CRS). ``glwd_tif``: the GLWD v2 main-class
    raster (geographic). ``sector_polygons``: {sector id: lon/lat polygon}, in the order
    that defines the sector codes 1, 2, .... ``deltaic``: the ids of the deltaic sectors.
    ``outline_npz``: the Caspian's February 2000 shoreline as a mask on the same grid
    (``lake2000``, ``transform``; scripts/rev3_outline.py, from the HydroLAKES polygon).

    Returns arrays on the extents grid: ``band`` (water in the 1995-96 extent and not
    water in the 2025-26 extent, ``high == 1`` and ``low == 0``, as the figures draw it),
    ``sector`` (code of the sector that contains the cell centre, 0 outside the domain),
    ``lake2000_raster`` (GLWD main class ``saline_lake`` at the cell centre: the form the
    amendment named, 15 arc-second cells, which also marks other saline lakes),
    ``lake2000`` (the polygon mask if ``outline_npz`` is given, else the raster) and
    ``not_seabed`` and ``not_seabed_raster`` (band cells of the deltaic sectors outside
    each: not drawn as emerged seabed); and ``transform``."""
    import pyproj
    import rasterio
    import shapely
    from rasterio.windows import from_bounds

    ext = np.load(extents_npz)
    band = (np.asarray(ext["high"]) == 1) & (np.asarray(ext["low"]) == 0)
    tr = [float(v) for v in ext["transform"][:6]]
    rows, cols = band.shape
    X, Y = np.meshgrid(tr[2] + tr[0] * (np.arange(cols) + 0.5), tr[5] + tr[4] * (np.arange(rows) + 0.5))
    lon, lat = pyproj.Transformer.from_crs(pyproj.CRS(str(ext["crs"])), "EPSG:4326", always_xy=True).transform(X, Y)
    with rasterio.open(glwd_tif) as ds:
        # whole pixels only: a fractional window would be read resampled, half a pixel off its transform
        win = from_bounds(float(lon.min()) - 0.1, float(lat.min()) - 0.1, float(lon.max()) + 0.1,
                          float(lat.max()) + 0.1, ds.transform).round_offsets(op="floor").round_lengths(op="ceil")
        g = ds.read(1, window=win)
        gt = ds.window_transform(win)
    ci, ri = np.floor((lon - gt.c) / gt.a).astype(int), np.floor((lat - gt.f) / gt.e).astype(int)
    if ci.min() < 0 or ri.min() < 0 or ci.max() >= g.shape[1] or ri.max() >= g.shape[0]:
        raise ValueError("the GLWD window does not cover the extents grid")
    raster = g[ri, ci] == saline_lake
    if outline_npz is None:
        lake2000 = raster
    else:
        o = np.load(outline_npz)
        lake2000 = np.asarray(o["lake2000"]).astype(bool)
        if lake2000.shape != band.shape or not np.allclose(np.asarray(o["transform"], float)[:6], tr):
            raise ValueError("the outline mask is not on the grid of the extents")
    pts = shapely.points(lon.ravel(), lat.ravel())
    sec = np.zeros(band.size, np.int8)
    names = list(sector_polygons)
    for k, name in enumerate(names, start=1):
        poly = sector_polygons[name]
        shapely.prepare(poly)
        sec[shapely.contains(poly, pts)] = k
    sec = sec.reshape(band.shape)
    dcodes = [names.index(s) + 1 for s in deltaic]
    in_delta = band & np.isin(sec, dcodes)
    return {"band": band, "sector": sec, "lake2000": lake2000, "lake2000_raster": raster,
            "not_seabed": in_delta & ~lake2000, "not_seabed_raster": in_delta & ~raster,
            "transform": np.asarray(tr, float)}


# --- item 2: pooled distances per block -------------------------------------------------------

def lattice_code(d, pixel: float) -> np.ndarray:
    """|distance| between cell centres as the squared lattice length round((|d| / pixel)^2)
    (int64); ``INF_K`` for an infinite distance. NaN is refused."""
    d = np.abs(np.asarray(d, float))
    if np.isnan(d).any():
        raise ValueError("NaN distance: drop undefined distances first")
    inf = np.isinf(d)
    k = np.rint((np.where(inf, 0.0, d) / pixel) ** 2).astype(np.int64)
    k[inf] = INF_K
    return k


def _decode(k, pixel: float) -> float:
    return float("inf") if int(k) == INF_K else float(math.sqrt(int(k)) * pixel)


def quantile_from_counts(k, counts, q: float, pixel: float) -> float:
    """The ``q`` quantile (numpy's default linear rule) of the distances sqrt(k) x pixel,
    each repeated ``counts`` times; ``k`` sorted ascending, ``INF_K`` standing for +inf.
    Equals numpy.percentile(expanded, 100 q) for finite values; +inf when an order
    statistic that enters with a positive weight is infinite."""
    k = np.asarray(k, np.int64)
    counts = np.asarray(counts, np.int64)
    if k.size == 0 or counts.sum() == 0:
        return float("nan")
    if np.any(np.diff(k) < 0):
        raise ValueError("k must be sorted ascending")
    cum = np.cumsum(counts)
    n = int(cum[-1])
    h = (n - 1) * q
    lo, hi = int(math.floor(h)), int(math.ceil(h))
    v_lo = _decode(k[np.searchsorted(cum, lo, side="right")], pixel)
    v_hi = _decode(k[np.searchsorted(cum, hi, side="right")], pixel)
    w = h - lo
    if w == 0.0:
        return v_lo
    if math.isinf(v_lo) or math.isinf(v_hi):
        return float("inf")
    return float(v_lo + w * (v_hi - v_lo))


class BlockDist:
    """Per key and block, how often each absolute distance occurs, summed over the units
    it is fed. ``add`` takes block indices and distances of measured waterline cells (as
    scripts/a7_hindcast._measured returns them); ``table`` gives n, the median and the
    95th percentile per key and block."""

    def __init__(self, pixel: float):
        self.pixel = float(pixel)
        self.s: dict = {}

    def add(self, key, block_i, block_j, d) -> None:
        d = np.asarray(d, float)
        if d.size == 0:
            return
        vc = pd.DataFrame({"block_i": np.asarray(block_i, np.int64), "block_j": np.asarray(block_j, np.int64),
                           "k": lattice_code(d, self.pixel)}).value_counts()
        self.s[key] = vc if key not in self.s else self.s[key].add(vc, fill_value=0)

    def table(self) -> pd.DataFrame:
        rows = []
        for key, s in sorted(self.s.items(), key=lambda kv: str(kv[0])):
            df = s.astype(np.int64).rename("count").reset_index().sort_values(["block_i", "block_j", "k"])
            for (i, j), g in df.groupby(["block_i", "block_j"], sort=True):
                kk, cc = g["k"].to_numpy(), g["count"].to_numpy()
                rows.append({"key": key, "block_i": int(i), "block_j": int(j), "n": int(cc.sum()),
                             "median_abs_m": quantile_from_counts(kk, cc, 0.5, self.pixel),
                             "p95_abs_m": quantile_from_counts(kk, cc, 0.95, self.pixel)})
        return pd.DataFrame(rows, columns=["key", "block_i", "block_j", "n", "median_abs_m", "p95_abs_m"])


def boundary_blocks(blocks: pd.DataFrame) -> pd.Series:
    """For a block table with integer ``i``, ``j`` and ``sector``: True where one of the
    block's eight neighbours in the table lies in another sector."""
    sec = {(int(r.i), int(r.j)): r.sector for r in blocks.itertuples()}
    out = []
    for r in blocks.itertuples():
        here = sec[(int(r.i), int(r.j))]
        out.append(any(sec.get((int(r.i) + di, int(r.j) + dj), here) != here
                       for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)))
    return pd.Series(out, index=blocks.index, dtype=bool)


# --- item 4: strata of the lidar check --------------------------------------------------------

SURFACE_EDGES = (0.1, 0.5)                      # E3 emergent-vegetation share of the cell
SURFACE_NAMES = ("bare", "mixed", "vegetated")


def surface_stratum(veg_share) -> np.ndarray:
    """"bare" (share < 0.1), "mixed" (0.1 <= share < 0.5) or "vegetated" (share >= 0.5);
    "" where the share is not finite."""
    v = np.asarray(veg_share, float)
    out = np.full(v.shape, "", dtype=object)
    out[v < SURFACE_EDGES[0]] = SURFACE_NAMES[0]
    out[(v >= SURFACE_EDGES[0]) & (v < SURFACE_EDGES[1])] = SURFACE_NAMES[1]
    out[v >= SURFACE_EDGES[1]] = SURFACE_NAMES[2]
    return out


def height_band(h, lowest: float, step: float = 0.5, n_bands: int = 6) -> np.ndarray:
    """Index 0..n_bands-1 of the ``step`` m band of the model height above ``lowest`` (band
    k holds lowest + k step <= h < lowest + (k + 1) step; the last band is open above);
    -1 below ``lowest`` or where h is not finite."""
    h = np.asarray(h, float)
    k = np.floor((h - lowest) / step)
    out = np.where(np.isfinite(h) & (k >= 0), np.minimum(k, n_bands - 1), -1)
    return out.astype(np.int64)


class HistBoot:
    """Block bootstrap of the mean, median and NMAD of errors from per-block histograms.

    The draws are those of caspian.validation.block_bootstrap (blocks in sorted label
    order, ``rng.integers(0, n_blocks, n_blocks)`` per replicate from default_rng(seed)).
    The mean of a replicate is exact (block sums and counts). The median and the NMAD are
    read from the summed histogram of bin ``width`` with linear interpolation inside a
    bin. Where the bins around the median hold several points each they lie within about
    one bin width of the exact values; with few points the difference can reach the gap
    between neighbouring order statistics. Millions of points are resampled thousands of
    times without touching the points again. Use it for intervals, not for the point
    estimates."""

    NMAD_SCALE = 1.4826

    def __init__(self, e, blocks, width: float = 1e-3):
        e = np.asarray(e, float)
        lab = np.asarray(blocks).astype(str)
        if e.size == 0 or e.size != lab.size:
            raise ValueError("need one block label per error, and at least one error")
        if not np.all(np.isfinite(e)):
            raise ValueError("errors must be finite")
        self.width = float(width)
        self.lo = math.floor(float(e.min()) / self.width) * self.width
        nb = int(math.floor((float(e.max()) - self.lo) / self.width)) + 1
        uniq, inv = np.unique(lab, return_inverse=True)
        self.k = int(uniq.size)
        j = np.minimum(np.floor((e - self.lo) / self.width).astype(np.int64), nb - 1)
        self.H = np.bincount(inv * nb + j, minlength=self.k * nb).reshape(self.k, nb).astype(np.float64)
        self.sum = np.bincount(inv, weights=e, minlength=self.k)
        self.n = np.bincount(inv, minlength=self.k).astype(np.float64)
        self.edges_lo = self.lo + self.width * np.arange(nb)

    @staticmethod
    def _median(h, lo: float, width: float) -> float:
        cum = np.cumsum(h)
        half = cum[-1] / 2.0
        i = int(np.searchsorted(cum, half, side="left"))
        below = cum[i - 1] if i > 0 else 0.0
        return float(lo + width * (i + (half - below) / h[i]))

    def stats(self, w_block) -> tuple[float, float, float]:
        """(mean, median, NMAD) of the points of the blocks weighted by ``w_block``."""
        w = np.asarray(w_block, float)
        h = w @ self.H
        mean = float((w * self.sum).sum() / (w * self.n).sum())
        med = self._median(h, self.lo, self.width)
        centres = self.edges_lo + self.width / 2.0
        a = np.abs(centres - med)
        ja = np.floor(a / self.width).astype(np.int64)
        ha = np.bincount(ja, weights=h, minlength=int(ja.max()) + 1)
        return mean, med, float(self.NMAD_SCALE * self._median(ha, 0.0, self.width))

    def replicates(self, n_boot: int, seed: int) -> np.ndarray:
        """[n_boot, 3]: mean, median and NMAD of each replicate."""
        rng = np.random.default_rng(seed)
        out = np.empty((n_boot, 3))
        for r in range(n_boot):
            out[r] = self.stats(np.bincount(rng.integers(0, self.k, self.k), minlength=self.k))
        return out


def interval(boot, conf: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap replicates (as caspian.validation)."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan"), float("nan")
    a = (1 - conf) / 2 * 100
    lo, hi = np.percentile(b, [a, 100 - a])
    return float(lo), float(hi)


# --- item 5: distance of points to a set of square blocks -------------------------------------

def distance_to_blocks(x, y, x0, y0, size: float) -> np.ndarray:
    """Distance (same units) from each point to the nearest of the squares
    [x0, x0 + size] x [y0, y0 + size]; zero inside a square; +inf with no square."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(x.shape, np.inf)
    for bx, by in zip(np.asarray(x0, float), np.asarray(y0, float)):
        dx = np.maximum(np.maximum(bx - x, x - (bx + size)), 0.0)
        dy = np.maximum(np.maximum(by - y, y - (by + size)), 0.0)
        np.minimum(out, np.hypot(dx, dy), out=out)
    return out


# --- item 6: practical size of the misses ------------------------------------------------------

EARTH_RADIUS_M = 6_371_008.8                    # mean radius (IUGG), for the cell's longer side


def native_cell_m(arc_seconds: float) -> float:
    """The longer (meridional) side of a geographic cell of ``arc_seconds``, in metres on a
    sphere of the Earth's mean radius. The zonal side is shorter by cos(latitude)."""
    return float(EARTH_RADIUS_M * math.radians(arc_seconds / 3600.0))


def half_metre_distance_m(gradient_m_per_km: float, height_m: float = 0.5) -> float:
    """The horizontal distance (m) that corresponds to ``height_m`` at a shelf gradient."""
    if not gradient_m_per_km > 0:
        return float("inf")
    return float(height_m / gradient_m_per_km * 1000.0)


TIERS = ("not_evaluable", "below_floor", "resolution", "material")


def tier_levels(D, Q, D_control, evaluable, cell_m: float, dist_m: float) -> np.ndarray:
    """Per level: "not_evaluable"; "below_floor" (D <= Q); "material" (D above the floor,
    the control's displacement, one native cell and the 0.5 m distance); otherwise
    "resolution". A level whose control displacement is not finite cannot be material.
    D = +inf (no predicted shoreline) is above every finite threshold."""
    D, Q, C = (np.asarray(a, float) for a in (D, Q, D_control))
    ev = np.asarray(evaluable, bool)
    out = np.full(D.shape, TIERS[0], dtype=object)
    above = ev & (D > Q)
    out[ev & ~above] = TIERS[1]
    material = above & np.isfinite(C) & (D > C) & (D > cell_m) & (D > dist_m)
    out[above & ~material] = TIERS[2]
    out[material] = TIERS[3]
    return out


def tier_sector(tiers) -> dict:
    """Counts and the class of a grid in a sector: "pass" when the registered rule passes
    (levels above the floor are not more than half of the evaluable ones); "material" when
    the material levels are more than half; "resolution" otherwise; None with no
    evaluable level."""
    t = np.asarray(tiers, dtype=object)
    n_ev = int((t != TIERS[0]).sum())
    n_above = int(np.isin(t, TIERS[2:]).sum())
    n_mat = int((t == TIERS[3]).sum())
    if n_ev == 0:
        cls = None
    elif not n_above > n_ev / 2:
        cls = "pass"
    elif n_mat > n_ev / 2:
        cls = "material"
    else:
        cls = "resolution"
    return {"n_evaluable": n_ev, "n_above_floor": n_above, "n_material": n_mat,
            "n_resolution": n_above - n_mat, "class": cls}


def mbb_median_interval(values, material, n_boot: int, seed: int, block: int = 3, conf: float = 0.95) -> dict:
    """Moving-block bootstrap over levels (caspian.rev_s.block_indices): the median of
    ``values`` (levels in order) with its percentile interval, and the share of resamples
    in which the True entries of ``material`` are more than half. Infinite values stay in
    (the median of a resample may be infinite)."""
    v = np.asarray(values, float)
    m = np.asarray(material, bool)
    n = v.size
    if n == 0:
        return {"n_levels": 0, "median": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_material_majority": float("nan")}
    rng = np.random.default_rng(seed)
    k = max(1, math.ceil(n / block))
    med = np.empty(n_boot)
    maj = np.zeros(n_boot, bool)
    for b in range(n_boot):
        idx = block_indices(n, rng.random(k), block)
        med[b] = np.median(v[idx])
        maj[b] = m[idx].sum() > n / 2
    a = (1 - conf) / 2 * 100
    lo, hi = np.percentile(med, [a, 100 - a])
    return {"n_levels": int(n), "median": float(np.median(v)), "lo": float(lo), "hi": float(hi),
            "p_material_majority": float(maj.mean())}


# --- item 7: clustering of the lidar points ----------------------------------------------------

def icc_oneway(values, clusters) -> dict:
    """Intra-cluster correlation of ``values`` from a one-way analysis of variance with
    unequal cluster sizes: rho = (MSB - MSW) / (MSB + (m0 - 1) MSW), with
    m0 = (N - sum(m_i^2) / N) / (k - 1); floored at zero. Returns rho, k clusters, N,
    m0 and the mean squares. NaN with fewer than two clusters or no within-cluster
    degrees of freedom."""
    v = np.asarray(values, float)
    lab = np.asarray(clusters)
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    uniq, inv = np.unique(lab.astype(str), return_inverse=True)
    k, n = int(uniq.size), int(v.size)
    out = {"k": k, "n": n, "rho": float("nan"), "m0": float("nan"), "msb": float("nan"), "msw": float("nan")}
    if k < 2 or n <= k:
        return out
    m = np.bincount(inv, minlength=k).astype(float)
    mean_i = np.bincount(inv, weights=v, minlength=k) / m
    grand = float(v.mean())
    ssb = float((m * (mean_i - grand) ** 2).sum())
    ssw = float(((v - mean_i[inv]) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - float((m ** 2).sum()) / n) / (k - 1)
    den = msb + (m0 - 1) * msw
    rho = (msb - msw) / den if den > 0 else float("nan")
    out.update(rho=float(max(rho, 0.0)) if np.isfinite(rho) else float("nan"), m0=float(m0), msb=float(msb),
               msw=float(msw))
    return out


def kish_effective_n(cluster_sizes, rho: float) -> dict:
    """Kish's design effect for unequal clusters, deff = 1 + (m' - 1) rho with
    m' = sum(m_i^2) / sum(m_i), and the effective sample size N / deff. With rho = 0 the
    effective size is N; with rho = 1 it is N / m'."""
    m = np.asarray(cluster_sizes, float)
    n = float(m.sum())
    if n <= 0 or not np.isfinite(rho):
        return {"n": int(n), "m_weighted": float("nan"), "deff": float("nan"), "n_eff": float("nan")}
    mw = float((m ** 2).sum() / n)
    deff = 1.0 + (mw - 1.0) * float(rho)
    return {"n": int(n), "m_weighted": mw, "deff": float(deff), "n_eff": float(n / deff)}


# --- item 8: summaries of absolute departures --------------------------------------------------

def abs_summary(x, thresholds=(0.10, 0.20)) -> dict:
    """Standard deviation, and the median and 95th percentile of |x|, with the shares of
    |x| above each threshold; non-finite values dropped."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "sd": float("nan"), "abs_median": float("nan"), "abs_p95": float("nan"),
                **{f"share_above_{t:g}": float("nan") for t in thresholds}}
    a = np.abs(x)
    return {"n": int(x.size), "sd": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
            "abs_median": float(np.median(a)), "abs_p95": float(np.percentile(a, 95)),
            **{f"share_above_{t:g}": float((a > t).mean()) for t in thresholds}}
