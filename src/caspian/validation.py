"""A6 validation: the waterline DEM against independent heights (PREREG sections 2 and
5-8; Amendments 2 and 5).

Reference heights
-----------------
ICESat-2 ATL06 and ATL08 segments (scripts/a6_icesat2.py) pass ordered quality steps
(``icesat2_steps``), each counted (``cascade``), and are kept only over seabed that was
dry at acquisition (``dry_at_acquisition``: the acquisition is later than the
emergence date plus design.EMERGENCE_LAG_DAYS for every DEM pixel within
``DRY_RADIUS_PX`` of the point). ATL06 and ATL08 measure largely the same photons, so
they are evaluated separately and never pooled. GEDI L2A shots (Amendment 2) use
``gedi_steps`` and the same dryness test.

Datum chain (docs/DATUMS.md; PREREG section 5)
---------------------------------------------
    h_ell (WGS84 ellipsoid, ITRF2014) - N (EGM2008)  = H_EGM2008       ``orthometric``
    H_EGM2008 + c                                    = DEM reference   ``apply_datum_constant``

The DEM is on the DAHITI reference (scripts/a6_dem.py). N is the ATL06 ``dem/geoid_h``.
ATL08 carries no geoid, so N is interpolated in time along the same beam of the ATL06
granule of the same orbit (``geoid_along_track``). c is the median of DEM - H_EGM2008
over training-block points, with a block-bootstrap σ (``fit_datum_constant``). The
DAHITI -> BHS-77 offset is common to the DEM, the products and the reference heights,
so it cancels in every metric and score computed here.

Products (PREREG section 5) are compared twice: ``matched``, after a constant offset
fitted on training blocks (product reference estimated, as for the DEM's c); and
``total``, on the documented reference where there is one (EGM2008 for ETOPO 2022,
docs/DATUMS.md, so z + c), otherwise on the estimated one (= ``matched``).

Separation of splits
--------------------
Every fitted quantity is computed by a function that takes the whole point table and
selects its own split through ``split_rows``; none of them takes a split argument:
- training blocks: c, the product offsets, the scale model β, the semivariogram;
- calibration blocks: q̂ (Mondrian with the pooled fallback, caspian.conformal);
- test blocks: accuracy metrics, the paired block bootstrap and coverage.
Points whose block is not in the block table (split NaN) enter nothing. ``run`` fits
everything before it reads a test row.

Point table columns: x, y (equal-area m), time, h_orth (H_EGM2008), dem_h (DEM height),
interp_dist (m), contour_density (points per km²), slope (m/m), block, sector, split,
and one column per product height (native reference). ``y_ref`` is added by
``apply_datum_constant``. Errors are model minus reference (positive: model too high).

The H2 decision rule (``h2_decision``) is a separate function. The validation driver
(scripts/a6_validate_dem.py) does not call it; the claim is made at G3 from the frozen
tables.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import blocking, conformal, design
from .geography import SECTOR_ORDER

SPLITS = ("train", "calib", "test")
NMAD_SCALE = 1.4826

# Defaults where the pre-registration is silent (proposed for docs/DECISIONS.md).
GROSS_ERROR_M = 50.0            # |h - product reference DEM| bound; mirrors Amendment 2 (GEDI)
GEOID_MAX_GAP_S = 0.1           # ATL06 geoid samples bracketing an ATL08 time (~0.7 km along track)
DRY_RADIUS_PX = {"ATL06": 1, "ATL08": 2, "GEDI": 1}   # segment half-length or footprint, in 30 m pixels
CONTOUR_RADIUS_M = 500.0        # contour-density radius (scale covariate, PREREG section 8)
ABS_RESID_FLOOR_M = 1e-3        # |residual| floor before the log in the scale model
VARIO_BIN_M = 2_500.0           # semivariogram bin width (PREREG section 6)
VARIO_MAX_LAG_M = 150_000.0
VARIO_MAX_PAIRS = 2_000_000
VARIO_SHORT_LAG_M = 7_500.0        # short-lag window of the spatial-structure test (D-069)
GEDI_EARLY_BEAMS = ("BEAM0000", "BEAM0001")
GEDI_BEAM_FIX = np.datetime64("2020-02-06")        # Amendment 2: earlier shots of these beams dropped
# Products whose vertical reference is documented as EGM2008 (docs/DATUMS.md); every other
# product's reference in the Caspian is estimated (PREREG section 5).
DOCUMENTED_EGM2008 = ("etopo_2022",)
SCALE_TERMS = ("intercept", "log1p_interp_dist", "contour_density", "slope")

ICESAT2_COLUMNS = {
    "ATL06": ["lat", "lon", "h", "h_sigma", "quality", "geoid_h", "dem_h", "time", "beam", "sc_orient"],
    "ATL08": ["lat", "lon", "h", "h_sigma", "watermask", "landcover", "cloud", "msw", "night", "dem_h",
              "time", "beam", "sc_orient"],
}


class ProductSampler(Protocol):
    """Product heights at points: metres, positive up, on the product's native vertical
    reference, at WGS84 lon/lat; NaN where the product has no value. The driver builds
    one from caspian.products (load_native, then sample)."""

    def __call__(self, name: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray: ...


# --- splits ---------------------------------------------------------------------------

def split_rows(points: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split. The only way the fitting and evaluation functions below read
    points, so the split each quantity may use is fixed here and in the caller's name."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    s = points["split"]
    bad = set(pd.unique(s.dropna())) - set(SPLITS)
    if bad:
        raise ValueError(f"unknown split labels {sorted(map(str, bad))}")
    return points.loc[(s == split).to_numpy(bool)]


def attach_blocks(points: pd.DataFrame, blocks: pd.DataFrame, size_m: float) -> pd.DataFrame:
    """Block id (``i_j``), sector and split of each point from a block table
    (scripts/a6_blocks.py). Points whose block is not in the table get NaN sector and
    split. Raises if the table is not on ``size_m`` blocks."""
    if not np.allclose(blocks["x0"], blocks["i"] * size_m) or not np.allclose(blocks["y0"], blocks["j"] * size_m):
        raise ValueError(f"block table is not on {size_m} m blocks")
    i, j = blocking.block_index(points["x"], points["y"], size_m)
    lut = blocks.set_index(["i", "j"])[["sector", "split"]]
    sub = lut.reindex(pd.MultiIndex.from_arrays([i, j]))
    return points.assign(block=[f"{a}_{b}" for a, b in zip(i, j)],
                         sector=sub["sector"].to_numpy(object), split=sub["split"].to_numpy(object))


# --- reference heights ----------------------------------------------------------------

def granule_key(name: str) -> str:
    """Orbit part of an ICESat-2 granule name, shared by ATL06 and ATL08 of one orbit:
    ``ATL08_20181015000939_02500106_007_01.h5`` -> ``20181015000939_02500106``."""
    parts = Path(name).name.split("_")
    return f"{parts[1]}_{parts[2]}"


def read_icesat2_csv(path, product: str) -> pd.DataFrame:
    """One granule file of scripts/a6_icesat2.py with the columns used here; time as
    datetime64[ns]."""
    d = pd.read_csv(path, usecols=ICESAT2_COLUMNS[product])
    d["time"] = pd.to_datetime(d["time"], format="ISO8601").to_numpy().astype("datetime64[ns]")
    d["beam"] = d["beam"].astype(str)
    return d


def _seconds(t) -> np.ndarray:
    return np.asarray(t, "datetime64[ns]").astype("int64") / 1e9


def _days(t) -> np.ndarray:
    return np.asarray(t, "datetime64[ns]").astype("int64") / 86_400e9


def geoid_along_track(t, beam, ref_t, ref_beam, ref_geoid, max_gap_s: float = GEOID_MAX_GAP_S) -> np.ndarray:
    """Geoid height at times ``t`` on ``beam``, linearly interpolated in time between
    reference samples of the same beam (the ATL06 dem/geoid_h, EGM2008, of the same
    orbit). NaN where no pair of reference samples at most ``max_gap_s`` apart brackets
    the time."""
    t, ref_t = _seconds(t), _seconds(ref_t)
    beam, ref_beam = np.asarray(beam).astype(str), np.asarray(ref_beam).astype(str)
    ref_geoid = np.asarray(ref_geoid, float)
    out = np.full(t.shape, np.nan)
    for b in np.unique(beam):
        sel = beam == b
        r = (ref_beam == b) & np.isfinite(ref_geoid) & np.isfinite(ref_t)
        if not r.any():
            continue
        o = np.argsort(ref_t[r], kind="stable")
        rt, rg = ref_t[r][o], ref_geoid[r][o]
        tt = t[sel]
        k = np.searchsorted(rt, tt, side="right")          # rt[k-1] <= tt < rt[k]
        lo = np.clip(k - 1, 0, rt.size - 1)
        hi = np.clip(k, 0, rt.size - 1)
        exact = (k >= 1) & (rt[lo] == tt)
        ok = (k >= 1) & (k < rt.size) & ((rt[hi] - rt[lo]) <= max_gap_s)
        span = np.where(rt[hi] > rt[lo], rt[hi] - rt[lo], 1.0)
        val = rg[lo] + (tt - rt[lo]) / span * (rg[hi] - rg[lo])
        out[sel] = np.where(exact, rg[lo], np.where(ok, val, np.nan))
    return out


def orthometric(h_ell, geoid) -> np.ndarray:
    """H = h - N: ellipsoidal height to height above the geoid model."""
    return np.asarray(h_ell, float) - np.asarray(geoid, float)


def strong_beam(beam, sc_orient) -> np.ndarray:
    """Strong beams: the left beams when sc_orient = 0 (backward), the right beams when
    1 (forward); none in transition (scripts/a6_icesat2.py)."""
    b = np.asarray(beam).astype(str)
    o = np.asarray(sc_orient)
    return ((o == 0) & np.char.endswith(b, "l")) | ((o == 1) & np.char.endswith(b, "r"))


def _gross(h, ref) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.abs(np.asarray(h, float) - np.asarray(ref, float)) <= GROSS_ERROR_M


def icesat2_steps(d: pd.DataFrame, product: str) -> dict[str, np.ndarray]:
    """Ordered quality steps of one product, each a boolean per row (True = passes).
    ATL08 needs a ``geoid_h`` column (``geoid_along_track``).

    Both: finite height; strong beam; finite geoid height; gross-error screen
    |h - dem_h| <= GROSS_ERROR_M (the product's reference DEM, ellipsoidal). ATL06:
    atl06_quality_summary = 0. ATL08 (atmospheric screens): msw_flag = 0,
    cloud_flag_atm = 0. segment_watermask is not used (D-069): whether a point was dry
    is decided by the dryness test against our own emergence dates (PREREG section 2),
    and a mask that does not follow the retreating shoreline would remove exactly the
    emerged seabed being validated. The height uncertainty is not screened (it is not
    used, and the ATL06 quality summary already covers it)."""
    if product not in ("ATL06", "ATL08"):
        raise ValueError(f"unknown product {product!r}")
    h = d["h"].to_numpy(float)
    steps = {"height": np.isfinite(h), "strong_beam": strong_beam(d["beam"], d["sc_orient"])}
    if product == "ATL06":
        steps["quality_summary_0"] = d["quality"].to_numpy(float) == 0
    else:
        steps["no_multiple_scattering"] = d["msw"].to_numpy(float) == 0
        steps["cloud_free"] = d["cloud"].to_numpy(float) == 0
    steps["geoid"] = np.isfinite(d["geoid_h"].to_numpy(float))
    steps["gross_error"] = _gross(h, d["dem_h"])
    return steps


def gedi_steps(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """GEDI L2A shot selection fixed by Amendment 2: quality_flag = 1, degrade_flag = 0,
    BEAM0000/BEAM0001 shots before 2020-02-06 dropped, and |elev_lowestmode -
    digital_elevation_model| <= 50 m."""
    t = np.asarray(d["time"], "datetime64[ns]")
    early = np.isin(d["beam"].astype(str).to_numpy(), GEDI_EARLY_BEAMS) & (t < GEDI_BEAM_FIX)
    return {"height": np.isfinite(d["elev_lowestmode"].to_numpy(float)),
            "quality_flag_1": d["quality_flag"].to_numpy(float) == 1,
            "degrade_flag_0": d["degrade_flag"].to_numpy(float) == 0,
            "beam_ranging": ~early,
            "gross_error": _gross(d["elev_lowestmode"], d["digital_elevation_model"])}


def cascade(steps: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply steps in order. Returns the kept mask and a table of the points remaining
    after each step and failing each step on its own."""
    keep = np.ones(n, bool)
    rows = [{"step": "input", "n_remaining": n, "n_fail_alone": 0}]
    for name, m in steps.items():
        m = np.asarray(m, bool)
        keep &= m
        rows.append({"step": name, "n_remaining": int(keep.sum()), "n_fail_alone": int((~m).sum())})
    return keep, pd.DataFrame(rows)


def prepare_icesat2(d: pd.DataFrame, product: str, ref: pd.DataFrame | None = None):
    """Quality steps for one granule. For ATL08, ``ref`` is the ATL06 granule of the same
    orbit (time, beam, geoid_h), or None if there is none (then no geoid, and every
    point fails the geoid step). Returns (kept points with h_orth, cascade table). The
    product's own reference DEM ``dem_h`` is renamed ``is2_dem_h``, so that ``dem_h``
    always means the waterline DEM."""
    d = d.copy()
    if product == "ATL08":
        d["geoid_h"] = (np.full(len(d), np.nan) if ref is None or ref.empty else
                        geoid_along_track(d["time"], d["beam"], ref["time"], ref["beam"], ref["geoid_h"]))
    keep, casc = cascade(icesat2_steps(d, product), len(d))
    out = d.loc[keep].rename(columns={"dem_h": "is2_dem_h"})
    out["h_orth"] = orthometric(out["h"], out["geoid_h"])
    out["source"] = product
    return out, casc


def dry_at_acquisition(t_acq, emergence_days, lag_days: float = design.EMERGENCE_LAG_DAYS) -> np.ndarray:
    """PREREG section 2 (and Amendment 2): True where the acquisition is later than the
    emergence date plus ``lag_days`` for every pixel of the point's neighbourhood.
    ``emergence_days``: (n, k) emergence dates (days since 1970-01-01) of the k pixels
    around each point. A NaN or infinite date (never emerged, not seabed, no data)
    fails, so only seabed that emerged in the record qualifies."""
    e = np.asarray(emergence_days, float)
    if e.ndim == 1:
        e = e[:, None]
    t = _days(t_acq)[:, None]
    with np.errstate(invalid="ignore"):
        return (np.isfinite(e) & (t > e + lag_days)).all(axis=1)


# --- rasters on the DEM tiling --------------------------------------------------------

class TileSet:
    """North-up GeoTIFF tiles ``<prefix>_<i>_<j>.tif`` on the equal-area grid, tile (i, j)
    starting at (i T, (j + 1) T) (scripts/a6_dem.py). A location takes the value of the
    pixel that contains it, read through the tile's own transform; NaN where no tile or
    pixel holds it (including any strip a tile does not reach)."""

    def __init__(self, directory, prefix: str = "dem", tile_m: float = 50_000,
                 res: float = design.DEM_RES_M):
        self.directory, self.prefix, self.tile_m, self.res = Path(directory), prefix, tile_m, res

    def sample(self, x, y, band: int = 1) -> np.ndarray:
        import rasterio

        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        shape = x.shape
        x, y = x.ravel(), y.ravel()
        out = np.full(x.shape, np.nan)
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return out.reshape(shape)
        idx = np.flatnonzero(ok)
        ti = np.floor(x[idx] / self.tile_m).astype(int)
        tj = np.floor(y[idx] / self.tile_m).astype(int)
        groups = pd.DataFrame({"i": ti, "j": tj}).groupby(["i", "j"]).indices
        for (i, j), g in groups.items():
            path = self.directory / f"{self.prefix}_{i}_{j}.tif"
            if not path.exists():
                continue
            k = idx[g]
            with rasterio.open(path) as src:
                arr = src.read(band).astype(float)
                if src.nodata is not None and not np.isnan(src.nodata):
                    arr[arr == src.nodata] = np.nan
                inv = ~src.transform
                col = inv.a * x[k] + inv.b * y[k] + inv.c
                row = inv.d * x[k] + inv.e * y[k] + inv.f
            col, row = np.floor(col).astype(int), np.floor(row).astype(int)
            inside = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
            vals = np.full(k.size, np.nan)
            vals[inside] = arr[row[inside], col[inside]]
            out[k] = vals
        return out.reshape(shape)

    def sample_window(self, x, y, band: int = 1, radius_px: int = 1) -> np.ndarray:
        """Values of the (2r + 1)² pixels around each location, shape (n, 2r + 1, 2r + 1);
        [:, a, b] is the pixel (r - a) rows north and (b - r) columns east."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        off = np.arange(-radius_px, radius_px + 1) * self.res
        dy, dx = np.meshgrid(off[::-1], off, indexing="ij")
        v = self.sample(x[:, None, None] + dx[None], y[:, None, None] + dy[None], band)
        return v.reshape(x.size, 2 * radius_px + 1, 2 * radius_px + 1)

    def slope(self, x, y, band: int = 1) -> np.ndarray:
        """Local slope (m/m): central differences over the 4-neighbours of the pixel. NaN
        where a neighbour has no value."""
        w = self.sample_window(x, y, band, radius_px=1)
        gx = (w[:, 1, 2] - w[:, 1, 0]) / (2 * self.res)
        gy = (w[:, 0, 1] - w[:, 2, 1]) / (2 * self.res)
        return np.hypot(gx, gy)


def contour_density(x, y, cx, cy, radius_m: float = CONTOUR_RADIUS_M) -> np.ndarray:
    """Contour points of the DEM (cx, cy) per km² within ``radius_m`` of each location
    (scale covariate, PREREG section 8). NaN at non-finite locations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    tree = cKDTree(np.column_stack([np.asarray(cx, float), np.asarray(cy, float)]))
    n = tree.query_ball_point(np.column_stack([x[ok], y[ok]]), r=radius_m, return_length=True)
    out[ok] = np.asarray(n, float) / (math.pi * radius_m ** 2 / 1e6)
    return out


# --- statistics -----------------------------------------------------------------------

def nmad(e) -> float:
    e = np.asarray(e, float)
    return float(NMAD_SCALE * np.median(np.abs(e - np.median(e)))) if e.size else float("nan")


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def bias(e) -> float:
    e = np.asarray(e, float)
    return float(np.mean(e)) if e.size else float("nan")


METRICS: dict[str, Callable] = {"nmad": nmad, "rmse": rmse, "bias": bias}


def error_metrics(e) -> dict:
    """PREREG section 7: bias (mean), RMSE, NMAD and the 5th/95th percentiles of the
    error, plus the median. Non-finite errors are dropped."""
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0, **{k: float("nan") for k in ("bias", "median", "rmse", "nmad", "p05", "p95")}}
    return {"n": int(e.size), "bias": bias(e), "median": float(np.median(e)), "rmse": rmse(e),
            "nmad": nmad(e), "p05": float(np.percentile(e, 5)), "p95": float(np.percentile(e, 95))}


def block_bootstrap(block_labels, stat: Callable[[np.ndarray], object], n_boot: int = design.BOOTSTRAP_N,
                    seed: int = design.SEED) -> np.ndarray:
    """Block bootstrap: each replicate draws as many blocks as there are, with
    replacement, and evaluates ``stat`` on the point indices of the drawn blocks (a block
    drawn twice enters twice). Blocks are taken in sorted label order, so the result
    depends only on the data, ``n_boot`` and ``seed``. One row per replicate."""
    lab = np.asarray(block_labels).astype(str)
    if lab.size == 0:
        raise ValueError("no points to resample")
    uniq, inv = np.unique(lab, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.searchsorted(inv[order], np.arange(uniq.size + 1))
    groups = [order[cuts[k]:cuts[k + 1]] for k in range(uniq.size)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(stat(np.concatenate([groups[k] for k in pick])))
    return np.asarray(out, float)


def percentile_interval(boot, conf: float = 0.95) -> tuple[float, float]:
    a = (1 - conf) / 2 * 100
    b = np.asarray(boot, float)
    if not np.isfinite(b).any():
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(b, [a, 100 - a], axis=0)
    return float(lo), float(hi)


# --- fitted on training blocks --------------------------------------------------------

def block_size_check(points: pd.DataFrame, base: float = design.BLOCK_SIZE_M, bin_width: float = VARIO_BIN_M,
                     max_lag: float = VARIO_MAX_LAG_M, max_pairs: int = VARIO_MAX_PAIRS,
                     seed: int = design.SEED) -> tuple[dict, pd.DataFrame]:
    """PREREG section 6: exponential semivariogram of DEM - H_EGM2008 in training blocks
    (the datum constant cancels in a semivariogram) and the block size it implies
    (caspian.blocking). Returns (record, empirical semivariogram).

    ``range_within_max_lag`` is False when the fitted practical range lies beyond the
    largest lag fitted. The range is then an extrapolation (for example, a flat,
    pure-nugget semivariogram gives an arbitrarily long range), and the implied block
    size is not usable without a decision; ``partial_sill_share`` = c1 / (c0 + c1) is
    recorded to inform it."""
    tr = split_rows(points, "train")
    r = (tr["dem_h"] - tr["h_orth"]).to_numpy(float)
    ok = np.isfinite(r)
    vario = blocking.empirical_semivariogram(tr["x"].to_numpy()[ok], tr["y"].to_numpy()[ok], r[ok],
                                             bin_width, max_lag, max_pairs, seed)
    fit = blocking.fit_exponential(vario)
    sill = fit["nugget"] + fit["partial_sill"]
    # D-069: the fitted range sets the block size only if spatial structure is detected
    test = blocking.spatial_structure_test(tr["x"].to_numpy()[ok], tr["y"].to_numpy()[ok], r[ok],
                                           VARIO_SHORT_LAG_M, max_lag, max_pairs=max_pairs, seed=seed)
    final = blocking.block_size_for_range(fit["practical_range"], base) if test["detected"] else float(base)
    return ({"split": "train", "n_points": int(ok.sum()), "n_blocks": int(tr.loc[ok, "block"].nunique()),
             "bin_width_m": bin_width, "max_lag_m": max_lag, "max_pairs": max_pairs, "seed": seed,
             "fit": fit, "partial_sill_share": float(fit["partial_sill"] / sill) if sill > 0 else float("nan"),
             "range_within_max_lag": bool(fit["practical_range"] <= vario["lag"].max()),
             "structure_test": test, "base_block_size_m": float(base), "final_block_size_m": final}, vario)


def fit_datum_constant(points: pd.DataFrame, n_boot: int = design.BOOTSTRAP_N, seed: int = design.SEED) -> dict:
    """PREREG section 5: the ICESat-2 -> waterline DEM constant c, the median over
    training-block points of DEM - H_EGM2008, with σ the standard deviation of that
    median over ``n_boot`` resamples of training blocks."""
    tr = split_rows(points, "train")
    d = (tr["dem_h"] - tr["h_orth"]).to_numpy(float)
    ok = np.isfinite(d)
    if not ok.any():
        raise ValueError("no co-located DEM and reference heights in training blocks")
    d, blk = d[ok], tr["block"].to_numpy()[ok]
    boot = block_bootstrap(blk, lambda ix: np.median(d[ix]), n_boot, seed)
    return {"split": "train", "estimator": "median of DEM - H_EGM2008", "constant_m": float(np.median(d)),
            "sigma_boot_m": float(np.std(boot, ddof=1)), "n_points": int(ok.sum()),
            "n_blocks": int(np.unique(blk).size), "n_boot": n_boot, "seed": seed}


def apply_datum_constant(points: pd.DataFrame, constant: dict) -> pd.DataFrame:
    """Reference height on the DEM reference: y_ref = H_EGM2008 + c."""
    return points.assign(y_ref=points["h_orth"].to_numpy(float) + constant["constant_m"])


def fit_product_offsets(points: pd.DataFrame, products: dict[str, str], n_boot: int = design.BOOTSTRAP_N,
                        seed: int = design.SEED) -> dict:
    """PREREG section 5: each product's reference estimated as one constant, the median
    over training-block points of y_ref - product height, with a block-bootstrap σ.
    ``products``: name -> column."""
    tr = split_rows(points, "train")
    out = {}
    for name, col in products.items():
        d = (tr["y_ref"] - tr[col]).to_numpy(float)
        ok = np.isfinite(d)
        if not ok.any():
            out[name] = {"split": "train", "offset_m": float("nan"), "sigma_boot_m": float("nan"),
                         "n_points": 0, "n_blocks": 0}
            continue
        dd, blk = d[ok], tr["block"].to_numpy()[ok]
        boot = block_bootstrap(blk, lambda ix: np.median(dd[ix]), n_boot, seed)
        out[name] = {"split": "train", "offset_m": float(np.median(dd)), "sigma_boot_m": float(np.std(boot, ddof=1)),
                     "n_points": int(ok.sum()), "n_blocks": int(np.unique(blk).size)}
    return out


def product_variants(points: pd.DataFrame, products: dict[str, str], offsets: dict, constant: dict,
                     documented=DOCUMENTED_EGM2008) -> pd.DataFrame:
    """Columns ``<name>|matched`` (product + its training offset) and ``<name>|total``
    (product on its documented reference: + c for EGM2008; otherwise the estimated
    reference, equal to ``matched``)."""
    new = {}
    for name, col in products.items():
        z = points[col].to_numpy(float)
        new[f"{name}|matched"] = z + offsets[name]["offset_m"]
        new[f"{name}|total"] = z + constant["constant_m"] if name in documented else new[f"{name}|matched"]
    return points.assign(**new)


def scale_design(points: pd.DataFrame) -> np.ndarray:
    """[1, log(1 + interpolation distance), contour density, local slope] (PREREG section 8)."""
    return np.column_stack([np.ones(len(points)), np.log1p(points["interp_dist"].to_numpy(float)),
                            points["contour_density"].to_numpy(float), points["slope"].to_numpy(float)])


def fit_scale_model(points: pd.DataFrame, floor_m: float = ABS_RESID_FLOOR_M) -> dict:
    """PREREG section 8: β by least squares of log |y_ref - DEM| on ``scale_design`` over
    training-block points; |residual| is floored at ``floor_m`` before the log."""
    tr = split_rows(points, "train")
    X = scale_design(tr)
    r = (tr["y_ref"] - tr["dem_h"]).to_numpy(float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(r)
    if ok.sum() <= X.shape[1]:
        raise ValueError("too few training points for the scale model")
    beta, *_ = np.linalg.lstsq(X[ok], np.log(np.maximum(np.abs(r[ok]), floor_m)), rcond=None)
    return {"split": "train", "terms": list(SCALE_TERMS), "beta": [float(b) for b in beta],
            "n_points": int(ok.sum()), "floor_m": floor_m}


def scale_values(points: pd.DataFrame, model: dict) -> np.ndarray:
    """s(x) = exp(β · design); NaN where a covariate is missing."""
    return np.exp(scale_design(points) @ np.asarray(model["beta"], float))


# --- fitted on calibration blocks -----------------------------------------------------

def calibrate(points: pd.DataFrame, model: dict, level: float = design.CONFORMAL_LEVEL,
              min_n: int = design.MONDRIAN_MIN_N, sectors=SECTOR_ORDER) -> dict:
    """PREREG section 8 and Amendment 5: q̂ per sector on calibration-block points
    (Mondrian with the pooled fallback below ``min_n`` usable points), with the number
    of calibration points excluded (no prediction, reference or scale) per sector."""
    cal = split_rows(points, "calib")
    s = scale_values(cal, model)
    y, yhat = cal["y_ref"].to_numpy(float), cal["dem_h"].to_numpy(float)
    sec = cal["sector"].to_numpy(object)
    rec = conformal.mondrian_calibration(conformal.scores(y, yhat, s), sec, 1 - level, min_n,
                                         all_groups=list(sectors))
    use = conformal.usable(y, yhat, s)
    rec["groups"] = {str(k): v for k, v in rec["groups"].items()}
    rec["n_excluded"] = {g: int(((sec == g) & ~use).sum()) for g in rec["groups"]}
    rec.update(split="calib", level=level)
    return rec


# --- evaluated on test blocks ---------------------------------------------------------

def _scopes(frame: pd.DataFrame, sectors) -> list[tuple[str, np.ndarray]]:
    sec = frame["sector"].to_numpy(object)
    return [("overall", np.ones(len(frame), bool))] + [(g, sec == g) for g in sectors]


def coverage_on_test(points: pd.DataFrame, model: dict, calibration: dict, n_boot: int = design.BOOTSTRAP_N,
                     seed: int = design.SEED, conf: float = 0.95, sectors=SECTOR_ORDER) -> pd.DataFrame:
    """PREREG section 8: empirical coverage of ŷ ± q̂ s(x) on test-block points, overall
    and per sector, with the Clopper-Pearson interval, whether it is consistent with the
    nominal level, and a block-bootstrap interval over test blocks. Points without a
    prediction, reference or scale are excluded and counted (Amendment 5)."""
    te = split_rows(points, "test")
    s = scale_values(te, model)
    q = conformal.mondrian_q(te["sector"].to_numpy(object).astype(str), calibration)
    y, yhat = te["y_ref"].to_numpy(float), te["dem_h"].to_numpy(float)
    lo, hi = conformal.intervals(yhat, 1.0, q * s)
    keep = ~np.isnan(y) & ~np.isnan(lo) & ~np.isnan(hi)
    inside = keep & (y >= lo) & (y <= hi)
    blk = te["block"].to_numpy()
    rows = []
    for scope, m in _scopes(te, sectors):
        c = conformal.coverage(y[m], lo[m], hi[m], conf)
        g = calibration["groups"].get(scope, calibration["pooled"]) if scope != "overall" else None
        mk = m & keep
        ins = inside[mk].astype(float)
        b_lo, b_hi = (percentile_interval(block_bootstrap(blk[mk], lambda ix: ins[ix].mean(), n_boot, seed), conf)
                      if mk.any() else (float("nan"), float("nan")))
        rows.append({"scope": scope, **c, "n_points": int(m.sum()),
                     "n_no_prediction": int((m & np.isnan(yhat)).sum()),
                     "frac_no_prediction": float((m & np.isnan(yhat)).sum() / m.sum()) if m.any() else float("nan"),
                     "consistent_with_nominal": (conformal.consistent_with_nominal(c["k"], c["n"], calibration["level"], conf)
                                                 if c["n"] else None),
                     "boot_low": b_lo, "boot_high": b_hi, "n_blocks": int(np.unique(blk[mk]).size),
                     "q": None if g is None else g["q"], "used_pooled": None if g is None else g.get("used_pooled")})
    return pd.DataFrame(rows)


def _paired(te: pd.DataFrame, cols) -> np.ndarray:
    ok = np.isfinite(te["y_ref"].to_numpy(float))
    for c in cols:
        ok &= np.isfinite(te[c].to_numpy(float))
    return ok


def accuracy_tables(points: pd.DataFrame, models: dict[str, str], sectors=SECTOR_ORDER):
    """PREREG section 7 on test-block points where the reference and every model have a
    value (the paired set). ``models``: name -> column. Returns (summary: pooled over
    test-block points, with the median of the per-block values; per-block table)."""
    te = split_rows(points, "test")
    ok = _paired(te, models.values())
    y = te["y_ref"].to_numpy(float)
    blk = te["block"].to_numpy()
    sec = te["sector"].to_numpy(object)
    summary, per_block = [], []
    for name, col in models.items():
        e = te[col].to_numpy(float) - y
        pb = pd.DataFrame([{"block": b, "sector": g["sector"].iloc[0], "model": name, **error_metrics(g["e"])}
                           for b, g in pd.DataFrame({"block": blk[ok], "sector": sec[ok], "e": e[ok]})
                           .groupby("block", sort=True)],
                          columns=["block", "sector", "model", "n", "bias", "median", "rmse", "nmad", "p05", "p95"])
        per_block.append(pb)
        for scope, m in _scopes(te, sectors):
            mk = m & ok
            row = {"scope": scope, "model": name, **error_metrics(e[mk]), "n_blocks": int(np.unique(blk[mk]).size),
                   "n_test_points": int(m.sum())}
            bm = pb if scope == "overall" else pb[pb["sector"] == scope]
            for k in ("bias", "rmse", "nmad"):
                row[f"block_median_{k}"] = float(bm[k].median()) if len(bm) else float("nan")
            summary.append(row)
    return pd.DataFrame(summary), pd.concat(per_block, ignore_index=True)


def paired_block_bootstrap(points: pd.DataFrame, dem_col: str, products: dict[str, str],
                           metrics=("nmad", "rmse"), n_boot: int = design.BOOTSTRAP_N, seed: int = design.SEED,
                           conf: float = 0.95, sectors=SECTOR_ORDER) -> pd.DataFrame:
    """PREREG section 7: metric(DEM error) - metric(product error) on the same test-block
    points (the paired set over the DEM and every product), with a percentile interval
    from ``n_boot`` resamples of test blocks. One draw of blocks serves every product
    and metric of a scope. ``products``: name -> column."""
    te = split_rows(points, "test")
    ok = _paired(te, [dem_col, *products.values()])
    y = te["y_ref"].to_numpy(float)
    e_dem = te[dem_col].to_numpy(float) - y
    e_p = {p: te[c].to_numpy(float) - y for p, c in products.items()}
    blk = te["block"].to_numpy()
    pairs = [(p, mt) for p in products for mt in metrics]
    rows = []
    for scope, m in _scopes(te, sectors):
        mk = m & ok
        ed = e_dem[mk]
        ep = {p: v[mk] for p, v in e_p.items()}
        if mk.any():
            boot = block_bootstrap(blk[mk], lambda ix: [METRICS[mt](ed[ix]) - METRICS[mt](ep[p][ix]) for p, mt in pairs],
                                   n_boot, seed)
        for k, (p, mt) in enumerate(pairs):
            dv, pv = METRICS[mt](ed), METRICS[mt](ep[p])
            lo, hi = percentile_interval(boot[:, k], conf) if mk.any() else (float("nan"), float("nan"))
            rows.append({"scope": scope, "product": p, "metric": mt, "dem_value": dv, "product_value": pv,
                         "diff": dv - pv, "ci_low": lo, "ci_high": hi, "n_points": int(mk.sum()),
                         "n_blocks": int(np.unique(blk[mk]).size), "n_boot": n_boot, "seed": seed})
    return pd.DataFrame(rows)


def run(points: pd.DataFrame, products: dict[str, str], documented=DOCUMENTED_EGM2008,
        n_boot: int = design.BOOTSTRAP_N, seed: int = design.SEED, level: float = design.CONFORMAL_LEVEL,
        min_n: int = design.MONDRIAN_MIN_N, sectors=SECTOR_ORDER) -> dict:
    """Every fit (training: c, product offsets, β; calibration: q̂), then every test-block
    evaluation. ``points`` must carry block, sector and split (``attach_blocks``)."""
    constant = fit_datum_constant(points, n_boot, seed)
    pts = apply_datum_constant(points, constant)
    offsets = fit_product_offsets(pts, products, n_boot, seed)
    pts = product_variants(pts, products, offsets, constant, documented)
    model = fit_scale_model(pts)
    calibration = calibrate(pts, model, level, min_n, sectors)
    # --- test blocks from here on ---
    coverage = coverage_on_test(pts, model, calibration, n_boot, seed, sectors=sectors)
    models = {"dem": "dem_h", **{f"{p}|{v}": f"{p}|{v}" for p in products for v in ("total", "matched")}}
    summary, per_block = accuracy_tables(pts, models, sectors)
    boots = []
    for v in ("total", "matched"):
        if products:
            b = paired_block_bootstrap(pts, "dem_h", {p: f"{p}|{v}" for p in products}, n_boot=n_boot, seed=seed,
                                       sectors=sectors)
            boots.append(b.assign(variant=v))
    return {"constant": constant, "product_offsets": offsets, "scale_model": model, "calibration": calibration,
            "coverage": coverage, "accuracy": summary, "accuracy_blocks": per_block,
            "paired_bootstrap": pd.concat(boots, ignore_index=True) if boots else pd.DataFrame(),
            "points": pts}


# --- H2 decision rule (synthetic use only until G3) -----------------------------------

def h2_decision(comparisons: pd.DataFrame, products, metrics=("nmad", "rmse")) -> dict:
    """PREREG section 7 rule, applied per scope ("overall" and each sector) of a
    ``paired_block_bootstrap`` table for one variant.

    For a product, the DEM wins on a metric if diff = DEM - product < 0 and the 95%
    interval of the difference lies below zero (ci_high < 0). A scope is "supported" if
    the DEM wins on every metric against every product in ``products``; "not supported"
    if it loses any evaluable comparison; otherwise "not evaluable" (a listed product or
    metric missing or NaN, so a product cannot drop out silently). Partial support (some
    products) is reported. The brief section 16 pivot applies when "overall" is not
    supported."""
    products = list(products)
    scopes = {}
    for scope, g in comparisons.groupby("scope", sort=False):
        per = {}
        for p in products:
            res = {}
            for mt in metrics:
                r = g[(g["product"] == p) & (g["metric"] == mt)]
                vals = r[["diff", "ci_low", "ci_high"]].to_numpy(float) if len(r) == 1 else None
                res[mt] = (None if vals is None or not np.isfinite(vals).all()
                           else bool(vals[0, 0] < 0 and vals[0, 2] < 0))
            evaluable = all(v is not None for v in res.values())
            per[p] = {**res, "evaluable": evaluable, "dem_better": evaluable and all(res.values())}
        lost = any(v is False for p in products for v in (per[p][mt] for mt in metrics))
        won = [p for p in products if per[p]["dem_better"]]
        status = ("not supported" if lost else "supported" if len(won) == len(products) else "not evaluable")
        scopes[scope] = {"status": status, "products": per, "dem_better_than": won,
                         "partial": 0 < len(won) < len(products)}
    overall = scopes.get("overall", {"status": "not evaluable"})
    return {"overall": overall["status"], "scopes": scopes,
            "sectors_supported": [s for s, v in scopes.items() if s != "overall" and v["status"] == "supported"],
            "pivot_brief_16": {"supported": False, "not supported": True}.get(overall["status"])}
