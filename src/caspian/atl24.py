"""ICESat-2 ATL24 (coastal and nearshore bathymetry, v002) as an independent check of the
near-term emergence surfaces (PREREG section 2 "ATL24 bathymetry where available",
section 10 and Amendment 17).

Reading (scripts/a8_atl24_fetch.py)
-----------------------------------
Each granule's ``metadata/extent`` polygon (``extent_polygon``) says where it holds
photons. Granules whose extent misses the domain's bounding box are not read further.
For the others, the class-40 (bathymetry) photons inside the box are kept with the
variables of ``PHOTON_VARS`` (``read_granule``), unchanged (data/raw/icesat2/atl24/).

Photon selection (Amendment 17, item 1; ``photon_steps``)
---------------------------------------------------------
Ordered steps, each counted: class_ph = 40; finite heights; low_confidence_flag = 0
(ensemble confidence at or above 0.6, the ATBD threshold); sensor_depth_exceeded = 0;
strong beam (``validation.strong_beam``); 0 < depth <= ``MAX_DEPTH_M``, with depth =
surface_h - ortho_h (instantaneous depth below ATL24's own sea surface); finite geoid.
invalid_kd and invalid_wind_speed flag only the uncertainty model (ATBD section 5.3)
and are counted, not used. night_flag defines the night-only sensitivity.

Datum (Amendment 17, item 2)
----------------------------
    H     = ellipse_h - N(EIGEN-6C4)            (as ICESat-2 ATL06/ATL08)
    y_ref = H + c                               (c: the ATL08 training-block constant)
ellipse_h is the ATL03 photon height (tide-free, ITRF of the ATL03 release) with the
refraction correction added; ATL24's own EGM08 orthometric heights are not used. The
sea surface on the same scale is surface_h + (ellipse_h - ortho_h) - N(EIGEN-6C4) + c
(``surface_working``), which gives the water-surface constant of the sensitivity
(``water_surface_constant``).

Aggregation (Amendment 17, item 4; ``aggregate``)
-------------------------------------------------
Photons -> 60 m hindcast cells, per track (granule x beam): the median y_ref of the
photons of a cell and track, kept when it has at least ``MIN_PHOTONS`` photons. One
cell-track is one observation.

Comparison and decision (Amendment 17, items 5-7; ``compare``, ``verdict``)
-------------------------------------------------------------------------
Errors are surface minus ATL24 (positive: the surface is too high). Bias (mean), RMSE
and NMAD per surface, overall and per sector, on the cell-tracks where both surfaces
of a pair have a value; the paired differences (fused minus uncorrected) of NMAD,
|bias| and RMSE get a percentile interval from ``BOOT_N`` resamples of the 50 km H2
blocks (validation.block_bootstrap; one draw per scope serves every metric).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import design, validation

BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
CLASS_BATHY = 40
CONFIDENCE_MIN = 0.6           # ATBD section 5.3: low_confidence_flag = 1 below 0.6
MAX_DEPTH_M = 6.5              # Amendment 17 item 1 (see there)
MIN_PHOTONS = 3                # photons per cell-track
MIN_BLOCKS = 5                 # blocks a scope needs for a verdict
BOOT_N = design.BOOTSTRAP_N
SEED = design.SEED
NEAR_TERM_DROP_M = 5.0         # near-term population: [L2026 - 5 m, L2026)
EPOCH = np.datetime64("2018-01-01T00:00:00", "ns")   # ATLAS SDP epoch (delta_time origin)

# Variables kept per photon (all exist in every gtx group of ATL24 v002).
PHOTON_VARS = ("lat_ph", "lon_ph", "delta_time", "ellipse_h", "ortho_h", "surface_h", "class_ph",
               "confidence", "low_confidence_flag", "night_flag", "sensor_depth_exceeded", "invalid_kd",
               "invalid_wind_speed", "sigma_tvu", "sigma_thu", "index_ph", "x_atc")

VERDICTS = ("correction improves", "correction worsens", "no clear difference", "not evaluable")


# --- reading ---------------------------------------------------------------------------------

def extent_polygon(extent_json: str | bytes):
    """The data extent of a granule from its ``metadata/extent`` JSON: the "polygon"
    string holds "lat lon" pairs. Returns a shapely Polygon in (lon, lat), or None when
    there is no polygon."""
    import json

    from shapely.geometry import Polygon

    if isinstance(extent_json, (bytes, np.bytes_)):
        extent_json = bytes(extent_json).decode("utf-8", "replace")
    rec = json.loads(extent_json)
    vals = [float(v) for v in re.split(r"\s+", str(rec.get("polygon", "")).strip()) if v]
    if len(vals) < 6 or len(vals) % 2:
        return None
    lat, lon = np.asarray(vals[0::2]), np.asarray(vals[1::2])
    return Polygon(np.column_stack([lon, lat]))


def box_mask(lat, lon, bounds) -> np.ndarray:
    """Inside the (west, south, east, north) box, edges included."""
    w, s, e, n = bounds
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)


def read_granule(h5, bounds) -> tuple[dict, dict]:
    """Class-40 photons inside ``bounds`` from an open ATL24 file (h5py.File), every beam,
    with the variables of PHOTON_VARS and a ``beam`` code (index into BEAMS). Returns
    (arrays, record) where the record holds the photons in the box per class, the
    spacecraft orientation, the ATL03 resource and the release."""
    out = {k: [] for k in (*PHOTON_VARS, "beam")}
    by_class: dict[str, int] = {}
    for k, gt in enumerate(BEAMS):
        if gt not in h5 or "lat_ph" not in h5[gt]:
            continue
        g = h5[gt]
        lat = g["lat_ph"][:]
        band = (lat >= bounds[1]) & (lat <= bounds[3])
        if not band.any():
            continue
        idx = np.flatnonzero(band)
        a, b = int(idx[0]), int(idx[-1]) + 1
        lon = g["lon_ph"][a:b]
        cls = g["class_ph"][a:b]
        inbox = box_mask(lat[a:b], lon, bounds)
        vals, cnt = np.unique(cls[inbox], return_counts=True)
        for v, c in zip(vals.tolist(), cnt.tolist()):
            by_class[str(v)] = by_class.get(str(v), 0) + int(c)
        keep = inbox & (cls == CLASS_BATHY)
        if not keep.any():
            continue
        kk = np.flatnonzero(keep)
        a2, b2 = a + int(kk[0]), a + int(kk[-1]) + 1
        sel = keep[kk[0]:kk[-1] + 1]
        for v in PHOTON_VARS:
            if v == "lat_ph":
                x = lat[a2:b2]
            elif v == "lon_ph":
                x = lon[a2 - a:b2 - a]
            elif v == "class_ph":
                x = cls[a2 - a:b2 - a]
            else:
                x = g[v][a2:b2]
            out[v].append(np.asarray(x)[sel])
        out["beam"].append(np.full(int(sel.sum()), k, np.int8))
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    sc = int(h5["orbit_info/sc_orient"][0]) if "orbit_info/sc_orient" in h5 else -1
    anc = h5["ancillary_data"] if "ancillary_data" in h5 else {}

    def _s(key):
        if key not in anc:
            return None
        v = anc[key][()]
        v = v[0] if isinstance(v, np.ndarray) and v.size else v
        return bytes(v).decode("utf-8", "replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
    rec = {"n_box_by_class": by_class, "n_saved": int(arrays["beam"].size), "sc_orient": sc,
           "atl03_resource": _s("resource"), "release": _s("release"), "data_start_utc": _s("data_start_utc")}
    return arrays, rec


def photon_time(delta_time) -> np.ndarray:
    """delta_time (seconds since the ATLAS SDP epoch, 2018-01-01) as datetime64[ns]; leap
    seconds are ignored (18 s, as scripts/a6_icesat2.py)."""
    s = np.asarray(delta_time, float)
    return EPOCH + np.round(s * 1e9).astype("int64").astype("timedelta64[ns]")


# --- photon selection -----------------------------------------------------------------------

def depth(surface_h, ortho_h) -> np.ndarray:
    """Instantaneous depth below ATL24's sea surface: surface_h - ortho_h (m, positive down)."""
    return np.asarray(surface_h, float) - np.asarray(ortho_h, float)


def photon_steps(d: pd.DataFrame, max_depth_m: float = MAX_DEPTH_M, *, strong_only: bool = True,
                 confidence_min: float = CONFIDENCE_MIN) -> dict[str, np.ndarray]:
    """Amendment 17 item 1, ordered (True = passes). ``d`` needs class_ph, ellipse_h,
    ortho_h, surface_h, low_confidence_flag, confidence, sensor_depth_exceeded, beam
    (name), sc_orient and geoid (N, m)."""
    eh, oh, sh = (d[c].to_numpy(float) for c in ("ellipse_h", "ortho_h", "surface_h"))
    dep = depth(sh, oh)
    with np.errstate(invalid="ignore"):
        steps = {"class_40": d["class_ph"].to_numpy() == CLASS_BATHY,
                 "finite_heights": np.isfinite(eh) & np.isfinite(oh) & np.isfinite(sh)
                 & (np.abs(eh) < 1e30) & (np.abs(oh) < 1e30) & (np.abs(sh) < 1e30),
                 "confidence": (d["low_confidence_flag"].to_numpy() == 0)
                 & (d["confidence"].to_numpy(float) >= confidence_min),
                 "sensor_depth_ok": d["sensor_depth_exceeded"].to_numpy() == 0}
        if strong_only:
            steps["strong_beam"] = validation.strong_beam(d["beam"], d["sc_orient"])
        steps["depth_range"] = (dep > 0) & (dep <= max_depth_m)
        steps["geoid"] = np.isfinite(d["geoid"].to_numpy(float))
    return steps


def reference_height(ellipse_h, geoid, constant_m: float) -> np.ndarray:
    """y_ref = ellipse_h - N(EIGEN-6C4) + c: the seabed on the working (DAHITI) reference."""
    return np.asarray(ellipse_h, float) - np.asarray(geoid, float) + float(constant_m)


def surface_working(surface_h, ellipse_h, ortho_h, geoid, constant_m: float) -> np.ndarray:
    """ATL24's instantaneous sea surface on the same scale as ``reference_height``:
    surface_h + (ellipse_h - ortho_h) - N(EIGEN-6C4) + c. ellipse_h - ortho_h is the
    EGM08 undulation ATL24 used at the photon."""
    return (np.asarray(surface_h, float) + np.asarray(ellipse_h, float) - np.asarray(ortho_h, float)
            - np.asarray(geoid, float) + float(constant_m))


# --- aggregation ------------------------------------------------------------------------------

def cell_index(x, y, transform) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the north-up grid cell holding (x, y) (equal-area metres)."""
    a, c, e, f = float(transform[0]), float(transform[2]), float(transform[4]), float(transform[5])
    col = np.floor((np.asarray(x, float) - c) / a).astype(np.int64)
    row = np.floor((np.asarray(y, float) - f) / e).astype(np.int64)
    return row, col


def aggregate(ph: pd.DataFrame, transform, min_photons: int = MIN_PHOTONS) -> pd.DataFrame:
    """Photons (x, y, y_ref, track, time, depth, surface, night) -> one row per (cell,
    track): the medians of y_ref, depth and surface, the median time, the night share
    and the photon count; rows with fewer than ``min_photons`` photons are dropped."""
    if ph.empty:
        return pd.DataFrame(columns=["row", "col", "track", "n_photons", "y_ref", "depth", "surface",
                                     "time", "night_share"])
    row, col = cell_index(ph["x"], ph["y"], transform)
    t = np.asarray(ph["time"], "datetime64[ns]").astype("int64")
    g = pd.DataFrame({"row": row, "col": col, "track": ph["track"].to_numpy(), "y_ref": ph["y_ref"].to_numpy(float),
                      "depth": ph["depth"].to_numpy(float), "surface": ph["surface"].to_numpy(float), "t": t,
                      "night": ph["night"].to_numpy(float)})
    agg = g.groupby(["row", "col", "track"], sort=True).agg(
        n_photons=("y_ref", "size"), y_ref=("y_ref", "median"), depth=("depth", "median"),
        surface=("surface", "median"), t=("t", "median"), night_share=("night", "mean")).reset_index()
    agg = agg[agg["n_photons"] >= min_photons].reset_index(drop=True)
    agg["time"] = agg["t"].round().astype("int64").to_numpy().astype("datetime64[ns]")
    return agg.drop(columns="t")


def cell_centres(row, col, transform) -> tuple[np.ndarray, np.ndarray]:
    a, c, e, f = float(transform[0]), float(transform[2]), float(transform[4]), float(transform[5])
    return c + (np.asarray(col, float) + 0.5) * a, f + (np.asarray(row, float) + 0.5) * e


# --- datum sensitivity: the water-surface constant ---------------------------------------------

def water_surface_constant(surface_minus_c, level_at_time, overpass, n_boot: int = BOOT_N,
                           seed: int = SEED) -> dict:
    """Sensitivity of Amendment 17 item 3: the constant that puts ATL24's own sea surface
    on the DAHITI level of the day. Per overpass (granule), the median of L_DAHITI(t) -
    (surface on the working scale without c); the constant is the median over
    overpasses, with the standard deviation of that median over ``n_boot`` resamples of
    overpasses."""
    d = np.asarray(level_at_time, float) - np.asarray(surface_minus_c, float)
    ok = np.isfinite(d)
    if not ok.any():
        return {"constant_m": float("nan"), "n_overpasses": 0}
    per = pd.Series(d[ok]).groupby(np.asarray(overpass)[ok]).median()
    v = per.to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.array([np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)]) if v.size > 1 else v
    return {"constant_m": float(np.median(v)), "sigma_boot_m": float(np.std(boot, ddof=1)) if v.size > 1 else float("nan"),
            "n_overpasses": int(v.size), "per_overpass_p05_m": float(np.percentile(v, 5)),
            "per_overpass_p95_m": float(np.percentile(v, 95)), "per_overpass_nmad_m": validation.nmad(v),
            "estimator": "median over overpasses of the per-overpass median of L_DAHITI(t) - ATL24 surface"}


# --- comparison --------------------------------------------------------------------------------

def _scopes(sector, sectors) -> list[tuple[str, np.ndarray]]:
    s = np.asarray(sector, object)
    return [("overall", np.ones(s.size, bool))] + [(g, s == g) for g in sectors]


def compare(table: pd.DataFrame, pairs: dict[str, tuple[str, str]], sectors, n_boot: int = BOOT_N,
            seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accuracy and paired differences.

    ``table``: one row per cell-track with y_ref, block, sector and one column per
    surface. ``pairs``: {name: (uncorrected column, corrected column)}. For each pair and
    scope (overall, then each sector), on the rows where y_ref and both surfaces are
    finite: bias, RMSE, NMAD (and the median, p05, p95) of each surface's error, and the
    differences corrected minus uncorrected of NMAD, |bias| and RMSE with percentile
    95% intervals over ``n_boot`` block resamples. Returns (accuracy, differences)."""
    y = table["y_ref"].to_numpy(float)
    blk = table["block"].astype(str).to_numpy()
    acc, dif = [], []
    for name, (cu, cf) in pairs.items():
        eu = table[cu].to_numpy(float) - y
        ef = table[cf].to_numpy(float) - y
        ok = np.isfinite(eu) & np.isfinite(ef)
        for scope, m in _scopes(table["sector"], sectors):
            mk = ok & m
            nb = int(np.unique(blk[mk]).size)
            for surf, e in (("uncorrected", eu), ("fused", ef)):
                acc.append({"pair": name, "surface": surf, "column": cu if surf == "uncorrected" else cf,
                            "scope": scope, **validation.error_metrics(e[mk]), "n_blocks": nb})
            a, b = eu[mk], ef[mk]
            stats = {"nmad": lambda u, f: validation.nmad(f) - validation.nmad(u),
                     "abs_bias": lambda u, f: abs(validation.bias(f)) - abs(validation.bias(u)),
                     "rmse": lambda u, f: validation.rmse(f) - validation.rmse(u)}
            if mk.any():
                boot = validation.block_bootstrap(blk[mk], lambda ix: [fn(a[ix], b[ix]) for fn in stats.values()],
                                                  n_boot, seed)
            for k, (metric, fn) in enumerate(stats.items()):
                lo, hi = (validation.percentile_interval(boot[:, k]) if mk.any() and nb > 1
                          else (float("nan"), float("nan")))
                dif.append({"pair": name, "scope": scope, "metric": metric,
                            "uncorrected": (validation.nmad(a) if metric == "nmad" else abs(validation.bias(a))
                                            if metric == "abs_bias" else validation.rmse(a)) if mk.any() else np.nan,
                            "fused": (validation.nmad(b) if metric == "nmad" else abs(validation.bias(b))
                                      if metric == "abs_bias" else validation.rmse(b)) if mk.any() else np.nan,
                            "diff": fn(a, b) if mk.any() else np.nan, "ci_low": lo, "ci_high": hi,
                            "n": int(mk.sum()), "n_blocks": nb, "n_boot": n_boot, "seed": seed})
    return pd.DataFrame(acc), pd.DataFrame(dif)


def verdict(diff_rows: pd.DataFrame, min_blocks: int = MIN_BLOCKS) -> dict:
    """Amendment 17 item 7 for one pair and scope, from its ``compare`` difference rows
    (metrics nmad and abs_bias).

    - "correction improves": the corrected surface's NMAD and |bias| are both lower and
      the 95% interval of the NMAD difference lies below zero;
    - "correction worsens": both higher and the interval lies above zero;
    - "not evaluable": fewer than ``min_blocks`` blocks, or a value missing;
    - "no clear difference": anything else. ``partial`` names what did hold."""
    r = diff_rows.set_index("metric")
    if not {"nmad", "abs_bias"} <= set(r.index):
        return {"verdict": "not evaluable", "reason": "metric missing", "partial": None}
    n_blocks = int(r.loc["nmad", "n_blocks"])
    vals = [r.loc["nmad", "diff"], r.loc["nmad", "ci_low"], r.loc["nmad", "ci_high"], r.loc["abs_bias", "diff"]]
    if n_blocks < min_blocks or not np.isfinite(np.asarray(vals, float)).all():
        return {"verdict": "not evaluable",
                "reason": f"{n_blocks} blocks (< {min_blocks})" if n_blocks < min_blocks else "value missing",
                "partial": None}
    dn, lo, hi, db = (float(v) for v in vals)
    nmad_lower, nmad_higher = dn < 0, dn > 0
    ci_below, ci_above = hi < 0, lo > 0
    if nmad_lower and db < 0 and ci_below:
        return {"verdict": "correction improves", "reason": None, "partial": None}
    if nmad_higher and db > 0 and ci_above:
        return {"verdict": "correction worsens", "reason": None, "partial": None}
    parts = []
    parts.append("NMAD lower" if nmad_lower else "NMAD higher" if nmad_higher else "NMAD equal")
    parts.append("NMAD interval below 0" if ci_below else "NMAD interval above 0" if ci_above
                 else "NMAD interval includes 0")
    parts.append("|bias| lower" if db < 0 else "|bias| higher" if db > 0 else "|bias| equal")
    return {"verdict": "no clear difference", "reason": None, "partial": "; ".join(parts)}
