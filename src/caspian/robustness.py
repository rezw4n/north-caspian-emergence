"""Robustness checks (brief section 10; PREREG section 11; the variants named in
docs/DECISIONS.md): library functions for the checks that need no new Earth Engine
run, and local plan builders for the checks that do. The plan, the cost arithmetic
and the run order are in docs/ROBUSTNESS_PLAN.md.

Nothing here applies a decision rule to real data. The drivers (scripts/r*.py) write
variant tables beside the main ones, for the G3 memo and Extended Data.

Sections
--------
count algebra   parse and merge the stage-2 count bands (w, v, o) of the main export:
                coarser level bins (0.20 m from 0.10 m), finer bins from half-bin
                classes (0.05 m), sums over or selections of group classes (sensor,
                season, threshold source, E1-end window), vegetation variants (D-071)
plan builders   compute-neutral group splits of the main export (one extra token per
                class, every acquisition in one group), the Amendment 3 acquisition
                sample, and screening deltas (main + ADD - SUB) for other screening
                rules
level shifts    first-order effect of another level series, or of a setup correction,
                on contour heights
interpolation   ordinary kriging (a Gaussian process with an exponential covariance)
                of contour points, the comparison method of brief section 10 check 7
validation      DEM-dependent columns from a variant DEM, the validation fits with
                fixed constants (check 6), split reassignment (check 8), the
                Amendment 8 exclusion, tide-system shifts (D-073), headline tables
hindcast        geoid-correction files (D-074), H1 headline tables, the morphology
                table (check 10), the area-level slope (check 11)
sampling        the stratified robustness block sample used by every block-sample
                Earth Engine alternative
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import blocking, design, levels
from .geography import SECTOR_ORDER

# --- count algebra ------------------------------------------------------------------------

# main-export band names (scripts/a6_stage2_export.py): <kind>_<epoch>_b<bin>_p<parity>,
# optionally followed by class tokens of a split plan (split_groups), e.g. E2_b043_p0_E7off_lo
BAND = re.compile(r"^([wvo])_(E[123])_b(\d{3})_p([01])(?:_([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*))?$")
KINDS = ("w", "v", "o")

# class tokens of split_groups; the vocabularies are disjoint, so a token's class is
# known from the token alone
SENSOR_CLASSES = ("TM", "E7on", "E7off", "OLI", "MSI")
HALVES = ("lo", "hi")
SEASONS = ("AMJ", "JASO")            # April-June, July-October (the open-water months, D-062)
THRESHOLD_SOURCES = ("otsu", "fb")   # pooled Otsu valid, or the D-061 fallback
SAMPLE_TOKEN = "a3"                  # acquisitions of the Amendment 3 robustness sample
WINDOW_PREFIX = "x"                  # E1-end window tokens x1, x2, ... (e1_window_index)
SLC_OFF = "2003-05-31"               # caspian.masks.L7_SLC_OFF (tested equal)


def parse_band(name: str) -> dict | None:
    """{kind, epoch, bin, parity, suffix} of a count band, or None."""
    m = BAND.match(name)
    if not m:
        return None
    kind, ep, b, p, suf = m.groups()
    return {"kind": kind, "epoch": ep, "bin": int(b), "parity": int(p), "suffix": suf or ""}


def group_name(epoch: str, b: int, parity: int, suffix: str = "") -> str:
    if not 0 <= int(b) <= 999:
        raise ValueError(f"bin {b} does not fit the 3-digit band name")
    g = f"{epoch}_b{int(b):03d}_p{int(parity)}"
    return f"{g}_{suffix}" if suffix else g


def tokens(suffix: str) -> list[str]:
    return [t for t in suffix.split("_") if t] if suffix else []


def merge_groups(arrays: Mapping[str, np.ndarray], mapper: Callable[[dict], dict | None]) -> dict:
    """Sum count bands whose groups map to the same target group.

    ``arrays``: {band name: 2-D count array} of one stage-2 tile (no other keys).
    ``mapper(group)`` gets {epoch, bin, parity, suffix} and returns the target group
    (same keys; suffix optional) or None to drop the group. Counts are summed, never
    frequencies averaged (caspian.frequency). Every source of a target group must carry
    the same kinds; the result is checked for w <= v (and v <= o) and for uint16
    overflow, and returned as uint16."""
    src: dict = {}
    for name, arr in arrays.items():
        p = parse_band(name)
        if p is None:
            raise ValueError(f"not a count band: {name}")
        src.setdefault((p["epoch"], p["bin"], p["parity"], p["suffix"]), {})[p["kind"]] = arr
    out: dict = {}
    kinds_of: dict = {}
    for (ep, b, par, suf), kinds in sorted(src.items()):
        if "w" not in kinds or "v" not in kinds:
            raise ValueError(f"group {group_name(ep, b, par, suf)}: needs both w and v")
        tgt = mapper({"epoch": ep, "bin": b, "parity": par, "suffix": suf})
        if tgt is None:
            continue
        name = group_name(tgt["epoch"], tgt["bin"], tgt["parity"], tgt.get("suffix", ""))
        ks = frozenset(kinds)
        if kinds_of.setdefault(name, ks) != ks:
            raise ValueError(f"target group {name}: sources carry different kinds")
        for k, a in kinds.items():
            band = f"{k}_{name}"
            a = np.asarray(a, np.int64)
            out[band] = a.copy() if band not in out else out[band] + a
    for name, ks in kinds_of.items():
        w, v = out[f"w_{name}"], out[f"v_{name}"]
        if (w > v).any():
            raise ValueError(f"group {name}: water count exceeds valid count")
        if "o" in ks and (v > out[f"o_{name}"]).any():
            raise ValueError(f"group {name}: valid count exceeds observed count")
    for band, a in out.items():
        if a.max(initial=0) > np.iinfo(np.uint16).max:
            raise ValueError(f"{band}: count overflows uint16")
    return {band: a.astype(np.uint16) for band, a in out.items()}


def rebin_mapper(factor: int) -> Callable[[dict], dict]:
    """Coarser bins on the same origin: fine bin b -> b // factor. With the PREREG
    section 4 origin (-30.00 m) and 0.10 m bins, factor 2 gives the 0.20 m bins of the
    sensitivity (bin k covers [-30 + 0.2 k, -30 + 0.2 (k + 1)) m). The parity halves
    are summed separately, so each half stays a set of whole acquisitions (no longer
    alternating in time across the merged bin)."""
    if int(factor) < 1:
        raise ValueError("factor must be a positive integer")
    return lambda g: {**g, "bin": g["bin"] // int(factor)}


def halves_to_fine_mapper(g: dict) -> dict:
    """0.05 m bins from a plan split by half-bin (split_groups(..., "half")): fine bin
    2 b for token lo, 2 b + 1 for hi; the half token is removed. Same origin."""
    tk = tokens(g["suffix"])
    h = [t for t in tk if t in HALVES]
    if len(h) != 1:
        raise ValueError(f"group {group_name(g['epoch'], g['bin'], g['parity'], g['suffix'])}: no half-bin token")
    rest = "_".join(t for t in tk if t not in HALVES)
    return {**g, "bin": 2 * g["bin"] + HALVES.index(h[0]), "suffix": rest}


def select_mapper(drop: Iterable[str] = (), keep_only: Iterable[str] | None = None,
                  strip: Iterable[str] | None = None) -> Callable[[dict], dict | None]:
    """Select groups by class token and sum over classes.

    ``drop``: groups carrying any of these tokens are dropped (e.g. "E7off" for the
    SLC-off check, "MSI" for Landsat only). ``keep_only``: if given, a group is kept
    only if it carries one of these tokens. ``strip``: the tokens removed from the
    suffix, so that groups differing only in them are summed (default: every token,
    giving the standard main-export group names)."""
    drop, keep = set(drop), None if keep_only is None else set(keep_only)

    def f(g):
        tk = tokens(g["suffix"])
        if drop & set(tk) or (keep is not None and not keep & set(tk)):
            return None
        rest = [] if strip is None else [t for t in tk if t not in set(strip)]
        return {**g, "suffix": "_".join(rest)}
    return f


def e1_end_mapper(ends_sorted, end_alt) -> Callable[[dict], dict]:
    """Epoch of the E1-end window groups under another E1 end (Amendment 4, D-037).

    ``ends_sorted``: the ascending E1-end times used to tag the plan
    (e1_window_index); ``end_alt`` one of them. A group with token x<j> holds
    acquisitions later than j of those ends; with the a4 rule (E1 if time <= end) it
    is E1 under ``end_alt`` = ends_sorted[a] exactly when j <= a. Its parity is kept
    (the halves of an alternative end are therefore approximate; the full-sample
    extents are exact). The window token is removed; other tokens are kept."""
    ends = [pd.Timestamp(t) for t in ends_sorted]
    a = ends.index(pd.Timestamp(end_alt))

    def f(g):
        tk = tokens(g["suffix"])
        w = [t for t in tk if re.fullmatch(rf"{WINDOW_PREFIX}\d+", t)]
        if not w:
            return g
        j = int(w[0][len(WINDOW_PREFIX):])
        rest = "_".join(t for t in tk if t not in w)
        return {**g, "epoch": "E1" if j <= a else "E2", "suffix": rest}
    return f


def compose(*mappers) -> Callable[[dict], dict | None]:
    def f(g):
        for m in mappers:
            g = m(g)
            if g is None:
                return None
        return g
    return f


VEGETATION_MODES = ("not_observed", "as_dry", "as_water")


def vegetation_variant(w, v, o, mode: str):
    """(w', v') under a reed-bed rule (brief section 10 check 9), from the main
    export's counts: w water, v water or dry, o water, dry or emergent vegetation
    (D-071), so o - v acquisitions saw vegetation.
    - not_observed (PREREG section 3, the main rule): (w, v);
    - as_dry: vegetation counts as land, (w, o);
    - as_water: vegetation counts as water (flooded reeds), (w + o - v, o)."""
    w, v, o = (np.asarray(a, np.int64) for a in (w, v, o))
    if (w > v).any() or (v > o).any():
        raise ValueError("counts must satisfy w <= v <= o")
    if mode == "not_observed":
        return w, v
    if mode == "as_dry":
        return w, o
    if mode == "as_water":
        return w + (o - v), o
    raise ValueError(f"mode must be one of {VEGETATION_MODES}")


def vegetation_tile(arrays: Mapping[str, np.ndarray], mode: str) -> dict:
    """vegetation_variant applied to every group of a main-export tile; returns w and
    v bands only (uint16)."""
    groups = {}
    for name in arrays:
        p = parse_band(name)
        if p is None:
            raise ValueError(f"not a count band: {name}")
        groups.setdefault(group_name(p["epoch"], p["bin"], p["parity"], p["suffix"]), set()).add(p["kind"])
    out = {}
    for g, ks in sorted(groups.items()):
        if not {"w", "v", "o"} <= ks:
            raise ValueError(f"group {g}: the vegetation variants need w, v and o (main export, D-071)")
        w2, v2 = vegetation_variant(arrays[f"w_{g}"], arrays[f"v_{g}"], arrays[f"o_{g}"], mode)
        out[f"w_{g}"], out[f"v_{g}"] = w2.astype(np.uint16), v2.astype(np.uint16)
    return out


def load_tile(path) -> tuple[np.ndarray, dict]:
    """(transform, {count band: array}) of a stage-2 tile file (tile_<key>.npz)."""
    with np.load(path) as z:
        return z["transform"], {k: z[k] for k in z.files if k != "transform"}


def write_tile(path, transform, arrays: Mapping[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, transform=np.asarray(transform), **arrays)


def sum_tiles(arrays_list) -> dict:
    """Band-wise sum of the counts of the same tile from several exports (e.g.
    Landsat and Sentinel-2, D-064 step 3, or main + ADD - SUB with signs applied by
    the caller), int64."""
    out: dict = {}
    for arrays in arrays_list:
        for k, a in arrays.items():
            a = np.asarray(a, np.int64)
            out[k] = a.copy() if k not in out else out[k] + a
    return out


def apply_delta(main: Mapping[str, np.ndarray], add: Mapping[str, np.ndarray] | None = None,
                sub: Mapping[str, np.ndarray] | None = None) -> dict:
    """main + ADD - SUB per band (screening_delta). Bands of ADD or SUB missing from
    main are added from zero. Raises if a count would go negative (SUB must be a
    subset of what main counted) or if w > v results."""
    add, sub = add or {}, sub or {}
    shape = next(iter(main.values())).shape
    keys = sorted(set(main) | set(add) | set(sub))
    out = {}
    for k in keys:
        a = np.asarray(main.get(k, np.zeros(shape)), np.int64)
        if k in add:
            a = a + np.asarray(add[k], np.int64)
        if k in sub:
            a = a - np.asarray(sub[k], np.int64)
        if (a < 0).any():
            raise ValueError(f"{k}: SUB removes more than main counted")
        out[k] = a
    for k in out:
        p = parse_band(k)
        if p and p["kind"] == "w":
            v = out.get("v" + k[1:])
            if v is not None and (out[k] > v).any():
                raise ValueError(f"{k}: water count exceeds valid count after the delta")
    return {k: a.astype(np.uint16) for k, a in out.items()}


def substitute_window(main: Mapping[str, np.ndarray], main_transform, alt: Mapping[str, np.ndarray],
                      alt_transform, region) -> tuple[dict, int]:
    """Replace the main counts by an alternative's inside ``region`` (boolean on the alt
    grid): the hybrid mosaic of a block-sample Earth Engine alternative (the robustness
    block sample R, docs/ROBUSTNESS_PLAN.md). Both grids must share the pixel size and
    be aligned (offsets a whole number of pixels); the alt window may extend beyond the
    main tile and is clipped. The alternative's bands must all be main bands; main bands
    it lacks (e.g. the observed count o, which no threshold changes) are kept. Returns
    (new bands, number of main cells replaced)."""
    a, _, c, _, e, f = (float(v) for v in main_transform[:6])
    a2, _, c2, _, e2, f2 = (float(v) for v in alt_transform[:6])
    if not (np.isclose(a, a2) and np.isclose(e, e2)):
        raise ValueError("pixel sizes differ")
    dc, dr = (c2 - c) / a, (f2 - f) / e
    if not (np.isclose(dc, round(dc)) and np.isclose(dr, round(dr))):
        raise ValueError("grids are not aligned")
    dc, dr = int(round(dc)), int(round(dr))
    if not set(alt) <= set(main):
        raise ValueError(f"alternative bands not in main: {sorted(set(alt) - set(main))[:5]}")
    shape = next(iter(main.values())).shape
    reg = np.asarray(region, bool)
    r0, c0 = max(dr, 0), max(dc, 0)
    r1, c1 = min(dr + reg.shape[0], shape[0]), min(dc + reg.shape[1], shape[1])
    out = {k: np.array(v) for k, v in main.items()}
    if r1 <= r0 or c1 <= c0:
        return out, 0
    win = reg[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    for k in alt:
        out[k][r0:r1, c0:c1][win] = np.asarray(alt[k])[r0 - dr:r1 - dr, c0 - dc:c1 - dc][win]
    return out, int(win.sum())


def contour_offsets(points_a: pd.DataFrame, points_b: pd.DataFrame, by=("epoch", "bin")) -> pd.DataFrame:
    """Per group, the symmetric nearest-point distance between two sets of contour points
    (x, y) of the same level bin: the median and p95 over both directions and the
    counts. Used where two waterline sets are compared directly (30 m vs 60 m, one
    sensor vs another, AMJ vs JASO scenes), within the cells both observed."""
    rows = []
    ga, gb = dict(tuple(points_a.groupby(list(by)))), dict(tuple(points_b.groupby(list(by))))
    for key in sorted(set(ga) & set(gb)):
        pa, pb = ga[key][["x", "y"]].to_numpy(float), gb[key][["x", "y"]].to_numpy(float)
        dab = cKDTree(pb).query(pa)[0]
        dba = cKDTree(pa).query(pb)[0]
        d = np.concatenate([dab, dba])
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "n_a": len(pa), "n_b": len(pb),
                     "median_m": float(np.median(d)), "p95_m": float(np.percentile(d, 95)),
                     "median_a_to_b_m": float(np.median(dab)), "median_b_to_a_m": float(np.median(dba))})
    return pd.DataFrame(rows)


# --- plan builders -------------------------------------------------------------------------

def _times(x) -> pd.Series:
    """Datetimes from timestamps or ISO 8601 strings (with or without a time part)."""
    s = pd.Series(np.asarray(x))
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        return pd.to_datetime(s, format="ISO8601")
    return pd.to_datetime(s)


def sensor_class(sensor, time) -> np.ndarray:
    """TM, E7on (ETM+ before the SLC failure), E7off, OLI (Landsat 8 and 9) or MSI."""
    s = pd.Series(np.asarray(sensor, object)).astype(str)
    t = _times(time)
    cls = np.select([s.eq("TM_L5"), s.eq("ETM_L7") & (t < pd.Timestamp(SLC_OFF)), s.eq("ETM_L7"),
                     s.isin(["OLI_L8", "OLI_L9"]), s.eq("MSI_S2")], list(SENSOR_CLASSES), default="")
    if (cls == "").any():
        raise ValueError(f"unknown sensors {sorted(set(s[cls == '']))}")
    return cls


def e1_window_index(times, ends_sorted) -> np.ndarray:
    """Number of the ascending E1-end times strictly earlier than each time: 0 means E1
    under every end; j means E2 under the j earliest ends (a4 rule: E1 if time <= end)."""
    t = _times(times).to_numpy("datetime64[ns]")
    e = np.array([pd.Timestamp(x).to_datetime64() for x in ends_sorted], "datetime64[ns]")
    if np.any(np.diff(e) <= np.timedelta64(0)):
        raise ValueError("ends must be strictly ascending")
    return np.searchsorted(e, t, side="left")


def split_groups(plan: pd.DataFrame, by: Iterable[str] = ("sensor", "half"), *, e1_ends=None,
                 sample_acqs=None, width: float = design.BIN_WIDTH_M, origin: float = design.BIN_ORIGIN_M,
                 time_col: str = "t") -> pd.DataFrame:
    """The main plan with each group split into classes (docs/ROBUSTNESS_PLAN.md, main-
    export design), for upload with scripts/a6_stage2_export.py.

    Each scene is still in exactly one group, so the export computes every scene once,
    as before; only the number of bands grows. Every class is constant within an
    acquisition (all its scenes take the class of the acquisition), so summing the
    classes of a group (select_mapper()) reproduces the unsplit group exactly, including
    the per-acquisition mosaic. Classes (tokens, in this order):
    - "sensor": SENSOR_CLASSES;
    - "half": lo or hi half of the 0.10 m bin, from the acquisition's median level in the
      bin (the 0.05 m sensitivity, halves_to_fine_mapper);
    - "season": AMJ or JASO, from the acquisition date;
    - "threshold": otsu or fb (D-061 threshold source of the acquisition);
    - e1_ends: the ascending alternative E1 ends of Amendment 4; E1 and E2 acquisitions
      later than the earliest end and not later than the latest get token x<j>
      (e1_window_index, e1_end_mapper);
    - sample_acqs: acquisitions of the Amendment 3 sample get token "a3", so that their
      main-threshold counts can be removed without another export.
    Columns kept; ``base_group`` holds the original group and ``group`` the split one."""
    d = plan.copy()
    t = _times(d[time_col].to_numpy())
    t.index = d.index
    acq_first = t.groupby(d["acq"]).transform("min")
    parts = []
    for key in by:
        if key == "sensor":
            parts.append(pd.Series(sensor_class(d["sensor"], acq_first), index=d.index))
        elif key == "half":
            centre = levels.bin_center(d["bin"].to_numpy(), width, origin)
            med = d.groupby(["epoch", "bin", "acq"])["level_m"].transform("median")
            parts.append(pd.Series(np.where(med.to_numpy() >= centre, "hi", "lo"), index=d.index))
        elif key == "season":
            parts.append(pd.Series(np.where(acq_first.dt.month <= 6, "AMJ", "JASO"), index=d.index))
        elif key == "threshold":
            src = d.groupby("acq")["threshold_source"].transform("first")
            parts.append(pd.Series(np.where(src.eq("pooled_otsu"), "otsu", "fb"), index=d.index))
        else:
            raise ValueError(f"unknown split {key!r}")
    if e1_ends is not None:
        j = e1_window_index(acq_first, sorted(pd.Timestamp(x) for x in e1_ends))
        tag = d["epoch"].isin(["E1", "E2"]).to_numpy() & (j > 0) & (j < len(e1_ends))
        parts.append(pd.Series(np.where(tag, [f"{WINDOW_PREFIX}{k}" for k in j], ""), index=d.index))
    if sample_acqs is not None:
        parts.append(pd.Series(np.where(d["acq"].isin(set(sample_acqs)), SAMPLE_TOKEN, ""), index=d.index))
    base = d["group"] if "group" in d.columns else (d["epoch"] + "_b" + d["bin"].astype(int).map("{:03d}".format)
                                                   + "_p" + d["parity"].astype(int).astype(str))
    d["base_group"] = base
    cols = [p.to_numpy(object) for p in parts]
    suffix = ["_".join(t for t in row if t) for row in zip(*cols)] if cols else [""] * len(d)
    d["group"] = [b if not s else f"{b}_{s}" for b, s in zip(base, suffix)]
    mixed = d.groupby(["base_group", "acq"])["group"].nunique()
    if (mixed > 1).any():
        raise AssertionError("an acquisition was split across classes")
    return d


def acquisition_sample(plan: pd.DataFrame, n_scenes_per_sensor: int = design.EDGE_OTSU_SAMPLE_N // 5,
                       seed: int = design.SEED) -> pd.DataFrame:
    """Amendment 3 robustness sample, drawn by whole acquisitions (the counting unit,
    Amendment 3 item 3; thresholds are per acquisition, D-061). Within each sensor, in
    sorted order, the acquisitions are permuted by one generator seeded once, and taken
    in that order until the sensor holds at least ``n_scenes_per_sensor`` scenes (100 per
    sensor: the 500 scenes of Amendment 3 over five sensors). Returns one row per
    sampled acquisition with sensor and n_scenes."""
    a = (plan.groupby("acq").agg(sensor=("sensor", "first"), n_scenes=("scene", "nunique"))
         .reset_index().sort_values(["sensor", "acq"]))
    rng = np.random.default_rng(seed)
    out = []
    for sensor, g in a.groupby("sensor", sort=True):
        g = g.iloc[rng.permutation(len(g))]
        k = int(np.searchsorted(g["n_scenes"].cumsum().to_numpy(), n_scenes_per_sensor) + 1)
        out.append(g.iloc[:k])
    return pd.concat(out, ignore_index=True)


def setup_columns(plan: pd.DataFrame) -> list[str]:
    cols = [f"setup_{s}_m" for s in SECTOR_ORDER]
    missing = [c for c in cols if c not in plan.columns]
    if missing:
        raise ValueError(f"plan lacks setup columns {missing}")
    return cols


def exclusion(plan: pd.DataFrame, tau: float, *, unknown_excluded: bool = True, cols=None,
              exempt_sectors: Iterable[str] = ()) -> np.ndarray:
    """(scenes, sectors) boolean: scene excluded in sector (|setup| > tau; NaN setup is
    excluded when ``unknown_excluded``, the main rule of D-066). Sectors in SECTOR_ORDER;
    ``exempt_sectors`` are never excluded (e.g. Komsomolets without screening, D-066)."""
    cols = cols or setup_columns(plan)
    a = plan[cols].to_numpy(float)
    ex = np.abs(a) > tau
    if unknown_excluded:
        ex |= ~np.isfinite(a)
    for s in exempt_sectors:
        ex[:, SECTOR_ORDER.index(s)] = False
    return ex


def _codes(mask_row) -> str:
    return ",".join(str(k + 1) for k in np.flatnonzero(mask_row))


def screening_delta(plan: pd.DataFrame, excl_main: np.ndarray, excl_alt: np.ndarray) -> dict:
    """Plans that turn the main export's counts into those of another screening rule:
    alt = main + ADD - SUB (apply_delta), without re-exporting the scenes on which the
    two rules agree.
    - ADD: scene-sector pairs excluded by the main rule and included by the alternative;
    - SUB: pairs included by the main rule and excluded by the alternative.
    Each is returned as plan rows (one per scene with at least one pair) with ``sectors``
    the comma-separated sector codes (SECTOR_ORDER index + 1) and mask_mode "include",
    in the main groups, ready for upload. The identity is exact when the decisions are
    constant within each acquisition and sector (each scene of an overpass is then in
    the same set): the number of (acquisition, sector) pairs with mixed decisions, under
    either rule, is reported in ``n_mixed_acq_sector``. Parity is unchanged (it is set
    before screening)."""
    if excl_main.shape != excl_alt.shape or excl_main.shape[0] != len(plan):
        raise ValueError("exclusion arrays must be (scenes, sectors) for this plan")
    add, sub = excl_main & ~excl_alt, ~excl_main & excl_alt

    def rows(m):
        keep = m.any(axis=1)
        r = plan.loc[keep].copy()
        r["sectors"] = [_codes(x) for x in m[keep]]
        r["mask_mode"] = "include"
        return r

    mixed = 0
    for ex in (excl_main, excl_alt):
        g = pd.DataFrame(ex.astype(np.int8), columns=SECTOR_ORDER).groupby(plan["acq"].to_numpy())
        mixed += int((g.min() != g.max()).to_numpy().sum())
    return {"add": rows(add), "sub": rows(sub), "n_add_pairs": int(add.sum()), "n_sub_pairs": int(sub.sum()),
            "n_mixed_acq_sector": mixed}


def delta_cost_share(rows: pd.DataFrame, scene_km2: pd.Series, total_km2: float,
                     sector_weight: Mapping[str, float] | None = None) -> dict:
    """Cost of a delta export as a share of the main export, from scene-weighted areas.
    ``upper``: every delta scene computed over its whole footprint in the cover (the
    sector mask is applied after the computation, D-026). ``per_sector``: the export split
    into sector regions, each scene computed only in its delta sectors, approximated with
    the sectors' shares of the cover (``sector_weight``, name -> share, summing to 1)."""
    km2 = scene_km2.reindex(rows.index).to_numpy(float)
    up = float(np.nansum(km2) / total_km2)
    out = {"upper": up, "n_scenes": int(len(rows))}
    if sector_weight is not None:
        w = np.array([sum(sector_weight[SECTOR_ORDER[int(c) - 1]] for c in s.split(",") if c)
                      for s in rows["sectors"].astype(str)])
        out["per_sector"] = float(np.nansum(km2 * w) / total_km2)
    return out


# --- level shifts -------------------------------------------------------------------------

def group_level_shift(plan: pd.DataFrame, alt_level, *, width: float = design.BIN_WIDTH_M,
                      origin: float = design.BIN_ORIGIN_M, by=("epoch", "bin")) -> pd.DataFrame:
    """First-order effect of another level series on the contour heights (brief section
    10 check 5): per group, the mean over its acquisitions of alt - main level (each
    acquisition the mean of its scenes). A contour keeps its location and takes the
    bin centre plus this shift. Second-order effects (acquisitions that would change
    bin, so the contour itself would change) are counted in ``n_acq_rebin``, with
    ``share_rebin``; they need a re-binned export (docs/ROBUSTNESS_PLAN.md)."""
    d = plan.assign(_alt=np.asarray(alt_level, float))
    a = d.groupby([*by, "acq"]).agg(main=("level_m", "mean"), alt=("_alt", "mean")).reset_index()
    a["delta"] = a["alt"] - a["main"]
    a["rebin"] = levels.bin_index(a["alt"].to_numpy(), width, origin) != levels.bin_index(
        a["main"].to_numpy(), width, origin)
    a.loc[~np.isfinite(a["alt"]), "rebin"] = False
    g = a.groupby(list(by)).agg(shift_m=("delta", "mean"), n_acq=("acq", "size"),
                                n_acq_alt=("alt", lambda s: int(np.isfinite(s).sum())),
                                n_acq_rebin=("rebin", "sum")).reset_index()
    g["share_rebin"] = g["n_acq_rebin"] / g["n_acq"]
    return g


def shift_points(points: pd.DataFrame, shifts: pd.DataFrame, *, on=("epoch", "bin"), col: str = "z_m",
                 shift_col: str = "shift_m") -> pd.DataFrame:
    """Contour points with ``col`` += the shift of their group (group_level_shift, or a
    per-sector setup correction with on=("epoch", "bin", "sector")). A point whose group
    has no finite shift raises: every contour of the DEM must be shifted."""
    s = shifts[[*on, shift_col]]
    if s.duplicated(list(on)).any():
        raise ValueError("one shift per group")
    out = points.merge(s, on=list(on), how="left", validate="many_to_one")
    if not np.isfinite(out[shift_col].to_numpy(float)).all():
        raise ValueError("points without a finite shift for their group")
    out[col] = out[col].to_numpy(float) + out[shift_col].to_numpy(float)
    return out.drop(columns=[shift_col])


def setup_shift(plan: pd.DataFrame, excl: np.ndarray, *, by=("epoch", "bin")) -> pd.DataFrame:
    """Setup correction instead of screening, first order (brief section 16 pivot;
    check 4): per group and sector, the mean predicted setup over the acquisitions
    included in that sector. A waterline seen under setup s lies at L + s, so the
    contour height becomes bin centre + this mean (``shift_m``). Groups with no
    included acquisition in a sector get NaN there."""
    cols = setup_columns(plan)
    rows = []
    for k, s in enumerate(SECTOR_ORDER):
        inc = ~excl[:, k]
        d = plan.loc[inc, [*by, "acq"]].assign(_s=plan.loc[inc, cols[k]].to_numpy(float))
        a = d.groupby([*by, "acq"])["_s"].mean().reset_index()
        g = a.groupby(list(by))["_s"].agg(["mean", "size"]).reset_index()
        rows.append(g.rename(columns={"mean": "shift_m", "size": "n_acq"}).assign(sector=s))
    return pd.concat(rows, ignore_index=True)


# --- interpolation: ordinary kriging -----------------------------------------------------

def exp_cov(h, sill: float, a: float):
    """Exponential covariance sill exp(-h / a) (Matern 1/2; practical range 3 a)."""
    return sill * np.exp(-np.asarray(h, float) / a)


def fit_covariance(xy, z, *, bin_width: float = 500.0, max_lag: float = 20_000.0, max_pairs: int = 400_000,
                   seed: int = design.SEED) -> dict:
    """Exponential model of contour heights from their empirical semivariogram
    (caspian.blocking): nugget c0, partial sill c1 and a. Fitted by the caller on
    training-block contour points only (PREREG section 12)."""
    xy = np.asarray(xy, float)
    v = blocking.empirical_semivariogram(xy[:, 0], xy[:, 1], np.asarray(z, float), bin_width, max_lag,
                                         max_pairs, seed)
    fit = blocking.fit_exponential(v)
    return {"nugget": fit["nugget"], "sill": fit["partial_sill"], "a": fit["a"],
            "practical_range": fit["practical_range"], "bin_width_m": bin_width, "max_lag_m": max_lag,
            "n_points": int(len(xy))}


def thin_points(xy, spacing: float) -> np.ndarray:
    """Indices keeping the first point of each ``spacing`` square cell, which removes
    the along-contour clustering (points every 60 m on lines kilometres apart) that
    makes kriging systems ill-conditioned."""
    xy = np.asarray(xy, float)
    key = np.floor(xy / spacing).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return np.sort(idx)


def ordinary_kriging(xy, z, qxy, *, sill: float, a: float, nugget: float = 0.0, k: int = 32,
                     per_quadrant: int | None = 8, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging (a Gaussian process with an unknown constant mean) of heights
    ``z`` at points ``xy`` to query points ``qxy``, with an exponential covariance and a
    measurement nugget, in a local neighbourhood.

    Neighbourhood: the ``k`` nearest points (``4 * per_quadrant`` candidates when a
    quadrant rule is used, of which at most ``per_quadrant`` per quadrant around the
    query are kept), so that a cell between two contours draws on both. Excluded
    candidates get zero weight exactly (identity rows). The nugget sits on the diagonal
    only (measurement error), so the predictor smooths rather than interpolates when it
    is positive. Returns (prediction, kriging variance of the noise-free field:
    sill - sum w c0 - mu)."""
    xy, z, q = np.asarray(xy, float), np.asarray(z, float), np.asarray(qxy, float)
    n = len(z)
    if n == 0:
        raise ValueError("no data points")
    cand = min(n, 4 * per_quadrant if per_quadrant else k)
    tree = cKDTree(xy)
    zhat, var = np.full(len(q), np.nan), np.full(len(q), np.nan)
    eps = 1e-10 * sill
    for s in range(0, len(q), chunk):
        qq = q[s:s + chunk]
        m = len(qq)
        d, idx = tree.query(qq, k=cand)
        if cand == 1:
            d, idx = d[:, None], idx[:, None]
        P = xy[idx]                                                   # (m, cand, 2)
        if per_quadrant:
            off = P - qq[:, None, :]
            quad = (off[..., 0] >= 0).astype(int) + 2 * (off[..., 1] >= 0).astype(int)
            rank = np.zeros(quad.shape, int)
            for qd in range(4):
                is_q = quad == qd
                rank = np.where(is_q, np.cumsum(is_q, axis=1) - 1, rank)
            use = rank < per_quadrant
        else:
            use = np.ones(d.shape, bool)
        D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        C = exp_cov(D, sill, a)
        C[:, np.arange(cand), np.arange(cand)] += nugget + eps
        c0 = exp_cov(d, sill, a)
        A = np.zeros((m, cand + 1, cand + 1))
        A[:, :cand, :cand] = C
        A[:, :cand, cand] = 1.0
        A[:, cand, :cand] = 1.0
        rhs = np.zeros((m, cand + 1))
        rhs[:, :cand] = c0
        rhs[:, cand] = 1.0
        # excluded candidates: weight 0 exactly (identity row and column, out of the constraint)
        ex = ~use
        if ex.any():
            mi, ci = np.nonzero(ex)
            A[mi, ci, :] = 0.0
            A[mi, :, ci] = 0.0
            A[mi, ci, ci] = 1.0
            rhs[mi, ci] = 0.0
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        w, mu = sol[:, :cand], sol[:, cand]
        zhat[s:s + m] = (w * z[idx]).sum(axis=1)
        var[s:s + m] = sill - (w * c0).sum(axis=1) - mu
    return zhat, var


# --- validation ----------------------------------------------------------------------------

DEM_TILE_M = 60_000      # scripts/a6_dem.py TILE_M (D-069 tiling fix)


def dem_columns(points: pd.DataFrame, dem_dir, contour_x, contour_y, tile_m: float = DEM_TILE_M) -> pd.DataFrame:
    """The DEM-dependent columns of a validation point table (scripts/a6_validate_dem.py
    prepare) recomputed from a variant DEM: dem_h, interp_dist, slope (caspian.validation.
    TileSet on the 60 km DEM tiling) and contour_density (the variant's contour points).
    The reference heights, the dryness selection and the product heights are kept, so a
    comparison with the main DEM is paired on the same points."""
    from . import validation
    ts = validation.TileSet(Path(dem_dir), "dem", tile_m=tile_m)
    x, y = points["x"].to_numpy(float), points["y"].to_numpy(float)
    return points.assign(dem_h=ts.sample(x, y, 1), interp_dist=ts.sample(x, y, 2), slope=ts.slope(x, y),
                         contour_density=validation.contour_density(x, y, contour_x, contour_y))


def run_fixed(points: pd.DataFrame, products: dict, *, constant: dict | None = None,
              product_offsets: dict | None = None, n_boot: int = design.BOOTSTRAP_N, seed: int = design.SEED,
              level: float = design.CONFORMAL_LEVEL, min_n: int = design.MONDRIAN_MIN_N,
              sectors=SECTOR_ORDER) -> dict:
    """caspian.validation.run with the training-block datum constant c and/or the product
    offsets supplied instead of fitted (check 6: c +- its bootstrap sigma; offsets +-
    sigma). Same order of operations; with neither supplied it equals validation.run."""
    from . import validation as V
    constant = constant or V.fit_datum_constant(points, n_boot, seed)
    pts = V.apply_datum_constant(points, constant)
    offsets = product_offsets or V.fit_product_offsets(pts, products, n_boot, seed)
    pts = V.product_variants(pts, products, offsets, constant)
    model = V.fit_scale_model(pts)
    calibration = V.calibrate(pts, model, level, min_n, sectors)
    coverage = V.coverage_on_test(pts, model, calibration, n_boot, seed, sectors=sectors)
    models = {"dem": "dem_h", **{f"{p}|{v}": f"{p}|{v}" for p in products for v in ("total", "matched")}}
    summary, per_block = V.accuracy_tables(pts, models, sectors)
    boots = [V.paired_block_bootstrap(pts, "dem_h", {p: f"{p}|{v}" for p in products}, n_boot=n_boot, seed=seed,
                                      sectors=sectors).assign(variant=v)
             for v in ("total", "matched") if products]
    return {"constant": constant, "product_offsets": offsets, "scale_model": model, "calibration": calibration,
            "coverage": coverage, "accuracy": summary, "accuracy_blocks": per_block,
            "paired_bootstrap": pd.concat(boots, ignore_index=True) if boots else pd.DataFrame(), "points": pts}


def shifted_constant(constant: dict, n_sigma: float) -> dict:
    """c + n_sigma x its bootstrap sigma, labelled."""
    return {**constant, "constant_m": constant["constant_m"] + n_sigma * constant["sigma_boot_m"],
            "robustness": f"c {n_sigma:+g} sigma"}


def shifted_offsets(offsets: dict, n_sigma: float) -> dict:
    return {k: {**v, "offset_m": v["offset_m"] + n_sigma * v["sigma_boot_m"]} for k, v in offsets.items()}


def reassign_splits(blocks: pd.DataFrame, seed: int, proportions: dict = design.SPLIT) -> pd.DataFrame:
    """The block table with a new random train/calib/test assignment (same stratified
    rule, caspian.blocking.assign_splits, another seed or proportions). For the
    calibration-split check of brief section 10 check 8, run after the main evaluation
    is frozen; it reuses test blocks for fitting and is reported as such."""
    return blocks.assign(split=blocking.assign_splits(blocks, proportions, seed).to_numpy())


def swap_splits(points: pd.DataFrame, a: str = "calib", b: str = "test") -> pd.DataFrame:
    """Split labels a and b exchanged (q-hat from the former test blocks, coverage on
    the former calibration blocks)."""
    s = points["split"].copy()
    return points.assign(split=np.where(s == a, b, np.where(s == b, a, s)))


def exclude_initial_training(points: pd.DataFrame, initial_blocks: pd.DataFrame,
                             initial_size: float = design.BLOCK_SIZE_M, splits=("calib", "test")) -> pd.DataFrame:
    """Amendment 8 clarification: drop the calibration and test points of the final
    assignment that lie in training blocks of the initial 25 km assignment, whose
    contour points fixed the static-contour rule (D-075). Needed only when PREREG
    section 6 enlarged the blocks."""
    i, j = blocking.block_index(points["x"], points["y"], initial_size)
    tr = set(zip(initial_blocks.loc[initial_blocks["split"] == "train", "i"],
                 initial_blocks.loc[initial_blocks["split"] == "train", "j"]))
    in_tr = np.array([(a, b) in tr for a, b in zip(i, j)], bool)
    drop = in_tr & points["split"].isin(list(splits)).to_numpy()
    return points.loc[~drop].reset_index(drop=True)


def tide_system_shift(lat_deg, recipe: str) -> np.ndarray:
    """Change of an ICESat-2 orthometric height H = h - geoid_h (tide-free) under the
    two mean-tide recipes of docs/DATUMS.md (D-073), from the product terms
    geoid_free2mean = 0.1287 - 0.3848 sin^2(phi) and tide_earth_free2mean =
    0.06029 - 0.180873 sin^2(phi) (m):
    - "both_mean": h and N both to mean tide, dH = tide_earth_free2mean - geoid_free2mean;
    - "geoid_mean": the Comparison Guide recipe h - (geoid_h + geoid_free2mean),
      dH = -geoid_free2mean.
    The constant part is absorbed by the training-block constant c; the north-south
    tilt is what the check tests."""
    s2 = np.sin(np.radians(np.asarray(lat_deg, float))) ** 2
    g = 0.1287 - 0.3848 * s2
    if recipe == "both_mean":
        return (0.06029 - 0.180873 * s2) - g
    if recipe == "geoid_mean":
        return -g
    raise ValueError("recipe must be 'both_mean' or 'geoid_mean'")


def validation_headline(r: dict) -> pd.DataFrame:
    """Long table of the H2 and coverage headline quantities of a validation run
    (validation.run or run_fixed): accuracy (n, bias, rmse, nmad, p05, p95 per scope and
    model), coverage (coverage, ci_low, ci_high, n per scope), the paired differences
    (diff, ci_low, ci_high per scope, product, metric, variant) and the datum constant."""
    rows = []
    for _, a in r["accuracy"].iterrows():
        for q in ("n", "bias", "rmse", "nmad", "p05", "p95"):
            rows.append({"table": "accuracy", "scope": a["scope"], "item": a["model"], "quantity": q,
                         "value": float(a[q])})
    for _, c in r["coverage"].iterrows():
        for q in ("coverage", "ci_low", "ci_high", "n"):
            rows.append({"table": "coverage", "scope": c["scope"], "item": "dem", "quantity": q, "value": float(c[q])})
    pb = r["paired_bootstrap"]
    for _, b in (pb.iterrows() if len(pb) else []):
        for q in ("diff", "ci_low", "ci_high"):
            rows.append({"table": "paired", "scope": b["scope"], "item": f"{b['product']}|{b['variant']}|{b['metric']}",
                         "quantity": q, "value": float(b[q])})
    rows.append({"table": "constant", "scope": "train", "item": "c", "quantity": "constant_m",
                 "value": float(r["constant"]["constant_m"])})
    return pd.DataFrame(rows)


def compare_headlines(main: pd.DataFrame, alt: pd.DataFrame, keys=("table", "scope", "item", "quantity")) -> pd.DataFrame:
    """Main and variant values side by side, with alt - main."""
    m = main.merge(alt, on=list(keys), how="outer", suffixes=("_main", "_alt"))
    m["delta"] = m["value_alt"] - m["value_main"]
    return m


# --- hindcast -----------------------------------------------------------------------------

def write_correction(geoid_npz, layer: str, out_path) -> dict:
    """A vertical-correction file for scripts/a7_hindcast.py --vertical-correction
    (caspian.products.load_correction: lon, lat, dN) from one layer of the geoid-
    difference file (results/study_area/geoid_egm2008_minus_eigen6c4.npz): dN_ngaEGM2008_m
    for the main total error (D-074) or dN_m for the sensitivity without NGA's -0.41 m
    height offset."""
    with np.load(geoid_npz) as z:
        if layer not in z.files:
            raise KeyError(f"{layer} not in {Path(geoid_npz).name}")
        lon, lat, dn = z["lon"], z["lat"], z[layer].astype(np.float32)
    if dn.shape != (lat.size, lon.size):
        raise ValueError("layer must be [lat, lon]")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, lon=lon, lat=lat, dN=dn,
                        definition=f"{layer} of {Path(geoid_npz).name} (D-074; robustness)")
    return {"file": out_path.name, "layer": layer, "mean_m": float(np.nanmean(dn)), "sd_m": float(np.nanstd(dn))}


def h1_tables(displacement: pd.DataFrame, noise_floor: pd.DataFrame):
    """(product_levels, noise_levels, noise_pooled) for caspian.h1.mispredicts from the
    tables of scripts/a7_hindcast.py: the epoch-pooled product rows per sector and bin,
    the per-level pooled noise floor, and the noise floor pooled over all levels (the
    D-032 sensitivity)."""
    pl = (displacement[(displacement["epoch"] == "pooled") & displacement["sector"].isin(SECTOR_ORDER)]
          .rename(columns={"median_m": "median"})[["product", "variant", "sector", "bin", "observed_level", "n",
                                                   "median"]])
    nl = (noise_floor[noise_floor["kind"] == "pooled_level"].rename(columns={"p95_m": "p95"})
          [["sector", "bin", "n", "p95"]])
    npool = (noise_floor[noise_floor["kind"] == "pooled_all_levels"].rename(columns={"p95_m": "p95"})
             [["sector", "n", "p95"]])
    pl = pl.assign(observed_level=pl["observed_level"].astype(str).str.lower().isin(["true", "1"]))
    return pl, nl, npool


def h1_headline(tables_dir) -> dict:
    """H1 headline tables of one hindcast run directory: the "mispredicts" summary per
    product, variant and sector (per-level floor, the deciding form, and the pooled
    floor), and per product and variant the domain area-bias summary over all units.
    The permutation test needs the sector labels and is run with caspian.h1.block_test
    on block_h1.csv by the caller."""
    from . import h1
    d = Path(tables_dir)
    disp, nf = pd.read_csv(d / "displacement.csv"), pd.read_csv(d / "noise_floor.csv")
    pl, nl, npool = h1_tables(disp, nf)
    s_level, _ = h1.mispredicts(pl, nl)
    s_pool, _ = h1.mispredicts(pl, npool)
    ext = pd.read_csv(d / "extent.csv")
    dom = ext[(ext["sector"] == "domain") & ext["epoch"].isin(["E1", "E2", "E3"])]
    area = (dom.groupby(["product", "variant"])
            .agg(n_units=("bin", "size"), sum_abs_area_bias_km2=("area_bias_total_km2", lambda s: float(np.abs(s).sum())),
                 median_area_bias_km2=("area_bias_total_km2", "median"),
                 median_rel_bias=("area_bias_rel_total", "median"), median_iou=("iou_total", "median"))
            .reset_index())
    return {"mispredicts": s_level, "mispredicts_pooled_floor": s_pool, "area": area}


def compare_h1(main: dict, alt: dict) -> pd.DataFrame:
    """Per product, variant and sector: the main and variant "mispredicts" outcomes and
    counts. ``in_both`` marks rows present in both runs (a variant may compute only
    the total or only the shape variant); ``changed`` is True where both runs have
    the row and the outcome differs, including evaluable in one run only."""
    keys = ["product", "variant", "sector"]
    m = main["mispredicts"].merge(alt["mispredicts"], on=keys, how="outer", suffixes=("_main", "_alt"),
                                  indicator=True)
    m["in_both"] = m.pop("_merge").eq("both")
    a, b = m["mispredicts_main"].astype("boolean"), m["mispredicts_alt"].astype("boolean")
    m["changed"] = m["in_both"] & (a != b).fillna(a.isna() != b.isna()).astype(bool)
    return m


def morphology_table(displacement: pd.DataFrame, noise_floor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check 10 (PREREG section 9, alternative explanation): at the levels observed in E1
    and in E2 or E3, does a product fit the older (rising-limb) seabed better than the
    later one, and how large is the product's displacement against the E1-vs-E2/E3
    displacement (the measured seabed change at that level)?

    Returns (detail per product, variant, sector and bin: median |d| against E1, E2, E3,
    their E2/E3 mean, e1_better, seabed_change_m (median of the e1_vs_e23 noise-floor
    row), ratio = E2/E3 displacement / seabed change; summary per product, variant and
    sector: n_bins, n_e1_better, a two-sided sign-test p-value and the median ratio).
    A product that fits E1 better at most shared levels, with a ratio near 1, is
    consistent with seabed change rather than bathymetry error."""
    from scipy.stats import binomtest
    d = displacement[displacement["epoch"].isin(["E1", "E2", "E3"]) & displacement["sector"].isin(SECTOR_ORDER)]
    piv = d.pivot_table(index=["product", "variant", "sector", "bin"], columns="epoch", values="median_m",
                        aggfunc="first")
    for e in ("E1", "E2", "E3"):
        if e not in piv.columns:
            piv[e] = np.nan
    piv = piv[np.isfinite(piv["E1"]) & (np.isfinite(piv["E2"]) | np.isfinite(piv["E3"]))].copy()
    piv["E23"] = piv[["E2", "E3"]].mean(axis=1)
    piv["e1_better"] = piv["E1"] < piv["E23"]
    ch = (noise_floor[(noise_floor["kind"] == "e1_vs_e23") & noise_floor["sector"].isin(SECTOR_ORDER)]
          [["sector", "bin", "median_m"]].rename(columns={"median_m": "seabed_change_m"}))
    det = piv.reset_index().merge(ch, on=["sector", "bin"], how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        det["ratio"] = det["E23"] / det["seabed_change_m"]
    rows = []
    for (p, v, s), g in det.groupby(["product", "variant", "sector"]):
        fin = np.isfinite(g["E1"]) & np.isfinite(g["E23"]) & (g["E1"] != g["E23"])
        n, k = int(fin.sum()), int((g.loc[fin, "e1_better"]).sum())
        rows.append({"product": p, "variant": v, "sector": s, "n_bins": n, "n_e1_better": k,
                     "sign_test_p": float(binomtest(k, n, 0.5).pvalue) if n else float("nan"),
                     "median_ratio": float(np.nanmedian(g["ratio"])) if np.isfinite(g["ratio"]).any() else float("nan")})
    return det, pd.DataFrame(rows)


def area_level_slope(observed: pd.DataFrame, *, lo: float = -29.0, hi: float = -26.0, sector: str = "all",
                     sample: str = "full", area_col: str = "area_sea_km2", min_valid_frac: float = 0.95,
                     n_boot: int = design.BOOTSTRAP_N, seed: int = design.SEED) -> dict:
    """Check 11: slope (km2 per m) of the observed sea area against level, by least
    squares over the (epoch, level) units with lo <= level <= hi, for comparison with a
    published area-level relation (Akbari et al. 2020: 7,103 km2 per m for the Northern
    Caspian, -29 < SWL < -26 m; refs/literature_values.csv). A constant datum offset
    between the level scales does not change a slope. Units whose valid area is below
    ``min_valid_frac`` of the largest are left out (cloud or low-count holes would bias
    the area), and counted. Bootstrap interval over units."""
    o = observed[(observed["sector"] == sector) & (observed["sample"] == sample)
                 & observed["epoch"].isin(["E1", "E2", "E3"])].copy()
    o = o[(o["level_m"] >= lo) & (o["level_m"] <= hi)]
    if o.empty:
        return {"n_units": 0, "slope_km2_per_m": float("nan")}
    full = o["area_valid_km2"] >= min_valid_frac * o["area_valid_km2"].max()
    u = o[full]
    x, y = u["level_m"].to_numpy(float), u[area_col].to_numpy(float)
    if len(u) < 3:
        return {"n_units": int(len(u)), "n_left_out": int((~full).sum()), "slope_km2_per_m": float("nan")}
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        k = rng.integers(0, len(x), len(x))
        boot.append(np.polyfit(x[k], y[k], 1)[0] if np.ptp(x[k]) > 0 else np.nan)
    lo_ci, hi_ci = np.nanpercentile(boot, [2.5, 97.5])
    return {"n_units": int(len(u)), "n_left_out": int((~full).sum()), "slope_km2_per_m": slope,
            "ci_low": float(lo_ci), "ci_high": float(hi_ci), "level_range_m": [float(x.min()), float(x.max())],
            "sector": sector, "sample": sample, "area_col": area_col}


# --- geometry -------------------------------------------------------------------------------

def load_cover(path, crs: str | None = None):
    """The stage-2 cover (results/study_area/zone_v6_cover_buffered.geojson) as one valid
    geometry: every feature made valid (shapely.make_valid) and united, in lon/lat, or
    in ``crs`` (made valid again after the transform). The committed cover is an invalid
    MultiPolygon in both CRSs (a self-intersection); make_valid changes its area by
    about 3e-6."""
    import json

    import shapely
    from shapely.geometry import shape
    feats = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    g = shapely.union_all([shapely.make_valid(shape(f["geometry"])) for f in feats])
    if crs is None:
        return g
    import pyproj
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shapely.make_valid(shapely.transform(g, lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1]))))


# --- sampling -------------------------------------------------------------------------------

def block_sample(blocks: pd.DataFrame, cover_km2, *, split: str = "test", n_per_sector: int = 2,
                 min_km2: float = 100.0, seed: int = design.SEED) -> pd.DataFrame:
    """The robustness block sample R (docs/ROBUSTNESS_PLAN.md): per sector, in sorted
    order, ``n_per_sector`` blocks of ``split`` with at least ``min_km2`` inside the
    stage-2 cover, drawn by a permutation from one generator seeded once (as
    caspian.blocking); all of them when fewer qualify. Every block-sample Earth Engine
    alternative uses the same R, and the main run restricted to R is its paired
    baseline."""
    b = blocks.assign(cover_km2=np.asarray(cover_km2, float))
    b = b[(b["split"] == split) & (b["cover_km2"] >= min_km2)]
    rng = np.random.default_rng(seed)
    out = []
    for sec in SECTOR_ORDER:
        g = b[b["sector"] == sec].sort_values(["i", "j"])
        if len(g):
            out.append(g.iloc[rng.permutation(len(g))[:n_per_sector]])
    return pd.concat(out, ignore_index=True) if out else b.iloc[:0]


def eecu_hours(scene_km2: float, rate_s_per_scene_km2: float) -> float:
    """EECU-hours = scene-weighted area (scene x km2) x rate (EECU-s per scene per km2) / 3600."""
    return float(scene_km2) * float(rate_s_per_scene_km2) / 3600.0
