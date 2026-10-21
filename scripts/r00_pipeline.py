"""Robustness (docs/ROBUSTNESS_PLAN.md): the downstream steps of a variant, with the
main run's frozen choices, and its paired comparison with the main run. No Earth
Engine. Every step reads the main run's outputs and refuses to run before they exist.

Steps (run in this order; --steps selects):
  points     contour points of the variant's stage-2 tiles (scripts/a6_waterlines.
             tile_points), heights = bin centres at the variant's bin width
             (r00_counts.bin_width_of), static flag from the main rho* (results/levels/
             static_contour_rule.json, D-075; frozen, not re-tuned on the variant)
  dem        TIN DEM of the E2 + E3 points (scripts/a6_dem.py on the variant tag: the
             same 60 km tiling, cover and interpolation-distance band)
  emergence  emergence dates of the variant DEM (scripts/a6_emergence.py)
  validate   the variant-minus-main DEM difference per sector (dem_difference), then,
             paired with the main run (compare_validation): the main prepared ICESat-2
             point table (same points, same dryness selection; D-069), the DEM columns
             from the variant DEM and contours (robustness.dem_columns), blocks at the
             main final size (block_size_check.json), caspian.validation.run on both;
             headline tables and their difference
  hindcast   scripts/a7_hindcast.run on the variant tiles, set up as the main run
             (run.json: products, resampling, tile, level and product offsets; the
             main vertical correction, checked by sha256), shape constant refitted on
             training blocks (or the main constants with --shape main); H1 headline
             tables and their comparison (robustness.h1_headline, compare_h1)

Outputs: data/interim/waterlines/<variant>/, data/processed/dem/<variant>/ (git-
ignored); results/robustness/<check>/<variant>/{validation,hindcast}/ with
*_headline_main.csv, *_headline_alt.csv, *_compare.csv and the variant's own tables.

Usage:
  python scripts/r00_pipeline.py <check> <variant> --main <tag> [--steps points,dem,emergence,validate,hindcast]
      [--sources ATL06,ATL08] [--vertical-correction PATH] [--shape fit|main|none]
      [--sector-labels CSV] [--n-boot N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels, robustness as R, validation
from caspian.config import INTERIM, PROCESSED, RESULTS, ROOT

ROBUST = RESULTS / "robustness"
STEPS = ("points", "dem", "emergence", "validate", "hindcast")
_SCRIPTS: dict = {}


def script(name: str):
    """A pipeline script as a module (cached), e.g. script("a7_hindcast")."""
    if name not in _SCRIPTS:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def out_dir(check: str, variant: str) -> Path:
    return ROBUST / check / variant


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _need(path: Path, what: str) -> Path:
    if not Path(path).exists():
        raise SystemExit(f"{what} not found: {path} (the main run must produce it first)")
    return Path(path)


def main_static_rule(main_tag: str) -> dict:
    r = json.loads(_need(RESULTS / "levels" / "static_contour_rule.json", "the static-contour rule (D-075)")
                   .read_text(encoding="utf-8"))
    if r.get("tag") != main_tag:
        raise SystemExit(f"the static-contour rule was fixed on tag {r.get('tag')!r}, not {main_tag!r}")
    return r


# --- points, DEM, emergence ------------------------------------------------------------------

def points_step(variant: str, main_tag: str, bin_width: float | None = None, static: str = "main") -> Path:
    """Contour points of the variant tiles; ``static``: "main" (the frozen main rho*) or
    "none" (no static flag, so the DEM keeps every point)."""
    a6w = script("a6_waterlines")
    files = sorted((INTERIM / "stage2" / variant).glob("tile_*.npz"))
    if not files:
        raise SystemExit(f"no stage-2 tiles for {variant}")
    width = float(bin_width or script("r00_counts").bin_width_of(variant, INTERIM / "stage2"))
    pts = []
    for f in files:
        with np.load(f) as z:
            spacing = abs(float(z["transform"][0]))
        pts.append(a6w.tile_points(f, spacing=spacing))
    d = pd.concat(pts, ignore_index=True)
    d["z_m"] = levels.bin_center(d["bin"].to_numpy(), width, design.BIN_ORIGIN_M)
    if static == "main":
        d["static"] = d["rho_level"].to_numpy(float) < main_static_rule(main_tag)["rho_star"]
    elif static != "none":
        raise ValueError("static must be 'main' or 'none'")
    out = INTERIM / "waterlines" / variant
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "points.csv.gz", index=False, float_format="%.6f")
    (out / "points.json").write_text(json.dumps({"variant": variant, "main": main_tag, "bin_width_m": width,
                                                 "static": static, "n_points": len(d), "created": _now()}, indent=1),
                                     encoding="utf-8")
    print(variant, "points:", len(d), flush=True)
    return out / "points.csv.gz"


def dem_step(variant: str, epochs: str = "E2,E3") -> None:
    script("a6_dem").main([variant, "--epochs", epochs])


def emergence_step(variant: str) -> None:
    script("a6_emergence").main([variant])


# --- validation ---------------------------------------------------------------------------------

def _products(points: pd.DataFrame) -> dict:
    return {c.split(":", 1)[1]: c for c in points.columns if c.startswith("prod:")}


def final_block_size(main_tag: str) -> float:
    chk = json.loads(_need(RESULTS / "validation" / main_tag / "block_size_check.json", "block_size_check.json")
                     .read_text(encoding="utf-8"))
    if chk.get("final_block_size_m") is None:
        raise SystemExit("the main block size is unresolved (PREREG section 6)")
    return float(chk["final_block_size_m"])


_MAIN_RUNS: dict = {}


def main_validation(main_tag: str, source: str, n_boot: int = design.BOOTSTRAP_N) -> tuple[pd.DataFrame, dict]:
    """(points with blocks, validation.run result) of the main DEM, cached per process."""
    key = (main_tag, source, n_boot)
    if key not in _MAIN_RUNS:
        a6v = script("a6_validate_dem")
        size = final_block_size(main_tag)
        pts = validation.attach_blocks(a6v._load_points(main_tag, source), a6v._blocks(size), size)
        _MAIN_RUNS[key] = (pts, validation.run(pts, _products(pts), n_boot=n_boot))
    return _MAIN_RUNS[key]


LIGHT_COLUMNS = ("lat", "lon", "time", "h_orth", "x", "y", "dem_h", "interp_dist", "slope", "contour_density")


def load_points_light(tag: str, source: str, population: str = "bare", chunk: int = 1_000_000,
                      row_filter=None) -> pd.DataFrame:
    """The rows of scripts/a6_validate_dem._load_points(tag, source, population), read in
    chunks with only the columns validation.run and the robustness variants use (the
    coordinates, time, reference height, the DEM columns and the product heights), so a
    10-million-row table fits in little memory. Needs the D-084 annotation cache
    (<source>_annotations.npz) of the primary, checked against the row count.
    ``row_filter(chunk) -> bool array`` keeps a further subset (e.g. some blocks)."""
    a6v = script("a6_validate_dem")
    path = a6v._cache(tag, source)
    ann = np.load(INTERIM / "validation" / tag / f"{source}_annotations.npz")
    in_band, veg = ann["in_band"], ann["veg_share"]
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in head if c in LIGHT_COLUMNS or c.startswith("prod:")]
    parts, k = [], 0
    for d in pd.read_csv(path, usecols=cols, chunksize=chunk):
        n = len(d)
        ib, vs = in_band[k:k + n], veg[k:k + n]
        if population == "bare":
            keep = ib & (vs < a6v.VEG_CELL_SHARE)
        elif population == "vegetated":
            keep = ib & (vs >= a6v.VEG_CELL_SHARE)
        elif population == "all":
            keep = np.ones(n, bool)
        else:
            raise ValueError(population)
        if row_filter is not None:
            keep = keep & np.asarray(row_filter(d), bool)
        parts.append(d.loc[keep].assign(in_band=ib[keep], veg_share=vs[keep]))
        k += n
    if k != len(in_band):
        raise ValueError(f"{source}: the annotation cache has {len(in_band)} rows, the table {k}")
    out = pd.concat(parts, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], format="ISO8601").to_numpy().astype("datetime64[ns]")
    return out


def main_from_files(main_tag: str, source: str) -> dict:
    """The primary validation result of one source as written by scripts/a6_validate_dem.py
    evaluate (results/validation/<main>/), in the form robustness.validation_headline
    reads, so that the primary is compared as registered and not recomputed."""
    d = _need(RESULTS / "validation" / main_tag, "the primary validation results")
    return {"accuracy": pd.read_csv(d / f"{source}_accuracy.csv"), "coverage": pd.read_csv(d / f"{source}_coverage.csv"),
            "paired_bootstrap": pd.read_csv(d / f"{source}_paired_bootstrap.csv"),
            "constant": json.loads((d / f"{source}_datum_constant.json").read_text(encoding="utf-8"))["constant"]}


def variant_dem_points(variant: str):
    """Callable for compare_validation: the DEM columns from the variant DEM and contours."""
    def f(points: pd.DataFrame, source: str) -> pd.DataFrame:
        cx, cy = script("a6_validate_dem")._contour_points(variant, ["E2", "E3"])
        return R.dem_columns(points, PROCESSED / "dem" / variant, cx, cy)
    return f


def _write_result(d: Path, prefix: str, r: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for k in ("coverage", "accuracy", "paired_bootstrap"):
        if isinstance(r.get(k), pd.DataFrame):
            r[k].to_csv(d / f"{prefix}_{k}.csv", index=False)
    (d / f"{prefix}_fits.json").write_text(json.dumps(
        {k: r[k] for k in ("constant", "product_offsets", "scale_model", "calibration")}, indent=1, default=float),
        encoding="utf-8")


def compare_validation(check: str, variant: str, main_tag: str, *, sources=("ATL06", "ATL08"), alt_points=None,
                       alt_block_size: float | None = None, alt_blocks: pd.DataFrame | None = None, post_blocks=None,
                       alt_run=None, n_boot: int = design.BOOTSTRAP_N, note: str = "", main_from: str = "files") -> dict:
    """Validation of a variant, paired with the main run on the same ICESat-2 points.

    ``alt_points(points, source)`` returns the variant's point table (default:
    unchanged; variant_dem_points for a variant DEM); ``alt_block_size`` another block
    size (check 8; its table, results/study_area/blocks_<km>km.csv, must exist), or
    ``alt_blocks`` a block table of that size (e.g. reassigned splits);
    ``post_blocks(points)`` edits the table after the blocks are attached (split swap,
    Amendment 8 exclusion); ``alt_run(points, products)`` replaces validation.run (e.g.
    robustness.run_fixed for check 6). ATL06 and ATL08 are never pooled (D-069).
    ``main_from``: "files" compares with the primary tables as written (default), or
    "recompute" reruns validation.run on the primary points with this n_boot."""
    out = out_dir(check, variant) / "validation"
    a6v = script("a6_validate_dem")
    rec = {"check": check, "variant": variant, "main": main_tag, "created": _now(), "note": note, "sources": {}}
    for source in sources:
        base = load_points_light(main_tag, source)
        if main_from == "files":
            main_pts, rm = base, main_from_files(main_tag, source)
        else:
            main_pts, rm = main_validation(main_tag, source, n_boot)
        alt = alt_points(base, source) if alt_points else base
        size = float(alt_block_size or final_block_size(main_tag))
        alt = validation.attach_blocks(alt, a6v._blocks(size) if alt_blocks is None else alt_blocks, size)
        if post_blocks is not None:
            alt = post_blocks(alt)
        ra = (alt_run or (lambda p, pr: validation.run(p, pr, n_boot=n_boot)))(alt, _products(alt))
        hm, ha = R.validation_headline(rm), R.validation_headline(ra)
        _write_result(out, f"{source}_alt", ra)
        hm.to_csv(out / f"{source}_headline_main.csv", index=False)
        ha.to_csv(out / f"{source}_headline_alt.csv", index=False)
        R.compare_headlines(hm, ha).to_csv(out / f"{source}_compare.csv", index=False)
        rec["sources"][source] = {"n_main_points": len(main_pts), "n_alt_points": len(alt), "block_size_m": size,
                                  "n_alt_with_dem": int(np.isfinite(alt["dem_h"]).sum()), "n_boot": n_boot,
                                  "main_from": main_from}
        print(check, variant, source, "validated", flush=True)
    prev = out / "run.json"
    if prev.exists():                      # keep the sources of earlier calls (one source at a time)
        old = json.loads(prev.read_text(encoding="utf-8"))
        rec["sources"] = {**old.get("sources", {}), **rec["sources"]}
    (out / "run.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# --- DEM difference --------------------------------------------------------------------------------

DIFF_EDGES = np.round(np.arange(-5.0, 5.0005, 0.001), 3)     # 1 mm bins over +-5 m, plus two open ends


def _hist_quantile(counts, q: float) -> float:
    """Quantile of a 1 mm histogram (bin centres; open end bins at +-5 m)."""
    c = np.cumsum(counts)
    if c[-1] == 0:
        return float("nan")
    k = int(np.searchsorted(c, q * c[-1]))
    centres = np.r_[DIFF_EDGES[0], (DIFF_EDGES[:-1] + DIFF_EDGES[1:]) / 2, DIFF_EDGES[-1]]
    return float(centres[min(k, centres.size - 1)])


def dem_difference(check: str, variant: str, main_tag: str, main_range: tuple | None = None,
                   name: str = "dem_difference.csv") -> pd.DataFrame:
    """Variant minus main DEM height on the cells both have, per sector and overall
    (sectors rasterised by cell centre on each 60 km tile): n, mean, RMS, p05, median,
    p95 (from 1 mm histograms), and the cells only one of the two DEMs has.
    ``main_range`` = (lo, hi): only cells whose main height lies in [lo, hi] (e.g. the
    two-limb band of check 10); the table is written as ``name``."""
    import rasterio
    import shapely
    from rasterio.features import rasterize
    from shapely.geometry import mapping
    from caspian.config import CRS_EQUAL_AREA
    from caspian.geography import SECTOR_ORDER, sectors
    import pyproj
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    polys = [shapely.transform(sectors()[n][1].segmentize(0.01),
                               lambda xy: np.column_stack(fwd.transform(xy[:, 0], xy[:, 1]))) for n in SECTOR_ORDER]
    names = ["outside", *SECTOR_ORDER]
    hist = {n: np.zeros(DIFF_EDGES.size + 1, np.int64) for n in names}
    acc = {n: {"n": 0, "sum": 0.0, "sum2": 0.0, "only_variant": 0, "only_main": 0} for n in names}
    for f in sorted((PROCESSED / "dem" / variant).glob("dem_*.tif")):
        g = PROCESSED / "dem" / main_tag / f.name
        if not g.exists():
            continue
        with rasterio.open(f) as a, rasterio.open(g) as b:
            if a.transform != b.transform or a.shape != b.shape:
                raise ValueError(f"{f.name}: the variant and main tiles are not on one grid")
            za, zb = a.read(1).astype(float), b.read(1).astype(float)
            lab = rasterize([(mapping(p), k) for k, p in enumerate(polys, 1)], out_shape=a.shape,
                            transform=a.transform, fill=0, dtype="uint8")
        fa, fb = np.isfinite(za), np.isfinite(zb)
        if main_range is not None:
            with np.errstate(invalid="ignore"):
                inr = (zb >= main_range[0]) & (zb <= main_range[1])
            fa, fb = fa & inr, fb & inr
        d = za - zb
        for k, n in enumerate(names):
            m = lab == k
            both = m & fa & fb
            v = d[both]
            hist[n] += np.bincount(np.searchsorted(DIFF_EDGES, v, side="right"), minlength=DIFF_EDGES.size + 1)
            acc[n]["n"] += int(v.size)
            acc[n]["sum"] += float(v.sum())
            acc[n]["sum2"] += float((v ** 2).sum())
            acc[n]["only_variant"] += int((m & fa & ~fb).sum())
            acc[n]["only_main"] += int((m & ~fa & fb).sum())
    rows = []
    for n in names + ["overall"]:
        h = sum(hist.values()) if n == "overall" else hist[n]
        a = ({k: sum(acc[x][k] for x in names) for k in acc[names[0]]} if n == "overall" else acc[n])
        rows.append({"sector": n, "n_cells": a["n"], "mean_m": a["sum"] / a["n"] if a["n"] else float("nan"),
                     "rms_m": float(np.sqrt(a["sum2"] / a["n"])) if a["n"] else float("nan"),
                     "p05_m": _hist_quantile(h, 0.05), "median_m": _hist_quantile(h, 0.5),
                     "p95_m": _hist_quantile(h, 0.95), "cells_only_variant": a["only_variant"],
                     "cells_only_main": a["only_main"]})
    out = out_dir(check, variant)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / name, index=False)
    return t


# --- hindcast -----------------------------------------------------------------------------------

@contextmanager
def bin_levels(a7, fn):
    """Temporarily replace scripts/a7_hindcast._bin_level (the level of a bin index),
    e.g. for 0.20 m bins or a per-bin level shift."""
    old = a7._bin_level
    a7._bin_level = fn
    try:
        yield
    finally:
        a7._bin_level = old


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hindcast_step(check: str, variant: str, main_tag: str, *, stage2_tag: str | None = None,
                  vertical_correction: str | None = None, correction_is_variant: bool = False,
                  resampling: str | None = None, shape: str = "fit", bin_level=None, blocks_csv=None,
                  block_m: float | None = None, sector_labels: str | None = None, note: str = "",
                  drop_epochs=(), envelope: bool = True) -> Path:
    """scripts/a7_hindcast.run for a variant, set up as the main run (see the module
    docstring), into results/robustness/<check>/<variant>/hindcast/, and the H1
    comparison. ``stage2_tag`` defaults to the variant (the main tag for variants of the
    products or the correction). ``bin_level(b)`` overrides the level of bin b (bin
    width or level-shift variants). ``drop_epochs``: epochs left out of every unit,
    product side and noise floor alike (E1 for the E2 + E3 check). ``envelope``: the
    record-low water envelope of the primary (D-082), from the same stage-2 tag.
    ``shape`` "loeo:<epoch>" takes the main run's leave-one-epoch-out shape constants
    without that epoch, which equal a refit on the remaining epochs' units."""
    from caspian import h1, products
    from caspian.geography import DOMAIN
    a7 = script("a7_hindcast")
    main_dir = RESULTS / "hindcast" / main_tag
    run_rec = json.loads(_need(main_dir / "run.json", "the main hindcast run").read_text(encoding="utf-8"))
    names = list(run_rec["products"])
    resampling = resampling or run_rec["resampling"]
    tile = int(run_rec["tile"])
    w, s, e, n = DOMAIN.bounds
    m = a7.MARGIN_DEG
    transform, shape_ = a7.grid((w - m, s - m, e + m, n + m))
    static = a7.production_static(transform, shape_, tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    blocks = pd.read_csv(blocks_csv or a7.BLOCKS)
    bm = float(block_m or design.BLOCK_SIZE_M)
    fit_region = static["domain"] & a7.train_mask(blocks, transform, shape_, bm)
    main_corr = run_rec["vertical_correction"]
    corr = corr_rec = None
    if vertical_correction:
        if isinstance(main_corr, dict) and not correction_is_variant and _sha(vertical_correction) != main_corr["sha256"]:
            raise SystemExit("--vertical-correction is not the main run's correction file (sha256 differs)")
        corr, corr_rec = products.correction(vertical_correction, transform, shape_)
    elif isinstance(main_corr, dict) and not correction_is_variant:
        raise SystemExit(f"the main run used the correction {main_corr['file']}: pass it with --vertical-correction")
    heights, flood_maps = {}, {}
    for p in names:
        h = products.heights(p, transform, shape_, resampling)
        heights[p] = h if corr is None else products.Corrected(h, corr)
        flood_maps[p] = a7.flood_map_cached(p, heights[p], static["sea_seed"],
                                            {"heights": Path(h.filename).name,
                                             "correction": corr_rec["sha256"] if corr_rec else None}, tile)
    if shape == "fit":
        shape_offsets = "fit"
    elif shape == "main":
        so = json.loads(_need(main_dir / "shape_offsets.json", "the main shape fit").read_text(encoding="utf-8"))
        shape_offsets = {p: {"offset_m": v["offset_m"], "source": f"main run {main_tag} (frozen)"}
                         for p, v in so["used_in_this_run"].items()}
    elif shape.startswith("loeo:"):
        ep = shape.split(":", 1)[1]
        so = json.loads(_need(main_dir / "shape_offsets.json", "the main shape fit").read_text(encoding="utf-8"))
        shape_offsets = {p: {"offset_m": v["leave_one_epoch_out"][ep]["offset_m"],
                             "source": f"main run {main_tag}, leave-one-epoch-out refit without {ep}"}
                         for p, v in so["products"].items()}
    elif shape == "none":
        shape_offsets = None
    else:
        raise ValueError("shape must be fit, main, loeo:<epoch> or none")
    po = run_rec["product_offsets"] if isinstance(run_rec["product_offsets"], dict) else {}
    out = out_dir(check, variant) / "hindcast"
    meta = {"robustness_check": check, "variant": variant, "main": main_tag, "note": note, "resampling": resampling,
            "tile": tile, "vertical_correction": corr_rec or "none", "block_m": bm,
            "blocks": str(blocks_csv or a7.BLOCKS), "shape": shape, "drop_epochs": list(drop_epochs)}
    stage2 = INTERIM / "stage2" / (stage2_tag or variant)
    a7.ENVELOPE = a7.record_low_water(a7.stage2_index(stage2), transform, shape_) if envelope else None
    meta["envelope"] = ({"rule": "D-082, as the primary", "cells": int(a7.ENVELOPE.sum())} if envelope
                        else "none (sensitivity)")
    orig_index = a7.stage2_index
    if drop_epochs:
        def filtered(src, _orig=orig_index, _drop=set(drop_epochs)):
            u = {b: {e: v for e, v in eps.items() if e not in _drop} for b, eps in _orig(src).items()}
            return {b: eps for b, eps in u.items() if eps}
        a7.stage2_index = filtered
    if bin_level is None:
        width = script("r00_counts").bin_width_of(stage2_tag or variant, INTERIM / "stage2")
        bin_level = lambda b: float(levels.bin_center(b, width, design.BIN_ORIGIN_M))  # noqa: E731
    try:
        with bin_levels(a7, bin_level):
            a7.run(stage2, out, static, heights, transform,
                   level_datum_offset=float(run_rec["level_datum"]["offset_m"]),
                   level_datum_source=run_rec["level_datum"]["source"], product_offsets=po,
                   shape_offsets=shape_offsets, fit_region=fit_region, flood_maps=flood_maps, blocks=blocks,
                   block_m=bm, tile=tile, meta=meta)
    finally:
        a7.stage2_index = orig_index
        a7.ENVELOPE = None
    hm, ha = R.h1_headline(main_dir), R.h1_headline(out)
    R.compare_h1(hm, ha).to_csv(out / "h1_compare.csv", index=False)
    hm["area"].merge(ha["area"], on=["product", "variant"], suffixes=("_main", "_alt")).to_csv(
        out / "area_compare.csv", index=False)
    if sector_labels:
        # the primary block table of the first run carries the D-089 sentinel; compare with the
        # fixed rerun (results/hindcast/<main>_fix) when it exists, else report the variant alone
        lab = pd.read_csv(sector_labels)
        fixed = RESULTS / "hindcast" / f"{main_tag}_fix"
        if not (fixed / "block_h1.csv").exists() and (main_dir / "d089_rerun_comparison.json").exists():
            fixed = main_dir                   # the D-089 rerun replaced the primary block table
        runs = [("alt", out)] + ([("main_fixed", fixed)] if (fixed / "block_h1.csv").exists() else [])
        bt = [h1.block_test(pd.read_csv(d / "block_h1.csv"), lab).assign(run=name) for name, d in runs]
        pd.concat(bt, ignore_index=True).to_csv(out / "block_test_compare.csv", index=False)
    print(check, variant, "hindcast compared ->", out, flush=True)
    return out


# --- driver -------------------------------------------------------------------------------------

def run_steps(check: str, variant: str, main_tag: str, steps=STEPS, *, sources=("ATL06", "ATL08"),
              vertical_correction=None, shape="fit", sector_labels=None, n_boot=design.BOOTSTRAP_N,
              bin_width=None, static="main", note="") -> None:
    if "points" in steps:
        points_step(variant, main_tag, bin_width, static)
    if "dem" in steps:
        dem_step(variant)
    if "emergence" in steps:
        emergence_step(variant)
    if "validate" in steps:
        dem_difference(check, variant, main_tag)
        compare_validation(check, variant, main_tag, sources=sources, alt_points=variant_dem_points(variant),
                           n_boot=n_boot, note=note)
    if "hindcast" in steps:
        hindcast_step(check, variant, main_tag, vertical_correction=vertical_correction, shape=shape,
                      sector_labels=sector_labels, note=note)


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--main", required=True, help="the main stage-2 / DEM tag")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--sources", default="ATL06,ATL08")
    ap.add_argument("--vertical-correction", default=None)
    ap.add_argument("--shape", default="fit", help="fit | main | loeo:<epoch> | none")
    ap.add_argument("--sector-labels", default=None)
    ap.add_argument("--n-boot", type=int, default=design.BOOTSTRAP_N)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("check")
    ap.add_argument("variant")
    add_common(ap)
    a = ap.parse_args(argv)
    run_steps(a.check, a.variant, a.main, a.steps.split(","), sources=a.sources.split(","),
              vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
              n_boot=a.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
