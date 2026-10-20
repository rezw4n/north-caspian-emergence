"""A6 validation: the waterline DEM against ICESat-2 (PREREG sections 2 and 5-8;
Amendments 2 and 5; caspian.validation).

Three stages, run in order; each refuses to run before the one it needs.
  prepare    ICESat-2 granules -> quality steps (each counted) -> geoid (EGM2008) -> DEM
             height, interpolation distance, slope and contour density, emergence dates
             and product heights at each point -> dry-at-acquisition test (PREREG
             section 2). Writes the point table and the filter cascade.
  blocksize  PREREG section 6: semivariogram of DEM - ICESat-2 in the training blocks of
             the initial 25 km assignment, per source; the final block size is the largest
             over the sources. Reads no calibration- or test-block value.
  evaluate   Needs block_size_check.json and the block table at the final size (made by
             scripts/a6_blocks.py <size>). Fits on training blocks (datum constant c,
             product offsets, scale model β), q̂ on calibration blocks, then accuracy
             metrics, the paired block bootstrap and coverage on test blocks
             (caspian.validation.run). ATL06 and ATL08 are evaluated separately. The H2
             decision rule is not applied here (validation.h2_decision, at G3).

Inputs
  data/interim/icesat2/<ATL06|ATL08>/<granule>.csv.gz (scripts/a6_icesat2.py): only granules
      recorded in results/provenance/icesat2_<product>.json, with the recorded row count
      (a file still being written is skipped and counted).
  data/processed/dem/<tag>/dem_<i>_<j>.tif (scripts/a6_dem.py; band 1 height on the DAHITI
      reference, band 2 interpolation distance).
  data/interim/waterlines/<tag>/points.csv.gz: the DEM's contour points (same epochs and
      static-contour rule as scripts/a6_dem.py), for the contour density.
  --emergence DIR: emergence-date tiles emergence_<i>_<j>.tif on the DEM tiling; band 1 is the
      date the pixel became permanently dry, in days since 1970-01-01, NaN where it never
      emerged or was never sea (the A6 emergence map; interface proposed here).
  caspian.products: every product in PRODUCTS (or --products), read with load_native and
      interpolated bilinearly at the points with sample, as distributed (no offset);
      skipped with --no-products.
  results/study_area/blocks_<size>km.csv (scripts/a6_blocks.py).

Outputs
  data/interim/validation/<tag>/<source>_points.csv.gz (git-ignored)
  results/validation/<tag>/ (a subfolder sensitivity_<size>m for a --block-size run):
      prepare.json, <source>_filter_cascade.csv; block_size_check.json,
      <source>_semivariogram.csv; per source from evaluate: _datum_constant.json,
      _product_offsets.json, _scale_model.json, _calibration.json, _counts.json,
      _coverage.csv, _accuracy.csv, _accuracy_blocks.csv, _paired_bootstrap.csv

Usage:
  python scripts/a6_validate_dem.py <tag> prepare --emergence DIR [--sources ATL06,ATL08]
         [--products a,b | --no-products] [--epochs E2,E3]
  python scripts/a6_validate_dem.py <tag> blocksize [--sources ATL06,ATL08]
  python scripts/a6_validate_dem.py <tag> evaluate [--sources ATL06,ATL08] [--block-size M]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj

from caspian import design, validation
from caspian.config import CRS_EQUAL_AREA, INTERIM, PROCESSED, RESULTS

SOURCES = ("ATL06", "ATL08")
CHUNK = 250_000          # points per emergence-window read


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(obj, indent=1, default=conv)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(obj), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _out(tag: str) -> Path:
    return RESULTS / "validation" / tag


def _cache(tag: str, source: str) -> Path:
    return INTERIM / "validation" / tag / f"{source}_points.csv.gz"


def _recorded_granules(product: str) -> dict[str, int]:
    p = RESULTS / "provenance" / f"icesat2_{product.lower()}.json"
    if not p.exists():
        return {}
    return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8"))["granules"].items() if int(v) > 0}


def load_icesat2(product: str, geoid_source: str = "eigen6c4") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every recorded granule of one product through validation.prepare_icesat2. Returns
    (kept points, summed filter cascade, granule counts)."""
    folder = INTERIM / "icesat2" / product
    rec = _recorded_granules(product)
    atl06 = {}
    if product == "ATL08":
        atl06 = {validation.granule_key(k): INTERIM / "icesat2" / "ATL06" / f"{k}.csv.gz"
                 for k in _recorded_granules("ATL06")}
    frames, cascades = [], []
    # N(EIGEN-6C4) for both products (D-078); "atl06" keeps the D-069 chain (ATL06 geoid_h)
    geoid = validation.EigenGeoid() if geoid_source == "eigen6c4" else None
    counts = {"recorded_with_points": len(rec), "read": 0, "missing_file": 0, "incomplete_file": 0,
              "atl08_without_atl06": 0}
    for name, n in sorted(rec.items()):
        path = folder / f"{name}.csv.gz"
        if not path.exists():
            counts["missing_file"] += 1
            continue
        try:
            d = validation.read_icesat2_csv(path, product)
        except (EOFError, OSError, ValueError, UnicodeDecodeError, zlib.error, pd.errors.ParserError):
            counts["incomplete_file"] += 1
            continue
        if len(d) != n:
            counts["incomplete_file"] += 1
            continue
        ref = None
        if geoid is None and product == "ATL08":
            p6 = atl06.get(validation.granule_key(name))
            if p6 is not None and p6.exists():
                ref = pd.read_csv(p6, usecols=["time", "beam", "geoid_h"])
                ref["time"] = pd.to_datetime(ref["time"], format="ISO8601").to_numpy().astype("datetime64[ns]")
                ref["beam"] = ref["beam"].astype(str)
            else:
                counts["atl08_without_atl06"] += 1
        kept, casc = validation.prepare_icesat2(d, product, ref, geoid=geoid)
        kept["granule"] = name
        frames.append(kept)
        cascades.append(casc.set_index("step"))
        counts["read"] += 1
    if not cascades:
        raise SystemExit(f"{product}: no complete granule file found under {folder}")
    casc = sum(cascades[1:], cascades[0]).reset_index()
    return pd.concat(frames, ignore_index=True), casc, counts


def _contour_points(tag: str, epochs) -> tuple[np.ndarray, np.ndarray]:
    """The DEM's contour points, selected as in scripts/a6_dem.py."""
    p = pd.read_csv(INTERIM / "waterlines" / tag / "points.csv.gz",
                    usecols=lambda c: c in {"x", "y", "epoch", "static"})
    p = p[p["epoch"].isin(epochs)]
    if "static" in p.columns:
        p = p[~p["static"].astype(bool)]
    return p["x"].to_numpy(float), p["y"].to_numpy(float)


def _product_sampler(names_arg: str | None, no_products: bool):
    """(names, sampler) with sampler(name, lon, lat) -> heights as distributed
    (validation.ProductSampler), from caspian.products: each product's native grid
    (``load_native``), interpolated bilinearly at the points (``sample``, the module's
    primary method). Default names: every product in ``products.PRODUCTS``."""
    if no_products:
        return [], None
    try:
        from caspian import products
    except ImportError:
        raise SystemExit("caspian.products is not available: pass --no-products to validate the DEM alone "
                         "(the H2 comparison needs the products)")
    names = names_arg.split(",") if names_arg else list(products.PRODUCTS)
    unknown = [n for n in names if n not in products.PRODUCTS]
    if unknown:
        raise SystemExit(f"unknown products {unknown}; caspian.products.PRODUCTS has {list(products.PRODUCTS)}")
    natives = {}

    def sampler(name, lon, lat):
        if name not in natives:
            natives[name] = products.load_native(name)
        return products.sample(natives[name], lon, lat)

    return names, sampler


def prepare(a) -> int:
    dem_dir = PROCESSED / "dem" / a.tag
    if not any(dem_dir.glob("dem_*.tif")):
        raise SystemExit(f"no DEM tiles in {dem_dir}")
    if a.emergence is None or not any(Path(a.emergence).glob("emergence_*.tif")):
        raise SystemExit("PREREG section 2 needs the emergence date of each pixel (the A6 emergence map): "
                         "pass --emergence DIR with emergence_<i>_<j>.tif tiles")
    names, sample = _product_sampler(a.products, a.no_products)
    dem = validation.TileSet(dem_dir, "dem")
    em = validation.TileSet(Path(a.emergence), "emergence")
    epochs = a.epochs.split(",")
    cx, cy = _contour_points(a.tag, epochs)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    meta = {"created": _now(), "geoid": a.geoid, "dem_tag": a.tag, "dem_dir": str(dem_dir), "emergence_dir": str(a.emergence),
            "epochs": epochs, "products": names,
            "product_interpolation": "bilinear, as distributed (caspian.products.sample)" if names else None,
            "emergence_lag_days": design.EMERGENCE_LAG_DAYS,
            "gross_error_m": validation.GROSS_ERROR_M, "geoid_max_gap_s": validation.GEOID_MAX_GAP_S,
            "contour_radius_m": validation.CONTOUR_RADIUS_M, "sources": {}}
    for source in a.sources:
        d, casc, counts = load_icesat2(source, a.geoid)
        x, y = fwd.transform(d["lon"].to_numpy(float), d["lat"].to_numpy(float))
        r = validation.DRY_RADIUS_PX[source]
        t = d["time"].to_numpy()
        dry = np.zeros(len(d), bool)
        for k in range(0, len(d), CHUNK):      # (2r + 1)² emergence dates per point, in chunks
            s = slice(k, k + CHUNK)
            dry[s] = validation.dry_at_acquisition(t[s], em.sample_window(x[s], y[s], 1, r).reshape(len(t[s]), -1))
        casc = pd.concat([casc, pd.DataFrame([{"step": "dry_at_acquisition", "n_remaining": int(dry.sum()),
                                               "n_fail_alone": int((~dry).sum())}])], ignore_index=True)
        d = d.loc[dry].assign(x=x[dry], y=y[dry]).reset_index(drop=True)
        d["dem_h"] = dem.sample(d["x"], d["y"], 1)
        d["interp_dist"] = dem.sample(d["x"], d["y"], 2)
        d["slope"] = dem.slope(d["x"], d["y"])
        d["contour_density"] = validation.contour_density(d["x"], d["y"], cx, cy)
        for name in names:
            d[f"prod:{name}"] = np.asarray(sample(name, d["lon"].to_numpy(float), d["lat"].to_numpy(float)), float)
        path = _cache(a.tag, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(path, index=False, float_format="%.9g")
        _out(a.tag).mkdir(parents=True, exist_ok=True)
        casc.to_csv(_out(a.tag) / f"{source}_filter_cascade.csv", index=False)
        meta["sources"][source] = {"granules": counts, "dry_radius_px": r, "n_points": len(d),
                                   "n_with_dem": int(np.isfinite(d["dem_h"]).sum())}
        print(source, counts, "->", len(d), "dry points,", meta["sources"][source]["n_with_dem"], "with a DEM height",
              flush=True)
    _write(_out(a.tag) / "prepare.json", meta)
    return 0


def _load_points(tag: str, source: str) -> pd.DataFrame:
    p = _cache(tag, source)
    if not p.exists():
        raise SystemExit(f"no point table {p}: run the prepare stage first")
    d = pd.read_csv(p)
    d["time"] = pd.to_datetime(d["time"], format="ISO8601").to_numpy().astype("datetime64[ns]")
    return d


def _blocks(size_m: float) -> pd.DataFrame:
    p = RESULTS / "study_area" / f"blocks_{int(size_m / 1000)}km.csv"
    if not p.exists():
        raise SystemExit(f"no block table {p}: run `python scripts/a6_blocks.py {size_m:g}` (same code and seed) "
                         "and log the size in docs/DECISIONS.md")
    return pd.read_csv(p)


def blocksize(a) -> int:
    base = float(design.BLOCK_SIZE_M)
    blocks = _blocks(base)
    _out(a.tag).mkdir(parents=True, exist_ok=True)
    recs = {}
    for source in a.sources:
        pts = validation.attach_blocks(_load_points(a.tag, source), blocks, base)
        rec, vario = validation.block_size_check(pts, base=base)
        vario.to_csv(_out(a.tag) / f"{source}_semivariogram.csv", index=False)
        recs[source] = rec
        print(source, "practical range", round(rec["fit"]["practical_range"]), "m ->",
              rec["final_block_size_m"], "m", "" if rec["range_within_max_lag"] else "(beyond the largest lag)",
              flush=True)
    resolved = all(r["range_within_max_lag"] for r in recs.values())
    final = max(r["final_block_size_m"] for r in recs.values()) if resolved else None
    _write(_out(a.tag) / "block_size_check.json",
           {"created": _now(), "dem_tag": a.tag, "rule": "PREREG section 6; largest size over the sources",
            "status": "resolved" if resolved else "unresolved: a fitted range lies beyond the largest lag",
            "base_block_size_m": base, "final_block_size_m": final, "sources": recs})
    if not resolved:
        print("block size unresolved: a fitted practical range lies beyond the largest lag fitted. The lead decides "
              "(docs/DECISIONS.md) before any test-block evaluation; evaluate refuses until then.")
        return 3
    print("final block size", final, "m" + ("" if final == base else
          f": run `python scripts/a6_blocks.py {final:g}` and log it in docs/DECISIONS.md before evaluate"))
    return 0


def evaluate(a) -> int:
    chk_path = _out(a.tag) / "block_size_check.json"
    if not chk_path.exists():
        raise SystemExit("PREREG section 6: the block size is checked before any test-block evaluation; "
                         "run the blocksize stage first")
    chk = json.loads(chk_path.read_text(encoding="utf-8"))
    missing = [s for s in a.sources if s not in chk["sources"]]
    if missing:
        raise SystemExit(f"block size not checked with {missing}; rerun the blocksize stage with these sources")
    if chk["final_block_size_m"] is None:
        raise SystemExit(f"block size check {chk['status']}; a decision is needed before any test-block evaluation")
    final = float(chk["final_block_size_m"])
    size = final if a.block_size is None else float(a.block_size)
    out = _out(a.tag) if size == final else _out(a.tag) / f"sensitivity_{int(size)}m"
    blocks = _blocks(size)
    out.mkdir(parents=True, exist_ok=True)
    for source in a.sources:
        pts = validation.attach_blocks(_load_points(a.tag, source), blocks, size)
        products = {c.split(":", 1)[1]: c for c in pts.columns if c.startswith("prod:")}
        r = validation.run(pts, products)
        head = {"created": _now(), "dem_tag": a.tag, "source": source, "block_size_m": size,
                "final_block_size_m": final, "sensitivity": size != final}
        for key in ("constant", "product_offsets", "scale_model", "calibration"):
            name = {"constant": "datum_constant"}.get(key, key)
            _write(out / f"{source}_{name}.json", {**head, key: r[key]})
        sp = pts["split"]
        _write(out / f"{source}_counts.json", {
            **head, "n_points": len(pts), "n_outside_block_table": int(sp.isna().sum()),
            "by_split_sector": pts.groupby([sp.fillna("none"), pts["sector"].fillna("none")]).size()
                                  .rename("n").reset_index().to_dict(orient="records"),
            "n_blocks_with_points": pts.loc[sp.notna()].groupby(sp)["block"].nunique().to_dict()})
        for key in ("coverage", "accuracy", "accuracy_blocks", "paired_bootstrap"):
            r[key].assign(source=source, block_size_m=size).to_csv(out / f"{source}_{key}.csv", index=False)
        print(source, "evaluated ->", out, flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("stage", choices=["prepare", "blocksize", "evaluate"])
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--emergence", default=None)
    ap.add_argument("--products", default=None)
    ap.add_argument("--no-products", action="store_true")
    ap.add_argument("--epochs", default="E2,E3")
    ap.add_argument("--geoid", choices=["eigen6c4", "atl06"], default="eigen6c4",
                    help="N for ICESat-2 heights: EIGEN-6C4 grid (D-078) or the ATL06 geoid_h chain (D-069)")
    ap.add_argument("--block-size", type=float, default=None, help="sensitivity run (PREREG section 6)")
    a = ap.parse_args(argv)
    a.sources = a.sources.split(",")
    bad = [s for s in a.sources if s not in SOURCES]
    if bad:
        ap.error(f"unknown sources {bad}")
    return {"prepare": prepare, "blocksize": blocksize, "evaluate": evaluate}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
