"""Robustness check 7: interpolation method and bin width. No Earth Engine.

Commands
  gp         The comparison method of PREREG section 4: ordinary kriging, i.e. a Gaussian
             process with an unknown constant mean and an exponential covariance
             (robustness.ordinary_kriging), of the same E2 + E3 non-static contour points.
             - covariance: fitted to the empirical semivariogram of the contour heights in
               the training blocks of the final assignment (robustness.fit_covariance);
             - nugget: the contour 1-sigma of PREREG section 4 squared (median level
               1-sigma, and the bin half-width as a uniform error, width^2 / 12), so the
               predictor smooths within the contours' own error;
             - neighbourhood: 32 nearest points, at most 8 per quadrant, after thinning
               each (epoch, bin) contour to one point per --thin-m cell;
             - support: the cells where the main TIN has a height, so the comparison is
               paired on the same cells and ICESat-2 points.
             Tiles: data/processed/dem/<main>__gp/dem_<i>_<j>.tif (band 1 height, band 2
             the main interpolation distance, band 3 the kriging standard deviation); then
             emergence and the paired validation (scripts/r00_pipeline.py).
  binwidth   0.20 m bins from the 0.10 m counts (r00_counts --bin-factor 2), or 0.05 m
             bins from a main export split by half-bin (--split, r00_counts
             --fine-from-halves); then every downstream step including the hindcast.
             0.05 m cannot be made from unsplit counts: a 0.10 m count sums its
             acquisitions, and which half of the bin each came from is not kept.

Usage:
  python scripts/r07_interpolation.py gp --main <tag> [--thin-m 90] [--steps dem,emergence,validate]
  python scripts/r07_interpolation.py binwidth --main <tag> --width 0.20
  python scripts/r07_interpolation.py binwidth --main <tag> --width 0.05 --split <tag>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import blocking, design, robustness as R
from caspian.config import INTERIM, PROCESSED, RESULTS

CHECK = "r07_interpolation"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def thinned(points: pd.DataFrame, spacing: float) -> pd.DataFrame:
    """One point per ``spacing`` cell within each (epoch, bin) contour set."""
    keep = []
    for _, g in points.groupby(["epoch", "bin"]):
        keep.append(g.iloc[R.thin_points(g[["x", "y"]].to_numpy(float), spacing)])
    return pd.concat(keep, ignore_index=True)


def gp_tiles(main_tag: str, variant: str, pts: pd.DataFrame, cov: dict, nugget: float, k: int = 32,
             per_quadrant: int = 8, chunk: int = 20_000) -> int:
    import rasterio
    src, out = PROCESSED / "dem" / main_tag, PROCESSED / "dem" / variant
    out.mkdir(parents=True, exist_ok=True)
    xy, z = pts[["x", "y"]].to_numpy(float), pts["z_m"].to_numpy(float)
    n = 0
    for f in sorted(src.glob("dem_*.tif")):
        with rasterio.open(f) as r:
            h, dist, prof, tr = r.read(1), r.read(2), r.profile, r.transform
        ok = np.isfinite(h)
        rows, cols = np.nonzero(ok)
        qx, qy = tr.c + tr.a * (cols + 0.5), tr.f + tr.e * (rows + 0.5)
        zk, var = R.ordinary_kriging(xy, z, np.column_stack([qx, qy]), sill=cov["sill"], a=cov["a"], nugget=nugget,
                                     k=k, per_quadrant=per_quadrant, chunk=chunk)
        g, sd = np.full(h.shape, np.nan, np.float32), np.full(h.shape, np.nan, np.float32)
        g[rows, cols], sd[rows, cols] = zk, np.sqrt(np.maximum(var, 0))
        prof.update(count=3, dtype="float32", nodata=np.nan)
        with rasterio.open(out / f.name, "w", **prof) as w:
            w.write(g, 1)
            w.write(dist.astype(np.float32), 2)
            w.write(sd, 3)
            for b, d in enumerate(("height_m_dahiti", "interpolation_distance_m", "kriging_sd_m"), 1):
                w.set_band_description(b, d)
        n += 1
        print(f"{f.name}: {int(ok.sum())} cells kriged", flush=True)
    return n


def gp(a, P) -> int:
    variant = f"{a.main}__gp"
    p = pd.read_csv(INTERIM / "waterlines" / a.main / "points.csv.gz")
    p = p[p["epoch"].isin(["E2", "E3"])]
    if "static" in p.columns:
        p = p[~p["static"].astype(bool)]
    size = P.final_block_size(a.main)
    blocks = pd.read_csv(RESULTS / "study_area" / f"blocks_{int(size / 1000)}km.csv")
    split = blocking.split_of_points(p["x"], p["y"], blocks, blocks["split"], size)
    t = thinned(p, a.thin_m)
    tr = thinned(p[split == "train"], a.thin_m)
    cov = R.fit_covariance(tr[["x", "y"]].to_numpy(float), tr["z_m"].to_numpy(float))
    sig = float(np.nanmedian(p["level_sigma_p50_m"])) if "level_sigma_p50_m" in p.columns else 0.05
    nugget = sig ** 2 + design.BIN_WIDTH_M ** 2 / 12
    rec = {"main": a.main, "covariance": cov, "nugget_m2": nugget, "level_sigma_m": sig, "thin_m": a.thin_m,
           "n_points_kriging": len(t), "n_points_train_fit": len(tr), "k": 32, "per_quadrant": 8,
           "block_size_m": size, "support": "cells where the main TIN has a height"}
    out = RESULTS / "robustness" / CHECK / variant
    out.mkdir(parents=True, exist_ok=True)
    (out / "gp_model.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    steps = a.steps.split(",")
    if "dem" in steps:
        gp_tiles(a.main, variant, t, cov, nugget)
        d = INTERIM / "waterlines" / variant
        d.mkdir(parents=True, exist_ok=True)
        p.to_csv(d / "points.csv.gz", index=False, float_format="%.6f")   # the same contours (density)
    P.run_steps(CHECK, variant, a.main, [s for s in steps if s in ("emergence", "validate")],
                sources=a.sources.split(","), n_boot=a.n_boot, note="ordinary kriging")
    return 0


def binwidth(a, P) -> int:
    C = P.script("r00_counts")
    if abs(a.width - 0.20) < 1e-9:
        variant = f"{a.main}__bw020"
        C.build(variant, [a.main], bin_factor=2)
    elif abs(a.width - 0.05) < 1e-9:
        if not a.split:
            raise SystemExit("0.05 m bins need a main export split by half-bin (--split); see the module docstring")
        variant = f"{a.main}__bw005"
        C.build(variant, a.split.split(","), fine_from_halves=True)
    else:
        raise SystemExit("width must be 0.20 or 0.05 (PREREG section 4 sensitivity)")
    P.run_steps(CHECK, variant, a.main, a.steps.split(","), sources=a.sources.split(","),
                vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
                n_boot=a.n_boot, note=f"bin width {a.width} m")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["gp", "binwidth"])
    ap.add_argument("--thin-m", type=float, default=90.0)
    ap.add_argument("--width", type=float)
    ap.add_argument("--split")
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    return gp(a, P) if a.cmd == "gp" else binwidth(a, P)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
