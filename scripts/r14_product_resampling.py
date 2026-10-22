"""Robustness: product resampling by nearest neighbour instead of bilinear. No
Earth Engine.

- H1: the hindcast with every product resampled by nearest neighbour to the 60 m grid
  (caspian.products.heights(..., "nearest")), set up as the main run otherwise, shape
  constant refitted (scripts/r00_pipeline.py hindcast_step, stage-2 tiles of the main).
- H2: the product heights at the ICESat-2 points by nearest neighbour
  (caspian.products.sample(..., "nearest")), DEM unchanged, paired comparison.

Usage:
  python scripts/r14_product_resampling.py --main <tag> --vertical-correction PATH [--steps validate,hindcast]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np

CHECK = "r14_product_resampling"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def nearest_products(points, source):
    from caspian import products
    out = points.copy()
    for c in [c for c in points.columns if c.startswith("prod:")]:
        nat = products.load_native(c.split(":", 1)[1])
        out[c] = np.asarray(products.sample(nat, points["lon"].to_numpy(float), points["lat"].to_numpy(float),
                                            "nearest"), float)
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    steps = a.steps.split(",")
    if "validate" in steps:
        P.compare_validation(CHECK, "nearest", a.main, sources=a.sources.split(","), alt_points=nearest_products,
                             n_boot=a.n_boot, note="products sampled by nearest neighbour")
    if "hindcast" in steps:
        P.hindcast_step(CHECK, "nearest", a.main, stage2_tag=a.main, vertical_correction=a.vertical_correction,
                        resampling="nearest", shape=a.shape, sector_labels=a.sector_labels,
                        note="products resampled by nearest neighbour")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
