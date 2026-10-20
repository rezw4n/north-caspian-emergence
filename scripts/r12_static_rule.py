"""Robustness (D-034, D-075; PREREG Amendment 8): the DEM without the static-contour
exclusion. No Earth Engine.

The main contour table (data/interim/waterlines/<main>/points.csv.gz, with the
`static` column of scripts/a6_static_rule.py) is copied without that column to the tag
<main>__nostatic, so that scripts/a6_dem.py keeps every point; then emergence and the
paired validation (scripts/r00_pipeline.py). The §9 observed extents never used the
rule (D-070), so the hindcast is unchanged and not rerun. The excluded fraction per
sector, split and epoch is already reported by the main run (results/levels/
static_contour_rule.json).

Usage:
  python scripts/r12_static_rule.py --main <tag> [--steps dem,emergence,validate]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import pandas as pd

from caspian.config import INTERIM, RESULTS

CHECK = "r12_static_rule"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    p = pd.read_csv(INTERIM / "waterlines" / a.main / "points.csv.gz")
    if "static" not in p.columns:
        raise SystemExit("the main contour table has no static column: run scripts/a6_static_rule.py first")
    variant = f"{a.main}__nostatic"
    d = INTERIM / "waterlines" / variant
    d.mkdir(parents=True, exist_ok=True)
    p.drop(columns=["static"]).to_csv(d / "points.csv.gz", index=False, float_format="%.6f")
    out = RESULTS / "robustness" / CHECK
    out.mkdir(parents=True, exist_ok=True)
    e23 = p["epoch"].isin(["E2", "E3"])
    (out / "points.json").write_text(json.dumps({"main": a.main, "n_points_e23": int(e23.sum()),
                                                 "n_static_e23": int(p.loc[e23, "static"].astype(bool).sum())},
                                                indent=1), encoding="utf-8")
    steps = [s for s in a.steps.split(",") if s in ("dem", "emergence", "validate")]
    P.run_steps(CHECK, variant, a.main, steps, sources=a.sources.split(","), n_boot=a.n_boot,
                note="no static-contour exclusion")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
