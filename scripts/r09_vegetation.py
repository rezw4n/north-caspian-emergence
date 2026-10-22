"""Robustness check 9: reed-bed handling. No Earth Engine.

PREREG section 3 counts emergent vegetation (NDVI > 0.30 while not water) as not
observed. The main export carries the observed count o, so two bounding rules
follow from the counts alone (robustness.vegetation_variant):
  as_dry    vegetation counted as land   (w, o)
  as_water  vegetation counted as water  (w + o - v, o): flooded reeds
Each variant tag is built with scripts/r00_counts.py --vegetation, then every downstream
step and the paired comparison runs (scripts/r00_pipeline.py). Other NDVI thresholds
change the classification itself and need Earth Engine.

Usage:
  python scripts/r09_vegetation.py --main <tag> --name as_dry|as_water [--from <tag>[,<tag>]]
      [--vertical-correction PATH] [--steps ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import caspian  # noqa: F401

CHECK = "r09_vegetation"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, choices=["as_dry", "as_water"])
    ap.add_argument("--from", dest="sources_tags", default=None,
                    help="stage-2 tags to sum (default: the main tag)")
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    variant = f"{a.main}__veg_{a.name}"
    P.script("r00_counts").build(variant, (a.sources_tags or a.main).split(","), vegetation=a.name)
    P.run_steps(CHECK, variant, a.main, a.steps.split(","), sources=a.sources.split(","),
                vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
                n_boot=a.n_boot, note=f"vegetation {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
