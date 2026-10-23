"""Second-round revision analysis A: PREREG Amendment 19 item 11, observation variants
through the hindcast (D-147; designed in D-119 and cut for time in the D-117/D-118/D-119
addendum).

Runs scripts/rev_r_audit.py `pass --configs variant` with its code unchanged, on all five
products at the primary area constants (shape variant), and writes to
results/revision2/a/audit/<name>/ instead of results/revision/r/audit/:

  veg_as_dry    stage-2 main__veg_as_dry, all 29 tiles, its own record-low envelope
  veg_as_water  stage-2 main__veg_as_water, likewise
  rR_awei       stage-2 rR_awei, hybrid: the variant's counts inside its 14 R-block tiles
                and the primary's elsewhere; only cells inside the tiles measured; the
                primary on the same footprint ("primary_fp") computed beside it
  rR_fixed0     stage-2 rR_fixed0 (fixed MNDWI 0), hybrid likewise

Usage: python scripts/rev2_a_variants.py veg_as_dry|veg_as_water|rR_awei|rR_fixed0 [--bins 12,38]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"veg_as_dry": ("main__veg_as_dry", False), "veg_as_water": ("main__veg_as_water", False),
        "rR_awei": ("rR_awei", True), "rR_fixed0": ("rR_fixed0", True)}
PRODUCTS = "gebco_2026,etopo_2022,srtm15p_v2.7,gebco_2022,etopo1"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", choices=sorted(RUNS))
    ap.add_argument("--bins", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    R = _load("rev_r_audit")
    from caspian.config import RESULTS
    R.OUT = RESULTS / "revision2" / "a" / "audit"          # results/revision2/a/audit/<name>/
    tag, hybrid = RUNS[a.name]
    ns = argparse.Namespace(command="pass", configs="variant", stage2=tag, hybrid=hybrid, bins=a.bins,
                            out=a.out or a.name, products=PRODUCTS, surfaces="products", c_step=0.0, constants="")
    return R.run_pass(ns)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
