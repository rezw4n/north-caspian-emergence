"""Robustness check 8 (brief section 10; PREREG section 6; Amendment 8): spatial
blocking and the conformal calibration split. No Earth Engine.

Commands
  tables      Block tables at 12.5 and 50 km with the same code and seed
              (scripts/a6_blocks.py; results/study_area/blocks_12km.csv, blocks_50km.csv).
  validate    The H2 and coverage headline at 12.5, 25 and 50 km (those other than the
              final size), paired on the same ICESat-2 points (scripts/r00_pipeline.py).
  hindcast    The H1 block values and permutation test with 12.5 and 50 km blocks: the
              hindcast rerun with that block table, which also sets the training blocks of
              the shape-constant fit (D-070); --sector-labels for the block test.
  splits      The calibration split: --n-seeds reassignments of the final-size blocks
              (seeds 1..N, same stratified rule and proportions), and the calibration /
              test swap. Run after the main evaluation is frozen: these reuse test blocks
              for fitting and are reported as a sensitivity, never as the result.
  amendment8  Only if PREREG section 6 enlarged the blocks: the headline recomputed
              without the calibration and test points that lie in training blocks of the
              initial 25 km assignment (whose contour points fixed the static rule, D-075).

Usage:
  python scripts/r08_blocking.py tables
  python scripts/r08_blocking.py validate --main <tag>
  python scripts/r08_blocking.py hindcast --main <tag> --vertical-correction PATH [--sector-labels CSV]
  python scripts/r08_blocking.py splits --main <tag> [--n-seeds 20]
  python scripts/r08_blocking.py amendment8 --main <tag>
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import caspian  # noqa: F401
import pandas as pd

from caspian import design, robustness as R
from caspian.config import RESULTS

CHECK = "r08_blocking"
SIZES = design.BLOCK_SIZE_SENS_M        # 12.5, 25, 50 km (PREREG section 6)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _table(size: float) -> Path:
    return RESULTS / "study_area" / f"blocks_{int(size / 1000)}km.csv"


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["tables", "validate", "hindcast", "splits", "amendment8"])
    ap.add_argument("--n-seeds", type=int, default=20)
    if argv and argv[0] == "tables":
        a6b = _load("a6_blocks")
        for s in SIZES:
            if s != design.BLOCK_SIZE_M and not _table(s).exists():
                a6b.main(float(s))
        return 0
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    sources = a.sources.split(",")
    final = P.final_block_size(a.main)
    if a.cmd == "validate":
        for s in SIZES:
            if float(s) != final:
                if not _table(s).exists():
                    raise SystemExit(f"{_table(s)} missing: run `python scripts/r08_blocking.py tables`")
                P.compare_validation(CHECK, f"blocks_{s / 1000:g}km", a.main, sources=sources, alt_block_size=float(s),
                                     n_boot=a.n_boot, note="block size sensitivity (PREREG section 6)")
    elif a.cmd == "hindcast":
        for s in SIZES:
            if float(s) != float(design.BLOCK_SIZE_M):     # the main hindcast uses the 25 km table
                P.hindcast_step(CHECK, f"blocks_{s / 1000:g}km", a.main, stage2_tag=a.main,
                                vertical_correction=a.vertical_correction, blocks_csv=_table(s), block_m=float(s),
                                sector_labels=a.sector_labels, note="block size sensitivity")
    elif a.cmd == "splits":
        blocks = pd.read_csv(_table(final))
        for seed in range(1, a.n_seeds + 1):
            P.compare_validation(CHECK, f"split_seed{seed:02d}", a.main, sources=sources,
                                 alt_blocks=R.reassign_splits(blocks, seed), n_boot=a.n_boot,
                                 note="split reassignment (sensitivity; reuses test blocks)")
        P.compare_validation(CHECK, "split_swap_calib_test", a.main, sources=sources, post_blocks=R.swap_splits,
                             n_boot=a.n_boot, note="calibration and test blocks swapped")
    else:
        if final == float(design.BLOCK_SIZE_M):
            print("final block size is the initial 25 km: Amendment 8's recomputation does not apply")
            return 0
        init = pd.read_csv(_table(design.BLOCK_SIZE_M))
        P.compare_validation(CHECK, "amendment8_exclusion", a.main, sources=sources,
                             post_blocks=lambda p: R.exclude_initial_training(p, init), n_boot=a.n_boot,
                             note="Amendment 8 clarification")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
