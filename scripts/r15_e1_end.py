"""Robustness (PREREG Amendment 4 item 1): every E1-based result with E1 ending
on other running-median maxima (k = 1, 5, 7, 9; k = 1, 5 and 7 share one date). No
Earth Engine, given a main export split with the E1-end window tokens
(scripts/r16_split_plan.py --e1-windows).

For each alternative end the variant tag reassigns the window acquisitions between E1
and E2 (robustness.e1_end_mapper; their parity labels are kept, so the half-sample
noise floor of the window bins is approximate while the full-sample extents are
exact). Then the contour points, the E2 + E3 DEM (the window acquisitions that move to
E2 enter it), the hindcast (the E1-vs-E2/E3 noise floor and the per-level H1 rule) and
the rising-versus-falling-limb difference (scripts/r10_morphology.limb_difference)
are rerun.

The ends come from results/levels/scene_levels_summary.json
(diagnostics.e1_end_sensitivity.by_k).

Usage:
  python scripts/r15_e1_end.py --main <tag> --split <tag>[,<tag>] --vertical-correction PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import pandas as pd

from caspian.config import RESULTS

CHECK = "r15_e1_end"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def e1_ends() -> dict:
    """{k: end time} of the E1-end sensitivity (scene-level summary)."""
    s = json.loads((RESULTS / "levels" / "scene_levels_summary.json").read_text(encoding="utf-8"))
    by_k = s["diagnostics"]["e1_end_sensitivity"]["by_k"]
    return {int(r["k"]): pd.Timestamp(r["time_of_max"]) for r in by_k}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True)
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    ends = e1_ends()
    applied = json.loads((RESULTS / "levels" / "scene_levels_summary.json").read_text(encoding="utf-8"))[
        "diagnostics"]["e1_end_sensitivity"]["applied_k"]
    distinct = sorted(set(ends.values()))
    C = P.script("r00_counts")
    for end in distinct:
        if end == ends[applied]:
            continue
        ks = "_".join(str(k) for k, v in sorted(ends.items()) if v == end)
        variant = f"{a.main}__e1k{ks}"
        C.build(variant, a.split.split(","), e1_end=end, e1_ends=distinct)
        steps = [s for s in a.steps.split(",") if s in ("points", "dem", "hindcast")]
        P.run_steps(CHECK, variant, a.main, steps, vertical_correction=a.vertical_correction, shape=a.shape,
                    sector_labels=a.sector_labels, n_boot=a.n_boot, note=f"E1 end for k = {ks}: {end}")
        if "dem" in steps:
            lim = _load("r10_morphology").limb_difference(variant, P)
            out = P.out_dir(CHECK, variant)
            out.mkdir(parents=True, exist_ok=True)
            lim.to_csv(out / "limb_difference_by_sector.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
