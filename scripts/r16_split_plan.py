"""Robustness (main-export design): the main stage-2 plan with
its groups split into classes, so that several robustness checks become local count
algebra (scripts/r00_counts.py) instead of new Earth Engine exports. Local; it only
writes a plan CSV for scripts/a6_stage2_export.py upload, and reports how many groups
and bands the export would carry.

Each scene stays in exactly one group, so the export evaluates every scene once, as
before; every class is constant within an acquisition, so the classes of a group sum
back to the unsplit group exactly, including the per-acquisition mosaic
(robustness.split_groups). What grows is the number of bands (3 per group: w, v, o)
and the per-group overhead, to be measured on a 0.3 deg micro-pilot box before the
design is adopted (budget-guard rule).

Classes (--by, in this order): sensor (TM, E7on, E7off, OLI, MSI: checks 2 and the
SLC-off question), half (the 0.05 m bins of PREREG section 4), season (AMJ, JASO:
check 3), threshold (otsu, fb: the fallback threshold). Optional tags: --e1-windows (the
Amendment 4 E1-end windows, from results/levels/scene_levels_summary.json) and
--a3-sample (the Amendment 3 acquisitions, scripts/r01_threshold.py sample).

Usage:
  python scripts/r16_split_plan.py --plan <main plan csv> [--by sensor,half] [--e1-windows]
      [--a3-sample data/interim/robustness/a3_sample.csv]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import pandas as pd

from caspian import robustness as R
from caspian.config import RESULTS


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--by", default="sensor,half")
    ap.add_argument("--e1-windows", action="store_true")
    ap.add_argument("--a3-sample", default=None)
    a = ap.parse_args(argv)
    plan = pd.read_csv(a.plan, low_memory=False, dtype={"sectors": str, "excluded_sectors": str})
    ends = sorted(set(_load("r15_e1_end").e1_ends().values())) if a.e1_windows else None
    acqs = pd.read_csv(a.a3_sample)["acq"].tolist() if a.a3_sample else None
    sp = R.split_groups(plan, by=[b for b in a.by.split(",") if b], e1_ends=ends, sample_acqs=acqs)
    out = Path(a.plan).with_name(Path(a.plan).stem + "_split.csv")
    sp.drop(columns=["base_group"]).to_csv(out, index=False)
    fam = sp["sensor"].eq("MSI_S2").map({True: "S2", False: "LANDSAT"})
    rec = {"plan": a.plan, "split_plan": str(out), "by": a.by, "e1_ends": [str(e) for e in ends] if ends else None,
           "a3_sample": a.a3_sample,
           "groups_before": {f: int(sp.loc[fam == f, "base_group"].nunique()) for f in ("LANDSAT", "S2")},
           "groups_after": {f: int(sp.loc[fam == f, "group"].nunique()) for f in ("LANDSAT", "S2")},
           "bands_after_wvo": {f: 3 * int(sp.loc[fam == f, "group"].nunique()) for f in ("LANDSAT", "S2")},
           "scenes": int(sp["scene"].nunique()),
           "note": "every scene in one group; classes constant within acquisitions (checked)"}
    o = RESULTS / "robustness" / "r16_split_plan"
    o.mkdir(parents=True, exist_ok=True)
    (o / (Path(a.plan).stem + "_split.json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
