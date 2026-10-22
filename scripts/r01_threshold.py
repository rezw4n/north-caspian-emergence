"""Robustness check 1: the per-scene threshold method. Local parts only; the Earth
Engine parts run separately.

Commands
  sample     The Amendment 3 robustness sample, by whole acquisitions, stratified by
             sensor (robustness.acquisition_sample, seed 20261020, 100 scenes per
             sensor): data/interim/robustness/a3_sample.csv. Drawn before the main
             export so that the main plan can tag it (scripts/r16_split_plan.py) and its
             main-threshold counts can be removed without another export.
  edge       After the Earth Engine export of the sample acquisitions classified with the
             edge-guided threshold (tag --edge-tag, main groups): variant = main with the
             a3-tagged groups dropped (split main) or minus --a3-main-tag (unsplit main),
             plus the edge counts; then the downstream steps and the paired comparison
             (scripts/r00_pipeline.py).
  fallback   With a main export split by threshold source: the variant without the
             fallback-threshold acquisitions (token fb), which bounds the effect of the
             fallback rule.
  block      A block-sample Earth Engine alternative (fixed MNDWI 0, AWEI_sh, ...; tag
             --alt-tag, 25 km block tiles of the robustness sample R): the hybrid variant
             (main counts with the alternative's inside R; r00_counts --substitute), then
             the downstream steps. Its comparison is read inside R only (the comparison
             tables carry block ids).

Usage:
  python scripts/r01_threshold.py sample --plan <main plan csv>
  python scripts/r01_threshold.py edge --main <tag> --edge-tag <tag> [--split-main <tag> | --a3-main-tag <tag>]
  python scripts/r01_threshold.py fallback --main <tag> --split-main <tag>
  python scripts/r01_threshold.py block --main <tag> --alt-tag <tag> --name <fixed0|awei|...>
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
from caspian.config import INTERIM, RESULTS

CHECK = "r01_threshold"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sample(plan_csv: str) -> int:
    plan = pd.read_csv(plan_csv, usecols=["scene", "sensor", "acq"]).drop_duplicates("scene")
    s = R.acquisition_sample(plan)
    out = INTERIM / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "a3_sample.csv", index=False)
    rec = {"plan": str(plan_csv), "n_acquisitions": len(s), "n_scenes": int(s["n_scenes"].sum()),
           "by_sensor": s.groupby("sensor")["n_scenes"].agg(["size", "sum"]).rename(
               columns={"size": "n_acquisitions", "sum": "n_scenes"}).to_dict(orient="index"),
           "rule": "Amendment 3 sample by whole acquisitions, 100 scenes per sensor, seed 20261020 "
                   "(robustness.acquisition_sample; proposed DECISIONS entry)"}
    (RESULTS / "robustness" / CHECK).mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness" / CHECK / "a3_sample.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["sample", "edge", "fallback", "block"])
    ap.add_argument("--plan")
    ap.add_argument("--edge-tag")
    ap.add_argument("--split-main")
    ap.add_argument("--a3-main-tag")
    ap.add_argument("--alt-tag")
    ap.add_argument("--name")
    if argv and argv[0] == "sample":
        a, _ = ap.parse_known_args(argv)
        return sample(a.plan)
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    C = P.script("r00_counts")
    if a.cmd == "edge":
        variant = f"{a.main}__a3edge"
        if a.split_main:
            C.build(variant, [a.split_main], add=a.edge_tag, drop=[R.SAMPLE_TOKEN])
        elif a.a3_main_tag:
            C.build(variant, [a.main], sub=a.a3_main_tag, add=a.edge_tag)
        else:
            raise SystemExit("edge needs --split-main (a3-tagged main) or --a3-main-tag")
    elif a.cmd == "fallback":
        variant = f"{a.main}__nofallback"
        C.build(variant, [a.split_main], drop=["fb"])
    else:
        variant = f"{a.main}__{a.name}"
        C.build(variant, [a.main], substitute=a.alt_tag)
    P.run_steps(CHECK, variant, a.main, a.steps.split(","), sources=a.sources.split(","),
                vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
                n_boot=a.n_boot, note=a.cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
