"""Robustness check 3: season and ice. Local parts; the November addition and the
Amendment 9 variant need Earth Engine.

Commands
  variant    From tags split by season (scripts/r16_split_plan.py; tokens AMJ, JASO):
             amj_only and jaso_only, then the downstream steps and the paired comparison
             (scripts/r00_pipeline.py). The month-sampling question is answered by
  offsets    the per-bin offsets between the AMJ and JASO waterlines (robustness.
             contour_offsets on the points of the two variants, after their points step),
             against the odd/even noise floor of the main run at the same bins.
  november   The scenes a 5% ice bar would add (November): levelled (E1-E3), scene
             cloud < 60%, in November, with a stage-1 run-2 zone count above zero where
             run 2 has one (run 3 covers April-October only). They need stage 1
             (thresholds) and a stage-2 ADD export before the variant can be built
             (r00_counts --add). Written to data/interim/robustness/november_scenes.csv.

Usage:
  python scripts/r03_season.py variant --main <tag> --name amj_only|jaso_only --split <tag>[,<tag>]
  python scripts/r03_season.py offsets --main <tag>
  python scripts/r03_season.py november
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import pandas as pd

from caspian import robustness as R
from caspian.config import INTERIM, RESULTS

CHECK = "r03_season"
KEEP = {"amj_only": ["AMJ"], "jaso_only": ["JASO"]}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def november() -> int:
    lv = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", low_memory=False)
    lv = lv[lv["epoch"].isin(["E1", "E2", "E3"]) & (lv["month"] == 11) & (lv["cloud_scene_pct"] < 60)]
    zp = []
    for f in sorted(glob.glob(str(INTERIM / "stage1_v2" / "*.csv"))):
        if "thresholds" not in f:
            zp.append(pd.read_csv(f, usecols=["scene", "zone_px"]))
    if zp:
        z = pd.concat(zp, ignore_index=True).drop_duplicates("scene")
        lv = lv.merge(z, left_on="system:index", right_on="scene", how="left")
        lv = lv[~(pd.to_numeric(lv["zone_px"], errors="coerce") <= 0)]
    out = INTERIM / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    lv.to_csv(out / "november_scenes.csv", index=False)
    rec = {"n_scenes": len(lv), "by_sensor": lv["sensor"].value_counts().to_dict(),
           "with_run2_zone_count": int(lv["zone_px"].notna().sum()) if "zone_px" in lv else 0,
           "rule": "D-062 sensitivity: a 5% bar keeps November"}
    (RESULTS / "robustness" / CHECK).mkdir(parents=True, exist_ok=True)
    (RESULTS / "robustness" / CHECK / "november.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def offsets(main_tag: str) -> int:
    pa = pd.read_csv(INTERIM / "waterlines" / f"{main_tag}__amj_only" / "points.csv.gz")
    pb = pd.read_csv(INTERIM / "waterlines" / f"{main_tag}__jaso_only" / "points.csv.gz")
    keep = lambda p: p[~p["static"].astype(bool)] if "static" in p else p  # noqa: E731
    d = R.contour_offsets(keep(pa), keep(pb))
    nf = pd.read_csv(RESULTS / "hindcast" / main_tag / "noise_floor.csv")
    nf = nf[(nf["kind"] == "halves") & (nf["sector"] == "all")][["epoch", "bin", "median_m", "p95_m"]]
    d = d.merge(nf.rename(columns={"median_m": "noise_median_m", "p95_m": "noise_p95_m"}), on=["epoch", "bin"],
                how="left")
    out = RESULTS / "robustness" / CHECK
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "amj_vs_jaso_offsets.csv", index=False)
    print(d.describe().to_string())
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["variant", "offsets", "november"])
    ap.add_argument("--name")
    ap.add_argument("--split")
    if argv and argv[0] == "november":
        return november()
    if argv and argv[0] == "offsets":
        ap.add_argument("--main", required=True)
        return offsets(ap.parse_args(argv).main)
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    if a.name not in KEEP:
        raise SystemExit(f"unknown variant {a.name}")
    variant = f"{a.main}__{a.name}"
    P.script("r00_counts").build(variant, a.split.split(","), keep_only=KEEP[a.name])
    P.run_steps(CHECK, variant, a.main, a.steps.split(","), sources=a.sources.split(","),
                vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
                n_boot=a.n_boot, note=a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
