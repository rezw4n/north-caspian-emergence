"""Robustness check 2 (brief section 10): sensor. Local parts; the same-day pair
export is specified in docs/ROBUSTNESS_PLAN.md.

Commands
  variant   Sensor variants from stage-2 tags, then the downstream steps and the paired
            comparison (scripts/r00_pipeline.py):
              landsat_only   the Landsat main tag alone (the main is Landsat +
                             Sentinel-2 summed, D-064 step 3)
            and, from tags split by sensor class (scripts/r16_split_plan.py):
              no_slcoff      without ETM+ SLC-off acquisitions (token E7off)
              no_tm, no_etm, no_oli, no_msi   leave one sensor family out
  pairs     The same-day / near-same-day pair list (local, from the main plan): pairs of
            acquisitions of two sensors at most --max-hours apart, with level difference
            <= --max-dlevel and |setup| <= tau in every sector, up to --n per sensor
            combination (seeded); written with a plan for the per-acquisition export
            (group = pair id and member, mask_mode "exclude" with no sector):
            data/interim/robustness/sensor_pairs.csv, plan_sensor_pairs.csv
  compare   After that export (tag --pairs-tag): per pair, the waterline displacement
            between the members over the cells both observed
            (caspian.hindcast.noise_floor_displacement) and the water-area difference;
            summary per sensor combination.

Usage:
  python scripts/r02_sensor.py variant --main <tag> --name landsat_only --landsat <tag>
  python scripts/r02_sensor.py variant --main <tag> --name no_slcoff --split <tag>[,<tag>]
  python scripts/r02_sensor.py pairs --plan <main plan csv> --tau <m> [--n 30] [--max-hours 24] [--max-dlevel 0.01]
  python scripts/r02_sensor.py compare --pairs-tag <tag>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, hindcast, robustness as R
from caspian.config import INTERIM, RESULTS

CHECK = "r02_sensor"
DROPS = {"no_slcoff": ["E7off"], "no_tm": ["TM"], "no_etm": ["E7on", "E7off"], "no_oli": ["OLI"], "no_msi": ["MSI"]}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pair_list(plan: pd.DataFrame, tau: float, n: int, max_hours: float, max_dlevel: float,
              seed: int = design.SEED) -> pd.DataFrame:
    """Cross-sensor acquisition pairs (see the module docstring)."""
    ex = R.exclusion(plan, tau)
    plan = plan.assign(_calm=~ex.any(axis=1), _t=pd.to_datetime(plan["t"], format="ISO8601"))
    a = (plan.groupby("acq").agg(sensor=("sensor", "first"), t=("_t", "min"), level=("level_m", "mean"),
                                 calm=("_calm", "all")).reset_index())
    a = a[a["calm"]].sort_values("t").reset_index(drop=True)
    rows = []
    tt = a["t"].to_numpy()
    for i in range(len(a)):
        j = i + 1
        while j < len(a) and (tt[j] - tt[i]) <= np.timedelta64(int(max_hours * 3600), "s"):
            if a.at[i, "sensor"] != a.at[j, "sensor"] and abs(a.at[i, "level"] - a.at[j, "level"]) <= max_dlevel:
                s1, s2 = sorted([a.at[i, "sensor"], a.at[j, "sensor"]])
                rows.append({"combo": f"{s1}|{s2}", "acq_a": a.at[i, "acq"], "acq_b": a.at[j, "acq"],
                             "dt_h": (tt[j] - tt[i]) / np.timedelta64(1, "h"),
                             "dlevel_m": abs(a.at[i, "level"] - a.at[j, "level"])})
            j += 1
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    rng = np.random.default_rng(seed)
    out = []
    for combo, g in sorted(p.groupby("combo"), key=lambda kv: kv[0]):
        g = g.sort_values(["acq_a", "acq_b"])
        out.append(g.iloc[rng.permutation(len(g))[:n]])
    p = pd.concat(out, ignore_index=True)
    p["pair"] = [f"P{k:03d}" for k in range(len(p))]
    return p


def pairs(plan_csv: str, tau: float, n: int, max_hours: float, max_dlevel: float) -> int:
    plan = pd.read_csv(plan_csv, low_memory=False)
    p = pair_list(plan, tau, n, max_hours, max_dlevel)
    out = INTERIM / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "sensor_pairs.csv", index=False)
    rows = []
    for _, r in p.iterrows():
        for m, acq in (("a", r["acq_a"]), ("b", r["acq_b"])):
            sc = plan[plan["acq"] == acq].copy()
            # group names follow the main pattern so that the export's band naming applies;
            # E3 is a label only, the bin index carries the pair number
            sc["group"] = f"E3_b{int(r['pair'][1:]):03d}_p{0 if m == 'a' else 1}"
            sc["sectors"], sc["mask_mode"] = "", "exclude"
            rows.append(sc)
    plan_out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    plan_out.to_csv(INTERIM / "stage2" / "plan_sensor_pairs.csv", index=False)
    print(p.groupby("combo").size().to_string() if len(p) else "no pairs", "\n", len(plan_out), "scenes in the plan")
    return 0


def compare(pairs_tag: str) -> int:
    p = pd.read_csv(INTERIM / "robustness" / "sensor_pairs.csv")
    files = sorted((INTERIM / "stage2" / pairs_tag).glob("tile_*.npz"))
    acc: dict = {}
    pixel = None
    for f in files:
        tr, arr = R.load_tile(f)
        pixel = abs(float(tr[0]))
        for k in arr:
            m = re.match(r"w_E3_b(\d{3})_p([01])$", k)
            if m:
                pid, member = int(m.group(1)), int(m.group(2))
                acc.setdefault((f.name, pid), {})[member] = (arr[k] > 0, arr["v" + k[1:]] > 0)
    rows = []
    for (tile, pid), mm in acc.items():
        if set(mm) != {0, 1}:
            continue
        (wa, va), (wb, vb) = mm[0], mm[1]
        both = va & vb
        if not both.any():
            continue
        r = hindcast.noise_floor_displacement(wa, va, wb, vb, pixel)
        rows.append({"pair": f"P{pid:03d}", "tile": tile, "median_m": r.get("median"), "p95_m": r.get("p95"),
                     "signed_median_m": r.get("signed_median"), "n": r.get("n"),
                     "water_km2_a": float((wa & both).sum()) * pixel ** 2 / 1e6,
                     "water_km2_b": float((wb & both).sum()) * pixel ** 2 / 1e6})
    d = pd.DataFrame(rows).merge(p, on="pair", how="left")
    out = RESULTS / "robustness" / CHECK
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "pairs_detail.csv", index=False)
    s = d.groupby("combo").agg(n_pairs=("pair", "nunique"), median_displacement_m=("median_m", "median"),
                               median_signed_m=("signed_median_m", "median"),
                               water_diff_km2=("water_km2_a", "sum"))
    s["water_diff_km2"] = d.groupby("combo")["water_km2_a"].sum() - d.groupby("combo")["water_km2_b"].sum()
    s.to_csv(out / "pairs_summary.csv")
    print(s.to_string())
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["variant", "pairs", "compare"])
    ap.add_argument("--name")
    ap.add_argument("--landsat")
    ap.add_argument("--split")
    ap.add_argument("--plan")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--max-dlevel", type=float, default=0.01)
    ap.add_argument("--pairs-tag")
    if argv and argv[0] in ("pairs", "compare"):
        a, _ = ap.parse_known_args(argv)
        return pairs(a.plan, a.tau, a.n, a.max_hours, a.max_dlevel) if a.cmd == "pairs" else compare(a.pairs_tag)
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    C = P.script("r00_counts")
    variant = f"{a.main}__{a.name}"
    if a.name == "landsat_only":     # a copy, so that no step writes under the Landsat tag itself
        C.build(variant, [a.landsat])
    elif a.name in DROPS:
        C.build(variant, a.split.split(","), drop=DROPS[a.name])
    else:
        raise SystemExit(f"unknown variant {a.name}")
    P.run_steps(CHECK, variant, a.main, a.steps.split(","), sources=a.sources.split(","),
                vertical_correction=a.vertical_correction, shape=a.shape, sector_labels=a.sector_labels,
                n_boot=a.n_boot, note=a.name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
