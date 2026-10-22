"""Robustness check 5: altimetry series and time-interpolation method, to first order
and without Earth Engine.

An alternative level for every scene of the main plan gives, per (epoch, bin), the
mean shift of its acquisitions' levels (robustness.group_level_shift). Each contour
keeps its location and takes the bin centre plus that shift; the DEM, emergence and the
paired validation follow (scripts/r00_pipeline.py). The hindcast floods each bin at its
centre plus the acquisition-weighted shift over its epochs. The share of acquisitions
that would change bin under the alternative (the second-order effect, which would
change the contours themselves and needs a re-binned export) is reported per group.

Alternatives (--name):
  hydroweb_const   Hydroweb (Theia) interpolated to the scene times with the same
                   Brownian-bridge machinery, shifted onto the DAHITI reference by ONE
                   constant (the median of all +-20-day pair differences), so that the
                   drift between the two products enters the levels (the main run
                   corrects for it with a moving median)
  dahiti_pchip     DAHITI with monotone cubic (PCHIP) instead of linear interpolation in
                   time, where the bracketing records are at most 60 days apart (the
                   Brownian-bridge mean is the linear interpolant)
  window1y         the gap-fill offset with a +-1 year moving-median window
                   instead of +-2 years (the candidate variant)
  external         another series (--series CSV with time, level_m; e.g. G-REALM when
                   downloaded), interpolated linearly and shifted by one constant
                   (median of the +-20-day differences from DAHITI)

Usage:
  python scripts/r05_levels.py --main <tag> --plan <main plan csv> --name hydroweb_const
      [--series CSV] [--steps dem,emergence,validate,hindcast] [--vertical-correction PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels, robustness as R
from caspian.config import INTERIM, RESULTS

CHECK = "r05_levels"
OUT = RESULTS / "robustness" / CHECK


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def alt_levels(name: str, times: pd.Series, series_csv: str | None = None) -> tuple[np.ndarray, dict]:
    a4 = _load("a4_scene_levels")
    if name == "window1y":
        a4.OFFSET_HALF_WINDOW_DAYS = 365.25
        prep = a4.prepare()
        lv, _ = a4.levels_at(prep, times)
        return lv["level_m"].to_numpy(float), {"offset_half_window_days": 365.25}
    prep = a4.prepare()
    d, hw, pairs = prep["d"], prep["hw"], prep["pairs"]
    if name == "hydroweb_const":
        off = float(np.median(pairs["diff"]))
        lh = levels.interpolate(hw["time"], hw["level_m"], prep["sig_h"], times, q=prep["q_h"],
                                max_gap_days=a4.MAX_GAP_DAYS)
        return lh["level_m"].to_numpy(float) + off, {"constant_offset_m": off, "n_pairs": int(len(pairs))}
    if name == "dahiti_pchip":
        from scipy.interpolate import PchipInterpolator
        td = (pd.to_datetime(d["time"]) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86400
        tq = (pd.to_datetime(times) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86400
        o = np.argsort(td)
        td, hd = td[o], d["level_m"].to_numpy(float)[o]
        keep = np.r_[True, np.diff(td) > 0]
        td, hd = td[keep], hd[keep]
        out = PchipInterpolator(td, hd, extrapolate=False)(tq)
        j = np.clip(np.searchsorted(td, tq), 1, td.size - 1)
        out[(td[j] - td[j - 1]) > a4.MAX_GAP_DAYS] = np.nan
        return out, {"method": "scipy PchipInterpolator over DAHITI records, 60-day gap rule"}
    if name == "external":
        s = pd.read_csv(series_csv, parse_dates=["time"])
        pr = levels.pair_series(d["time"], d["level_m"], s["time"], s["level_m"], 20.0)
        off = float(np.median(pr["diff"]))
        lx = levels.interpolate(s["time"], s["level_m"], np.zeros(len(s)), times, q=0.0, max_gap_days=a4.MAX_GAP_DAYS)
        return lx["level_m"].to_numpy(float) + off, {"series": series_csv, "constant_offset_m": off,
                                                     "n_pairs": int(len(pr))}
    raise SystemExit(f"unknown alternative {name}")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--name", required=True, choices=["hydroweb_const", "dahiti_pchip", "window1y", "external"])
    ap.add_argument("--series")
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    plan = pd.read_csv(a.plan, usecols=["scene", "acq", "t", "level_m", "epoch", "bin"]).drop_duplicates("scene")
    times = pd.to_datetime(plan["t"], format="ISO8601")
    alt, info = alt_levels(a.name, times.reset_index(drop=True), a.series)
    # where the alternative has no level (outside its record or in a gap), the main level is kept
    alt = np.where(np.isfinite(alt), alt, plan["level_m"].to_numpy(float))
    sh = R.group_level_shift(plan.reset_index(drop=True), alt)
    OUT.mkdir(parents=True, exist_ok=True)
    sh.to_csv(OUT / f"{a.name}_shifts.csv", index=False)
    rec = {"name": a.name, **info, "n_scenes": len(plan),
           "shift_m": {"p05": float(np.percentile(sh["shift_m"], 5)), "p50": float(np.median(sh["shift_m"])),
                       "p95": float(np.percentile(sh["shift_m"], 95))},
           "share_acq_rebin": float(sh["n_acq_rebin"].sum() / sh["n_acq"].sum())}
    (OUT / f"{a.name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    variant = f"{a.main}__lv_{a.name}"
    pts = pd.read_csv(INTERIM / "waterlines" / a.main / "points.csv.gz")
    d = INTERIM / "waterlines" / variant
    d.mkdir(parents=True, exist_ok=True)
    R.shift_points(pts, sh).to_csv(d / "points.csv.gz", index=False, float_format="%.6f")
    steps = a.steps.split(",")
    P.run_steps(CHECK, variant, a.main, [s for s in steps if s in ("dem", "emergence", "validate")],
                sources=a.sources.split(","), n_boot=a.n_boot, note=a.name)
    if "hindcast" in steps:
        w = sh.assign(ws=sh["shift_m"] * sh["n_acq"]).groupby("bin")[["ws", "n_acq"]].sum()
        by_bin = (w["ws"] / w["n_acq"]).to_dict()
        P.hindcast_step(CHECK, variant, a.main, stage2_tag=a.main, vertical_correction=a.vertical_correction,
                        shape=a.shape, sector_labels=a.sector_labels, note=a.name,
                        bin_level=lambda b: float(levels.bin_center(b, design.BIN_WIDTH_M, design.BIN_ORIGIN_M)
                                                  + by_bin.get(int(b), 0.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
