"""Observed level-binned waterlines for the ERL "Then and now" figure (figure 1, panel b).

The figure's shorelines were contours of the waterline DEM's emergence dates, which failed
validation at the Volga delta front. They are replaced by observed shorelines: for each of
the E3 level bins in BINS, the observed extent exactly as scripts/a7_hindcast.py builds it
(observed_raw with the record-low envelope, classify with the sea connection and the
outside states; the same functions and cached static layers as
scripts/f1_erl_display.py's hero map, figure 2), reduced to the share of sea-connected
water per 240 m display block. Nothing here is a new result.

Bin choice (a display rule fixed before any of these extents was drawn): four E3
bins 0.6 m apart spanning the E3 record, each fully observed (valid area of the domain at
least 90% of the median over E3 bins, results/hindcast/main/observed.csv): bins 32, 26, 20
and 14 (-26.75, -27.35, -27.95 and -28.55 m DAHITI). The partly observed bins (12, 13, 22,
33) are excluded by the completeness test, which the script checks.

Outputs
  data/interim/figures/erl/waterlines_main.npz   per bin, the water share per 240 m block
                                                 (git-ignored)
  results/display/erl_waterlines.json            bins, levels, acquisitions and their
                                                 years, the completeness check
                                                 (tools/metrics_rev_fig.py)
Usage: python scripts/f3_erl_waterlines.py [--tag main]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("f1_erl_display", ROOT / "scripts" / "f1_erl_display.py")
f1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f1)
a7 = f1.a7

EPOCH = "E3"
BINS = (32, 26, 20, 14)          # high to low
COMPLETE_FRAC = 0.90             # domain valid area >= this share of the E3 median


def completeness(tag: str) -> dict:
    o = pd.read_csv(f1.RESULTS / "hindcast" / tag / "observed.csv")
    o = o[(o["sample"] == "full") & (o["epoch"] == EPOCH) & (o["sector"] != "all")]
    tot = o.groupby("bin")["area_valid_km2"].sum()
    ratio = tot / tot.median()
    return {int(b): {"complete": bool(ratio[b] >= COMPLETE_FRAC), "valid_ratio": round(float(ratio[b]), 3)}
            for b in tot.index}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tile", type=int, default=f1.ht.TILE)
    a = ap.parse_args(argv)
    comp = completeness(a.tag)
    bad = [b for b in BINS if not comp.get(b, {}).get("complete")]
    if bad:
        raise SystemExit(f"bins {bad} are not fully observed")
    hrun = json.loads((f1.RESULTS / "hindcast" / a.tag / "run.json").read_text(encoding="utf-8"))
    transform, shape = f1.caspian_grid(hrun)
    static = f1.static_main(a.tag, transform, shape, a.tile)
    if int(static["envelope"].sum()) != int(hrun["envelope"]["cells"]):
        raise SystemExit("the envelope differs from the hindcast run's")
    a7.ENVELOPE = static["envelope"]
    units = a7.stage2_index(f1.INTERIM / "stage2" / a.tag)
    plan = pd.read_csv(f1.PLAN, usecols=["acq", "t", "epoch", "bin"])
    obs = pd.read_csv(f1.RESULTS / "hindcast" / a.tag / "observed.csv")
    obs = obs[(obs["sample"] == "full") & (obs["epoch"] == EPOCH) & (obs["sector"] != "all")]
    fields, rec = {}, {"epoch": EPOCH, "bins": [], "display_res_m": f1.RES * f1.FACTOR,
                       "rule": "D-140: four fully observed E3 bins 0.6 m apart spanning E3; complete = domain valid area "
                               f">= {COMPLETE_FRAC:.0%} of the median over E3 bins (observed.csv)",
                       "completeness": {str(k): v for k, v in comp.items()}, "plan": f1.PLAN.name}
    t0 = time.time()
    for b in BINS:
        level = a7._bin_level(b)
        f1.log(f"{EPOCH} bin {b} ({level:.2f} m): observed extent")
        raw, info = a7.observed_raw(a7.groups_for(units, b, [EPOCH]), transform, shape)
        code = a7.classify(raw, static, a.tile)
        # cross-check: sea-connected water area in the domain equals observed.csv
        sea_km2 = float(((code == 1) & static["domain"]).sum()) * f1.CELL_KM2
        ref = float(obs.loc[obs["bin"] == b, "area_sea_km2"].sum())
        if abs(sea_km2 - ref) > 0.01 * max(ref, 1.0):
            raise SystemExit(f"bin {b}: sea area {sea_km2:.1f} km2 differs from observed.csv {ref:.1f} km2")
        del raw
        st = f1.obs_state(code, static["outside"], static["domain"], a.tile)
        del code
        fields[f"obs_b{b:03d}"] = f1.block_fraction(st)
        del st
        acq = plan[(plan["epoch"] == EPOCH) & (plan["bin"] == b)].drop_duplicates("acq")
        t = pd.to_datetime(acq["t"], format="ISO8601")
        rec["bins"].append({"bin": b, "level_m": round(level, 4), "n_acquisitions": int(len(acq)),
                            "first_year": int(t.min().year), "last_year": int(t.max().year),
                            "first_date": str(t.min().date()), "last_date": str(t.max().date()),
                            "median_date": str(t.median().date()), "sea_km2": round(sea_km2, 1),
                            "observed_csv_sea_km2": round(ref, 1), "envelope_cells": info.get("envelope_cells")})
    r_, c_ = (shape[0] // f1.FACTOR) * f1.FACTOR, (shape[1] // f1.FACTOR) * f1.FACTOR
    ta, _, tc, _, te, tf = (float(v) for v in transform[:6])
    np.savez_compressed(f1.CACHE / f"waterlines_{a.tag}.npz", extent=np.array([tc, tc + ta * c_, tf + te * r_, tf]),
                        **fields)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["hindcast_run"] = f"results/hindcast/{a.tag}/run.json"
    f1.OUT.mkdir(parents=True, exist_ok=True)
    (f1.OUT / "erl_waterlines.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    f1.log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
