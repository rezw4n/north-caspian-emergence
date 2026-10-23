"""Second-review analysis D, observation bias from all ICESat-2 blocks (PREREG Amendment 19
item 10). Descriptive, post-review; the validated-sector rule of Amendment 18 item 1 and the
H2 verdict are unchanged. Defaults in D-149.

Per source (ATL08, ATL06), the H2 point set exactly as the registered evaluation built it
(scripts/a6_validate_dem.py evaluate, caspian.validation):
  - the prepared points (data/interim/validation/main/<source>_points.csv.gz) in the H2
    population: inside the emerged band and with an E3 vegetation share below
    VEG_CELL_SHARE (D-084; the cached annotations next to the point table);
  - the 50 km blocks of results/study_area/blocks_50km.csv (D-085): block, sector and
    split of each point; points outside the block table are dropped and counted;
  - the reference on the DEM's datum, y_ref = H_EGM2008 + c, with the registered training
    constant c (results/validation/main/<source>_datum_constant.json; not refitted);
  - the paired set: y_ref, the DEM and all five products finite (as accuracy_tables).
Error e = DEM - y_ref. Check first: on test blocks the sector statistics reproduce the DEM
rows of results/validation/main/<source>_accuracy.csv (the run stops otherwise).

Then over ALL blocks (training, calibration and test), per sector and overall: mean,
median, NMAD, RMSE, n points and n blocks, with block-bootstrap 95% percentile intervals
(blocks of the scope drawn with replacement, B = 2,000, seed 20261023;
caspian.rev2_d.block_bootstrap_stats, the draws of caspian.validation.block_bootstrap). The
number of blocks is fixed under this bootstrap and has no interval. Because c is the median
over training points of all sectors, the pooled training-block median error is near zero by
construction; sector values are relative to that pooled constant.

Outputs: results/revision2/d/obsbias/{obs_bias_by_sector.csv, obs_bias_by_split.csv,
obs_bias_blocks.csv, summary.json}
Usage: python scripts/rev2_d_obsbias.py [ATL08,ATL06]
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import rev2_d, validation
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision2" / "d" / "obsbias"
VAL = RESULTS / "validation" / "main"
SEED = 20261023
N_BOOT = 2000
BLOCK_M = 50_000.0
LIMIT_M = 0.5            # the A18 item 1 threshold, for the descriptive flags only
STATS = ("mean", "median", "nmad", "rmse")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def points(source: str, veg_share_max: float) -> tuple[pd.DataFrame, dict]:
    path = INTERIM / "validation" / "main" / f"{source}_points.csv.gz"
    cols = list(pd.read_csv(path, nrows=0).columns)
    prods = [c for c in cols if c.startswith("prod:")]
    d = pd.read_csv(path, usecols=["x", "y", "h_orth", "dem_h", *prods])
    ann = np.load(INTERIM / "validation" / "main" / f"{source}_annotations.npz")
    if len(ann["in_band"]) != len(d):
        raise SystemExit(f"{source}: the annotations do not match the point table")
    n_prepared = len(d)
    keep = ann["in_band"] & (ann["veg_share"] < veg_share_max)
    d = d.loc[keep].reset_index(drop=True)
    n_bare = len(d)
    d = validation.attach_blocks(d, pd.read_csv(RESULTS / "study_area" / "blocks_50km.csv"), BLOCK_M)
    n_outside = int(d["split"].isna().sum())
    d = d[d["split"].notna()].reset_index(drop=True)
    c = json.loads((VAL / f"{source}_datum_constant.json").read_text(encoding="utf-8"))["constant"]["constant_m"]
    y = d["h_orth"].to_numpy(float) + c
    ok = np.isfinite(y) & np.isfinite(d["dem_h"].to_numpy(float))
    for p in prods:
        ok &= np.isfinite(d[p].to_numpy(float))
    n_unpaired = int((~ok).sum())
    d = d.loc[ok, ["block", "sector", "split"]].assign(e=(d["dem_h"].to_numpy(float) - y)[ok]).reset_index(drop=True)
    return d, {"n_prepared": n_prepared, "n_bare_population": n_bare, "n_outside_block_table": n_outside,
               "n_not_paired": n_unpaired, "n_used": len(d), "datum_constant_m": c, "products": prods}


def check_test_blocks(d: pd.DataFrame, source: str) -> float:
    acc = pd.read_csv(VAL / f"{source}_accuracy.csv")
    acc = acc[acc["model"] == "dem"].set_index("scope")
    te = d[d["split"] == "test"]
    worst = 0.0
    for scope in ["overall"] + SECTOR_ORDER:
        e = te["e"].to_numpy() if scope == "overall" else te.loc[te["sector"] == scope, "e"].to_numpy()
        m = validation.error_metrics(e)
        if int(acc.loc[scope, "n"]) != m["n"]:
            raise SystemExit(f"{source} {scope}: {m['n']} test points, registered {int(acc.loc[scope, 'n'])}")
        if m["n"] == 0:
            continue
        for k in ("bias", "median", "rmse", "nmad"):
            dev = abs(m[k] - float(acc.loc[scope, k]))
            worst = max(worst, dev)
            if dev > 1e-9:
                raise SystemExit(f"{source} {scope} {k}: {m[k]} against registered {acc.loc[scope, k]}")
    return worst


def summarise(e, blocks) -> dict:
    m = validation.error_metrics(e)
    row = {"n_points": m["n"], "n_blocks": int(np.unique(blocks).size), "mean": m["bias"], "median": m["median"],
           "nmad": m["nmad"], "rmse": m["rmse"]}
    boot = rev2_d.block_bootstrap_stats(e, blocks, N_BOOT, SEED)
    for k, name in enumerate(STATS):
        row[f"{name}_lo"], row[f"{name}_hi"] = rev2_d.percentile_interval(boot[:, k])
    for name in ("mean", "median"):
        lo, hi = row[f"{name}_lo"], row[f"{name}_hi"]
        row[f"{name}_within_0p5"] = bool(abs(row[name]) <= LIMIT_M)
        row[f"{name}_ci_inside_0p5"] = bool(lo >= -LIMIT_M and hi <= LIMIT_M)
        row[f"{name}_ci_excludes_0p5_band"] = bool(lo > LIMIT_M or hi < -LIMIT_M)
    return row


def main(argv) -> int:
    sources = argv[0].split(",") if argv else ["ATL08", "ATL06"]
    a6v = _load("a6_validate_dem")
    rows, split_rows, block_rows, summ = [], [], [], {}
    for source in sources:
        t0 = time.time()
        d, info = points(source, a6v.VEG_CELL_SHARE)
        worst = check_test_blocks(d, source)
        info["test_block_check_max_abs_dev_m"] = worst
        print(f"{source}: {info['n_used']} points, test-block statistics reproduced (max dev {worst:.2e} m), "
              f"{time.time() - t0:.0f} s", flush=True)
        for scope in ["overall"] + SECTOR_ORDER:
            g = d if scope == "overall" else d[d["sector"] == scope]
            if g.empty:
                rows.append({"source": source, "sector": scope, "n_points": 0, "n_blocks": 0})
                continue
            r = summarise(g["e"].to_numpy(), g["block"].to_numpy())
            rows.append({"source": source, "sector": scope, **r,
                         **{f"n_blocks_{s}": int(g.loc[g["split"] == s, "block"].nunique()) for s in validation.SPLITS}})
            for s in validation.SPLITS:
                gs = g[g["split"] == s]
                if gs.empty:
                    continue
                m = validation.error_metrics(gs["e"].to_numpy())
                split_rows.append({"source": source, "sector": scope, "split": s, "n_points": m["n"],
                                   "n_blocks": int(gs["block"].nunique()), "mean": m["bias"], "median": m["median"],
                                   "nmad": m["nmad"], "rmse": m["rmse"]})
            print(f"  {source} {scope}: n={r['n_points']} blocks={r['n_blocks']} mean={r['mean']:.3f} "
                  f"[{r['mean_lo']:.3f}, {r['mean_hi']:.3f}] median={r['median']:.3f} "
                  f"[{r['median_lo']:.3f}, {r['median_hi']:.3f}] nmad={r['nmad']:.3f} rmse={r['rmse']:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        for (blk, sec, sp), g in d.groupby(["block", "sector", "split"], sort=True):
            m = validation.error_metrics(g["e"].to_numpy())
            block_rows.append({"source": source, "block": blk, "sector": sec, "split": sp, "n_points": m["n"],
                               "mean": m["bias"], "median": m["median"], "nmad": m["nmad"], "rmse": m["rmse"]})
        summ[source] = info
        del d
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "obs_bias_by_sector.csv", index=False, float_format="%.6g")
    pd.DataFrame(split_rows).to_csv(OUT / "obs_bias_by_split.csv", index=False, float_format="%.6g")
    pd.DataFrame(block_rows).to_csv(OUT / "obs_bias_blocks.csv", index=False, float_format="%.6g")
    (OUT / "summary.json").write_text(json.dumps({
        "rule": "A19 item 10 (D-149): DEM (main) - (H_EGM2008 + c) on the H2 point set over all 50 km blocks; "
                f"block bootstrap B = {N_BOOT}, seed {SEED}, 95% percentile intervals; descriptive",
        "sources": summ}, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
