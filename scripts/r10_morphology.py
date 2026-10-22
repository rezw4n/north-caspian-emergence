"""Robustness check 10 (PREREG section 9): does seabed change between
epochs explain the hindcast errors? No Earth Engine.

Two parts, from the main run's outputs:
1. The morphology table (robustness.morphology_table) from results/hindcast/<main>/
   displacement.csv and noise_floor.csv: at every level observed in E1 and in E2 or E3,
   per product, variant and sector, the product's displacement against E1 and against
   E2/E3, whether it fits the older seabed better (sign test over levels), and the ratio
   of its E2/E3 displacement to the E1-vs-E2/E3 displacement (the measured seabed
   change at that level).
2. The rising-versus-falling-limb difference map (a by-product of the DEM step): a TIN DEM of the
   E1 contours (scripts/a6_dem.py --epochs E1 on a copied point table, tag <main>__e1)
   minus the main E2 + E3 DEM, where both have a height; summary per sector (median,
   p05, p95, cells) and the difference tiles in data/processed/dem/<main>__e1/diff/.

Usage:
  python scripts/r10_morphology.py --main <tag> [--no-map]
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

from caspian import robustness as R
from caspian.config import INTERIM, PROCESSED, RESULTS

CHECK = "r10_morphology"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def limb_difference(main_tag: str, P) -> pd.DataFrame:
    import rasterio
    r04 = _load("r04_screening")
    e1 = f"{main_tag}__e1"
    p = pd.read_csv(INTERIM / "waterlines" / main_tag / "points.csv.gz")
    d = INTERIM / "waterlines" / e1
    d.mkdir(parents=True, exist_ok=True)
    p.to_csv(d / "points.csv.gz", index=False, float_format="%.6f")
    P.dem_step(e1, epochs="E1")
    out = PROCESSED / "dem" / e1 / "diff"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted((PROCESSED / "dem" / e1).glob("dem_*.tif")):
        g = PROCESSED / "dem" / main_tag / f.name
        if not g.exists():
            continue
        with rasterio.open(f) as a, rasterio.open(g) as b:
            diff = (a.read(1) - b.read(1)).astype(np.float32)
            prof, tr = a.profile, a.transform
        ok = np.isfinite(diff)
        if not ok.any():
            continue
        prof.update(count=1)
        with rasterio.open(out / f.name.replace("dem_", "diff_"), "w", **prof) as w:
            w.write(diff, 1)
            w.set_band_description(1, "E1_minus_E2E3_m")
        rr, cc = np.nonzero(ok)
        sec = r04.point_sectors(tr.c + tr.a * (cc + 0.5), tr.f + tr.e * (rr + 0.5))
        rows.append(pd.DataFrame({"sector": sec, "diff_m": diff[ok]}))
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    return d.groupby("sector")["diff_m"].agg(n_cells="size", median_m="median",
                                             p05_m=lambda s: float(np.percentile(s, 5)),
                                             p95_m=lambda s: float(np.percentile(s, 95))).reset_index()


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", required=True)
    ap.add_argument("--no-map", action="store_true")
    a = ap.parse_args(argv)
    src = RESULTS / "hindcast" / a.main
    det, summ = R.morphology_table(pd.read_csv(src / "displacement.csv"), pd.read_csv(src / "noise_floor.csv"))
    out = RESULTS / "robustness" / CHECK
    out.mkdir(parents=True, exist_ok=True)
    det.to_csv(out / "morphology_detail.csv", index=False)
    summ.to_csv(out / "morphology_summary.csv", index=False)
    print(summ.to_string())
    if not a.no_map:
        s = limb_difference(a.main, _load("r00_pipeline"))
        s.to_csv(out / "limb_difference_by_sector.csv", index=False)
        print(s.to_string())
    (out / "run.json").write_text(json.dumps({"main": a.main, "map": not a.no_map}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
