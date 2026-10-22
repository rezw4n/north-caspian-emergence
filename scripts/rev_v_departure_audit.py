"""Revision, workstream V: the H1 audit with each sector's levels shifted by its SWOT
departure from DAHITI (PREREG Amendment 18 item 9, first bullet; D-120).

run:     the primary run's machinery (scripts/a7_hindcast.py: static layers, record-low
         envelope, units, cached corrected heights and cached flood-level maps; no new
         flood fill). A cell of sector k is predicted water at bin level L when
         F < L + d_k (caspian.watersurface.ShiftedFloodMap). d_k = departure_m of
         results/revision/v/sector_departures.csv; a sector without a value takes the
         median of the reported sector departures (no relative shift; D-120). The D-070
         shape constant is refitted on the shifted maps; both variants are computed.
         Output: results/revision/v/hindcast_departure/ (the a7_hindcast tables), then
         the H1 rule (scripts/a7_h1_decision.py) on that directory.
compare: verdicts, evaluable and exceeding levels, and sector medians against the
         primary (results/hindcast/main), per product, variant and sector:
         results/revision/v/departure_audit_comparison.csv and .json.

Usage: python scripts/rev_v_departure_audit.py run --vertical-correction <npz>
       python scripts/rev_v_departure_audit.py compare
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

from caspian import products
from caspian import watersurface as ws
from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision" / "v"
TAG = "hindcast_departure"
RUN_DIR = OUT / TAG
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]   # A18 item 1


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def departures(path=OUT / "sector_departures.csv") -> tuple[dict, dict]:
    """{sector: d (m)} for the rerun, and a record of where each value comes from."""
    t = pd.read_csv(path).set_index("sector")
    meas = {s: float(t.loc[s, "departure_m"]) for s in SECTOR_ORDER
            if s in t.index and np.isfinite(t.loc[s, "departure_m"])}
    fill = float(np.median(list(meas.values())))
    d, rec = {}, {}
    for s in SECTOR_ORDER:
        if s in meas:
            d[s], rec[s] = meas[s], "measured (sector_departures.csv)"
        else:
            d[s], rec[s] = fill, "no value: median of the measured sector departures (D-120 filler)"
    return d, rec


def run(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical-correction", required=True)
    a = ap.parse_args(argv)
    a7 = _load("a7_hindcast")
    main_dir = RESULTS / "hindcast" / "main"
    run_rec = json.loads((main_dir / "run.json").read_text(encoding="utf-8"))
    names, tile, resampling = list(run_rec["products"]), int(run_rec["tile"]), run_rec["resampling"]
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    static = a7.production_static(transform, shape, tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    blocks = pd.read_csv(a7.BLOCKS)
    fit_region = static["domain"] & a7.train_mask(blocks, transform, shape)
    corr, corr_rec = products.correction(a.vertical_correction, transform, shape)
    if corr_rec["sha256"] != run_rec["vertical_correction"]["sha256"]:
        raise SystemExit("--vertical-correction is not the primary run's correction (sha256 differs)")
    d, d_rec = departures()
    lut = ws.departure_lut(d, SECTOR_ORDER)
    heights, shifted = {}, {}
    cache = INTERIM / "hindcast" / "flood"
    for p in names:
        raw = products.heights(p, transform, shape, resampling)
        heights[p] = products.Corrected(raw, corr)
        before = set(cache.glob(f"{p}_*.npy"))
        F = a7.flood_map_cached(p, heights[p], static["sea_seed"],
                                {"heights": Path(raw.filename).name, "correction": corr_rec["sha256"]}, tile)
        if set(cache.glob(f"{p}_*.npy")) != before:
            raise SystemExit(f"{p}: the primary flood-level map was not in the cache (a new one was built)")
        shifted[p] = ws.ShiftedFloodMap(F, static["sector"], lut)
        print("shifted flood-level map", p, flush=True)
    stage2 = INTERIM / "stage2" / "main"
    a7.ENVELOPE = a7.record_low_water(a7.stage2_index(stage2), transform, shape)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"tag": TAG, "variant": "A18 item 9: sector levels shifted by the SWOT departure (D-120)",
            "resampling": resampling, "tile": tile, "envelope": run_rec.get("envelope"),
            "vertical_correction": corr_rec, "references": run_rec.get("references"),
            "sector_departures_m": d, "sector_departure_sources": d_rec,
            "note": "flood level of sector k at bin level L is L + d_k (threshold of the primary's cached flood-level "
                    "maps); the shape constant is refitted (D-070) on the shifted maps"}
    a7.run(stage2, RUN_DIR, static, heights, transform, shape_offsets="fit", fit_region=fit_region,
           flood_maps=shifted, blocks=blocks, tile=tile, meta=meta)
    # the H1 rule exactly as for the primary, on this directory
    _load("a7_h1_decision").main("../revision/v/" + TAG)
    return 0


def compare(argv) -> int:
    prim, dep = RESULTS / "hindcast" / "main", RUN_DIR
    a, b = pd.read_csv(prim / "h1_mispredicts.csv"), pd.read_csv(dep / "h1_mispredicts.csv")
    key = ["product", "variant", "sector"]
    cols = [c for c in a.columns if c not in key]
    t = a.merge(b, on=key, suffixes=("_primary", "_departure"))
    t["verdict_changed"] = t["mispredicts_primary"].astype(str) != t["mispredicts_departure"].astype(str)
    sm = _load("rev_v_planar").sector_medians
    for name, d in (("primary", prim), ("departure", dep)):
        parts = [sm(d, v).assign(variant=v) for v in ("shape", "total")]
        med = pd.concat(parts, ignore_index=True).rename(columns={c: f"{c}_{name}" for c in
                                                                  ("sector_median_m", "floor_p95_m", "dq_median")})
        t = t.merge(med, on=key, how="left")
    t.to_csv(OUT / "departure_audit_comparison.csv", index=False)
    so_p = json.loads((prim / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    so_d = json.loads((dep / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    run_d = json.loads((dep / "run.json").read_text(encoding="utf-8"))
    out = {"departures_m": run_d.get("sector_departures_m"), "sources": run_d.get("sector_departure_sources"),
           "shape_offsets_m": {p: {"primary": so_p[p]["offset_m"], "departure": so_d[p]["offset_m"]} for p in so_p},
           "columns_compared": cols, "by_variant": {}}
    for v, g in t.groupby("variant"):
        rv = {}
        for p, gp in g.groupby("product"):
            mp = sorted(gp.loc[gp["mispredicts_primary"] == True, "sector"])        # noqa: E712
            md = sorted(gp.loc[gp["mispredicts_departure"] == True, "sector"])      # noqa: E712
            rv[p] = {"n_mispredicted_primary": len(mp), "n_mispredicted_departure": len(md),
                     "n_validated_primary": len(set(mp) & set(VALIDATED)),
                     "n_validated_departure": len(set(md) & set(VALIDATED)),
                     "changed": sorted(gp.loc[gp["verdict_changed"], "sector"]),
                     "mispredicted_departure": md}
        out["by_variant"][v] = rv
    dec_d = json.loads((dep / "h1_decision.json").read_text(encoding="utf-8"))
    dec_p = json.loads((prim / "h1_decision.json").read_text(encoding="utf-8"))
    out["block_test"] = {"primary": dec_p.get("block_test"), "departure": dec_d.get("block_test")}
    (OUT / "departure_audit_comparison.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out["by_variant"], indent=1))
    print(json.dumps(out["shape_offsets_m"], indent=1))
    return 0


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    sys.exit({"run": run, "compare": compare}[cmd](rest))
