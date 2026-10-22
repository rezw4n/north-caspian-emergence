"""Revision analysis S: analyses that need the cached rasters (PREREG Amendment 18
item 4). No Earth Engine.

Reuses scripts/a7_hindcast.py unchanged (static layers, record-low envelope, cached
flood-level maps, observed extents with their sea connection, waterline distances and
cell counts) with the main run's shape constants, and computes per (epoch, level) unit:

1. Out-of-sample area bias (A18 item 4): for E3 units, the known cells (the area fit's
   definition, a7_hindcast.unit_counts) predicted and observed as sea, per region code
   (train or calibration/test blocks x validated or other sectors, inside the domain),
   at the all-epoch constant and at the constant fitted on E1 + E2 training-block units
   (shape_offsets.json -> leave_one_epoch_out.E3).
2. Overlap in the emerged band (A18 item 4): extent counts and IoU/F1 (with
   the known outside states) in the emerged band, per sector, over the domain band,
   the validated-sector band; and the area counts over the domain in the validated
   sectors (per-level area bias there) and over the domain (reproduction check).
3. Held-out displacement: the waterline distances whose cells lie in test blocks,
   per unit and sector, and pooled over the epochs of each level.
4. Floor without between-epoch distances: at the bins crossed by E1, the pooled
   per-level floor of the parity halves only, beside the registered floor rebuilt with the
   E1-vs-(E2 + E3) distances.

Reproduction checks (the run stops on a mismatch): per-unit all-cell displacement
medians per sector against displacement.csv; domain area-bias counts against
extent.csv; the rebuilt registered floor at E1 bins against noise_floor.csv.

Outputs: results/revision/s/rasters/{oos_area,band_overlap,test_displacement,
floor_e1_bins}.csv, checks.json and run.json.
Usage: python scripts/rev_s_rasters.py [--tag main] [--bins 12,38] [--out-suffix _pilot]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, products, rev_s
from caspian import hindcast_tiled as ht
from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a7 = _load("a7_hindcast")
ss = _load("a7_sector_shape")

VALIDATED = ("west", "interfluve", "northeast", "open_shelf")     # A18 item 1
OUT = RESULTS / "revision" / "s" / "rasters"
REGCODES = {1: "train_validated", 2: "train_other", 3: "calibtest_validated", 4: "calibtest_other",
            5: "no_block"}


def split_lut(blocks: pd.DataFrame, transform, shape, splits, block_m: float = design.BLOCK_SIZE_M):
    """(lut, bi, bj, i0, j0): lut[j - j0, i - i0] is True for blocks whose split is in
    ``splits``; bi per column and bj per row are the cells' block indices (as
    a7_hindcast.train_mask)."""
    bi, bj = a7.block_index(transform, shape, block_m)
    i0, j0 = int(bi.min()), int(bj.min())
    lut = np.zeros((int(bj.max()) - j0 + 1, int(bi.max()) - i0 + 1), bool)
    t = blocks[blocks["split"].isin(list(splits))]
    ii, jj = t["i"].to_numpy() - i0, t["j"].to_numpy() - j0
    ok = (ii >= 0) & (ii < lut.shape[1]) & (jj >= 0) & (jj < lut.shape[0])
    lut[jj[ok], ii[ok]] = True
    return lut, bi, bj, i0, j0


def split_mask(blocks, transform, shape, splits) -> np.ndarray:
    lut, bi, bj, i0, j0 = split_lut(blocks, transform, shape, splits)
    return lut[(bj - j0)[:, None], (bi - i0)[None, :]]


def same(a, b, tol: float = 1e-6) -> bool:
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(b))


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--bins", default="", help="comma-separated bins (pilot); default every bin")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args(argv)
    t_start = time.time()
    main_dir = RESULTS / "hindcast" / a.tag
    run_rec = json.loads((main_dir / "run.json").read_text(encoding="utf-8"))
    so = json.loads((main_dir / "shape_offsets.json").read_text(encoding="utf-8"))
    names, tile, resampling = list(run_rec["products"]), int(run_rec["tile"]), run_rec["resampling"]
    c_all = {p: float(so["used_in_this_run"][p]["offset_m"]) for p in names}
    c_e12 = {p: float(so["products"][p]["leave_one_epoch_out"]["E3"]["offset_m"]) for p in names}
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    if [float(v) for v in transform[:6]] != run_rec["grid"]["transform"] or list(shape) != run_rec["grid"]["shape"]:
        raise SystemExit("grid differs from the main run")
    static = a7.production_static(transform, shape, tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    if static["counts"] != {k: v for k, v in run_rec["static_counts"].items()}:
        raise SystemExit(f"static layers differ from the main run: {static['counts']} vs {run_rec['static_counts']}")
    print("static layers", static["counts"], f"{time.time() - t_start:.0f} s", flush=True)
    sector, outside, domain = static["sector"], static["outside"], static["domain"]
    blocks = pd.read_csv(a7.BLOCKS)
    train = split_mask(blocks, transform, shape, ["train"])
    calibtest = split_mask(blocks, transform, shape, ["calib", "test"])
    tlut, tbi, tbj, ti0, tj0 = split_lut(blocks, transform, shape, ["test"])

    def keep_test(rows, cols):
        return tlut[tbj[np.asarray(rows)] - tj0, tbi[np.asarray(cols)] - ti0]

    sid = {nm: k + 1 for k, nm in enumerate(SECTOR_ORDER)}
    val = np.isin(sector, [sid[x] for x in VALIDATED])
    regcode = np.zeros(shape, np.uint8)
    regcode[domain] = 5                       # domain cells in no listed block (kept for the check)
    regcode[domain & train & val] = 1
    regcode[domain & train & ~val] = 2
    regcode[domain & calibtest & val] = 3
    regcode[domain & calibtest & ~val] = 4
    del train, calibtest
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band250 = ((ex["high"] == 1) & (ex["low"] == 0)).astype(np.int8)
    band = (a7.nearest(band250, ex["transform"], transform, shape, 0) > 0) & domain
    labels_band = np.where(band, sector, 0).astype(np.uint8)
    regions = {"band_domain": band, "band_validated": band & val, "domain_validated": domain & val, "domain": domain}
    print("band cells", int(band.sum()), "validated", int((band & val).sum()), flush=True)

    corr_path = RESULTS / "study_area" / run_rec["vertical_correction"]["file"]
    corr, corr_rec = products.correction(corr_path, transform, shape)
    if corr_rec["sha256"] != run_rec["vertical_correction"]["sha256"]:
        raise SystemExit("vertical correction differs from the main run (sha256)")
    heights, flood_maps = {}, {}
    for p in names:
        h = products.heights(p, transform, shape, resampling)
        heights[p] = products.Corrected(h, corr)
        flood_maps[p] = a7.flood_map_cached(p, heights[p], static["sea_seed"],
                                            {"heights": Path(h.filename).name, "correction": corr_rec["sha256"]}, tile)
    finite = {p: (lambda hh: lambda r0, r1, c0, c1: np.isfinite(np.asarray(hh[r0:r1, c0:c1], float)))(hh)
              for p, hh in heights.items()}
    stage2 = INTERIM / "stage2" / a.tag
    units = a7.stage2_index(stage2)
    a7.ENVELOPE = a7.record_low_water(units, transform, shape)
    if int(a7.ENVELOPE.sum()) != int(run_rec["envelope"]["cells"]):
        raise SystemExit("record-low envelope differs from the main run")
    print("envelope cells", int(a7.ENVELOPE.sum()), f"{time.time() - t_start:.0f} s", flush=True)

    disp_reg = pd.read_csv(main_dir / "displacement.csv")
    disp_reg = disp_reg[disp_reg["epoch"].isin(["E1", "E2", "E3"])].set_index(["product", "variant", "epoch", "bin", "sector"])
    ext_reg = pd.read_csv(main_dir / "extent.csv")
    ext_reg = ext_reg[ext_reg["sector"] == "domain"].set_index(["product", "variant", "epoch", "bin"])
    nf_reg = pd.read_csv(main_dir / "noise_floor.csv")
    nf_reg = nf_reg[nf_reg["kind"] == "pooled_level"]
    nf_reg = nf_reg.assign(bin=nf_reg["bin"].astype(int)).set_index(["bin", "sector"])

    bins = [int(x) for x in a.bins.split(",") if x] or sorted(units)
    out = OUT.with_name(OUT.name + a.out_suffix) if a.out_suffix else OUT
    out.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("oos_area", "band_overlap", "test_displacement", "floor_e1_bins")}
    checks = {"displacement_compared": 0, "displacement_mismatch": [], "domain_counts_compared": 0,
              "domain_counts_mismatch": [], "oos_total_compared": 0, "oos_total_mismatch": [],
              "floor_compared": 0, "floor_mismatch": []}
    ids = list(range(1, len(SECTOR_ORDER) + 1))
    name = {k + 1: nm for k, nm in enumerate(SECTOR_ORDER)}
    variants = [(p, "total", 0.0) for p in names] + [(p, "shape", c_all[p]) for p in names]
    cidx = {p: (int(np.flatnonzero(np.isclose(a7.C_GRID, c_all[p]))[0]),
                int(np.flatnonzero(np.isclose(a7.C_GRID, c_e12[p]))[0])) for p in names}

    def write():
        for k, v in rows.items():
            pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
        (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str), encoding="utf-8")

    for b in bins:
        t_bin = time.time()
        level = a7._bin_level(b)
        epochs = sorted(units[b])
        base = {"bin": b, "level_m": round(level, 4)}
        full = {}
        for ep in epochs:
            raw, _ = a7.observed_raw(a7.groups_for(units, b, [ep]), transform, shape)
            full[ep] = a7.classify(raw, static, tile)
            del raw
        # 1. out-of-sample area bias on E3 units
        if "E3" in epochs:
            T = products.flood_level(level, 0.0, 0.0, a7.C_GRID)
            for p in names:
                cnt, water = ss.unit_counts_by_sector(flood_maps[p], full["E3"], outside, finite[p], domain, regcode,
                                                      5, T, tile=tile)
                for code, nm in REGCODES.items():
                    rows["oos_area"].append({"product": p, "epoch": "E3", **base, "region": nm,
                                             "c_all_m": c_all[p], "c_e12_m": c_e12[p],
                                             "cells_pred_c_all": int(cnt[code, cidx[p][0]]),
                                             "cells_pred_c_e12": int(cnt[code, cidx[p][1]]),
                                             "cells_water": int(water[code])})
                tot = int(cnt[1:5, cidx[p][0]].sum() - water[1:5].sum())
                tot_d = int(cnt[1:6, cidx[p][0]].sum() - water[1:6].sum())      # every domain cell
                reg = ext_reg.loc[(p, "shape", "E3", b)]
                want = (reg["n_tp"] + reg["n_fp"] + reg["n_fp_out"]) - (reg["n_tp"] + reg["n_fn"] + reg["n_fn_out"])
                checks["oos_total_compared"] += 1
                if tot_d != int(want):
                    checks["oos_total_mismatch"].append({"product": p, "bin": b, "mine": tot_d, "registered": int(want)})
                rows["oos_area"].append({"product": p, "epoch": "E3", **base, "region": "domain_all_blocks_check",
                                         "c_all_m": c_all[p], "c_e12_m": c_e12[p], "cells_pred_c_all": None,
                                         "cells_pred_c_e12": None, "cells_water": None,
                                         "bias_cells_regions_1to4": tot, "bias_cells_domain": tot_d})
        # 2 and 3. band overlap and held-out displacement
        for prod, var, c in variants:
            fl = float(products.flood_level(level, 0.0, 0.0, c))
            pred = ht.flooded(flood_maps[prod], fl)
            pool = a7.Pool()
            for ep in epochs:
                obs, valid = full[ep], a7._valid(full[ep])
                cnt = ht.extent_counts(pred, obs, valid, labels_band, ids, regions=regions, pred_valid=finite[prod],
                                       outside_state=outside, shape=shape, tile=tile)
                for k, cc in cnt.items():
                    m = ht.metrics_from_counts(cc, a7.CELL_KM2)
                    scope = f"band_{name[k]}" if isinstance(k, int) else k
                    rows["band_overlap"].append({"product": prod, "variant": var, "epoch": ep, **base, "scope": scope,
                                                 **{f"n_{x}": v for x, v in cc.items()},
                                                 "area_obs_km2": m["area_obs"], "area_pred_km2": m["area_pred"],
                                                 "area_bias_rel_total": m["area_bias_rel_total"],
                                                 "iou_total": m["iou_total"], "f1_total": m["f1_total"]})
                reg = ext_reg.loc[(prod, var, ep, b)]
                checks["domain_counts_compared"] += 1
                if any(int(cnt["domain"][x]) != int(reg[f"n_{x}"]) for x in ht.COUNT_KEYS):
                    checks["domain_counts_mismatch"].append({"product": prod, "variant": var, "epoch": ep, "bin": b})
                cells = ht.displacement_cells(pred, obs, valid, sector, a7.RES, pred_valid=finite[prod],
                                              outside_state=outside, shape=shape, tile=tile)
                for k, res in ht.summarise(cells, ids).items():
                    key = (prod, var, ep, b, name[k])
                    checks["displacement_compared"] += 1
                    if key in disp_reg.index:
                        r = disp_reg.loc[key]
                        if not (same(res["median"], r["median_m"]) and int(res["n"]) == int(r["n"])):
                            checks["displacement_mismatch"].append({"key": key, "mine": res["median"], "n": res.get("n"),
                                                                    "registered": r["median_m"], "n_reg": r["n"]})
                    else:
                        checks["displacement_mismatch"].append({"key": key, "missing_registered_row": True})
                tcells = rev_s.mask_labels(cells, keep_test)
                for k, res in ht.summarise(tcells, ids, return_distances=True).items():
                    rows["test_displacement"].append({"product": prod, "variant": var, "epoch": ep, **base,
                                                      "sector": name[k], **a7._disp_row(res)})
                    pool.add(k, res)
                del cells, tcells
            for k in ids:
                rows["test_displacement"].append({"product": prod, "variant": var, "epoch": "pooled",
                                                  "epochs": "+".join(epochs), **base, "sector": name[k],
                                                  **a7._disp_row(pool.summary(k))})
        # 4. floors at bins crossed by E1
        if "E1" in epochs:
            halves, regd = a7.Pool(), a7.Pool()
            for ep in epochs:
                if set(units[b][ep]) != {0, 1}:
                    continue
                hv = []
                for par in (0, 1):
                    raw, _ = a7.observed_raw(a7.groups_for(units, b, [ep], (par,)), transform, shape)
                    hv.append(a7.classify(raw, static, tile))
                    del raw
                cells = ht.noise_floor_cells(hv[0], a7._valid(hv[0]), hv[1], a7._valid(hv[1]), sector, a7.RES,
                                             outside_state=outside, shape=shape, tile=tile)
                for k, res in ht.summarise(cells, ids, return_distances=True).items():
                    halves.add(k, res)
                    regd.add(k, res)
                del hv, cells
            e23 = [ep for ep in epochs if ep in ("E2", "E3")]
            n_between = {}
            if e23:
                raw, _ = a7.observed_raw(a7.groups_for(units, b, e23), transform, shape)
                later = a7.classify(raw, static, tile)
                del raw
                cells = ht.noise_floor_cells(full["E1"], a7._valid(full["E1"]), later, a7._valid(later), sector, a7.RES,
                                             outside_state=outside, shape=shape, tile=tile)
                for k, res in ht.summarise(cells, ids, return_distances=True).items():
                    regd.add(k, res)
                    n_between[k] = int(res["n"])
                del later, cells
            for k in ids:
                hs = halves.summary(k) if k in halves.d else None
                rs = regd.summary(k) if k in regd.d else None
                row = {**base, "sector": name[k], "epochs": "+".join(epochs),
                       "halves_median_m": hs["median"] if hs else np.nan, "halves_p95_m": hs["p95"] if hs else np.nan,
                       "halves_n": hs["n"] if hs else 0,
                       "registered_rebuilt_p95_m": rs["p95"] if rs else np.nan, "registered_rebuilt_n": rs["n"] if rs else 0,
                       "between_epoch_n": n_between.get(k, 0)}
                rows["floor_e1_bins"].append(row)
                if (b, name[k]) in nf_reg.index:
                    r = nf_reg.loc[(b, name[k])]
                    checks["floor_compared"] += 1
                    if not same(row["registered_rebuilt_p95_m"], r["p95_m"]):
                        checks["floor_mismatch"].append({"bin": b, "sector": name[k], "mine": row["registered_rebuilt_p95_m"],
                                                         "registered": r["p95_m"]})
        del full
        write()
        bad = {k: v for k, v in checks.items() if k.endswith("mismatch") and v}
        print(f"bin {b} ({level:.2f} m) epochs {epochs}: {time.time() - t_bin:.0f} s, total {time.time() - t_start:.0f} s; "
              f"checks {checks['displacement_compared']}/{checks['domain_counts_compared']}/{checks['oos_total_compared']}/"
              f"{checks['floor_compared']}", flush=True)
        if bad:
            write()
            raise SystemExit(f"reproduction check failed at bin {b}: {json.dumps(bad, default=str)[:2000]}")
    rec = {"tag": a.tag, "bins": bins, "c_all_m": c_all, "c_e12_m": c_e12, "validated_sectors": list(VALIDATED),
           "band": "D-071: extents_v6_250m.npz high == 1 and low == 0, nearest onto the 60 m grid, inside the domain",
           "band_cells": int(band.sum()), "regions": REGCODES, "envelope_cells": int(a7.ENVELOPE.sum()),
           "test_blocks": int((blocks["split"] == "test").sum()), "seconds": round(time.time() - t_start, 1),
           "decisions": "D-114"}
    (out / "run.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    write()
    print("done", json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in checks.items()}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
