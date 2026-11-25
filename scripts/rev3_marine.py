"""Journal revision, PREREG Amendment 20 items 1 (a-c) and 2 (b) (D-161): the audit on the
marine domain, and a view by 25 km block. Exploratory, post-review; the registered verdicts
and Eq. (rule) are unchanged.

The marine domain is the study domain less the sectors registered as deltaic
(results/levels/sector_labels.csv: the Volga delta and the Ural delta).

fit    Item 1a. Each grid's D-070 area constant refitted on the training-block cells of the
       marine domain, by exact counts on the cached per-unit masks of the fit region
       (data/interim/revision_v/fit_masks.npz, D-121; rev_v_planar.exact_bias). First, the
       same counts on the full fit region must reproduce the primary constant and objective.
       results/revision3/marine/marine_constants.json.
pass   Items 1b, 1c and 2b. One pass over the level bins of the main stage-2 counts,
       rebuilding the observed extents exactly as scripts/a7_hindcast.py does (observed_raw,
       the record-low envelope, classify), with the waterline engine of caspian.rev_r (as
       scripts/rev2_a_audit.py). Per grid:
         shape    the primary shape variant (c_area); the replica
         marine   the shape variant at c_marine (skipped, and recorded as identical, when
                  c_marine equals c_area)
       and the registered floor (halves; E1 against E2 + E3). Beside them:
         - cell counts by sector inside the domain (hindcast_tiled.extent_counts) at the
           thresholds of the grid as distributed (c = 0), at c_area and at c_marine;
         - per 25 km block, the pooled absolute distances of the shape variant and of the
           floor (caspian.rev3.BlockDist).
       results/revision3/marine/audit/<out>/{levels_D.csv.gz, levels_Q.csv, extent_counts.csv,
       blocks_D.csv, blocks_Q.csv, run.json}.
decide Checks the replica against results/hindcast/main (per-level medians and floors,
       verdicts, domain area bias, block medians), then applies caspian.h1.mispredicts to
       both configurations and writes the comparison tables.
       results/revision3/marine/{verdicts.csv, verdict_changes.csv, sector_medians.csv,
       area_bias_units.csv, area_bias.csv, blocks.csv, block_shares.csv, summary.json}.

Usage:
  python scripts/rev3_marine.py fit
  python scripts/rev3_marine.py pass [--bins 12,38] [--out main]
  python scripts/rev3_marine.py decide [--out main]
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

from caspian import design, h1, products, rev3, rev_r, rev_s
from caspian import hindcast_tiled as ht
from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a7 = _load("a7_hindcast")

TAG = "main"
TILE = 2048
RES = a7.RES
MAIN = RESULTS / "hindcast" / TAG
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
OUT = RESULTS / "revision3" / "marine"
MASKS = INTERIM / "revision_v" / "fit_masks.npz"
LABELS = RESULTS / "levels" / "sector_labels.csv"
IDS = list(range(1, len(SECTOR_ORDER) + 1))
NAME = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}
CHECKED = ("west", "interfluve", "northeast", "open_shelf")        # Amendment 18 item 1
BLOCK_M = float(design.BLOCK_SIZE_M)
MIN_BLOCK_N = h1.MIN_BLOCK_N                                        # the block test's minimum
TOL = 1e-6


def deltaic_sectors() -> list[str]:
    lab = pd.read_csv(LABELS)
    return [s for s in SECTOR_ORDER if bool(lab.set_index("sector").loc[s, "deltaic"])]


def grid_setup():
    w, s, e, n = DOMAIN.bounds
    return a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))


def _main_run():
    return json.loads((MAIN / "run.json").read_text(encoding="utf-8"))


def _c_area() -> dict:
    so = json.loads((MAIN / "shape_offsets.json").read_text(encoding="utf-8"))
    return {p: float(v["offset_m"]) for p, v in so["used_in_this_run"].items()}


def _static(transform, shape):
    static = a7.production_static(transform, shape, TILE)
    for k in ("zone", "land_seed"):
        static.pop(k)
    return static


def _finite(h, shape) -> np.ndarray:
    fin = np.zeros(shape, bool)
    for r0, r1, c0, c1 in ht.tiles(shape, TILE):
        fin[r0:r1, c0:c1] = np.isfinite(np.asarray(h[r0:r1, c0:c1], np.float32))
    return fin


def primary_flood_map(p, transform, shape, seed, corr, corr_rec):
    """The primary run's corrected heights and their cached flood-level map (never rebuilt)."""
    h = products.heights(p, transform, shape, _main_run()["resampling"])
    hc = products.Corrected(h, corr)
    cache = INTERIM / "hindcast" / "flood"
    before = set(cache.glob(f"{p}_*.npy"))
    F = a7.flood_map_cached(p, hc, seed, {"heights": Path(h.filename).name, "correction": corr_rec["sha256"]}, TILE)
    if set(cache.glob(f"{p}_*.npy")) != before:
        raise SystemExit(f"{p}: the primary flood-level map was not in the cache (a new one was built)")
    return h, hc, F


def _correction(transform, shape):
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != _main_run()["vertical_correction"]["sha256"]:
        raise SystemExit("the correction is not the primary run's")
    return corr, corr_rec


# --- fit (item 1a) --------------------------------------------------------------------------------

def cmd_fit(a) -> int:
    t0 = time.time()
    planar = _load("rev_v_planar")
    transform, shape = grid_setup()
    static = _static(transform, shape)
    fd = planar.FitData(MASKS)
    prim = json.loads((MAIN / "shape_offsets.json").read_text(encoding="utf-8"))
    if [(u["bin"], u["epoch"]) for u in prim["units"]] != list(zip(fd.bins.tolist(), fd.epochs)):
        raise SystemExit("the fit masks' units differ from the primary area fit")
    if fd.n != prim["fit_region_cells"] or fd.shape != tuple(shape):
        raise SystemExit("the fit masks' region or grid differs from the primary")
    corr, corr_rec = _correction(transform, shape)
    flat_corr = np.asarray(corr).reshape(-1)
    delt = deltaic_sectors()
    codes = rev3.marine_codes(SECTOR_ORDER, delt)
    sec_fit = np.asarray(static["sector"])[fd.rows, fd.cols]
    marine = np.isin(sec_fit, codes)
    n_by_sector = {NAME[k]: int((sec_fit == k).sum()) for k in IDS}
    rec = {"rule": "D-070 (0.01 m grid over +-5 m, the 43 units, ties to the smallest |c| then the smaller c) on "
                   "the known training-block cells of the marine domain: the domain less the deltaic sectors",
           "deltaic_sectors": delt, "marine_sectors": [NAME[k] for k in codes],
           "fit_region_cells": fd.n, "fit_region_cells_marine": int(marine.sum()),
           "fit_region_cells_by_sector": n_by_sector, "products": {}}
    for p in _main_run()["products"]:
        t1 = time.time()
        h, hc, F = primary_flood_map(p, transform, shape, static["sea_seed"], corr, corr_rec)
        flat_h = np.asarray(h).reshape(-1)
        fin = np.zeros(fd.n, bool)
        for s0 in range(0, fd.n, 1 << 24):
            j = fd.idx[s0:s0 + (1 << 24)]
            fin[s0:s0 + j.size] = np.isfinite(flat_h[j].astype(np.float32) + flat_corr[j].astype(np.float32))
        Fv = np.asarray(F).reshape(-1)[fd.idx]
        b0 = planar.exact_bias(fd, Fv, fin)
        r0 = a7.fit_offset(b0, fd.epochs)
        want = prim["products"][p]
        if r0["offset_m"] != want["offset_m"] or abs(r0["objective_km2_at_offset"] - want["objective_km2_at_offset"]) > 1e-3:
            raise SystemExit(f"{p}: exact counts give c={r0['offset_m']} obj={r0['objective_km2_at_offset']}, "
                             f"primary c={want['offset_m']} obj={want['objective_km2_at_offset']}")
        b1 = planar.exact_bias(fd, Fv, fin & marine)
        r1 = a7.fit_offset(b1, fd.epochs)
        k0 = int(np.flatnonzero(np.isclose(a7.C_GRID, want["offset_m"]))[0])
        rec["products"][p] = {
            "c_area_primary_m": want["offset_m"], "objective_primary_km2": want["objective_km2_at_offset"],
            "primary_reproduced": True,
            "c_marine_m": r1["offset_m"], "objective_marine_km2": r1["objective_km2_at_offset"],
            "objective_marine_at_c_area_km2": float(r1["objective_km2"][k0]),
            "n_tied_minima": r1["n_tied_minima"], "c_change_m": round(r1["offset_m"] - want["offset_m"], 2),
            "leave_one_epoch_out_m": {e: a7.fit_offset(b1, fd.epochs, exclude=e)["offset_m"]
                                      for e in sorted(set(fd.epochs))},
            "seconds": round(time.time() - t1, 1)}
        print(json.dumps({p: rec["products"][p]}), flush=True)
        del b0, b1, Fv, F
    OUT.mkdir(parents=True, exist_ok=True)
    rec["seconds"] = round(time.time() - t0, 1)
    (OUT / "marine_constants.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


# --- the pass (items 1b, 1c, 2b) --------------------------------------------------------------------

class Acc:
    """Signed distances and counts per key and sector, pooled over the epochs of a bin
    (as scripts/rev2_a_audit.Acc)."""

    def __init__(self):
        self.d = {}

    def add(self, key, res: dict):
        for k in IDS:
            r = res[k]
            e = self.d.setdefault((key, k), {"dist": [], "n_pred": 0, "n_obs": 0, "n_pred_unknown": 0,
                                              "n_obs_unknown": 0, "observed_level": False})
            e["dist"].append(np.asarray(r.get("distances", np.empty(0)), float))
            for f in ("n_pred", "n_obs", "n_pred_unknown", "n_obs_unknown"):
                e[f] += int(r[f])
            e["observed_level"] = e["observed_level"] or bool(r["observed_level"])

    def rows(self, b: int, level: float):
        out = []
        for (key, k), e in self.d.items():
            d = [x[~np.isnan(x)] for x in e["dist"]]
            p = ht.pooled(d, e["n_pred"], e["n_obs"], e["n_pred_unknown"], e["n_obs_unknown"], e["observed_level"])
            out.append({**dict(key), "bin": b, "level_m": round(level, 4), "sector": NAME[k], "n": p["n"],
                        "median_m": p["median"], "p95_m": p["p95"], "signed_median_m": p["signed_median"],
                        "n_pred": p["n_pred"], "n_obs": p["n_obs"], "n_pred_unknown": p["n_pred_unknown"],
                        "n_obs_unknown": p["n_obs_unknown"], "observed_level": p["observed_level"]})
        return out


def _summ(cells):
    return ht.summarise(cells, IDS, return_distances=True)


def cmd_pass(a) -> int:
    t0 = time.time()
    out = OUT / "audit" / a.out
    out.mkdir(parents=True, exist_ok=True)
    transform, shape = grid_setup()
    static = _static(transform, shape)
    outside, sector, seed, domain = static["outside"], static["sector"], static["sea_seed"], static["domain"]
    lab_in = np.where(np.asarray(domain, bool), np.asarray(sector), 0).astype(np.uint8)
    units = a7.stage2_index(INTERIM / "stage2" / TAG)
    env = a7.record_low_water(units, transform, shape)
    c_area = _c_area()
    mc = json.loads((OUT / "marine_constants.json").read_text(encoding="utf-8"))
    c_marine = {p: float(v["c_marine_m"]) for p, v in mc["products"].items()}
    names = list(_main_run()["products"])
    corr, corr_rec = _correction(transform, shape)
    fmaps, Fmap, fin_of = {}, {}, {}
    for p in names:
        h, hc, F = primary_flood_map(p, transform, shape, seed, corr, corr_rec)
        fin_of[p] = _finite(hc, shape)
        Fmap[p] = F
        fmaps[p] = rev_r.FloodMap(F, fin_of[p], tile=256)
        print(f"flood map {p} ready, {time.time() - t0:.0f} s", flush=True)
    configs = []          # (config, product, threshold as a function of the bin level)
    for p in names:
        configs.append(("shape", p, lambda L, p=p: float(products.flood_level(L, 0.0, 0.0, c_area[p]))))
        if c_marine[p] != c_area[p]:
            configs.append(("marine", p, lambda L, p=p: float(products.flood_level(L, 0.0, 0.0, c_marine[p]))))
    area_cfg = {p: [("total0", 0.0), ("shape", c_area[p])] + ([("marine", c_marine[p])] if c_marine[p] != c_area[p] else [])
                for p in names}
    bins = sorted(units) if not a.bins else [int(x) for x in a.bins.split(",")]
    rows_D, rows_Q, rows_E = [], [], []
    blockD, blockQ = rev3.BlockDist(RES), rev3.BlockDist(RES)

    def sample(b, eps, parities=(0, 1)):
        raw, _ = a7.observed_raw(a7.groups_for(units, b, eps, parities), transform, shape, envelope=False)
        raw[(raw == 0) & env] = 1
        code = a7.classify(raw, static, TILE)
        del raw
        return code, rev_r.obs_waterline(code, outside)

    for b in bins:
        level = a7._bin_level(b)
        epochs = sorted(units[b])
        accD, accQ = Acc(), Acc()
        pred_cache = {}

        def pred(p, T):
            if (p, T) not in pred_cache:
                pred_cache[(p, T)] = fmaps[p].waterline(T)
            return pred_cache[(p, T)]

        def floor(sa, sb, kind):
            (ca, rca), (cb, rcb) = sa, sb
            cells = rev_r.cells_from(rca, rcb, rev_r.obs_at(ca, outside), rev_r.obs_at(cb, outside), sector)
            cells = ht._distances(cells, RES)
            res = _summ(cells)
            accQ.add((("config", "registered"), ("kind", kind)), res)
            accQ.add((("config", "registered"), ("kind", "pooled_level")), res)
            blockQ.add("floor", *a7._measured(cells, transform, BLOCK_M))

        e1 = None
        for ep in epochs:
            code, rc = sample(b, [ep])
            oat = rev_r.obs_at(code, outside)
            for cfg, p, thr in configs:
                T = thr(level)
                fm = fmaps[p]
                cells = rev_r.cells_from(pred(p, T), rc, lambda r, c, fm=fm, T=T: fm.at(r, c, T), oat, sector)
                cells = ht._distances(cells, RES)
                accD.add((("config", cfg), ("product", p), ("threshold_m", T)), _summ(cells))
                if cfg == "shape":
                    blockD.add(p, *a7._measured(cells, transform, BLOCK_M))
                del cells
            for p in names:
                for cfg, c in area_cfg[p]:
                    T = float(products.flood_level(level, 0.0, 0.0, c))
                    cnt = ht.extent_counts(ht.flooded(Fmap[p], T), code, a7._valid(code), lab_in, IDS,
                                           pred_valid=fin_of[p], outside_state=outside, shape=shape, tile=TILE)
                    for k in IDS:
                        rows_E.append({"config": cfg, "product": p, "epoch": ep, "bin": b, "level_m": round(level, 4),
                                       "threshold_m": T, "sector": NAME[k], **{f"n_{n}": v for n, v in cnt[k].items()}})
            if ep == "E1":
                e1 = (code, rc)
            del code, rc
            if set(units[b][ep]) == {0, 1}:
                floor(sample(b, [ep], (0,)), sample(b, [ep], (1,)), "halves")
        e23 = [ep for ep in epochs if ep in ("E2", "E3")]
        if "E1" in epochs and e23:
            floor(e1, sample(b, e23), "e1_vs_e23")
        e1 = None
        rows_D += accD.rows(b, level)
        rows_Q += accQ.rows(b, level)
        print(f"bin {b} ({level:.2f} m) {epochs} done, {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows_D).to_csv(out / "levels_D.csv.gz", index=False)
    pd.DataFrame(rows_Q).to_csv(out / "levels_Q.csv", index=False)
    pd.DataFrame(rows_E).to_csv(out / "extent_counts.csv", index=False)
    blockD.table().rename(columns={"key": "product"}).to_csv(out / "blocks_D.csv", index=False)
    blockQ.table().to_csv(out / "blocks_Q.csv", index=False)
    rec = {"stage2": TAG, "products": names, "bins": bins, "c_area_m": c_area, "c_marine_m": c_marine,
           "configs": sorted({c[0] for c in configs}), "marine_identical_to_shape": [p for p in names
                                                                                     if c_marine[p] == c_area[p]],
           "floor": "registered", "block_m": BLOCK_M, "envelope_cells": int(env.sum()),
           "seconds": round(time.time() - t0, 1)}
    (out / "run.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec), flush=True)
    return 0


# --- decide ---------------------------------------------------------------------------------------

def _close(a, b) -> bool:
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= TOL


def cmd_decide(a) -> int:
    src = OUT / "audit" / a.out
    run = json.loads((src / "run.json").read_text(encoding="utf-8"))
    names, same = run["products"], set(run["marine_identical_to_shape"])
    D = pd.read_csv(src / "levels_D.csv.gz")
    Q = pd.read_csv(src / "levels_Q.csv")
    Q = Q[(Q["config"] == "registered") & (Q["kind"] == "pooled_level")]
    delt = deltaic_sectors()
    marine_sec = [s for s in SECTOR_ORDER if s not in delt]
    checks = {}

    # 1. the replica reproduces the primary per level and sector
    det = pd.read_csv(MAIN / "h1_mispredicts_detail.csv")
    det = det[(det["variant"] == "shape") & (det["epoch"] == "pooled")]
    full = set(det["bin"]) <= set(D["bin"])
    rep = D[D["config"] == "shape"].merge(det[det["bin"].isin(D["bin"])], on=["product", "bin", "sector"],
                                           how="outer", suffixes=("", "_prim"), indicator=True)
    if (rep["_merge"] != "both").any():
        raise SystemExit("replica: rows without a primary counterpart (or the reverse)")
    bad = [i for i, r in rep.iterrows() if not (_close(r["median_m"], r["median"]) and int(r["n"]) == int(r["n_prim"]))]
    if bad:
        raise SystemExit(f"replica: {len(bad)} per-level displacement rows differ from the primary")
    qq = Q.merge(det.drop_duplicates(["bin", "sector"])[["bin", "sector", "n_noise", "p95_noise"]],
                 on=["bin", "sector"], how="left")
    badq = [i for i, r in qq.iterrows() if not (int(r["n"]) == int(r["n_noise"]) and
                                                (r["n"] == 0 or _close(r["p95_m"], r["p95_noise"])))]
    if badq:
        raise SystemExit(f"replica: {len(badq)} floor rows differ from the primary")
    checks["replica_levels"] = {"rows": int(len(rep)), "floor_rows": int(len(qq)), "tolerance_m": TOL,
                                "all_bins": bool(full)}

    # 2. verdicts by Eq. (rule) for both configurations
    noise = Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]]
    lev = D.rename(columns={"config": "variant", "median_m": "median"})
    cfgs = [lev]
    for p in same:        # an identical constant: the marine rows are the shape rows
        cfgs.append(lev[(lev["variant"] == "shape") & (lev["product"] == p)].assign(variant="marine"))
    lev = pd.concat(cfgs, ignore_index=True)
    summ, detail = h1.mispredicts(lev[["product", "variant", "sector", "bin", "level_m", "observed_level", "n",
                                       "median"]], noise)
    prim = pd.read_csv(MAIN / "h1_mispredicts.csv")
    prim = prim[prim["variant"] == "shape"].set_index(["product", "sector"])
    if full:
        for r in summ[summ["variant"] == "shape"].itertuples():
            w = prim.loc[(r.product, r.sector)]
            if (int(r.n_evaluable), int(r.n_exceeding), bool(r.mispredicts)) != (
                    int(w["n_evaluable"]), int(w["n_exceeding"]), bool(w["mispredicts"])):
                raise SystemExit(f"replica verdict differs from the primary: {r.product} {r.sector}")
        checks["replica_verdicts"] = "identical to results/hindcast/main/h1_mispredicts.csv (35 cases)"
    OUT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / "verdicts.csv", index=False)
    wide = summ.pivot(index=["product", "sector"], columns="variant", values=["n_evaluable", "n_exceeding", "mispredicts"])
    wide.columns = [f"{a_}_{b_}" for a_, b_ in wide.columns]
    wide = wide.reset_index()
    wide["checked"] = wide["sector"].isin(CHECKED)
    wide["changed"] = wide["mispredicts_shape"].astype(bool) != wide["mispredicts_marine"].astype(bool)
    wide.to_csv(OUT / "verdict_changes.csv", index=False)

    # 3. sector medians over evaluable levels (as results/revision/s/item2_dq.csv)
    rows = []
    for (p, v, s), g in detail.groupby(["product", "variant", "sector"], sort=True):
        e = g[g["evaluable"]]
        rs = rev_s.ratio_summary(e["median"].to_numpy(float), e["p95_noise"].to_numpy(float))
        rows.append({"product": p, "variant": v, "sector": s, "n_levels": int(len(e)),
                     "d_median_m": float(np.median(e["median"].to_numpy(float))) if len(e) else np.nan,
                     "q_median_m": float(np.median(e["p95_noise"].to_numpy(float))) if len(e) else np.nan,
                     "dq_median": rs["median"]})
    med = pd.DataFrame(rows)
    if full:
        ref = pd.read_csv(RESULTS / "revision" / "s" / "item2_dq.csv")
        ref = ref[ref["variant"] == "shape"].set_index(["product", "sector"])
        for r in med[med["variant"] == "shape"].itertuples():
            w = ref.loc[(r.product, r.sector)]
            if not (_close(r.d_median_m, w["d_median_m"]) and _close(r.q_median_m, w["q_median_m"])):
                raise SystemExit(f"sector medians differ from results/revision/s/item2_dq.csv: {r.product} {r.sector}")
        checks["replica_sector_medians"] = "identical to results/revision/s/item2_dq.csv"
    med.to_csv(OUT / "sector_medians.csv", index=False)

    # 4. area bias over the domain (check) and the marine domain
    E = pd.read_csv(src / "extent_counts.csv")
    ncol = [c for c in E.columns if c.startswith("n_")]
    urows = []
    for region, secs in (("domain", SECTOR_ORDER), ("marine", marine_sec), ("deltaic", delt)):
        g = E[E["sector"].isin(secs)].groupby(["config", "product", "epoch", "bin", "level_m"], as_index=False)[ncol].sum()
        g["region"] = region
        g["area_bias_rel_total"] = [rev3.area_bias_rel({k[2:]: int(r[k]) for k in ncol}) for _, r in g.iterrows()]
        g["area_obs_km2"] = (g["n_tp"] + g["n_fn"]) * a7.CELL_KM2
        g["area_bias_total_km2"] = (g["n_fp"] - g["n_fn"] + g["n_fp_out"] - g["n_fn_out"]) * a7.CELL_KM2
        urows.append(g)
    U = pd.concat(urows, ignore_index=True)
    ex = pd.read_csv(MAIN / "extent.csv")
    ex = ex[(ex["sector"] == "domain") & (ex["epoch"] != "pooled")]
    key = {"total0": "total", "shape": "shape"}
    worst = 0.0
    for cfg, var in key.items():
        m = U[(U["region"] == "domain") & (U["config"] == cfg)].merge(
            ex[ex["variant"] == var], on=["product", "epoch", "bin"], suffixes=("", "_prim"))
        if len(m) != len(U[(U["region"] == "domain") & (U["config"] == cfg)]):
            raise SystemExit("area bias: units without a primary counterpart")
        worst = max(worst, float((m["area_bias_rel_total"] - m["area_bias_rel_total_prim"]).abs().max()))
    if worst > 1e-9:
        raise SystemExit(f"area bias: the domain rows differ from results/hindcast/main/extent.csv by {worst}")
    checks["replica_domain_area_bias_max_abs_dev"] = worst
    U.to_csv(OUT / "area_bias_units.csv", index=False)
    ab = []
    for (region, cfg, p), g in U.groupby(["region", "config", "product"], sort=True):
        rel = 100 * g["area_bias_rel_total"].astype(float)
        ab.append({"region": region, "config": cfg, "product": p, "n_units": int(len(g)),
                   "median_pct": float(rel.median()), "max_abs_pct": float(rel.abs().max()),
                   "median_km2": float(g["area_bias_total_km2"].median()),
                   "median_obs_km2": float(g["area_obs_km2"].median())})
    for p in same:        # identical constant
        for r in [x for x in ab if x["config"] == "shape" and x["product"] == p]:
            ab.append({**r, "config": "marine"})
    pd.DataFrame(ab).to_csv(OUT / "area_bias.csv", index=False)

    # 5. blocks (item 2b)
    bD = pd.read_csv(src / "blocks_D.csv")
    bQ = pd.read_csv(src / "blocks_Q.csv").rename(columns={"n": "n_floor", "p95_abs_m": "Q_b_m"})[
        ["block_i", "block_j", "n_floor", "Q_b_m"]]
    if full:
        bh = pd.read_csv(MAIN / "block_h1.csv")
        bh = bh[bh["variant"] == "shape"]
        m = bD.merge(bh, on=["product", "block_i", "block_j"], how="outer", suffixes=("", "_prim"), indicator=True)
        if (m["_merge"] != "both").any() or (m["n"] != m["n_prim"]).any() or any(
                not _close(x, y) for x, y in zip(m["median_abs_m"], m["median_abs_m_prim"])):
            raise SystemExit("blocks: the pooled block medians differ from results/hindcast/main/block_h1.csv")
        checks["replica_block_medians"] = f"identical to block_h1.csv ({len(m)} rows)"
    tab = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    tab["boundary"] = rev3.boundary_blocks(tab)
    lab = tab.rename(columns={"i": "block_i", "j": "block_j"})[["block_i", "block_j", "sector", "split", "boundary"]]
    B = bD.rename(columns={"n": "n_D", "median_abs_m": "D_b_m"})[["product", "block_i", "block_j", "n_D", "D_b_m"]]
    B = B.merge(bQ, on=["block_i", "block_j"], how="left").merge(lab, on=["block_i", "block_j"], how="left")
    B["n_floor"] = B["n_floor"].fillna(0).astype(int)
    B["evaluable"] = (B["n_D"] >= MIN_BLOCK_N) & (B["n_floor"] >= MIN_BLOCK_N) & B["sector"].notna()
    B["exceeds"] = B["evaluable"] & (B["D_b_m"] > B["Q_b_m"])
    B["ratio"] = np.where(B["evaluable"], B["D_b_m"] / B["Q_b_m"], np.nan)
    B.to_csv(OUT / "blocks.csv", index=False)
    srows = []
    scopes = {"all": B["evaluable"], "checked": B["evaluable"] & B["sector"].isin(CHECKED),
              "marine": B["evaluable"] & B["sector"].isin(marine_sec),
              "boundary": B["evaluable"] & B["boundary"].fillna(False).astype(bool),
              "interior": B["evaluable"] & ~B["boundary"].fillna(True).astype(bool),
              "checked_boundary": B["evaluable"] & B["sector"].isin(CHECKED) & B["boundary"].fillna(False).astype(bool),
              "checked_interior": B["evaluable"] & B["sector"].isin(CHECKED) & ~B["boundary"].fillna(True).astype(bool)}
    for s in SECTOR_ORDER:
        scopes[f"sector:{s}"] = B["evaluable"] & (B["sector"] == s)
    for name, mask in scopes.items():
        for p, g in B[mask].groupby("product", sort=True):
            srows.append({"scope": name, "product": p, "n_blocks": int(len(g)), "n_exceeding": int(g["exceeds"].sum()),
                          "share_exceeding": float(g["exceeds"].mean()) if len(g) else np.nan,
                          "ratio_median": float(g["ratio"].median()) if len(g) else np.nan})
    pd.DataFrame(srows).to_csv(OUT / "block_shares.csv", index=False)

    ch = wide[wide["changed"]]
    summary = {"amendment": "PREREG Amendment 20 items 1 (a-c) and 2 (b)", "decision": "D-161",
               "deltaic_sectors": delt, "marine_sectors": marine_sec, "checked_sectors": list(CHECKED),
               "c_area_m": run["c_area_m"], "c_marine_m": run["c_marine_m"],
               "c_change_m": {p: round(run["c_marine_m"][p] - run["c_area_m"][p], 2) for p in names},
               "bins": run["bins"], "all_bins": bool(full), "checks": checks,
               "n_cases": int(len(wide)), "n_changed": int(len(ch)),
               "n_cases_checked": int(wide["checked"].sum()), "n_changed_checked": int(ch["checked"].sum()),
               "changed": [f"{r.product}|{r.sector}: {bool(r.mispredicts_shape)} -> {bool(r.mispredicts_marine)}"
                           for r in ch.itertuples()],
               "min_block_n": MIN_BLOCK_N, "block_m": BLOCK_M}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["fit", "pass", "decide"])
    ap.add_argument("--bins", default="")
    ap.add_argument("--out", default="main")
    a = ap.parse_args(argv)
    return {"fit": cmd_fit, "pass": cmd_pass, "decide": cmd_decide}[a.command](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
