"""Second-round revision analysis A: PREREG Amendment 19 items 7 and 8 (D-147).

flood  Item 7(b): flood-level maps of the heights as distributed (no per-cell
       N(EGM2008) - N(EIGEN-6C4)) for GEBCO_2026, GEBCO_2022, SRTM15+ and, as a labelled
       extra, ETOPO1, from the primary sea seed; cached in data/interim/revision2/a/flood/.
fit    Item 7(b): the D-070 area constant refitted on those maps, by exact counts on the
       cached per-unit known and water masks of the fit region (data/interim/revision_v/
       fit_masks.npz, D-121; rev_v_planar.exact_bias). First, the same counts on each
       product's primary corrected map must reproduce its c_area and objective.
       results/revision2/a/nogeoid_constants.json.
pass   One pass over the 32 level bins of the main stage-2 counts, rebuilding the observed
       extents exactly as scripts/a7_hindcast.py does (observed_raw, the record-low
       envelope, classify), with the waterline engine of caspian.rev_r. Configs per
       product:
         shape          the primary shape variant (c_area); replica check
         total0         the primary total variant (c = 0); replica check
         total_cref     item 7(a): F < L - c_ref, c_ref = ATL08 constant (0.7098 m)
         total_cref06   sensitivity: the ATL06 constant (0.7230 m)
         total_swot     sensitivity: the SWOT check (0.8345 m)
         nogeoid        item 7(b): heights as distributed, refitted constant
         shape_i        item 8 (i): unknown predicted cells measured
         shape_ii       item 8 (ii): observed waterline cells within 2 cells of an
                        unknown cell not measured
       Floors: registered (must reproduce the primary), matched_i and matched_ii (the
       same rules on the floor pairs; D-147).
       results/revision2/a/audit/<out>/{levels_D.csv.gz, levels_Q.csv, units_D.csv.gz,
       run.json}.

Usage:
  python scripts/rev2_a_audit.py flood
  python scripts/rev2_a_audit.py fit
  python scripts/rev2_a_audit.py pass [--bins 12,38] [--out main]
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

from caspian import hindcast_tiled as ht
from caspian import products, rev2_a, rev_r
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
OUT = RESULTS / "revision2" / "a"
WORK = INTERIM / "revision2" / "a"
FLOOD = WORK / "flood"
MASKS = INTERIM / "revision_v" / "fit_masks.npz"
NOGEOID = ["gebco_2026", "gebco_2022", "srtm15p_v2.7", "etopo1"]     # etopo1: labelled extra (D-147)
REGISTERED_7B = ["gebco_2026", "gebco_2022", "srtm15p_v2.7"]
IDS = list(range(1, len(SECTOR_ORDER) + 1))
NAME = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}


def grid_setup():
    w, s, e, n = DOMAIN.bounds
    return a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))


def constants_cref() -> dict:
    d = json.loads((RESULTS / "revision" / "v" / "datum_terms.json").read_text(encoding="utf-8"))
    return {"total_cref": float(d["constants"]["ATL08"]["c_m"]),
            "total_cref06": float(d["constants"]["ATL06"]["c_m"]),
            "total_swot": float(d["swot_check"]["nga_realisation_m"])}


def _main_run():
    return json.loads((MAIN / "run.json").read_text(encoding="utf-8"))


def _c_area() -> dict:
    so = json.loads((MAIN / "shape_offsets.json").read_text(encoding="utf-8"))
    return {p: float(v["offset_m"]) for p, v in so["used_in_this_run"].items()}


def _finite(h, shape) -> np.ndarray:
    fin = np.zeros(shape, bool)
    for r0, r1, c0, c1 in ht.tiles(shape, TILE):
        fin[r0:r1, c0:c1] = np.isfinite(np.asarray(h[r0:r1, c0:c1], np.float32))
    return fin


def raw_flood_map(p, transform, shape, seed, build: bool):
    """The flood-level map of product p's heights as distributed (cached)."""
    h = products.heights(p, transform, shape, _main_run()["resampling"])
    before = set(FLOOD.glob(f"{p}_asdist_*.npy")) if FLOOD.exists() else set()
    F = a7.flood_map_cached(f"{p}_asdist", h, seed, {"heights": Path(h.filename).name, "correction": None},
                            TILE, cache=FLOOD)
    if not build and set(FLOOD.glob(f"{p}_asdist_*.npy")) != before:
        raise SystemExit(f"{p}: the as-distributed flood-level map was not cached (run `flood` first)")
    return h, F


def primary_flood_map(p, transform, shape, seed, corr, corr_rec):
    h = products.heights(p, transform, shape, _main_run()["resampling"])
    hc = products.Corrected(h, corr)
    cache = INTERIM / "hindcast" / "flood"
    before = set(cache.glob(f"{p}_*.npy"))
    F = a7.flood_map_cached(p, hc, seed, {"heights": Path(h.filename).name, "correction": corr_rec["sha256"]}, TILE)
    if set(cache.glob(f"{p}_*.npy")) != before:
        raise SystemExit(f"{p}: the primary flood-level map was not in the cache (a new one was built)")
    return h, hc, F


def _static(transform, shape):
    static = a7.production_static(transform, shape, TILE)
    for k in ("zone", "land_seed"):
        static.pop(k)
    return static


# --- flood and fit (item 7b) ----------------------------------------------------------------------

def cmd_flood(a) -> int:
    t0 = time.time()
    transform, shape = grid_setup()
    static = _static(transform, shape)
    FLOOD.mkdir(parents=True, exist_ok=True)
    for p in NOGEOID:
        raw_flood_map(p, transform, shape, static["sea_seed"], build=True)
        print(f"flood-level map (as distributed) {p}, {time.time() - t0:.0f} s", flush=True)
    return 0


class _FitData:
    """The fit-region masks of D-121 (rev_v_planar.FitData without the coordinates)."""

    def __init__(self, path=MASKS):
        z = np.load(path)
        self.idx, self.n = z["idx"], int(z["n"])
        self.known, self.water = z["known"], z["water"]
        self.levels, self.epochs, self.bins = z["levels"], [str(e) for e in z["epochs"]], z["bins"]
        self.shape = tuple(int(v) for v in z["shape"])

    def unit(self, u, which="known"):
        return np.unpackbits(getattr(self, which)[u], count=self.n).astype(bool)


def cmd_fit(a) -> int:
    t0 = time.time()
    planar = _load("rev_v_planar")
    transform, shape = grid_setup()
    static = _static(transform, shape)
    fd = _FitData()
    prim = json.loads((MAIN / "shape_offsets.json").read_text(encoding="utf-8"))
    if [(u["bin"], u["epoch"]) for u in prim["units"]] != list(zip(fd.bins.tolist(), fd.epochs)):
        raise SystemExit("the fit masks' units differ from the primary area fit")
    if fd.n != prim["fit_region_cells"] or fd.shape != tuple(shape):
        raise SystemExit("the fit masks' region or grid differs from the primary")
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != _main_run()["vertical_correction"]["sha256"]:
        raise SystemExit("the correction is not the primary run's")
    flat_corr = np.asarray(corr).reshape(-1)
    rec = {"rule": "D-070 on the as-distributed flood-level maps (D-147): 0.01 m grid over +-5 m, 43 units, "
                   "known cells of the domain in training blocks, ties to the smallest |c| then the smaller c",
           "fit_region_cells": fd.n, "registered_products": REGISTERED_7B,
           "extra_products": [p for p in NOGEOID if p not in REGISTERED_7B], "products": {}}
    for p in NOGEOID:
        t1 = time.time()
        h, hc, F = primary_flood_map(p, transform, shape, static["sea_seed"], corr, corr_rec)
        flat_h = np.asarray(h).reshape(-1)
        fin_c = np.zeros(fd.n, bool)
        fin_r = np.zeros(fd.n, bool)
        dn = np.zeros(fd.n, np.float64)
        for s0 in range(0, fd.n, 1 << 24):
            j = fd.idx[s0:s0 + (1 << 24)]
            hv, cv = flat_h[j].astype(np.float32), flat_corr[j].astype(np.float32)
            fin_c[s0:s0 + j.size] = np.isfinite(hv + cv)
            fin_r[s0:s0 + j.size] = np.isfinite(hv)
            dn[s0:s0 + j.size] = cv
        # 0. the exact counts on the primary corrected map reproduce the primary fit
        b0 = planar.exact_bias(fd, np.asarray(F).reshape(-1)[fd.idx], fin_c)
        r0 = a7.fit_offset(b0, fd.epochs)
        want = prim["products"][p]
        if r0["offset_m"] != want["offset_m"] or abs(r0["objective_km2_at_offset"] - want["objective_km2_at_offset"]) > 1e-3:
            raise SystemExit(f"{p}: exact counts give c={r0['offset_m']} obj={r0['objective_km2_at_offset']}, "
                             f"primary c={want['offset_m']} obj={want['objective_km2_at_offset']}")
        del b0, F
        # 1. the refit on the as-distributed map
        _, Fr = raw_flood_map(p, transform, shape, static["sea_seed"], build=False)
        b1 = planar.exact_bias(fd, np.asarray(Fr).reshape(-1)[fd.idx], fin_r)
        r1 = a7.fit_offset(b1, fd.epochs)
        del b1, Fr
        ok = fin_c & np.isfinite(dn)
        rec["products"][p] = {
            "registered": p in REGISTERED_7B,
            "c_area_primary_m": want["offset_m"], "objective_primary_km2": want["objective_km2_at_offset"],
            "primary_reproduced": True,
            "c_nogeoid_m": r1["offset_m"], "objective_nogeoid_km2": r1["objective_km2_at_offset"],
            "n_tied_minima": r1["n_tied_minima"],
            "c_change_m": round(r1["offset_m"] - want["offset_m"], 2),
            "field_mean_over_fit_cells_m": float(dn[ok].mean()),
            "field_range_over_fit_cells_m": [float(dn[ok].min()), float(dn[ok].max())],
            "finite_cells_equal": bool(np.array_equal(fin_c, fin_r)),
            "seconds": round(time.time() - t1, 1)}
        print(json.dumps({p: rec["products"][p]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rec["seconds"] = round(time.time() - t0, 1)
    (OUT / "nogeoid_constants.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


# --- the pass (items 7 and 8) ----------------------------------------------------------------------

class Acc:
    """Signed distances and counts per key and sector, pooled over the epochs of a bin."""

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
    outside, sector, seed = static["outside"], static["sector"], static["sea_seed"]
    units = a7.stage2_index(INTERIM / "stage2" / TAG)
    env = a7.record_low_water(units, transform, shape)
    c_area = _c_area()
    cref = constants_cref()
    ng = json.loads((OUT / "nogeoid_constants.json").read_text(encoding="utf-8"))["products"]
    names = list(_main_run()["products"])
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != _main_run()["vertical_correction"]["sha256"]:
        raise SystemExit("the correction is not the primary run's")
    fmaps, fin_of = {}, {}
    for p in names:
        h, hc, F = primary_flood_map(p, transform, shape, seed, corr, corr_rec)
        fin = _finite(hc, shape)
        fin_of[p] = fin
        fmaps[p] = rev_r.FloodMap(F, fin, tile=256)
        print(f"flood map {p} ready, {time.time() - t0:.0f} s", flush=True)
    for p in NOGEOID:
        h, Fr = raw_flood_map(p, transform, shape, seed, build=False)
        fin = _finite(h, shape)
        if np.array_equal(fin, fin_of[p]):
            fin = fin_of[p]                      # share the mask (memory)
        fmaps[f"{p}|nogeoid"] = rev_r.FloodMap(Fr, fin, tile=256)
        print(f"as-distributed flood map {p} ready, {time.time() - t0:.0f} s", flush=True)
    del fin_of
    # (config, product, surface, threshold function of the bin level)
    configs = []
    for p in names:
        configs.append(("shape", p, p, lambda L, p=p: float(products.flood_level(L, 0.0, 0.0, c_area[p]))))
        configs.append(("total0", p, p, lambda L: float(products.flood_level(L, 0.0, 0.0, 0.0))))
        for cfg, cv in cref.items():
            configs.append((cfg, p, p, lambda L, cv=cv: float(rev2_a.total_threshold(L, cv))))
    for p in NOGEOID:
        cn = float(ng[p]["c_nogeoid_m"])
        configs.append(("nogeoid", p, f"{p}|nogeoid", lambda L, cn=cn: float(products.flood_level(L, 0.0, 0.0, cn))))
    bins = sorted(units) if not a.bins else [int(x) for x in a.bins.split(",")]
    rows_D, rows_Q, rows_U = [], [], []

    def sample(b, eps, parities=(0, 1)):
        raw, _ = a7.observed_raw(a7.groups_for(units, b, eps, parities), transform, shape, envelope=False)
        raw[(raw == 0) & env] = 1
        code = a7.classify(raw, static, TILE)
        del raw
        rc = rev_r.obs_waterline(code, outside)
        return code, rc

    for b in bins:
        level = a7._bin_level(b)
        epochs = sorted(units[b])
        accD, accQ = Acc(), Acc()
        pred_cache = {}

        def pred(surf, T):
            if (surf, T) not in pred_cache:
                pred_cache[(surf, T)] = fmaps[surf].waterline(T)
            return pred_cache[(surf, T)]

        def unit_rows(cfg, p, ep, res):
            for k in IDS:
                r = res[k]
                rows_U.append({"config": cfg, "product": p, "epoch": ep, "bin": b, "sector": NAME[k], "n": r["n"],
                               "median_m": r["median"], "n_pred": r["n_pred"], "n_pred_unknown": r["n_pred_unknown"],
                               "n_obs": r["n_obs"], "n_obs_unknown": r["n_obs_unknown"]})

        def floor(sa, sb, kind):
            (ca, rca), (cb, rcb) = sa, sb
            cells = rev_r.cells_from(rca, rcb, rev_r.obs_at(ca, outside), rev_r.obs_at(cb, outside), sector)
            cells = ht._distances(cells, RES)
            near_a = rev2_a.near_unknown(cells["pred"]["rows"], cells["pred"]["cols"], ca, outside, 2)
            near_b = rev2_a.near_unknown(cells["obs"]["rows"], cells["obs"]["cols"], cb, outside, 2)
            variants = {"registered": cells,
                        "matched_i": rev2_a.measure_all(cells, ("pred", "obs")),
                        "matched_ii": rev2_a.drop_measured(rev2_a.drop_measured(cells, "pred", near_a), "obs", near_b)}
            for name, cc in variants.items():
                res = _summ(cc)
                accQ.add((("config", name), ("kind", kind)), res)
                accQ.add((("config", name), ("kind", "pooled_level")), res)

        e1 = None
        for ep in epochs:
            code, rc = sample(b, [ep])
            oat = rev_r.obs_at(code, outside)
            near_o = rev2_a.near_unknown(rc[0], rc[1], code, outside, 2)
            for cfg, p, surf, thr in configs:
                T = thr(level)
                fm = fmaps[surf]
                cells = rev_r.cells_from(pred(surf, T), rc, lambda r, c, fm=fm, T=T: fm.at(r, c, T), oat, sector)
                cells = ht._distances(cells, RES)
                res = _summ(cells)
                accD.add((("config", cfg), ("product", p), ("threshold_m", T)), res)
                unit_rows(cfg, p, ep, res)
                if cfg == "shape":
                    for name, cc in (("shape_i", rev2_a.measure_all(cells)),
                                     ("shape_ii", rev2_a.drop_measured(cells, "obs", near_o))):
                        r2 = _summ(cc)
                        accD.add((("config", name), ("product", p), ("threshold_m", T)), r2)
                        unit_rows(name, p, ep, r2)
                del cells
            if ep == "E1":
                e1 = (code, rc)
            del code, rc, near_o
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
    pd.DataFrame(rows_U).to_csv(out / "units_D.csv.gz", index=False)
    rec = {"stage2": TAG, "products": names, "bins": bins, "c_area_m": c_area, "c_ref_m": cref,
           "c_nogeoid_m": {p: ng[p]["c_nogeoid_m"] for p in NOGEOID}, "nogeoid_registered": REGISTERED_7B,
           "configs": sorted({c[0] for c in configs} | {"shape_i", "shape_ii"}),
           "floors": ["registered", "matched_i", "matched_ii"], "near_unknown_radius_cells": 2,
           "envelope_cells": int(env.sum()), "seconds": round(time.time() - t0, 1)}
    (out / "run.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec), flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["flood", "fit", "pass"])
    ap.add_argument("--bins", default="")
    ap.add_argument("--out", default="main")
    a = ap.parse_args(argv)
    return {"flood": cmd_flood, "fit": cmd_fit, "pass": cmd_pass}[a.command](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
