"""Revision R: the A18 item 3 and item 6 audits that reuse the hindcast's units with the
waterline engine of caspian.rev_r.

One pass over the level bins of a stage-2 tag rebuilds the observed extents exactly as
scripts/a7_hindcast.py does (observed_raw with the record-low envelope, classify) and
feeds them to "configs":

  replica  the primary audit at each product's area constant c_area (shape variant) and
           the primary floor; must reproduce results/hindcast/main exactly (checked by
           `decide`)
  static   the static-contour filter (the registered rule, rho* of
           results/levels/static_contour_rule.json) removes static cells from every
           observed waterline (full extents, halves, E1 and E2 + E3) before distances
  res      observed extents (full and halves) aggregated by majority to each product's
           native cell (15", or 1' for ETOPO1) before the sea connection and distances
  cd       the displacement-fitted constant search: c = c_area + 0.10 k, k = -10..10;
           per unit the median |d| of the measured cells in training blocks (domain and
           split == train), and per level and sector the pooled median
  variant  another stage-2 tag at c_area. With --hybrid (the R-block
           threshold variants): the variant's counts inside its tiles, the primary's
           elsewhere, and only cells inside the variant's tiles measured; the primary on
           the same footprint is computed beside it ("primary_fp")

Outputs: results/revision/r/audit/<name>/{levels_D.csv.gz, levels_Q.csv, units_D.csv.gz,
cd_objective.csv, static_cells.csv, run.json}; intermediates in data/interim/rev_r/.

Usage:
  python scripts/rev_r_audit.py prep
  python scripts/rev_r_audit.py pass --configs replica,static,res,cd [--bins 12,38] [--out main]
  python scripts/rev_r_audit.py pass --configs variant --stage2 main__veg_as_dry --out veg_as_dry
  python scripts/rev_r_audit.py pass --configs variant --stage2 rR_awei --hybrid --out rR_awei
  python scripts/rev_r_audit.py pass --configs replica --surfaces controls       --constants results/revision/r/control/refit_dem_range.json --out control_refit
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
import pyproj

from caspian import design, products, rev_r
from caspian import hindcast_tiled as ht
from caspian.config import CRS_EQUAL_AREA, INTERIM, RESULTS
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
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
WORK = INTERIM / "rev_r"
OUT = RESULTS / "revision" / "r" / "audit"
K_GRID = np.arange(-10, 11)            # c = c_area + 0.10 k (A18 item 3)
C_STEP = 0.10
GRID_OF = {"gebco_2026": "15s", "gebco_2022": "15s", "srtm15p_v2.7": "15s", "etopo_2022": "15s", "etopo1": "1m"}
NATIVE_OF = {"15s": "gebco_2026", "1m": "etopo1"}
IDS = list(range(1, len(SECTOR_ORDER) + 1))
NAME = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}


def _a6w():
    return _load("a6_waterlines")


# --- preparation ---------------------------------------------------------------------------

def grid_setup():
    w, s, e, n = DOMAIN.bounds
    return a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))


def static_map_path(tag: str, ep: str) -> Path:
    return WORK / f"static_{tag}_{ep}.npy"


def prep_static(tag: str, transform, shape, rho_star: float):
    """Per epoch, int8 on the grid: 1 static (rho_level < rho*), 0 not static, -1 no rho
    (fewer than 4 valid bins). rho_level as scripts/a6_waterlines.py computes it per
    stage-2 tile (both parities summed per bin); where tiles overlap, the first tile with a
    value is kept (as a7_hindcast.observed_raw keeps the first valid cell)."""
    a6w = _a6w()
    out = {}
    x0, y1 = float(transform[2]), float(transform[5])
    files = sorted((INTERIM / "stage2" / tag).glob("tile_*.npz"))
    for ep in ("E1", "E2", "E3"):
        path = static_map_path(tag, ep)
        if path.exists():
            out[ep] = path
            continue
        m = np.full(shape, -1, np.int8)
        for f in files:
            with np.load(f) as z:
                tr = z["transform"]
                groups = {}
                for k in z.files:
                    g = a7.BAND.match(k)
                    if g and g.group(1) == "w" and g.group(2) == ep and f"v{k[1:]}" in z.files:
                        groups.setdefault(int(g.group(3)), []).append((z[k], z["v" + k[1:]]))
            if not groups:
                continue
            stack = []
            for b in sorted(groups):
                w = sum(np.asarray(x[0], np.int64) for x in groups[b])
                v = sum(np.asarray(x[1], np.int64) for x in groups[b])
                stack.append((b, a6w.frequency.water_frequency(w, v)))
            rho = a6w._rho(stack)
            code = np.where(np.isnan(rho), -1, (rho < rho_star).astype(np.int8)).astype(np.int8)
            r0, c0 = int(round((y1 - float(tr[5])) / RES)), int(round((float(tr[2]) - x0) / RES))
            dst = m[r0:r0 + code.shape[0], c0:c0 + code.shape[1]]
            put = (dst == -1) & (code >= 0)
            dst[put] = code[put]
            print(f"static map {tag} {ep} {f.name}", flush=True)
        np.save(path, m)
        out[ep] = path
    return out


def native_id_path(g: str) -> Path:
    return WORK / f"nid_{g}.npy"


def prep_native_ids(transform, shape):
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    xs = c + a * (np.arange(shape[1]) + 0.5)
    for g, prod in NATIVE_OF.items():
        path = native_id_path(g)
        if path.exists():
            continue
        nat = products.load_native(prod)
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.int32, shape=tuple(shape))
        for r0 in range(0, shape[0], 256):
            r1 = min(r0 + 256, shape[0])
            ys = f + e * (np.arange(r0, r1) + 0.5)
            lon, lat = inv.transform(*np.meshgrid(xs, ys))
            i, j = rev_r.native_cell_index(lon, lat, nat.lon, nat.lat)
            mm[r0:r1] = np.where(i >= 0, i * len(nat.lon) + j, -1).astype(np.int32)
        mm.flush()
        del mm
        tmp.replace(path)
        (WORK / f"nid_{g}.json").write_text(json.dumps({"product_grid": prod, "n_native": int(nat.z.size),
                                                        "shape": list(nat.z.shape)}), encoding="utf-8")
        print("native ids", g, flush=True)


def cmd_prep(a) -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    transform, shape = grid_setup()
    rule = json.loads((RESULTS / "levels" / "static_contour_rule.json").read_text(encoding="utf-8"))
    prep_static(TAG, transform, shape, float(rule["rho_star"]))
    prep_native_ids(transform, shape)
    return 0


# --- the pass --------------------------------------------------------------------------------

class Acc:
    """Signed distances per key and sector, pooled over the epochs of a bin."""

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
                        "n_pred": p["n_pred"], "n_obs": p["n_obs"], "observed_level": p["observed_level"]})
        return out


def _summ(cells, labels_ids=IDS):
    cells = ht._distances(cells, RES)
    return cells, ht.summarise(cells, labels_ids, return_distances=True)


def train_median(cells, train) -> tuple[int, float]:
    """Median |d| of the measured cells (both directions) that lie in training blocks."""
    p, o = cells["pred"], cells["obs"]
    mp = p["known"] & (np.asarray(train)[p["rows"], p["cols"]] if len(p["rows"]) else np.empty(0, bool))
    mo = o["known"] & (np.asarray(train)[o["rows"], o["cols"]] if len(o["rows"]) else np.empty(0, bool))
    d = np.abs(np.r_[cells["signed_p"][mp], cells["signed_o"][mo]])
    d = d[~np.isnan(d)]
    if not d.size:
        return 0, float("nan")
    return int(d.size), float(np.inf) if np.isinf(d).any() else float(np.median(d))


def run_pass(a) -> int:
    global K_GRID, C_STEP
    t0 = time.time()
    if a.c_step:                       # coarser search, same +-1.0 m range
        C_STEP = float(a.c_step)
        n = int(round(1.0 / C_STEP))
        K_GRID = np.arange(-n, n + 1)
    configs = [c for c in a.configs.split(",") if c]
    tag = a.stage2
    out = OUT / a.out
    out.mkdir(parents=True, exist_ok=True)
    transform, shape = grid_setup()
    static = a7.production_static(transform, shape, TILE)
    outside, sector, domain = static["outside"], static["sector"], static["domain"]
    blocks = pd.read_csv(a7.BLOCKS)
    train = domain & a7.train_mask(blocks, transform, shape)
    units_main = a7.stage2_index(INTERIM / "stage2" / TAG)
    units = a7.stage2_index(INTERIM / "stage2" / tag)
    hybrid = bool(a.hybrid)
    footprint = None
    if hybrid:                                   # variant tiles override the primary's counts
        footprint = np.zeros(shape, bool)
        x0, y1 = float(transform[2]), float(transform[5])
        for f in sorted((INTERIM / "stage2" / tag).glob("tile_*.npz")):
            with np.load(f) as z:
                tr = z["transform"]
                k0 = next(k for k in z.files if k.startswith("w_"))
                hh, ww = z[k0].shape
            r0, c0 = int(round((y1 - float(tr[5])) / RES)), int(round((float(tr[2]) - x0) / RES))
            footprint[r0:r0 + hh, c0:c0 + ww] = True
    # observed extents: raw without envelope, then the envelope of the sample's own tag
    def raw_of(us, b, eps, parities=(0, 1)):
        g = a7.groups_for(us, b, eps, parities) if b in us else {}
        if not g:
            return np.full(shape, -2, np.int8)
        r, _ = a7.observed_raw(g, transform, shape, envelope=False)
        return r

    def merged(b, eps, parities=(0, 1)):
        r = raw_of(units_main if hybrid else units, b, eps, parities)
        if hybrid:
            rv = raw_of(units, b, eps, parities)
            r[footprint] = rv[footprint]
        return r

    # record-low water exactly as a7_hindcast.record_low_water: the counts of the
    # lowest E3 bins pooled into one extent; for a hybrid, each footprint from its own tag
    env = {"variant": a7.record_low_water(units_main if hybrid else units, transform, shape)}
    if hybrid:
        env["primary_fp"] = env["variant"].copy()
        env["variant"][footprint] = a7.record_low_water(units, transform, shape)[footprint]

    def with_env(raw, which="variant"):
        r = raw.copy()
        r[(r == 0) & env[which]] = 1
        return r

    def classify(raw):
        return a7.classify(raw, static, TILE)

    shape_c = json.loads((RESULTS / "hindcast" / TAG / "shape_offsets.json").read_text(encoding="utf-8"))
    c_area = {p: float(v["offset_m"]) for p, v in shape_c["used_in_this_run"].items()}
    names = [p for p in a.products.split(",") if p]
    corr, _ = products.correction(CORRECTION, transform, shape)
    fmaps, finite = {}, {}
    if a.surfaces == "controls":            # the control surfaces at given constants
        C = _load("rev_r_control")
        surf, flood, _ = C.build(transform, shape, static, units_main)
        cj = json.loads(Path(a.constants).read_text(encoding="utf-8"))
        cj = cj.get("constants", cj)
        c_area = {n: float(v["offset_m"]) for n, v in cj.items()}
        names = [n for n in surf if n in c_area]
        for n in names:
            fin = np.zeros(shape, bool)
            for r0, r1, c0, c1 in ht.tiles(shape, TILE):
                fin[r0:r1, c0:c1] = np.isfinite(np.asarray(surf[n][r0:r1, c0:c1], float))
            finite[n] = fin
            fmaps[n] = rev_r.FloodMap(flood[n], fin, tile=256)
            GRID_OF.setdefault(n, "15s")
        print(f"control surfaces ready: {names}", flush=True)
    for p in (names if a.surfaces == "products" else []):
        h = products.heights(p, transform, shape, "bilinear")
        hc = products.Corrected(h, corr)
        F = a7.flood_map_cached(p, hc, static["sea_seed"], {"heights": Path(h.filename).name,
                                                              "correction": json.loads(
                                                                  (RESULTS / "hindcast" / TAG / "run.json").read_text(
                                                                      encoding="utf-8"))["vertical_correction"]["sha256"]},
                                TILE)
        fin = np.zeros(shape, bool)
        for r0, r1, c0, c1 in ht.tiles(shape, TILE):
            fin[r0:r1, c0:c1] = np.isfinite(np.asarray(hc[r0:r1, c0:c1], float))
        finite[p] = fin
        fmaps[p] = rev_r.FloodMap(F, fin, tile=256)
        print(f"flood map {p} ready, {time.time() - t0:.0f} s", flush=True)
    smap = {}
    if "static" in configs:
        rule = json.loads((RESULTS / "levels" / "static_contour_rule.json").read_text(encoding="utf-8"))
        for ep in ("E1", "E2", "E3"):
            smap[ep] = np.load(static_map_path(TAG, ep)) == 1
    nids = {}
    if "res" in configs:
        for g in NATIVE_OF:
            nids[g] = (np.load(native_id_path(g), mmap_mode="r"),
                       json.loads((WORK / f"nid_{g}.json").read_text(encoding="utf-8"))["n_native"])
    labels_main = sector
    labels_fp = np.where(footprint, sector, 0).astype(sector.dtype) if hybrid else None
    bins = sorted(units) if not a.bins else [int(x) for x in a.bins.split(",")]
    rows_D, rows_Q, rows_U, rows_obj, rows_static = [], [], [], [], []

    def flag_static(eps):
        m = np.zeros(shape, bool)
        for ep in eps:
            m |= smap[ep]
        return m

    for b in bins:
        level = a7._bin_level(b)
        epochs = sorted(units[b]) if b in units else []
        accD, accQ = Acc(), Acc()
        T0 = {p: float(products.flood_level(level, 0.0, 0.0, c_area[p])) for p in names}
        pred_cache = {}

        def pred(p, T):
            if (p, T) not in pred_cache:
                pred_cache[(p, T)] = fmaps[p].waterline(T)
            return pred_cache[(p, T)]

        def samples_for(eps, parities=(0, 1)):
            """{config: (code, waterline cells, labels)} of one observed sample."""
            out_s = {}
            raw = merged(b, eps, parities)
            if "replica" in configs or "static" in configs or "cd" in configs or "variant" in configs:
                code = classify(with_env(raw))
                rc = rev_r.obs_waterline(code, outside)
                lab = labels_fp if hybrid else labels_main
                if "variant" in configs:
                    out_s["variant"] = (code, rc, lab)
                if "replica" in configs or "cd" in configs:
                    out_s["replica"] = (code, rc, lab)
                if "static" in configs:
                    fl = flag_static(eps)
                    rcs = rev_r.drop_cells(rc, fl)
                    out_s["static"] = (code, rcs, lab)
                    if len(parities) == 2 and len(eps) == 1:
                        r_, c_ = rc
                        st = fl[r_, c_] if len(r_) else np.empty(0, bool)
                        for k in IDS:
                            m = sector[r_, c_] == k if len(r_) else np.empty(0, bool)
                            rows_static.append({"epoch": eps[0], "bin": b, "sector": NAME[k], "n_cells": int(m.sum()),
                                                "n_static": int((m & st).sum())})
            if hybrid:                           # the primary on the same footprint
                rawp = raw_of(units_main, b, eps, parities)
                codep = classify(with_env(rawp, "primary_fp"))
                out_s["primary_fp"] = (codep, rev_r.obs_waterline(codep, outside), labels_fp)
            if "res" in configs:
                wr = with_env(raw)
                for g, (nid, n_nat) in nids.items():
                    agg = rev_r.majority_to_native(wr, np.asarray(nid), n_nat)
                    code_g = classify(agg)
                    out_s[f"res_{g}"] = (code_g, rev_r.obs_waterline(code_g, outside), labels_main)
            return out_s

        def floor(sa, sb, tag_q):
            for cfg in sa:
                ca, rca, lab = sa[cfg]
                cb, rcb, _ = sb[cfg]
                cells = rev_r.cells_from(rca, rcb, rev_r.obs_at(ca, outside), rev_r.obs_at(cb, outside), lab)
                _, res = _summ(cells)
                accQ.add((("config", cfg), ("kind", tag_q)), res)
                accQ.add((("config", cfg), ("kind", "pooled_level")), res)

        e1_samples = None
        for ep in epochs:
            full = samples_for([ep])
            for cfg, (code, rc, lab) in full.items():
                prods = names
                if cfg.startswith("res_"):
                    prods = [p for p in names if GRID_OF[p] == cfg[4:]]
                for p in prods:
                    ks = K_GRID if (cfg == "replica" and "cd" in configs) else [0]
                    for k in ks:
                        c = round(c_area[p] + C_STEP * k, 2) if k else c_area[p]
                        T = float(products.flood_level(level, 0.0, 0.0, c)) if k else T0[p]
                        cells = rev_r.cells_from(pred(p, T), rc, lambda r, cc, p=p, T=T: fmaps[p].at(r, cc, T),
                                                 rev_r.obs_at(code, outside), lab)
                        cells, res = _summ(cells)
                        if k == 0:
                            accD.add((("config", "res" if cfg.startswith("res_") else cfg), ("product", p),
                                      ("k", 0), ("c_m", c)), res)
                            if cfg in ("replica", "variant", "primary_fp"):
                                for kk in IDS:
                                    r = res[kk]
                                    rows_U.append({"config": cfg, "product": p, "epoch": ep, "bin": b,
                                                   "sector": NAME[kk], "n": r["n"], "median_m": r["median"]})
                        if cfg == "replica" and "cd" in configs:
                            if k != 0:
                                accD.add((("config", "cd"), ("product", p), ("k", int(k)), ("c_m", c)), res)
                            n_tr, med_tr = train_median(cells, train)
                            rows_obj.append({"product": p, "k": int(k), "c_m": c, "epoch": ep, "bin": b,
                                             "n_train": n_tr, "median_train_m": med_tr})
            if ep == "E1":
                e1_samples = full
            del full
            if set(units[b][ep]) == {0, 1}:
                floor(samples_for([ep], (0,)), samples_for([ep], (1,)), "halves")
        e23 = [ep for ep in epochs if ep in ("E2", "E3")]
        if "E1" in epochs and e23:
            floor(e1_samples, samples_for(e23), "e1_vs_e23")
        e1_samples = None
        rows_D += accD.rows(b, level)
        rows_Q += accQ.rows(b, level)
        print(f"bin {b} ({level:.2f} m) {epochs} done, {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows_D).to_csv(out / "levels_D.csv.gz", index=False)
    pd.DataFrame(rows_Q).to_csv(out / "levels_Q.csv", index=False)
    pd.DataFrame(rows_U).to_csv(out / "units_D.csv.gz", index=False)
    if rows_obj:
        pd.DataFrame(rows_obj).to_csv(out / "cd_objective.csv", index=False)
    if rows_static:
        pd.DataFrame(rows_static).to_csv(out / "static_cells.csv", index=False)
    rec = {"stage2": tag, "hybrid": hybrid, "configs": configs, "products": names, "bins": bins,
           "surfaces": a.surfaces, "constants_file": a.constants or None,
           "c_area_m": c_area, "k_grid": K_GRID.tolist(), "c_step_m": C_STEP,
           "envelope_cells": {k: int(v.sum()) for k, v in env.items()},
           "footprint_cells": int(footprint.sum()) if hybrid else None, "seconds": round(time.time() - t0, 1)}
    (out / "run.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec), flush=True)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["prep", "pass"])
    ap.add_argument("--configs", default="replica")
    ap.add_argument("--stage2", default=TAG)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--bins", default="")
    ap.add_argument("--out", default="main")
    ap.add_argument("--products", default=",".join(products.PRODUCTS))
    ap.add_argument("--surfaces", default="products", choices=["products", "controls"])
    ap.add_argument("--c-step", type=float, default=0.0, help="c search step (default 0.10 m)")
    ap.add_argument("--constants", default="", help="with --surfaces controls: JSON of constants")
    a = ap.parse_args(argv)
    return cmd_prep(a) if a.command == "prep" else run_pass(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
