"""Revision, workstream V: a planar offset field per product, c(x, y) = c0 + a x + b y,
fitted on training-block units by the Eq. S11 criterion, and the H1 audit with it
(PREREG Amendment 18 item 9, second bullet; D-121).

masks:   per (epoch, level) unit of the primary fit (the 43 units of D-070), the known
         and water cells of the fit region (domain and training blocks), bit-packed,
         with the record-low envelope and sea connection exactly as
         scripts/a7_hindcast.unit_counts sees them. data/interim/revision_v/fit_masks.npz.
fit:     per product, on the primary's cached flood-level map F (corrected heights):
         0. check: exact counts on F reproduce the primary D-070 constant and objective;
         1. search (a, b) with c0 profiled out (D-070 rule on the 0.01 m grid over
            +-5 m), using the threshold approximation F_ab ~ F - (a x + b y), on a coarse
            grid at 1/16 of the cells, then a finer grid at 1/4 of the cells;
         2. the exact flood-level map of the corrected heights less a x + b y at the
            chosen (a, b), and the exact D-070 profile of c0 on it.
         results/revision/v/planar/fit_<product>.json; the exact map is cached under
         data/interim/hindcast/flood/.
audit:   scripts/a7_hindcast.run on the exact maps with the constant c0 (as a product
         offset of -c0, so its "total" rows are the planar variant), then the H1 rule
         (scripts/a7_h1_decision.py). results/revision/v/hindcast_planar/.
compare: verdicts, sector medians and the Eq. S11 objective against the primary
         (domain constant) and the sector-constant run R2 (results/hindcast/main_sectorshape).

x and y are the equal-area grid coordinates in km (origin 45.5 N, 50 E); a and b in m
per km. Usage: python scripts/rev_v_planar.py masks|fit|audit|compare|all|sector-all
[--products p,...] [--criterion domain|sector]. all = fit, audit and compare in one process;
sector-all = the per-sector sensitivity of the D-121 addendum (GEBCO_2026 and ETOPO1).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import hindcast_tiled as ht
from caspian import products
from caspian import watersurface as ws
from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision" / "v"
FIT_DIR = OUT / "planar"
RUN_DIR = OUT / "hindcast_planar"
CACHE = INTERIM / "revision_v"
MASKS = CACHE / "fit_masks.npz"
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]
# D-121 search: coarse grid, its one-time extension, and the refinement
COARSE_STEP, COARSE_MAX, COARSE_MAX_EXT, COARSE_SUB = 0.001, 0.010, 0.020, 4     # m/km; sub = every 4th row/col
FINE_STEP, FINE_HALF, FINE_SUB = 0.0002, 0.001, 2


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a7 = _load("a7_hindcast")


def setup():
    run_rec = json.loads((RESULTS / "hindcast" / "main" / "run.json").read_text(encoding="utf-8"))
    tile = int(run_rec["tile"])
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    static = a7.production_static(transform, shape, tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    blocks = pd.read_csv(a7.BLOCKS)
    fit_region = static["domain"] & a7.train_mask(blocks, transform, shape)
    return run_rec, tile, transform, shape, static, blocks, fit_region


def cell_xy_km(idx, transform, shape):
    rows, cols = np.divmod(np.asarray(idx, np.int64), shape[1])
    x = (float(transform[2]) + float(transform[0]) * (cols + 0.5)) / 1000.0
    y = (float(transform[5]) + float(transform[4]) * (rows + 0.5)) / 1000.0
    return x, y, rows, cols


# --- masks ------------------------------------------------------------------------------

def masks(argv) -> int:
    run_rec, tile, transform, shape, static, blocks, fit_region = setup()
    stage2 = INTERIM / "stage2" / "main"
    units = a7.stage2_index(stage2)
    a7.ENVELOPE = a7.record_low_water(units, transform, shape)
    ulist = [(b, ep) for b in sorted(units) for ep in sorted(units[b])]
    prim = json.loads((RESULTS / "hindcast" / "main" / "shape_offsets.json").read_text(encoding="utf-8"))
    if [(u["bin"], u["epoch"]) for u in prim["units"]] != ulist:
        raise SystemExit("the units differ from the primary D-070 fit")
    idx = np.flatnonzero(fit_region.ravel()).astype(np.int64)
    if idx.size != prim["fit_region_cells"]:
        raise SystemExit("the fit region differs from the primary")
    outside = static["outside"]
    known_b, water_b = [], []
    t0 = time.time()
    for i, (b, ep) in enumerate(ulist):
        raw, _ = a7.observed_raw(a7.groups_for(units, b, [ep]), transform, shape)
        code = a7.classify(raw, static, tile)
        del raw
        k = np.zeros(idx.size, bool)
        wtr = np.zeros(idx.size, bool)
        flat_c, flat_o = code.ravel(), np.asarray(outside).ravel()
        for s0 in range(0, idx.size, 1 << 24):
            j = idx[s0:s0 + (1 << 24)]
            c, o = flat_c[j], flat_o[j]
            k[s0:s0 + j.size] = (c >= 0) | ((c == -2) & (o >= 0))
            wtr[s0:s0 + j.size] = (c == 1) | ((c == -2) & (o == 1))
        known_b.append(np.packbits(k))
        water_b.append(np.packbits(wtr))
        del code
        print(f"masks: unit {ep} bin {b} ({i + 1}/{len(ulist)}), {time.time() - t0:.0f} s", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(MASKS, idx=idx, known=np.stack(known_b), water=np.stack(water_b), n=np.int64(idx.size),
             bins=np.array([b for b, _ in ulist]), epochs=np.array([ep for _, ep in ulist]),
             levels=np.array([a7._bin_level(b) for b, _ in ulist]), shape=np.array(shape),
             transform=np.array([float(v) for v in transform[:6]]))
    print("masks written", MASKS, f"{time.time() - t0:.0f} s", flush=True)
    return 0


# --- counting -----------------------------------------------------------------------------

class FitData:
    def __init__(self, path=MASKS):
        z = np.load(path)
        self.idx, self.n = z["idx"], int(z["n"])
        self.known, self.water = z["known"], z["water"]
        self.levels, self.epochs, self.bins = z["levels"], [str(e) for e in z["epochs"]], z["bins"]
        self.shape, self.transform = tuple(int(v) for v in z["shape"]), z["transform"]
        x, y, rows, cols = cell_xy_km(self.idx, self.transform, self.shape)
        # compact: km coordinates in float32 (resolution < 0.1 m at 400 km), indices in int32
        self.x, self.y = x.astype(np.float32), y.astype(np.float32)
        self.rows, self.cols = rows.astype(np.int32), cols.astype(np.int32)

    def unit(self, u, which="known"):
        return np.unpackbits(getattr(self, which)[u], count=self.n).astype(bool)


def exact_bias(fd: FitData, Fv, finite, c_grid=a7.C_GRID) -> np.ndarray:
    """[unit, c] signed area bias (cells) exactly as scripts/a7_hindcast.unit_counts:
    known cells with F below L + c (compared in float64, searchsorted side right)."""
    bias = np.zeros((len(fd.levels), len(c_grid)), np.int64)
    for u, L in enumerate(fd.levels):
        T = products.flood_level(float(L), 0.0, 0.0, c_grid)
        k = fd.unit(u) & finite
        v = np.asarray(Fv[k], float)
        cnt = np.cumsum(np.bincount(np.searchsorted(T, v, side="right"), minlength=T.size + 1))[:T.size]
        bias[u] = cnt - int((fd.unit(u, "water") & finite).sum())
    return bias


def exact_bias_sector(fd: FitData, Fv, finite, sector, c_grid=a7.C_GRID) -> np.ndarray:
    """[unit, sector code, c]: exact_bias split by the sector code of each fit cell
    (D-121 addendum); summed over sectors it equals exact_bias."""
    sector = np.asarray(sector, np.int64)
    ns = int(sector.max()) + 1
    out = np.zeros((len(fd.levels), ns, len(c_grid)), np.int64)
    for u, L in enumerate(fd.levels):
        T = products.flood_level(float(L), 0.0, 0.0, c_grid)
        k = fd.unit(u) & finite
        idx = np.searchsorted(T, np.asarray(Fv[k], float), side="right")
        h = np.bincount(sector[k] * (T.size + 1) + idx, minlength=ns * (T.size + 1)).reshape(ns, T.size + 1)
        cnt = np.cumsum(h, axis=1)[:, :T.size]
        wtr = np.bincount(sector[fd.unit(u, "water") & finite], minlength=ns)
        out[u] = cnt - wtr[:, None]
    return out


class Approx:
    """The threshold approximation on a subsample: per unit, known & finite masks and
    water counts; bias(a, b) from the quantised F - a x - b y (0.01 m bins)."""

    def __init__(self, fd: FitData, Fv, finite, sub: int, c_grid=a7.C_GRID, sector=None):
        sel = (fd.rows % sub == 0) & (fd.cols % sub == 0) if sub > 1 else np.ones(fd.n, bool)
        self.s = None if sector is None else np.asarray(sector, np.int64)[sel]
        self.F = np.where(np.isfinite(Fv[sel]), Fv[sel], 1e6).astype(np.float64)
        self.x, self.y = fd.x[sel].astype(np.float64), fd.y[sel].astype(np.float64)
        fin = finite[sel]
        self.k = [fd.unit(u)[sel] & fin for u in range(len(fd.levels))]
        self.w = np.array([int((fd.unit(u, "water")[sel] & fin).sum()) for u in range(len(fd.levels))])
        self.c_grid = np.asarray(c_grid, float)
        self.m = [np.rint((float(L) + self.c_grid) * 100).astype(np.int64) for L in fd.levels]
        self.qlo = min(int(m.min()) for m in self.m)
        self.qhi = max(int(m.max()) for m in self.m)
        self.scale = sub * sub
        if self.s is not None:      # D-121 addendum: per-sector counts
            self.ns = int(self.s.max()) + 1
            self.ws = [np.bincount(self.s[fd.unit(u, "water")[sel] & fin], minlength=self.ns)
                       for u in range(len(fd.levels))]

    def bias(self, a: float, b: float) -> np.ndarray:
        """[unit, c] signed bias (cells); with sectors, [unit x sector, c]."""
        v = self.F - (a * self.x + b * self.y)
        q = np.clip(np.floor(v * 100.0).astype(np.int64) + 1, self.qlo, self.qhi + 1) - self.qlo
        nb = self.qhi - self.qlo + 2
        if self.s is not None:
            out3 = np.zeros((len(self.k), self.ns, self.c_grid.size), np.int64)
            for u, k in enumerate(self.k):
                h = np.bincount(self.s[k] * nb + q[k], minlength=self.ns * nb).reshape(self.ns, nb)
                out3[u] = np.cumsum(h, axis=1)[:, self.m[u] - self.qlo] - self.ws[u][:, None]
            return out3.reshape(-1, self.c_grid.size)
        out = np.zeros((len(self.k), self.c_grid.size), np.int64)
        for u, k in enumerate(self.k):
            cum = np.cumsum(np.bincount(q[k], minlength=nb))
            out[u] = cum[self.m[u] - self.qlo] - self.w[u]       # cells with q <= m, i.e. v < m / 100
        return out

    def profile(self, a: float, b: float) -> dict:
        r = ws.profile_c0(self.bias(a, b), self.c_grid)
        r["objective_km2_scaled"] = r["objective"] * a7.CELL_KM2 * self.scale
        return r


def grid_search(ap: Approx, centre, half, step, log) -> tuple[dict, list]:
    vals = np.round(np.arange(-half, half + step / 2, step), 6)
    rows = []
    for da in vals:
        for db in vals:
            a, b = round(centre[0] + da, 6), round(centre[1] + db, 6)
            r = ap.profile(a, b)
            rows.append({"a": a, "b": b, "c0_m": r["c0_m"], "objective_km2": r["objective_km2_scaled"]})
    best = min(rows, key=lambda r: (r["objective_km2"], abs(r["a"]) + abs(r["b"])))
    log(f"  grid +-{half} step {step} around {centre}: best a={best['a']} b={best['b']} c0={best['c0_m']} "
        f"obj={best['objective_km2']:.1f}", flush=True)
    return best, rows


def plane_corr(corr, a: float, b: float, transform, shape, tile, path: Path):
    """The correction less a x + b y, as a float32 memory map (cached)."""
    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
        for r0, r1, c0, c1 in ht.tiles(shape, tile):
            x = (float(transform[2]) + float(transform[0]) * (np.arange(c0, c1) + 0.5)) / 1000.0
            y = (float(transform[5]) + float(transform[4]) * (np.arange(r0, r1) + 0.5)) / 1000.0
            pl = a * x[None, :] + b * y[:, None]
            mm[r0:r1, c0:c1] = (np.asarray(corr[r0:r1, c0:c1], np.float64) - pl).astype(np.float32)
        mm.flush()
        del mm
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def _key(d) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _profile_exact(bx, epochs, criterion):
    """c0 by the D-070 rule on exact counts: domain ([unit, c]) or per sector ([unit, sector, c])."""
    if criterion == "domain":
        r = a7.fit_offset(bx, epochs)
        return r["offset_m"], r["objective_km2_at_offset"], r["n_tied_minima"]
    r = ws.profile_c0(bx.reshape(-1, bx.shape[-1]), a7.C_GRID)
    return r["c0_m"], round(r["objective"] * a7.CELL_KM2, 4), r["n_tied"]


def fit(argv) -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--products", default=None)
    ap_.add_argument("--criterion", default="domain", choices=["domain", "sector"],
                     help="domain: the registered Eq. S11 (D-121); sector: the per-sector sensitivity (D-121 addendum)")
    args = ap_.parse_args(argv)
    crit = args.criterion
    sfx = "" if crit == "domain" else "sec"
    run_rec, tile, transform, shape, static, blocks, fit_region = setup()
    del fit_region
    fd = FitData()
    sec = np.asarray(static["sector"]).reshape(-1)[fd.idx].astype(np.int64) if crit == "sector" else None
    prim = json.loads((RESULTS / "hindcast" / "main" / "shape_offsets.json").read_text(encoding="utf-8"))
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != run_rec["vertical_correction"]["sha256"]:
        raise SystemExit("the correction is not the primary run's")
    names = args.products.split(",") if args.products else list(run_rec["products"])
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    cache = INTERIM / "hindcast" / "flood"
    for p in names:
        t0 = time.time()
        raw = products.heights(p, transform, shape, run_rec["resampling"])
        h = products.Corrected(raw, corr)
        before = set(cache.glob(f"{p}_*.npy"))
        F = a7.flood_map_cached(p, h, static["sea_seed"], {"heights": Path(raw.filename).name,
                                                           "correction": corr_rec["sha256"]}, tile)
        if set(cache.glob(f"{p}_*.npy")) != before:
            raise SystemExit(f"{p}: the primary flood-level map was not cached")
        flatF = np.asarray(F).reshape(-1)
        Fv = flatF[fd.idx]
        finite = np.zeros(fd.n, bool)       # as a7_hindcast: the corrected height (float32 sum) is finite
        flat_h = np.asarray(raw).reshape(-1)
        flat_c = np.asarray(corr).reshape(-1)
        for s0 in range(0, fd.n, 1 << 24):
            j = fd.idx[s0:s0 + (1 << 24)]
            finite[s0:s0 + j.size] = np.isfinite(flat_h[j].astype(np.float32) + flat_c[j].astype(np.float32))
        # 0. exact counts on the primary map reproduce the primary D-070 fit
        b0 = exact_bias(fd, Fv, finite)
        r0 = a7.fit_offset(b0, fd.epochs)
        want = prim["products"][p]
        if r0["offset_m"] != want["offset_m"] or abs(r0["objective_km2_at_offset"] - want["objective_km2_at_offset"]) > 1e-3:
            raise SystemExit(f"{p}: exact counts give c={r0['offset_m']} obj={r0['objective_km2_at_offset']}, "
                             f"primary c={want['offset_m']} obj={want['objective_km2_at_offset']}")
        print(f"{p}: primary fit reproduced (c={r0['offset_m']}, {r0['objective_km2_at_offset']:.1f} km2), "
              f"{time.time() - t0:.0f} s", flush=True)
        rec = {"product": p, "primary": {"c_m": r0["offset_m"], "objective_km2": r0["objective_km2_at_offset"]},
               "units": len(fd.levels), "fit_region_cells": fd.n, "criterion": crit}
        if crit == "sector":
            bs0 = exact_bias_sector(fd, Fv, finite, sec)
            if not np.array_equal(bs0.sum(axis=1), b0):
                raise SystemExit(f"{p}: the per-sector counts do not sum to the domain counts")
            k0 = int(np.flatnonzero(np.isclose(a7.C_GRID, r0["offset_m"]))[0])
            rec["primary"]["objective_sector_km2"] = round(float(np.abs(bs0[:, :, k0]).sum()) * a7.CELL_KM2, 4)
            c_s, o_s, _ = _profile_exact(bs0, fd.epochs, "sector")
            rec["constant_by_sector_criterion"] = {"c_m": c_s, "objective_sector_km2": o_s}
            del bs0
        # 1. approximate search
        apx = Approx(fd, Fv, finite, COARSE_SUB, sector=sec)
        z = apx.profile(0.0, 0.0)
        rec["approx_check_at_zero"] = {"c0_m": z["c0_m"], "objective_km2_scaled": z["objective_km2_scaled"],
                                       "subsample": COARSE_SUB}
        best, rows = grid_search(apx, (0.0, 0.0), COARSE_MAX, COARSE_STEP, print)
        extended = False
        if max(abs(best["a"]), abs(best["b"])) >= COARSE_MAX - 1e-9:
            extended = True
            best, rows2 = grid_search(apx, (0.0, 0.0), COARSE_MAX_EXT, COARSE_STEP, print)
            rows += rows2
        rec["coarse"] = {"best": best, "extended": extended, "subsample": COARSE_SUB,
                         "on_edge": bool(max(abs(best["a"]), abs(best["b"])) >=
                                         (COARSE_MAX_EXT if extended else COARSE_MAX) - 1e-9)}
        pd.DataFrame(rows).to_csv(FIT_DIR / f"search_coarse{sfx}_{p}.csv", index=False)
        del apx
        apx = Approx(fd, Fv, finite, FINE_SUB, sector=sec)
        fbest, frows = grid_search(apx, (best["a"], best["b"]), FINE_HALF, FINE_STEP, print)
        pd.DataFrame(frows).to_csv(FIT_DIR / f"search_fine{sfx}_{p}.csv", index=False)
        rec["fine"] = {"best": fbest, "subsample": FINE_SUB}
        del apx
        a, b = fbest["a"], fbest["b"]
        # the threshold approximation at full resolution, counted exactly unit by unit (low memory)
        va = Fv - (a * fd.x.astype(np.float64) + b * fd.y.astype(np.float64))
        bf = exact_bias(fd, va, finite) if crit == "domain" else exact_bias_sector(fd, va, finite, sec)
        cf, of, _ = _profile_exact(bf, fd.epochs, crit)
        rec["approx_full_at_best"] = {"c0_m": cf, "objective_km2": of}
        del bf
        # 2. exact flood-level map at (a, b) and the exact c0 profile
        key = _key({"product": p, "a": a, "b": b, "corr": corr_rec["sha256"], "shape": list(shape)})
        cname = f"planar{sfx}_corr_{p}_{key}.npy"
        cp = plane_corr(corr, a, b, transform, shape, tile, CACHE / cname)
        hp = products.Corrected(raw, cp)
        Fp = a7.flood_map_cached(f"{p}_planar{sfx}", hp, static["sea_seed"],
                                 {"heights": Path(raw.filename).name, "correction": cname}, tile)
        Fpv = np.asarray(Fp).reshape(-1)[fd.idx]
        bx = exact_bias(fd, Fpv, finite)
        rx = a7.fit_offset(bx, fd.epochs)
        if crit == "domain":
            c0x, ox, tx = rx["offset_m"], rx["objective_km2_at_offset"], rx["n_tied_minima"]
        else:
            bxs = exact_bias_sector(fd, Fpv, finite, sec)
            c0x, ox, tx = _profile_exact(bxs, fd.epochs, "sector")
            k0 = int(np.flatnonzero(np.isclose(a7.C_GRID, c0x))[0])
            rec["exact_domain_objective_at_c0_km2"] = round(float(np.abs(bx[:, k0]).sum()) * a7.CELL_KM2, 4)
        rec["exact"] = {"a_m_per_km": a, "b_m_per_km": b, "c0_m": c0x, "objective_km2": ox, "n_tied_minima": tx,
                        "criterion": crit, "flood_map": f"{p}_planar{sfx} (key in data/interim/hindcast/flood)",
                        "correction_cache": cname}
        base = (r0["objective_km2_at_offset"] if crit == "domain"
                else rec["constant_by_sector_criterion"]["objective_sector_km2"])
        rec["objective_reduction_vs_primary_pct"] = round(100 * (1 - ox / base), 2)
        rec["seconds"] = round(time.time() - t0, 1)
        (FIT_DIR / f"fit{sfx}_{p}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(json.dumps({k: rec[k] for k in ("primary", "exact", "objective_reduction_vs_primary_pct", "seconds")}),
              flush=True)
    return 0


# --- audit and comparison --------------------------------------------------------------------

RUN_DIR_SEC = OUT / "hindcast_planar_sector"


def audit(argv) -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--products", default=None)
    ap_.add_argument("--criterion", default="domain", choices=["domain", "sector"])
    args = ap_.parse_args(argv)
    sfx = "" if args.criterion == "domain" else "sec"
    run_dir = RUN_DIR if args.criterion == "domain" else RUN_DIR_SEC
    run_rec, tile, transform, shape, static, blocks, fit_region = setup()
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    names = args.products.split(",") if args.products else list(run_rec["products"])
    heights, fmaps, offs, fits = {}, {}, {}, {}
    cache = INTERIM / "hindcast" / "flood"
    for p in names:
        rec = json.loads((FIT_DIR / f"fit{sfx}_{p}.json").read_text(encoding="utf-8"))
        ex = rec["exact"]
        raw = products.heights(p, transform, shape, run_rec["resampling"])
        heights[p] = products.Corrected(raw, corr)
        cp = np.load(CACHE / ex["correction_cache"], mmap_mode="r")
        before = set(cache.glob(f"{p}_planar{sfx}_*.npy"))
        fmaps[p] = a7.flood_map_cached(f"{p}_planar{sfx}", products.Corrected(raw, cp), static["sea_seed"],
                                       {"heights": Path(raw.filename).name, "correction": ex["correction_cache"]}, tile)
        if set(cache.glob(f"{p}_planar{sfx}_*.npy")) != before:
            raise SystemExit(f"{p}: the planar flood-level map was not cached by the fit")
        offs[p] = {"offset_m": -float(ex["c0_m"]), "source": "minus the planar c0 (D-121), so the total rows "
                                                             "are the planar variant"}
        fits[p] = {k: ex[k] for k in ("a_m_per_km", "b_m_per_km", "c0_m")}
    stage2 = INTERIM / "stage2" / "main"
    a7.ENVELOPE = a7.record_low_water(a7.stage2_index(stage2), transform, shape)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tag": run_dir.name, "variant": f"A18 item 9: planar offset field c0 + a x + b y (D-121; criterion "
                                            f"{args.criterion}); total rows = the planar variant", "planar_fits": fits,
            "resampling": run_rec["resampling"], "tile": tile, "envelope": run_rec.get("envelope"),
            "vertical_correction": corr_rec}
    a7.run(stage2, run_dir, static, heights, transform, product_offsets=offs, shape_offsets=None,
           fit_region=fit_region, flood_maps=fmaps, blocks=blocks, tile=tile, meta=meta)
    _load("a7_h1_decision").main("../revision/v/" + run_dir.name)
    return 0


def _objective(d: Path, variant: str) -> dict:
    """Sum over (epoch, level) units of |area bias| (km2) on the fit region (domain_train
    rows of extent.csv), per product: the Eq. S11 objective."""
    e = pd.read_csv(d / "extent.csv")
    e = e[(e["variant"] == variant) & (e["sector"] == "domain_train")]
    col = "area_bias_total_km2" if "area_bias_total_km2" in e.columns else None
    if col is None:
        return {}
    return {p: round(float(g[col].abs().sum()), 2) for p, g in e.groupby("product")}


def sector_medians(d: Path, variant: str) -> pd.DataFrame:
    """Table S8's sector median (h1.disp_median): the median over the evaluable levels of
    the per-level median displacement, with the median of the per-level floor (p95) and
    of their ratio D/Q."""
    det = pd.read_csv(d / "h1_mispredicts_detail.csv")
    det = det[(det["evaluable"] == True) & (det["variant"] == variant)].copy()      # noqa: E712
    det["dq"] = det["median"] / det["p95_noise"]
    return (det.groupby(["product", "sector"])
            .agg(sector_median_m=("median", "median"), floor_p95_m=("p95_noise", "median"), dq_median=("dq", "median"))
            .reset_index())


def compare(argv) -> int:
    runs = {"domain_constant": (RESULTS / "hindcast" / "main", "shape"),
            "sector_constants_R2": (RESULTS / "hindcast" / "main_sectorshape", "total"),
            "planar": (RUN_DIR, "total")}
    if (RUN_DIR_SEC / "h1_mispredicts.csv").exists():
        runs["planar_sector_sensitivity"] = (RUN_DIR_SEC, "total")
    rows, med = [], []
    for name, (d, var) in runs.items():
        m = pd.read_csv(d / "h1_mispredicts.csv")
        m = m[m["variant"] == var].assign(run=name)
        rows.append(m)
        med.append(sector_medians(d, var).assign(run=name))
    t = pd.concat(rows, ignore_index=True).merge(pd.concat(med, ignore_index=True), on=["run", "product", "sector"],
                                                  how="left")
    t.to_csv(OUT / "planar_comparison.csv", index=False)
    summ = {}
    for (run, p), g in t.groupby(["run", "product"]):
        mp = sorted(g.loc[g["mispredicts"] == True, "sector"])     # noqa: E712
        summ.setdefault(p, {})[run] = {"n_mispredicted": len(mp), "n_validated": len(set(mp) & set(VALIDATED)),
                                       "mispredicted": mp,
                                       "median_of_sector_medians_m": round(float(g["sector_median_m"].median()), 1)}
    obj = {name: _objective(d, var) for name, (d, var) in runs.items()}
    fits, fits_sec = {}, {}
    for f in sorted(FIT_DIR.glob("fitsec_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        fits_sec[r["product"]] = {"exact": r["exact"], "constant_by_sector_criterion": r["constant_by_sector_criterion"],
                                  "objective_reduction_vs_constant_pct": r["objective_reduction_vs_primary_pct"],
                                  "exact_domain_objective_at_c0_km2": r.get("exact_domain_objective_at_c0_km2"),
                                  "coarse_on_edge": r["coarse"]["on_edge"]}
    for f in sorted(FIT_DIR.glob("fit_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        fits[r["product"]] = {"exact": r["exact"], "primary": r["primary"],
                              "objective_reduction_vs_primary_pct": r["objective_reduction_vs_primary_pct"],
                              "coarse_on_edge": r["coarse"]["on_edge"]}
    out = {"verdicts": summ, "objective_eq_s11_km2_extent_domain_train": obj, "fits": fits,
           "fits_sector_sensitivity": fits_sec}
    (OUT / "planar_comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def run_all(argv) -> int:
    """fit, audit and compare in one process (one heavy slot)."""
    for step in (fit, audit, compare):
        rc = step(argv if step is not compare else [])
        if rc:
            return rc
    return 0


SENSITIVITY_PRODUCTS = "gebco_2026,etopo1"     # D-121 addendum


def run_sector(argv) -> int:
    """The per-sector sensitivity (D-121 addendum): fit, audit and compare in one process."""
    args = ["--criterion", "sector", "--products", SENSITIVITY_PRODUCTS]
    for step in (fit, audit):
        rc = step(args)
        if rc:
            return rc
    return compare([])


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    sys.exit({"masks": masks, "fit": fit, "audit": audit, "compare": compare, "all": run_all,
              "sector-all": run_sector}[cmd](rest))
