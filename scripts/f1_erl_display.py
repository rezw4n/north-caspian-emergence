"""Display preparations for the ERL main figures: the shorelines, product-
agreement counts and arrows that figures/erl_fig*.py draw, rebuilt with the hindcast's and
the H3 driver's own functions from their own inputs, and cross-checked against the frozen
result tables. Nothing here is a new result: every count and distance is checked against
results/hindcast/<tag>/ or results/emergence/<tag>/, and the run stops on any difference.

Commands
  caspian   (1) Hero map (figure A). For two hindcast units, the observed extent exactly as
            scripts/a7_hindcast.py builds it (observed_raw with the record-low envelope,
            classify with the sea connection and the outside states) and
            the shape-variant prediction of GEBCO_2026, ETOPO 2022 and SRTM15+ (the
            product's flood-level map from the hindcast cache, thresholded at the bin
            centre plus its shape constant, results/hindcast/<tag>/shape_offsets.json).
            The per-sector median displacement of every (unit, product) is recomputed with
            caspian.hindcast_tiled and must equal displacement.csv. For each zoom inset
            and product, one arrow: from the observed waterline cell inside the inset,
            in the inset's sector, whose distance to the nearest predicted waterline cell
            is closest to that sector's median, to that nearest cell.
            (2) Product agreement (figure C). Per cell of the domain, the number of the
            five products, uncorrected (total-variant heights, as the H3 driver reads
            them), that place it in the band that emerges between the 2026 level and 1 m
            (2 m) below it: L26 - D <= F < L26 on the product's flood-level map. Each
            product's count must equal its emergence.csv domain estimate, cell for cell.
  gsl       (3) Great Salt Lake map (figure D). Per arm, the static layers, envelope and
            flood-level maps of scripts/g1_hindcast.py, rebuilt with its code, at one
            representative level: the observed extent and the shape-variant predictions
            of the five products and of the 3DEP control; every per-arm median
            displacement must equal the arm's displacement.csv.

Outputs
  data/interim/figures/erl/*.npz   display rasters (git-ignored): main-map fields on a
                                   240 m grid, insets at 60 m, agreement counts
  results/display/erl_hero.json, erl_agreement.json, erl_gsl.json
                                   the units, constants, arrows, class areas and the
                                   cross-checks (tools/build_metrics.py, display_erl_metrics)

Memory: whole-grid arrays are 1-byte masks and codes; heights and flood-level maps are
memory maps; peak about 2 GB.
Usage: python scripts/f1_erl_display.py caspian|gsl [--tag main]
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
from scipy.spatial import cKDTree

from caspian import hindcast_tiled as ht
from caspian import products
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a7_hindcast", ROOT / "scripts" / "a7_hindcast.py")
a7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a7)

CACHE = INTERIM / "figures" / "erl"
OUT = RESULTS / "display"
RES = a7.RES
CELL_KM2 = a7.CELL_KM2
FACTOR = 4                       # main-map display grid: 4 x 60 m = 240 m
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
PLAN = INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv"     # the main stage-2 plan

# Figure A. Two units on either side of the -27.7 m band edge, below which ETOPO 2022's
# emerged band comes from GEBCO lake depths and above which from FABDEM: the E3
# bins nearest -28.0 m and -27.0 m. E2 has no fully observed bin near -27.0 m: bin 31
# (-26.85 m) has no valid cell, and bin 32 (-26.75 m) 39 % of the valid area of bin 33
# (observed.csv), with unobserved levels in two sectors; so both units are E3.
HERO_UNITS = (("E3", 19), ("E3", 29))
HERO_PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7")
# Zoom insets (equal-area metres, x0, y0, x1, y1) and the sector whose arrows they carry,
# chosen on quick looks of the observed shorelines (cartographic choice). The 60 m
# rasters are kept for a larger region around each inset.
INSETS = {
    "volga": {"sector": "volga", "box": (-170_000.0, -40_000.0, -90_000.0, 30_000.0),
              "region": (-230_000.0, -60_000.0, -60_000.0, 70_000.0)},
    "northeast": {"sector": "northeast", "box": (190_000.0, 50_000.0, 270_000.0, 120_000.0),
                  "region": (170_000.0, 0.0, 300_000.0, 170_000.0)},
    "komsomolets": {"sector": "komsomolets", "box": (180_000.0, -35_000.0, 260_000.0, 35_000.0),
                    "region": (180_000.0, -80_000.0, 380_000.0, 50_000.0)},
}
AGREE_SCENARIOS = (1.0, 2.0)


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp(paths) -> str:
    return json.dumps([[Path(p).name, Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in sorted(map(str, paths))])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --- Caspian: grid, static layers, envelope, flood-level maps ---------------------------------

def caspian_grid(hrun: dict):
    from caspian.geography import DOMAIN
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    if [float(v) for v in transform[:6]] != hrun["grid"]["transform"][:6] or list(shape) != hrun["grid"]["shape"]:
        raise SystemExit("the hindcast grid differs from this one")
    return transform, shape


def static_main(tag: str, transform, shape, tile: int) -> dict:
    """a7.production_static (the hindcast's own static layers) and the record-low envelope,
    cached as compressed 1-byte arrays; the cache is keyed by the static inputs."""
    inputs = [a7.COVER, a7.SEA, a7.BEYOND, a7.STUDY / "extents_v6_250m.npz", ROOT / "src" / "caspian" / "geography.py",
              ROOT / "scripts" / "a7_hindcast.py", *sorted((INTERIM / "stage2" / tag).glob("tile_*.npz"))]
    path = CACHE / f"static_{tag}.npz"
    st = stamp(inputs)
    if path.exists():
        z = np.load(path)
        if str(z["_stamp"]) == st:
            log("static layers from cache")
            return {"domain": z["domain"], "sector": z["sector"], "sea_seed": z["sea_seed"], "outside": z["outside"],
                    "envelope": z["envelope"]}
    log("static layers: building (a7.production_static)")
    s = a7.production_static(transform, shape, tile)
    out = {"domain": s["domain"], "sector": s["sector"].astype(np.uint8), "sea_seed": s["sea_seed"],
           "outside": s["outside"].astype(np.int8)}
    del s
    log("record-low envelope")
    out["envelope"] = a7.record_low_water(a7.stage2_index(INTERIM / "stage2" / tag), transform, shape)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, _stamp=np.array(st), **out)
    return out


def flood_maps(names, hrun: dict, static: dict, transform, shape, tile: int) -> tuple[dict, dict]:
    """The hindcast's cached flood-level maps (total-variant heights: product + per-cell geoid
    correction). Stops if a cache is missing rather than building a new map."""
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != hrun["vertical_correction"]["sha256"]:
        raise SystemExit("the vertical correction differs from the hindcast run's")
    seed_sha = hashlib.sha256(np.packbits(static["sea_seed"])).hexdigest()
    F, fin = {}, {}
    for p in names:
        h = products.heights(p, transform, shape, hrun.get("resampling", "bilinear"))
        meta = {"heights": Path(h.filename).name, "correction": corr_rec["sha256"]}
        key = hashlib.sha256(json.dumps({**meta, "seed": seed_sha, "connectivity": 8}, sort_keys=True).encode()).hexdigest()[:12]
        if not (INTERIM / "hindcast" / "flood" / f"{p}_{key}.npy").exists():
            raise SystemExit(f"{p}: the hindcast's flood-level map {p}_{key}.npy is not cached")
        hh = products.Corrected(h, corr)
        F[p] = a7.flood_map_cached(p, hh, static["sea_seed"], meta, tile)
        fin[p] = (lambda g: lambda r0, r1, c0, c1: np.isfinite(np.asarray(g[r0:r1, c0:c1], float)))(hh)
    return F, fin


# --- display reductions ------------------------------------------------------------------------

def block_fraction(state: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Share of water among the known cells (state 1 water, 0 land, < 0 unknown) of each
    factor x factor block, NaN where a block has no known cell; float16. Trailing rows and
    columns that do not fill a block are dropped."""
    r, c = (state.shape[0] // factor) * factor, (state.shape[1] // factor) * factor
    out = np.full((r // factor, c // factor), np.nan, np.float16)
    step = 256 * factor
    for r0 in range(0, r, step):
        s = state[r0:min(r0 + step, r), :c]
        k = (s >= 0).reshape(s.shape[0] // factor, factor, c // factor, factor).sum(axis=(1, 3))
        w = (s == 1).reshape(s.shape[0] // factor, factor, c // factor, factor).sum(axis=(1, 3))
        out[r0 // factor:r0 // factor + k.shape[0]] = np.where(k > 0, w / np.maximum(k, 1), np.nan)
    return out


def window(transform, box):
    """Row and column slices of the 60 m grid covering box = (x0, y0, x1, y1)."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    x0, y0, x1, y1 = box
    c0, c1 = int(np.floor((x0 - c) / a)), int(np.ceil((x1 - c) / a))
    r0, r1 = int(np.floor((f - y1) / -e)), int(np.ceil((f - y0) / -e))
    return slice(r0, r1), slice(c0, c1)


def cell_xy(transform, rows, cols):
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    return c + a * (np.asarray(cols) + 0.5), f + e * (np.asarray(rows) + 0.5)


def obs_state(code: np.ndarray, outside: np.ndarray, domain: np.ndarray, tile: int) -> np.ndarray:
    """1 known water, 0 known land, -1 unknown (hindcast.known_states with the outside
    states), inside the domain; -1 outside it."""
    out = np.full(code.shape, -1, np.int8)
    for r0, r1, c0, c1 in ht.tiles(code.shape, tile):
        cd, st, dm = code[r0:r1, c0:c1], outside[r0:r1, c0:c1], domain[r0:r1, c0:c1]
        kw = (cd == 1) | ((cd == -2) & (st == 1))
        kl = (cd == 0) | ((cd == -2) & (st == 0))
        o = out[r0:r1, c0:c1]
        o[kw & dm] = 1
        o[kl & dm] = 0
    return out


def pred_state(F, level: float, domain: np.ndarray, tile: int, finite=None) -> np.ndarray:
    """As the hindcast's predicted states (hindcast.known_states with pred_valid): 1 flooded
    from the open sea at ``level`` (F < level, float64), 0 a finite product height not
    flooded (also where F is +inf, no path from the sea), -1 no product height; inside the
    domain, -1 outside it. ``finite``: window callable of the finite-height mask (default:
    F not NaN)."""
    out = np.full(domain.shape, -1, np.int8)
    for r0, r1, c0, c1 in ht.tiles(domain.shape, tile):
        f = np.asarray(F[r0:r1, c0:c1], float)
        ok = (~np.isnan(f)) if finite is None else np.asarray(finite(r0, r1, c0, c1), bool)
        dm = domain[r0:r1, c0:c1]
        o = out[r0:r1, c0:c1]
        o[dm & ok & (f < level)] = 1
        o[dm & ok & ~(f < level)] = 0
    return out


# --- (1) hero map ------------------------------------------------------------------------------

def hero(tag: str, hrun: dict, static: dict, F: dict, fin: dict, transform, shape, tile: int) -> dict:
    disp = pd.read_csv(RESULTS / "hindcast" / tag / "displacement.csv")
    so = json.loads((RESULTS / "hindcast" / tag / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    units = a7.stage2_index(INTERIM / "stage2" / tag)
    a7.ENVELOPE = static["envelope"]
    plan = pd.read_csv(PLAN, usecols=["acq", "t", "epoch", "bin"])
    ids = list(range(1, len(SECTOR_ORDER) + 1))
    name = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}
    code_of = {s: k + 1 for k, s in enumerate(SECTOR_ORDER)}
    rec = {"units": [], "products": list(HERO_PRODUCTS), "display_res_m": RES * FACTOR,
           "insets": {k: {"sector": v["sector"], "box_m": list(v["box"]), "region_m": list(v["region"])}
                      for k, v in INSETS.items()},
           "shape_offsets_m": {p: float(so[p]["offset_m"]) for p in HERO_PRODUCTS}, "checks": [], "arrows": []}
    fields, insets = {}, {}
    for ep, b in HERO_UNITS:
        level = a7._bin_level(b)
        log(f"hero unit {ep} bin {b} ({level:.2f} m): observed extent")
        raw, info = a7.observed_raw(a7.groups_for(units, b, [ep]), transform, shape)
        code = a7.classify(raw, static, tile)
        del raw
        ustate = obs_state(code, static["outside"], static["domain"], tile)
        u = f"{ep}_b{b:03d}"
        fields[f"obs_{u}"] = block_fraction(ustate)
        for k, v in INSETS.items():
            rs, cs = window(transform, v["region"])
            insets[f"obs_{u}_{k}"] = ustate[rs, cs].copy()
        del ustate
        acq = plan[(plan["epoch"] == ep) & (plan["bin"] == b)].drop_duplicates("acq")
        t = pd.to_datetime(acq["t"], format="ISO8601")
        rec["units"].append({"epoch": ep, "bin": b, "level_m": round(level, 4), "envelope_cells": info.get("envelope_cells"),
                             "n_acquisitions": int(len(acq)), "first_year": int(t.min().year),
                             "last_year": int(t.max().year), "plan": PLAN.name})
        for p in HERO_PRODUCTS:
            fl = float(products.flood_level(level, 0.0, 0.0, float(so[p]["offset_m"])))
            pst = pred_state(F[p], fl, static["domain"], tile, fin[p])
            fields[f"pred_{u}_{p}"] = block_fraction(pst)
            for k, v in INSETS.items():
                rs, cs = window(transform, v["region"])
                insets[f"pred_{u}_{p}_{k}"] = pst[rs, cs].copy()
            del pst
            cells = ht.displacement_cells(ht.flooded(F[p], fl), code, a7._valid(code), static["sector"], RES,
                                          pred_valid=fin[p], outside_state=static["outside"], shape=shape, tile=tile)
            summ = ht.summarise(cells, ids)
            ref = disp[(disp["product"] == p) & (disp["variant"] == "shape") & (disp["epoch"] == ep) & (disp["bin"] == b)]
            for k in ids:
                r = ref[ref["sector"] == name[k]]
                mine, theirs = float(summ[k]["median"]), float(r["median_m"].iloc[0])
                same = (np.isnan(mine) and np.isnan(theirs)) or abs(mine - theirs) < 1e-6
                rec["checks"].append({"unit": u, "product": p, "sector": name[k], "median_m": mine,
                                      "displacement_csv_m": theirs, "n": int(summ[k]["n"]), "equal": bool(same)})
                if not same:
                    raise SystemExit(f"{u} {p} {name[k]}: median {mine} differs from displacement.csv {theirs}")
            # arrows: per inset, the observed waterline cell nearest the sector median
            o, pr = cells["obs"], cells["pred"]
            ok_p = pr["known"]
            tree = cKDTree(np.c_[pr["rows"][ok_p], pr["cols"][ok_p]].astype(float))
            for k, v in INSETS.items():
                sec = v["sector"]
                med = float(summ[code_of[sec]]["median"])
                ox, oy = cell_xy(transform, o["rows"], o["cols"])
                x0, y0, x1, y1 = v["box"]
                inb = o["known"] & (o["label"] == code_of[sec]) & (ox >= x0) & (ox < x1) & (oy >= y0) & (oy < y1)
                d = np.abs(cells["signed_o"])
                sel = np.flatnonzero(inb & np.isfinite(d))
                if sel.size:     # keep cells whose nearest predicted waterline cell lies in the box too
                    dd, jj = tree.query(np.c_[o["rows"][sel], o["cols"][sel]].astype(float))
                    exs, eys = cell_xy(transform, pr["rows"][ok_p][jj], pr["cols"][ok_p][jj])
                    keep = (exs >= x0) & (exs < x1) & (eys >= y0) & (eys < y1)
                    sel, jj, dd = sel[keep], jj[keep], dd[keep]
                if not sel.size or not np.isfinite(med):
                    rec["arrows"].append({"unit": u, "product": p, "inset": k, "sector": sec, "note": "no cell"})
                    continue
                m_ = int(np.argmin(np.abs(d[sel] - med)))
                i, j, dist = sel[m_], jj[m_], dd[m_]
                ex, ey = cell_xy(transform, pr["rows"][ok_p][j], pr["cols"][ok_p][j])
                rec["arrows"].append({"unit": u, "product": p, "inset": k, "sector": sec,
                                      "x0": float(ox[i]), "y0": float(oy[i]), "x1": float(ex), "y1": float(ey),
                                      "distance_m": round(float(dist * RES), 3), "sector_median_m": round(med, 3),
                                      "signed": "observed waterline inside predicted water (too much water)"
                                      if cells["signed_o"][i] > 0 else "observed waterline outside predicted water"})
            del cells
            log(f"  {p}: c = {so[p]['offset_m']} m, sector medians equal displacement.csv")
        del code
    r_, c_ = (shape[0] // FACTOR) * FACTOR, (shape[1] // FACTOR) * FACTOR
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    ext = [c, c + a * c_, f + e * r_, f]
    win = {}
    for k, v in INSETS.items():
        rs, cs = window(transform, v["region"])
        win[k] = [c + a * cs.start, c + a * cs.stop, f + e * rs.stop, f + e * rs.start]
    np.savez_compressed(CACHE / f"hero_{tag}.npz", extent=np.array(ext), **fields,
                        **{f"ext_{k}": np.array(w_) for k, w_ in win.items()}, **insets)
    return rec


# --- (2) product agreement ----------------------------------------------------------------------

def agreement(tag: str, static: dict, F: dict, transform, shape, tile: int) -> dict:
    em = pd.read_csv(RESULTS / "emergence" / tag / "emergence.csv")
    em = em[(em["functional"] == "emergence") & (em["region"] == "domain") & (em["surface"] == "uncorrected")]
    l26 = float(em["level_2026_m"].iloc[0])
    if not np.allclose(em["level_2026_m"], l26):
        raise SystemExit("emergence.csv: more than one 2026 level")
    lab_in = static["domain"] & (static["sector"] > 0)
    rec = {"level_2026_m": l26, "products": list(F), "scenarios": {}, "checks": []}
    out = {}
    for D in AGREE_SCENARIOS:
        row = em[np.isclose(em["scenario_m"], D)].set_index("product")
        L = float(row["level_m"].iloc[0])
        count = np.zeros(shape, np.uint8)
        for p, Fp in F.items():
            n = 0
            for r0, r1, c0, c1 in ht.tiles(shape, tile):
                f = np.asarray(Fp[r0:r1, c0:c1], float)
                m = lab_in[r0:r1, c0:c1] & (f < l26) & ~(f < L)
                count[r0:r1, c0:c1] += m.astype(np.uint8)
                n += int(m.sum())
            est = float(row.loc[p, "estimate_km2"])
            same = abs(n * CELL_KM2 - est) < 0.5 * CELL_KM2
            rec["checks"].append({"scenario_m": D, "product": p, "cells": n, "km2": n * CELL_KM2,
                                  "emergence_csv_km2": est, "equal": bool(same)})
            if not same:
                raise SystemExit(f"agreement {D} m {p}: {n * CELL_KM2} km2 differs from emergence.csv {est}")
        hist = np.bincount(count[lab_in], minlength=len(F) + 1)
        rec["scenarios"][f"{D:g}"] = {"level_m": L, "area_km2_by_count": {str(k): float(v * CELL_KM2)
                                                                            for k, v in enumerate(hist)},
                                      "area_km2_any": float(hist[1:].sum() * CELL_KM2),
                                      "area_km2_all": float(hist[len(F)] * CELL_KM2),
                                      "area_km2_majority": float(hist[(len(F) // 2 + 1):].sum() * CELL_KM2)}
        # display: per 240 m block, the most frequent count among its domain cells (ties to the
        # higher count), 255 where the block holds no domain cell
        r_, c_ = (shape[0] // FACTOR) * FACTOR, (shape[1] // FACTOR) * FACTOR
        best = np.full((r_ // FACTOR, c_ // FACTOR), 255, np.uint8)
        top = np.zeros(best.shape, np.int32)
        for k in range(len(F) + 1):
            s = np.zeros(best.shape, np.int32)
            step = 256 * FACTOR
            for r0 in range(0, r_, step):
                blk = (count[r0:min(r0 + step, r_), :c_] == k) & lab_in[r0:min(r0 + step, r_), :c_]
                s[r0 // FACTOR:r0 // FACTOR + blk.shape[0] // FACTOR] = blk.reshape(
                    blk.shape[0] // FACTOR, FACTOR, c_ // FACTOR, FACTOR).sum(axis=(1, 3))
            take = (s > 0) & (s >= top)
            best[take], top[take] = k, s[take]
        out[f"count_{D:g}m"] = best
        log(f"agreement {D:g} m: {rec['scenarios'][f'{D:g}']['area_km2_by_count']}")
        del count
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    r_, c_ = (shape[0] // FACTOR) * FACTOR, (shape[1] // FACTOR) * FACTOR
    np.savez_compressed(CACHE / f"agreement_{tag}.npz", extent=np.array([c, c + a * c_, f + e * r_, f]), **out)
    return rec


def run_caspian(tag: str, tile: int) -> int:
    hrun = json.loads((RESULTS / "hindcast" / tag / "run.json").read_text(encoding="utf-8"))
    if hrun.get("envelope") in (None, "none (sensitivity)"):
        raise SystemExit("the hindcast run has no record-low envelope")
    transform, shape = caspian_grid(hrun)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    static = static_main(tag, transform, shape, tile)
    if int(static["envelope"].sum()) != int(hrun["envelope"]["cells"]):
        raise SystemExit("the envelope differs from the hindcast run's")
    F, fin = flood_maps(list(hrun["products"]), hrun, static, transform, shape, tile)
    t0 = time.time()
    h = hero(tag, hrun, static, F, fin, transform, shape, tile)
    h["seconds"] = round(time.time() - t0, 1)
    h["hindcast_run"] = f"results/hindcast/{tag}/run.json"
    (OUT / "erl_hero.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
    t0 = time.time()
    g = agreement(tag, static, F, transform, shape, tile)
    g["seconds"] = round(time.time() - t0, 1)
    g["emergence_table"] = f"results/emergence/{tag}/emergence.csv"
    (OUT / "erl_agreement.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
    log("done")
    return 0


# --- (3) Great Salt Lake ---------------------------------------------------------------------------

GSL = "great_salt_lake"
GSL_MAP_PRODUCTS = ("gebco_2026", "etopo_2022", "srtm15p_v2.7")   # drawn; every product is checked


def _g1():
    spec = importlib.util.spec_from_file_location("g1_hindcast", ROOT / "scripts" / "g1_hindcast.py")
    g1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g1)
    return g1


def gsl_unit(d: Path, arms) -> tuple[tuple[str, int], dict]:
    """The representative unit (a display choice): among the (epoch, bin) units at
    which, in both arms, every global product and the 3DEP control has a finite positive
    shape-variant median displacement at an observed level, the one whose medians are
    closest to the summary medians of the audit (h1_decision.json: per arm, the median
    over evaluable levels), by the sum over arms and surfaces of |ln(unit / summary)|; ties
    to the later epoch, then the higher bin. Returns the unit and the score table."""
    dec = json.loads((d / "h1_decision.json").read_text(encoding="utf-8"))
    summ = {(arm, p): float(v["shape"]["arms"][arm]["median_displacement_m"])
            for p, v in dec["products"].items() for arm in arms}
    summ.update({(arm, "dep3"): float(dec["control_arms"][arm]["median_displacement_m"]) for arm in arms})
    keys, val = None, {}
    for arm in arms:
        rows = []
        for sub in ("hindcast", "hindcast_dep3"):
            x = pd.read_csv(d / arm / sub / "displacement.csv")
            x = x[(x["variant"] == "shape") & (x["sector"] == arm) & x["epoch"].isin(["E1", "E2", "E3"])]
            rows.append(x)
        x = pd.concat(rows)
        k = set()
        for key, g in x.groupby(["epoch", "bin"]):
            med = g["median_m"].to_numpy(float)
            if len(g) == 6 and np.isfinite(med).all() and (med > 0).all() and g["observed_level"].astype(bool).all():
                k.add((key[0], int(key[1])))
                for p, v in zip(g["product"], med):
                    val[(key[0], int(key[1]), arm, p)] = float(v)
        keys = k if keys is None else keys & k
    score = {u: sum(abs(np.log(val[(u[0], u[1], arm, p)] / summ[(arm, p)])) for (arm, p) in summ) for u in keys}
    best = sorted(keys, key=lambda u: (round(score[u], 9), -int(u[0][1]), -u[1]))[0]
    return best, {"n_candidates": len(keys), "score": round(float(score[best]), 4),
                  "rule": "closest to the audit's summary medians: min sum over arms and surfaces of |ln(unit/summary)|"}


def run_gsl(tag: str) -> int:
    from shapely.geometry import shape as as_shape
    from shapely.ops import unary_union

    from caspian import generality as G
    from caspian import hindcast
    from caspian.config import RAW
    g1 = _g1()
    A7 = g1.A7
    B = G.BASINS[GSL]
    crs = B["crs"]
    A7.CRS_EQUAL_AREA = crs
    transform, shape = G.grid(B["bbox"], crs, g1.RES)
    d = RESULTS / "generality" / GSL
    arms_gj = json.loads((d / "arms_laea.geojson").read_text(encoding="utf-8"))
    arms = {f["properties"]["arm"]: as_shape(f["geometry"]) for f in arms_gj["features"]
            if f["properties"]["role"] == "sector"}
    cover = unary_union([as_shape(f["geometry"]) for f in
                         json.loads((d / "cover_laea.geojson").read_text(encoding="utf-8"))["features"]])
    ponds = unary_union([as_shape(f["geometry"]) for f in
                         json.loads((d / "ponds_laea.geojson").read_text(encoding="utf-8"))["features"]])
    st = np.load(INTERIM / "generality" / GSL / "static_250m.npz")
    tr250 = [float(v) for v in st["transform"]]
    pond = g1.burn(ponds, transform, shape)
    corr_native, _ = products.load_correction(RAW / "generality" / GSL / "vdatum_navd88_minus_egm2008.npz")
    corr = products.to_grid(corr_native, transform, shape, "bilinear", crs=crs)
    natives = {p: g1.load_native(RAW / "generality" / GSL / "bathymetry" / g1.FILES[p][0], g1.FILES[p][1])
               for p in G.GLOBAL_PRODUCTS}
    dep3 = np.load(INTERIM / "generality" / GSL / "dep3_60m.npy")
    rim_cells = int(round(g1.RIM_M / g1.RES))
    (ep, b), sel = gsl_unit(d, list(arms))
    level = A7._bin_level(b)
    log(f"GSL unit {ep} bin {b}: level {level + B['level_offset_m']:.2f} m NAVD 88")
    rec = {"basin": GSL, "unit": {"epoch": ep, "bin": b, "level_bin_m": round(level, 4),
                                  "level_navd88_m": round(level + B["level_offset_m"], 4), "selection": sel},
           "map_products": list(GSL_MAP_PRODUCTS), "arms": {}, "checks": []}
    obs_all = np.full(shape, -1, np.int8)
    pred_all = {p: np.full(shape, -1, np.int8) for p in [*G.GLOBAL_PRODUCTS, "dep3"]}
    for arm in arms:
        # static layers, envelope and heights exactly as scripts/g1_hindcast.py main()
        dom = g1.burn(arms[arm], transform, shape)
        k = 1 + list(arms).index(arm)
        armlab = np.asarray(st["armlab"]) == k
        hi250 = np.asarray(st["high"]) & armlab
        lo250 = np.asarray(st["low"]) & armlab
        zone = g1.burn(cover.intersection(arms[arm]), transform, shape) & dom
        zone |= A7.coast_band({"high": hi250.astype(np.int8), "low": lo250.astype(np.int8), "transform": tr250},
                              transform, shape) & dom
        rim_in = ht.eroded(dom, rim_cells)
        sea_seed = (g1.nearest250(np.asarray(st["seed"]) & armlab, tr250, transform, shape, False) > 0) & rim_in
        land250 = hindcast.beyond_land_state(np.asarray(st["high"]), np.ones(hi250.shape, bool), 250.0, 2000.0)
        bst = g1.nearest250(np.nan_to_num(land250, nan=-1).astype(np.int8), tr250, transform, shape, -1)
        land_seed = (bst == 0) & ~zone & rim_in
        if (sea_seed & ~zone).any():
            outside = ht.outside_state(zone, sea_seed & ~zone, land_seed, domain=dom, beyond_state=None,
                                       domain_margin=rim_cells)
        else:
            land = ht.components_any(~zone & rim_in, [land_seed])[0]
            outside = np.full(shape, -1, np.int8)
            outside[land] = 0
        outside[pond] = -1
        sector = np.ones(shape, np.uint8)
        static = {"domain": dom, "sector": sector, "sea_seed": sea_seed, "outside": outside}
        s2 = g1.masked_stage2(GSL, tag, arm, dom & ~pond, transform)
        units = A7.stage2_index(s2)
        A7.ENVELOPE = A7.record_low_water(units, transform, shape)
        hrun = json.loads((d / arm / "hindcast" / "run.json").read_text(encoding="utf-8"))
        if int(A7.ENVELOPE.sum()) != int(hrun["envelope"]["cells"]):
            raise SystemExit(f"{arm}: envelope differs from the hindcast run's")
        raw, _ = A7.observed_raw(A7.groups_for(units, b, [ep]), transform, shape)
        code = A7.classify(raw, static, ht.TILE)
        del raw
        ost = obs_state(code, outside, dom, ht.TILE)
        obs_all[dom] = ost[dom]
        so = {**hrun["shape_offsets"],
              **json.loads((d / arm / "hindcast_dep3" / "run.json").read_text(encoding="utf-8"))["shape_offsets"]}
        disp = {"global": pd.read_csv(d / arm / "hindcast" / "displacement.csv"),
                "dep3": pd.read_csv(d / arm / "hindcast_dep3" / "displacement.csv")}
        arec = {"shape_offsets_m": {p: float(v["offset_m"]) for p, v in so.items()}, "medians_m": {}}
        for p in [*G.GLOBAL_PRODUCTS, "dep3"]:
            if p == "dep3":
                surf = dep3.copy()
                surf[~dom] = np.nan
            else:
                h = products.to_grid(natives[p], transform, shape, "bilinear", crs=crs)
                h[~dom] = np.nan
                surf = products.Corrected(h, corr)
            F = ht.flood_level_map(surf, sea_seed)
            fin = (lambda g: lambda r0, r1, c0, c1: np.isfinite(np.asarray(g[r0:r1, c0:c1], float)))(surf)
            fl = float(products.flood_level(level, B["level_offset_m"], 0.0, float(so[p]["offset_m"])))
            cells = ht.displacement_cells(ht.flooded(F, fl), code, A7._valid(code), sector, g1.RES, pred_valid=fin,
                                          outside_state=outside, shape=shape, tile=ht.TILE)
            mine = float(ht.summarise(cells, [1])[1]["median"])
            ref = disp["dep3" if p == "dep3" else "global"]
            r = ref[(ref["product"] == p) & (ref["variant"] == "shape") & (ref["epoch"] == ep) & (ref["bin"] == b)
                    & (ref["sector"] == arm)]
            theirs = float(r["median_m"].iloc[0])
            same = abs(mine - theirs) < 1e-6
            rec["checks"].append({"arm": arm, "product": p, "median_m": mine, "displacement_csv_m": theirs,
                                  "equal": bool(same)})
            if not same:
                raise SystemExit(f"GSL {arm} {p}: median {mine} differs from displacement.csv {theirs}")
            arec["medians_m"][p] = mine
            pst = pred_state(F, fl, dom, ht.TILE, fin)
            pred_all[p][dom] = pst[dom]
            del F, cells, pst
            log(f"  {arm} {p}: c = {so[p]['offset_m']} m, median {mine:.1f} m equals displacement.csv")
        rec["arms"][arm] = arec
        del code, ost, outside, zone
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    np.savez_compressed(CACHE / "gsl.npz", extent=np.array([c, c + a * shape[1], f + e * shape[0], f]),
                        crs=np.array(crs), obs=obs_all, **{f"pred_{p}": v for p, v in pred_all.items()})
    (OUT / "erl_gsl.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log("done")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["caspian", "gsl"])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tile", type=int, default=ht.TILE)
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == "caspian":
        return run_caspian(a.tag, a.tile)
    return run_gsl(a.tag)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
