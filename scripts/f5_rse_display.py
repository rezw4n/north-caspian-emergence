"""Display preparations for the RSE figures rse_fig_shorelines.py and rse_fig_method.py:
observed and predicted shorelines of all five grids at one example level, the distance of
every predicted shoreline cell to the observed shoreline, the two interleaved halves of the
observations, and one cross-shore profile with ICESat-2 heights. Nothing here is a new
result: the per-sector median displacement of every grid and the noise floor of the
example sector are recomputed with the hindcast's own functions and must equal
results/hindcast/main/displacement.csv and noise_floor.csv; the run stops on any difference.

Example level (a display choice, by a fixed rule): among the late-fall (E3) level bins
that are evaluable for all five grids (shape variant) in the Kizlyar-Dagestan, Volga-Ural
interfluve and NE Kazakh coasts, the bin whose per-level median displacements are jointly
closest to each grid's median over levels (smallest sum over grids of |log(D_level /
D_median)|), per sector (results/hindcast/main/h1_mispredicts_detail.csv). When the
three sectors pick different bins, each is computed.

Profile track (a display choice, by a fixed rule): among the ICESat-2 ATL08 beams of the
H2 population (dry at acquisition, in the emerged band, vegetation share below 0.5) that
have at least one point in the interfluve test block (results/study_area/blocks_50km.csv),
the beam with the most such points inside the interfluve region. The transect is the
straight line fitted through that beam's points (principal axis), extended seaward and
landward; along it, every 30 m: the waterline DEM (data/processed/dem/main, band 1), each
grid's height as the hindcast reads it (bilinear on the 60 m grid, plus the per-cell
datum correction, minus its area-fitted constant c), and the observed and predicted
states at the example level.

Outputs
  data/interim/figures/rse/shorelines_main.npz   states, cells and windows (git-ignored)
  data/interim/figures/rse/profile_main.npz      the transect
  results/display/rse_shorelines.json            units, regions, constants, track, checks

Heavy (about 2 GB peak, a few minutes): run through tools/heavy_slot.py.
Usage: python scripts/f5_rse_display.py [--tag main]
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

from caspian import frequency
from caspian import hindcast_tiled as ht
from caspian import products
from caspian.config import DATA, INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("f1_erl_display", ROOT / "scripts" / "f1_erl_display.py")
f1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f1)
a7 = f1.a7

CACHE = INTERIM / "figures" / "rse"
OUT = RESULTS / "display"
RES = a7.RES
PRODUCTS = ("gebco_2026", "gebco_2022", "etopo_2022", "srtm15p_v2.7", "etopo1")
ZOOM_SECTORS = ("west", "interfluve", "northeast")
METHOD_SECTOR, METHOD_PRODUCT = "interfluve", "gebco_2026"
# Regions kept at 60 m around each zoom (equal-area metres x0, y0, x1, y1); the figure
# chooses its zoom boxes inside them (cartographic choice).
REGIONS = {"west": (-300_000.0, -170_000.0, -160_000.0, 0.0),
           "interfluve": (-30_000.0, 90_000.0, 110_000.0, 210_000.0),
           "northeast": (150_000.0, -20_000.0, 300_000.0, 170_000.0)}
DEM_DIR = DATA / "processed" / "dem" / "main"
PTS = INTERIM / "validation" / "main"
STEP_M = 30.0                    # profile sampling step
EXTEND_SEA_M, EXTEND_LAND_M = 8_000.0, 3_000.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pick_units() -> dict:
    """{sector: (epoch, bin, score table)} by the rule of the module docstring."""
    det = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts_detail.csv")
    d = det[(det["variant"] == "shape") & (det["evaluable"] == True)]  # noqa: E712
    out = {}
    for s in ZOOM_SECTORS:
        g = d[d["sector"] == s]
        piv = g.pivot_table(index=["bin", "level_m", "epochs"], columns="product", values="median").dropna()
        piv = piv[list(PRODUCTS)]
        med = piv.median()
        score = np.abs(np.log(piv / med)).sum(axis=1)
        e3 = [k for k in score.index if k[2] == "E3"]
        sc = score.loc[e3].sort_values()
        b = int(sc.index[0][0])
        out[s] = {"epoch": "E3", "bin": b, "level_m": float(sc.index[0][1]), "score": float(sc.iloc[0]),
                  "runner_up_bin": int(sc.index[1][0]), "runner_up_score": float(sc.iloc[1]),
                  "n_candidates": int(len(sc)), "grid_medians_m": {p: float(med[p]) for p in PRODUCTS},
                  "level_D_m": {p: float(piv.loc[sc.index[0], p]) for p in PRODUCTS}}
    return out


def window(transform, box, shape=None):
    """f1.window clipped to the grid (a region may reach beyond the grid's edge)."""
    rs, cs = f1.window(transform, box)
    if shape is None:
        shape = SHAPE
    return (slice(max(rs.start, 0), min(rs.stop, shape[0])), slice(max(cs.start, 0), min(cs.stop, shape[1])))


SHAPE = None


def win_extent(transform, rs, cs):
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    return [c + a * cs.start, c + a * cs.stop, f + e * rs.stop, f + e * rs.start]


def cells_table(part: dict, signed: np.ndarray, transform) -> dict:
    x, y = f1.cell_xy(transform, part["rows"], part["cols"])
    return {"x": x.astype(np.float32), "y": y.astype(np.float32), "d": np.abs(signed).astype(np.float32),
            "known": part["known"].astype(bool), "label": part["label"].astype(np.uint8)}


def freq_window(units: dict, b: int, ep: str, transform, box) -> tuple[np.ndarray, list]:
    """Water frequency of the unit's full sample (both parity halves) on the 60 m grid
    inside ``box``; NaN where fewer than design.MIN_VALID_OBS valid acquisitions."""
    rs, cs = window(transform, box)
    w = np.zeros((rs.stop - rs.start, cs.stop - cs.start), np.int64)
    v = np.zeros_like(w)
    x0, y1 = float(transform[2]), float(transform[5])
    seen = np.zeros(w.shape, bool)
    for f, groups in a7.groups_for(units, b, [ep]).items():
        with np.load(f) as z:
            tr = z["transform"]
            counts = {g: (z[f"w_{g}"], z[f"v_{g}"]) for g in groups}
        tw, tv = frequency.combine(counts, groups)
        r0 = int(round((y1 - float(tr[5])) / RES))
        c0 = int(round((float(tr[2]) - x0) / RES))
        # intersection of the tile with the window
        rr0, rr1 = max(r0, rs.start), min(r0 + tv.shape[0], rs.stop)
        cc0, cc1 = max(c0, cs.start), min(c0 + tv.shape[1], cs.stop)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        sw = tw[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        sv = tv[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        dst = (slice(rr0 - rs.start, rr1 - rs.start), slice(cc0 - cs.start, cc1 - cs.start))
        new = ~seen[dst] & (sv > 0)        # first tile wins where tiles overlap (as observed_raw)
        w[dst][new] = sw[new]
        v[dst][new] = sv[new]
        seen[dst] |= new
    return frequency.water_frequency(w, v).astype(np.float32), win_extent(transform, rs, cs)


def pick_track(region) -> tuple[pd.DataFrame, dict]:
    ann = np.load(PTS / "ATL08_annotations.npz")
    cols = ["x", "y", "h_orth", "dem_h", "granule", "beam", "time"]
    x0, y0, x1, y1 = region
    parts, r0 = [], 0
    for ch in pd.read_csv(PTS / "ATL08_points.csv.gz", usecols=cols, chunksize=1_000_000):
        n = len(ch)
        sel = ann["in_band"][r0:r0 + n] & (ann["veg_share"][r0:r0 + n] < 0.5)
        r0 += n
        ch = ch[sel]
        parts.append(ch[(ch["x"] >= x0) & (ch["x"] < x1) & (ch["y"] >= y0) & (ch["y"] < y1)])
    if r0 != len(ann["in_band"]):
        raise SystemExit("ATL08 annotations and points differ in length")
    d = pd.concat(parts, ignore_index=True)
    b50 = pd.read_csv(RESULTS / "study_area" / "blocks_50km.csv")
    tb = b50[(b50["split"] == "test") & (b50["sector"] == METHOD_SECTOR)]
    in_test = np.zeros(len(d), bool)
    for _, r in tb.iterrows():
        in_test |= (d["x"] >= r["x0"]) & (d["x"] < r["x0"] + 50_000) & (d["y"] >= r["y0"]) & (d["y"] < r["y0"] + 50_000)
    d["in_test_block"] = in_test
    g = d.groupby(["granule", "beam"]).agg(n=("x", "size"), n_test=("in_test_block", "sum")).reset_index()
    g = g[g["n_test"] > 0].sort_values(["n", "n_test"], ascending=False)
    best = g.iloc[0]
    t = d[(d["granule"] == best["granule"]) & (d["beam"] == best["beam"])].copy()
    c = json.loads((RESULTS / "validation" / "main" / "ATL08_datum_constant.json").read_text(encoding="utf-8"))
    t["y_ref"] = t["h_orth"] + float(c["constant"]["constant_m"])
    rec = {"granule": str(best["granule"]), "beam": str(best["beam"]), "n_points": int(best["n"]),
           "n_points_test_block": int(best["n_test"]), "n_candidate_beams": int(len(g)),
           "datum_constant_m": float(c["constant"]["constant_m"]), "time_first": str(t["time"].min()),
           "test_blocks": [f"{int(r['i'])}_{int(r['j'])}" for _, r in tb.iterrows()]}
    return t, rec


def transect(track: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Points every STEP_M along the principal axis of the track, oriented from sea (low
    DEM) to land (high DEM); returns (s, x, y, record)."""
    xy = track[["x", "y"]].to_numpy(float)
    c = xy.mean(axis=0)
    u = np.linalg.svd(xy - c, full_matrices=False)[2][0]
    s_pts = (xy - c) @ u
    if np.corrcoef(s_pts, track["dem_h"].to_numpy(float))[0, 1] < 0:
        u = -u
        s_pts = -s_pts
    s0, s1 = s_pts.min() - EXTEND_SEA_M, s_pts.max() + EXTEND_LAND_M
    s = np.arange(s0, s1 + STEP_M / 2, STEP_M)
    x, y = c[0] + s * u[0], c[1] + s * u[1]
    off = (xy - c) @ np.array([-u[1], u[0]])
    rec = {"origin_xy_m": c.tolist(), "direction": u.tolist(), "s0_m": float(s0), "s1_m": float(s1),
           "track_s_min_m": float(s_pts.min()), "track_s_max_m": float(s_pts.max()),
           "track_offset_abs_max_m": float(np.abs(off).max())}
    track["s"] = s_pts
    track["offset"] = off
    return s, x, y, rec


def sample_dem(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    import rasterio
    out = np.full(x.shape, np.nan)
    ti, tj = np.floor(x / 60_000.0).astype(int), np.floor(y / 60_000.0).astype(int)
    for i, j in set(zip(ti.tolist(), tj.tolist())):
        f = DEM_DIR / f"dem_{i}_{j}.tif"
        if not f.exists():
            continue
        k = (ti == i) & (tj == j)
        with rasterio.open(f) as r:
            rows, cols = rasterio.transform.rowcol(r.transform, x[k], y[k])
            a = r.read(1)
            rows, cols = np.asarray(rows), np.asarray(cols)
            ok = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])
            v = np.full(k.sum(), np.nan)
            v[ok] = a[rows[ok], cols[ok]]
            if r.nodata is not None:
                v[v == r.nodata] = np.nan
            out[k] = v
    return out


def band_at(x, y) -> np.ndarray:
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    a, _, c0, _, e, f0 = ex["transform"]
    ci = np.floor((x - c0) / a).astype(int)
    ri = np.floor((y - f0) / e).astype(int)
    band = (ex["high"] == 1) & (ex["low"] == 0)
    ok = (ri >= 0) & (ri < band.shape[0]) & (ci >= 0) & (ci < band.shape[1])
    out = np.zeros(x.shape, bool)
    out[ok] = band[ri[ok], ci[ok]]
    return out


def run(tag: str, tile: int) -> int:
    hrun = json.loads((RESULTS / "hindcast" / tag / "run.json").read_text(encoding="utf-8"))
    transform, shape = f1.caspian_grid(hrun)
    global SHAPE
    SHAPE = tuple(shape)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    picks = pick_units()
    unit_list = sorted({(v["epoch"], v["bin"]) for v in picks.values()})
    log("example units", picks)
    static = f1.static_main(tag, transform, shape, tile)
    if int(static["envelope"].sum()) != int(hrun["envelope"]["cells"]):
        raise SystemExit("the envelope differs from the hindcast run's")
    F, fin = f1.flood_maps(list(PRODUCTS), hrun, static, transform, shape, tile)
    so = json.loads((RESULTS / "hindcast" / tag / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    disp = pd.read_csv(RESULTS / "hindcast" / tag / "displacement.csv")
    nf = pd.read_csv(RESULTS / "hindcast" / tag / "noise_floor.csv")
    units = a7.stage2_index(INTERIM / "stage2" / tag)
    a7.ENVELOPE = static["envelope"]
    ids = list(range(1, len(SECTOR_ORDER) + 1))
    name = {k + 1: s for k, s in enumerate(SECTOR_ORDER)}
    code_of = {s: k + 1 for k, s in enumerate(SECTOR_ORDER)}
    plan = pd.read_csv(f1.PLAN, usecols=["acq", "t", "epoch", "bin", "level_m", "parity"])
    rec = {"rule_units": picks, "units": [], "products": list(PRODUCTS), "regions_m": {k: list(v) for k, v in REGIONS.items()},
           "shape_offsets_m": {p: float(so[p]["offset_m"]) for p in PRODUCTS}, "checks": [], "method": {},
           "hindcast_run": f"results/hindcast/{tag}/run.json"}
    out: dict = {}
    track, trec = pick_track(REGIONS[METHOD_SECTOR])
    s, tx, ty, prec = transect(track)
    trec.update(prec)
    a_, _, c_, _, e_, f_ = (float(v) for v in transform[:6])
    t_col = np.floor((tx - c_) / a_).astype(int)
    t_row = np.floor((ty - f_) / e_).astype(int)
    prof = {"s": s, "x": tx, "y": ty, "dem": sample_dem(tx, ty), "band": band_at(tx, ty),
            "is2_s": track["s"].to_numpy(float), "is2_h": track["y_ref"].to_numpy(float),
            "is2_dem": track["dem_h"].to_numpy(float), "is2_offset": track["offset"].to_numpy(float),
            "is2_test": track["in_test_block"].to_numpy(bool)}
    corr, _ = products.correction(f1.CORRECTION, transform, shape)
    for p in PRODUCTS:
        h = products.heights(p, transform, shape, hrun.get("resampling", "bilinear"))
        z = np.asarray(h[t_row, t_col], float) + np.asarray(corr[t_row, t_col], float)
        prof[f"z_{p}"] = z - float(so[p]["offset_m"])      # on the DAHITI frame after the shape constant

    for ep, b in unit_list:
        level = a7._bin_level(b)
        u = f"{ep}_b{b:03d}"
        log(f"unit {u} ({level:.2f} m): observed extent")
        raw, info = a7.observed_raw(a7.groups_for(units, b, [ep]), transform, shape)
        code = a7.classify(raw, static, tile)
        del raw
        ustate = f1.obs_state(code, static["outside"], static["domain"], tile)
        out[f"obs_{u}"] = f1.block_fraction(ustate)
        for k, box in REGIONS.items():
            rs, cs = window(transform, box)
            out[f"obs_{u}_{k}"] = ustate[rs, cs].copy()
            out[f"ext_{k}"] = np.array(win_extent(transform, rs, cs))
        prof[f"obs_{u}"] = ustate[t_row, t_col].astype(np.int8)
        acq = plan[(plan["epoch"] == ep) & (plan["bin"] == b)].drop_duplicates("acq")
        t = pd.to_datetime(acq["t"], format="ISO8601")
        out[f"acq_time_{u}"] = np.array([x.timestamp() for x in t], float)        # seconds since 1970 (UTC)
        out[f"acq_level_{u}"] = acq["level_m"].to_numpy(float)
        out[f"acq_parity_{u}"] = acq["parity"].to_numpy(int)
        rec["units"].append({"epoch": ep, "bin": b, "level_m": round(level, 4), "n_acquisitions": int(len(acq)),
                             "first_year": int(t.min().year), "last_year": int(t.max().year),
                             "envelope_cells": info.get("envelope_cells")})
        del ustate
        ref = disp[(disp["variant"] == "shape") & (disp["epoch"] == ep) & (disp["bin"] == b)]
        for p in PRODUCTS:
            fl = float(products.flood_level(level, 0.0, 0.0, float(so[p]["offset_m"])))
            pst = f1.pred_state(F[p], fl, static["domain"], tile, fin[p])
            out[f"pred_{u}_{p}"] = f1.block_fraction(pst)
            for k, box in REGIONS.items():
                rs, cs = window(transform, box)
                out[f"pred_{u}_{p}_{k}"] = pst[rs, cs].copy()
            prof[f"pred_{u}_{p}"] = pst[t_row, t_col].astype(np.int8)
            if p == METHOD_PRODUCT:
                rs, cs = window(transform, REGIONS[METHOD_SECTOR])
                hh = products.Corrected(products.heights(p, transform, shape, hrun.get("resampling", "bilinear")), corr)
                out[f"z_{p}_{METHOD_SECTOR}"] = (np.asarray(hh[rs, cs], np.float32) - np.float32(so[p]["offset_m"]))
                out[f"F_{p}_{METHOD_SECTOR}"] = np.asarray(F[p][rs, cs], np.float32) - np.float32(so[p]["offset_m"])
            del pst
            cells = ht.displacement_cells(ht.flooded(F[p], fl), code, a7._valid(code), static["sector"], RES,
                                          pred_valid=fin[p], outside_state=static["outside"], shape=shape, tile=tile)
            summ = ht.summarise(cells, ids, return_distances=True)
            for k in ids:
                r = ref[(ref["product"] == p) & (ref["sector"] == name[k])]
                mine, theirs = float(summ[k]["median"]), float(r["median_m"].iloc[0])
                same = (np.isnan(mine) and np.isnan(theirs)) or abs(mine - theirs) < 1e-6
                rec["checks"].append({"unit": u, "product": p, "sector": name[k], "median_m": mine,
                                      "displacement_csv_m": theirs, "n": int(summ[k]["n"]), "equal": bool(same)})
                if not same:
                    raise SystemExit(f"{u} {p} {name[k]}: median {mine} differs from displacement.csv {theirs}")
            for side, sg in (("pred", "signed_p"), ("obs", "signed_o")):
                tab = cells_table(cells[side], cells[sg], transform)
                for key, v in tab.items():
                    out[f"{side}cells_{u}_{p}_{key}"] = v
            if p == METHOD_PRODUCT:
                out[f"dist_{u}_{p}_{METHOD_SECTOR}"] = np.asarray(summ[code_of[METHOD_SECTOR]]["distances"], np.float32)
            del cells, summ
            log(f"  {p}: c = {so[p]['offset_m']} m, sector medians equal displacement.csv")
        # the two interleaved halves (acquisition parity) and their distances (the noise floor)
        if set(units[b][ep]) == {0, 1}:
            half = []
            for par in (0, 1):
                raw, _ = a7.observed_raw(a7.groups_for(units, b, [ep], (par,)), transform, shape)
                half.append(a7.classify(raw, static, tile))
                del raw
            cells = ht.noise_floor_cells(half[0], a7._valid(half[0]), half[1], a7._valid(half[1]), static["sector"], RES,
                                         outside_state=static["outside"], shape=shape, tile=tile)
            summ = ht.summarise(cells, ids, return_distances=True)
            k = code_of[METHOD_SECTOR]
            r = nf[(nf["kind"] == "halves") & (nf["epoch"] == ep) & (nf["bin"] == b) & (nf["sector"] == METHOD_SECTOR)]
            mine, theirs = float(summ[k]["p95"]), float(r["p95_m"].iloc[0])
            same = abs(mine - theirs) < 1e-6
            rec["checks"].append({"unit": u, "what": "halves p95", "sector": METHOD_SECTOR, "p95_m": mine,
                                  "noise_floor_csv_m": theirs, "equal": bool(same)})
            if not same:
                raise SystemExit(f"{u} halves {METHOD_SECTOR}: p95 {mine} differs from noise_floor.csv {theirs}")
            out[f"halfdist_{u}_{METHOD_SECTOR}"] = np.asarray(summ[k]["distances"], np.float32)
            rs, cs = window(transform, REGIONS[METHOD_SECTOR])
            for par in (0, 1):
                st = f1.obs_state(half[par], static["outside"], static["domain"], tile)
                out[f"half{par}_{u}_{METHOD_SECTOR}"] = st[rs, cs].copy()
                del st
            del half, cells, summ
            log(f"  halves: {METHOD_SECTOR} p95 equals noise_floor.csv")
        fw, fext = freq_window(units, b, ep, transform, REGIONS[METHOD_SECTOR])
        out[f"freq_{u}_{METHOD_SECTOR}"] = fw
        out[f"freqext_{METHOD_SECTOR}"] = np.array(fext)
        del code
    r_, c2 = (shape[0] // f1.FACTOR) * f1.FACTOR, (shape[1] // f1.FACTOR) * f1.FACTOR
    out["extent"] = np.array([c_, c_ + a_ * c2, f_ + e_ * r_, f_])
    rec["track"] = trec
    rec["display_res_m"] = RES * f1.FACTOR
    np.savez_compressed(CACHE / f"shorelines_{tag}.npz", **out)
    np.savez_compressed(CACHE / f"profile_{tag}.npz", **prof)
    (OUT / "rse_shorelines.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log("done")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--tile", type=int, default=ht.TILE)
    a = ap.parse_args(argv)
    return run(a.tag, a.tile)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
