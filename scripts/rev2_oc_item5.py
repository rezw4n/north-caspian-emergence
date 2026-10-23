"""Amendment 19 item 5: operating characteristic of the H1 decision rule (D-148). Exploratory.

The "perfect but degraded" control of Amendment 18 item 6 (scripts/rev_r_control.py, D-116),
one per resolution class (15": ctrl_r15_gebco; 1': ctrl_r60_etopo1), plus a spatially correlated
Gaussian vertical error field sigma * Z_r with the D-125 covariance (exponential, practical range
150 km; simulated on a 2 km grid by circulant embedding and interpolated bilinearly to the 60 m
cell centres). Z_r is the same unit-variance field for every sigma and both classes (common
random numbers; stream SeedSequence(20261023).spawn(2)[0].spawn(20)[r]).

Each realisation, on the 60 m hindcast grid:
1. heights = control + sigma * Z_r (cells without a height stay without one);
2. its flood-level map from the open-sea seed (caspian.hindcast_tiled.flood_level_map);
3. its own D-070 area constant (0.01 m grid over +-5 m, all 43 units, known cells of the domain
   in training blocks; the D-121 fit masks; ties to the smallest |c|, then the smaller c);
4. the audit at L + c in the four validated sectors exactly as the products are audited
   (whole-grid waterlines and distances, epochs pooled per level), and the H1 rule with the
   primary floor Q (all evaluable levels; the DEM-range version beside it).

Commands:
  check      sigma = 0: the cached D-116 control flood maps must reproduce the D-116 constants and
             the control's D in the four sectors (results/revision/r/control/hindcast/)
  pilot      one realisation (15", sigma 0.25 m, r = 1), timed by step -> results/revision2/oc/item5_pilot.json
  run        all realisations [--R 20] [--workers 2] (resumable; data/interim/revision2/oc/item5/)
  summarise  -> results/revision2/oc/item5_pflag.csv, item5_realisations.csv, item5_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev2_oc_common as C  # noqa: E402

from caspian import hindcast_tiled as ht  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian import products, rev2_oc as oc  # noqa: E402
from caspian.config import INTERIM, RESULTS  # noqa: E402

CLASSES = {"15s": "ctrl_r15_gebco", "1m": "ctrl_r60_etopo1"}
SIGMAS = [0.10, 0.25, 0.50, 1.00]
R_MAX = 20
RANGE_M = 150000.0
GRF_DX = 2000.0
CTRL_DIR = INTERIM / "rev_r" / "control"
CTRL_FLOOD = INTERIM / "rev_r" / "flood"
FIT_MASKS = INTERIM / "revision_v" / "fit_masks.npz"
RUN = C.WORK / "item5"
SCRATCH = C.WORK / "work"
TOL_M = 1e-9          # floating-point rounding of the median (D-148 addendum)


def _one(pattern_dir: Path, name: str) -> Path:
    c = [p for p in pattern_dir.glob(f"{name}_*.npy") if not p.name.endswith(".tmp.npy")]
    if len(c) != 1:
        raise SystemExit(f"expected one cached {name} in {pattern_dir}, found {[p.name for p in c]}")
    return c[0]


def control(name: str):
    return np.load(_one(CTRL_DIR, name), mmap_mode="r")


def finite_of(name: str, h) -> np.ndarray:
    path = C.WORK / f"finite_{name}.npy"
    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=bool, shape=tuple(h.shape))
        for r0 in range(0, h.shape[0], 1024):
            mm[r0:r0 + 1024] = np.isfinite(np.asarray(h[r0:r0 + 1024]))
        mm.flush()
        del mm
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


# --- fit data ---------------------------------------------------------------------------------

def fit_data(S: dict) -> dict:
    """The D-121 fit masks as memory maps (idx int32; known and water bit-packed per unit),
    checked against the primary fit's units and fit region."""
    paths = {k: C.WORK / f"fit_{k}.npy" for k in ("idx", "known", "water")}
    if not all(p.exists() for p in paths.values()):
        z = np.load(FIT_MASKS)
        ul = [(int(b), str(e)) for b, e in zip(z["bins"], z["epochs"])]
        if ul != S["units"]:
            raise SystemExit("the D-121 fit masks' units differ from the primary fit")
        if int(z["n"]) != int(S["shape_offsets"]["fit_region_cells"]):
            raise SystemExit("the D-121 fit region differs from the primary's")
        if tuple(int(v) for v in z["shape"]) != tuple(S["shape"]):
            raise SystemExit("the D-121 fit masks' grid differs")
        C.WORK.mkdir(parents=True, exist_ok=True)
        np.save(paths["idx"], z["idx"].astype(np.int32))
        np.save(paths["known"], z["known"])
        np.save(paths["water"], z["water"])
    d = {k: np.load(p, mmap_mode="r") for k, p in paths.items()}
    d["n"] = int(d["idx"].size)
    d["T"] = [products.flood_level(S["level"][b], 0.0, 0.0, S["a7"].C_GRID) for b, _ in S["units"]]
    d["counter"] = oc.ThresholdCounter(d["T"])
    d["epochs"] = [ep for _, ep in S["units"]]
    return d


def unpack(packed_row, n):
    return np.unpackbits(np.asarray(packed_row), count=n).astype(bool)


class ClassFit:
    """Per class: the finite mask at the fit cells and each unit's water count (fixed, since the
    field never changes which cells have a height)."""

    def __init__(self, fd: dict, fin):
        self.fd = fd
        self.fin_fit = np.asarray(fin).ravel()[np.asarray(fd["idx"])]
        self.water = np.array([int((unpack(fd["water"][u], fd["n"]) & self.fin_fit).sum())
                               for u in range(len(fd["epochs"]))], np.int64)

    def constant(self, F, a7) -> dict:
        fd = self.fd
        Fv = np.asarray(F).ravel()[np.asarray(fd["idx"])]
        q = fd["counter"].locate(Fv)
        del Fv
        bias = np.zeros((len(fd["epochs"]), a7.C_GRID.size), np.int64)
        for u in range(len(fd["epochs"])):
            k = unpack(fd["known"][u], fd["n"]) & self.fin_fit
            bias[u] = fd["counter"].counts(q[k], u) - self.water[u]
        return a7.fit_offset(bias, fd["epochs"])


# --- the audit --------------------------------------------------------------------------------

def audit(S: dict, F, fin, c: float) -> dict:
    """D and n [sector, bin] at L + c in the validated sectors (module docstring, step 4)."""
    sector = S["static"]["sector"]
    T = {b: float(products.flood_level(S["level"][b], 0.0, 0.0, c)) for b in S["bins"]}
    cand = oc.waterline_candidates(F, fin, min(T.values()) - 1e-9, max(T.values()) + 1e-9)
    parts = {b: {s: [] for s in C.VIDS} for b in S["bins"]}
    for b in S["bins"]:
        for ep in S["epochs_of"][b]:
            ob = C.Obs(b, ep)
            fin_obs = np.asarray(fin[ob.rows, ob.cols], bool) if ob.rows.size else np.empty(0, bool)
            got = ob.codes(cand, T[b], sector, fin_obs)
            for s in C.VIDS:
                parts[b][s].append(got[s])
    D, n = C.pooled_D(parts, S)
    return {"D": D, "n": n, "candidates": int(cand["idx"].size)}


def verdicts(S: dict, lv: pd.DataFrame, D, n) -> dict:
    f, nex, nev = C.flags(lv, D, n)
    fd, nexd, nevd = C.flags(lv, D, n, restrict_dem=True)
    return {"flag": f.astype(bool).tolist(), "n_exceeding": nex.tolist(), "n_evaluable": nev.tolist(),
            "flag_dem": fd.astype(bool).tolist(), "n_exceeding_dem": nexd.tolist(), "n_evaluable_dem": nevd.tolist()}


# --- fields -----------------------------------------------------------------------------------

def field_grid(S: dict):
    tr = S["transform"]
    x0, y0 = float(tr[2]), float(tr[5])
    rows, cols = S["shape"]
    ys = y0 - 60.0 * (np.arange(rows) + 0.5)
    xs = x0 + 60.0 * (np.arange(cols) + 0.5)
    ny = int(np.ceil((y0 - ys[-1]) / GRF_DX)) + 2
    nx = int(np.ceil((xs[-1] - x0) / GRF_DX)) + 2
    return y0, x0, ys, xs, ny, nx


def unit_field(S: dict, r: int):
    """Z_r on the 2 km grid (unit variance) and the negative-eigenvalue share."""
    y0, x0, ys, xs, ny, nx = field_grid(S)
    child = C.seeds()[0].spawn(R_MAX)[r - 1]
    f, neg = nt.grf_exponential(ny, nx, GRF_DX, 1.0, RANGE_M, np.random.default_rng(child), n=1)
    return f[0], neg


def perturbed(S: dict, ctrl, Z, sigma: float, path: Path):
    """control + sigma * Z (bilinear to the 60 m centres), float32 memory map at ``path``."""
    y0, x0, ys, xs, _, _ = field_grid(S)
    tmp = path.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(S["shape"]))
    for r0 in range(0, S["shape"][0], 512):
        r1 = min(r0 + 512, S["shape"][0])
        eps = oc.bilinear_rows(Z, y0, x0, GRF_DX, ys, xs, r0, r1)
        mm[r0:r1] = (np.asarray(ctrl[r0:r1], np.float64) + sigma * eps).astype(np.float32)
    mm.flush()
    del mm
    tmp.replace(path)
    return np.load(path, mmap_mode="r")


# --- one realisation --------------------------------------------------------------------------

_W: dict = {}


def _cleanup(wid: str):
    """Remove this worker's scratch rasters (a file still mapped is left for the next call)."""
    import gc
    gc.collect()
    for f in SCRATCH.glob(f"*_{wid}_*.npy"):
        try:
            f.unlink()
        except OSError:
            pass


def _init_worker():
    S = C.setup()
    fd = fit_data(S)
    lv = C.level_frame(S, C.observed_levels(S))
    ctrl = {k: control(n) for k, n in CLASSES.items()}
    fin = {k: finite_of(n, ctrl[k]) for k, n in CLASSES.items()}
    cf = {k: ClassFit(fd, fin[k]) for k in CLASSES}
    _W.update(S=S, fd=fd, lv=lv, ctrl=ctrl, fin=fin, cf=cf)


def realisation(cls: str, sigma: float, r: int, wid: str = "0") -> dict:
    if not _W:
        _init_worker()
    S, lv = _W["S"], _W["lv"]
    t = {"start": time.time()}
    Z, neg = unit_field(S, r)
    t["field"] = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    _cleanup(wid)
    tag = f"{wid}_{cls}_s{int(round(sigma * 100)):03d}_r{r:02d}"
    hp, Fp = SCRATCH / f"h_{tag}.npy", SCRATCH / f"F_{tag}.npy"
    h = perturbed(S, _W["ctrl"][cls], Z, sigma, hp)
    t["heights"] = time.time()
    F = np.lib.format.open_memmap(Fp, mode="w+", dtype=np.float32, shape=tuple(S["shape"]))
    ht.flood_level_map(h, S["static"]["sea_seed"], tile=C.TILE, out=F)
    F.flush()
    del h
    t["flood"] = time.time()
    fit = _W["cf"][cls].constant(F, S["a7"])
    c = fit["offset_m"]
    t["fit"] = time.time()
    A = audit(S, F, _W["fin"][cls], c)
    t["audit"] = time.time()
    del F
    _cleanup(wid)
    v = verdicts(S, lv, A["D"], A["n"])
    keys = ["start", "field", "heights", "flood", "fit", "audit"]
    rec = {"class": cls, "control": CLASSES[cls], "sigma_m": sigma, "r": r, "c_m": c,
           "objective_km2": fit["objective_km2_at_offset"], "n_tied_minima": fit["n_tied_minima"],
           "field_mean": float(Z.mean()), "field_sd": float(Z.std()), "negative_eigenvalue_share": neg,
           "candidates": A["candidates"], "D_m": np.where(np.isfinite(A["D"]), A["D"], -1).tolist(),
           "D_is_inf": np.isinf(A["D"]).tolist(), "n": A["n"].tolist(), **v,
           "seconds": {k2: round(t[k2] - t[k1], 1) for k1, k2 in zip(keys, keys[1:])},
           "seconds_total": round(t["audit"] - t["start"], 1)}
    return rec


def _task(args):
    cls, sigma, r, wid = args
    path = RUN / f"res_{cls}_s{int(round(sigma * 100)):03d}_r{r:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rec = realisation(cls, sigma, r, wid)
    C.write_json(path, rec)
    C.log(f"done {cls} sigma {sigma} r {r}: c {rec['c_m']}, flags {rec['flag']}, {rec['seconds_total']} s "
          f"{rec['seconds']}")
    return rec


def _pool_task(args):
    import os
    return _task((*args, str(os.getpid())))


# --- commands ---------------------------------------------------------------------------------

def cmd_check(a) -> int:
    """sigma = 0 on the cached D-116 flood maps."""
    _init_worker()
    S, lv = _W["S"], _W["lv"]
    disp = pd.read_csv(RESULTS / "revision" / "r" / "control" / "hindcast" / "displacement.csv")
    so = json.loads((RESULTS / "revision" / "r" / "control" / "hindcast" / "shape_offsets.json").read_text(encoding="utf-8"))
    out, rows = {}, []
    for cls, name in CLASSES.items():
        t0 = time.time()
        F = np.load(_one(CTRL_FLOOD, name), mmap_mode="r")
        fit = _W["cf"][cls].constant(F, S["a7"])
        c = fit["offset_m"]
        c_ref = float(so["used_in_this_run"][name]["offset_m"])
        A = audit(S, F, _W["fin"][cls], c)
        ref = disp[(disp["product"] == name) & (disp["variant"] == "shape") & (disp["epoch"] == "pooled")]
        n_diff, max_d = 0, 0.0
        for i, s in enumerate(C.VIDS):
            for j, b in enumerate(S["bins"]):
                r = ref[(ref["sector"] == C.NAME[s]) & (ref["bin"] == b)]
                rD = float(r["median_m"].iloc[0]) if len(r) else np.nan
                rn = int(r["n"].iloc[0]) if len(r) else 0
                same = rn == int(A["n"][i, j]) and ((np.isnan(rD) and np.isnan(A["D"][i, j])) or rD == A["D"][i, j]
                                                     or abs(rD - A["D"][i, j]) <= TOL_M)
                n_diff += int(not same)
                if np.isfinite(rD) and np.isfinite(A["D"][i, j]):
                    max_d = max(max_d, abs(rD - A["D"][i, j]))
                rows.append({"class": cls, "control": name, "sector": C.NAME[s], "bin": b, "D_d116_m": rD,
                             "D_here_m": A["D"][i, j], "n_d116": rn, "n_here": int(A["n"][i, j]), "identical": same})
        v = verdicts(S, lv, A["D"], A["n"])
        out[cls] = {"control": name, "c_m": c, "c_d116_m": c_ref, "constant_reproduced": c == c_ref,
                    "objective_km2": fit["objective_km2_at_offset"], "levels_not_identical": n_diff,
                    "max_abs_diff_m": max_d, "seconds": round(time.time() - t0, 1), **v,
                    "sectors": C.VALIDATED}
        C.log(cls, json.dumps(out[cls]))
    C.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.OUT / "item5_check_sigma0.csv", index=False)
    C.write_json(C.OUT / "item5_sigma0.json", out)
    ok = all(o["constant_reproduced"] and o["levels_not_identical"] == 0 for o in out.values())
    print("CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_pilot(a) -> int:
    t0 = time.time()
    _init_worker()
    t1 = time.time()
    rec = realisation("15s", 0.25, 1, "pilot")
    rec["setup_seconds"] = round(t1 - t0, 1)
    C.write_json(C.OUT / "item5_pilot.json", rec)
    print(json.dumps({k: rec[k] for k in ("c_m", "flag", "n_exceeding", "n_evaluable", "seconds", "seconds_total",
                                          "setup_seconds", "candidates")}, indent=1))
    return 0


def tasks(R: int):
    """Ordered by realisation index, so a cut keeps the first realisations of every (class, sigma)."""
    return [(cls, s, r) for r in range(1, R + 1) for s in SIGMAS for cls in CLASSES]


def cmd_run(a) -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    todo = tasks(a.R)
    C.log(f"{len(todo)} realisations, {a.workers} workers")
    if a.workers <= 1:
        for tk in todo:
            _task((*tk, "0"))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker) as pool:
            for rec in pool.imap_unordered(_pool_task, todo):
                pass
    C.log("all realisations done")
    return 0


def cmd_summarise(a) -> int:
    S = C.setup(load_static=False)
    s0 = json.loads((C.OUT / "item5_sigma0.json").read_text(encoding="utf-8"))
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUN.glob("res_*.json"))]
    rows = []
    for rec in recs:
        for i, sec in enumerate(C.VALIDATED):
            D = np.array(rec["D_m"][i], float)
            D[np.array(rec["D_is_inf"][i], bool)] = np.inf
            D[D < 0] = np.nan
            rows.append({"class": rec["class"], "control": rec["control"], "sigma_m": rec["sigma_m"], "r": rec["r"],
                         "sector": sec, "c_m": rec["c_m"], "flag": rec["flag"][i],
                         "n_exceeding": rec["n_exceeding"][i], "n_evaluable": rec["n_evaluable"][i],
                         "flag_dem": rec["flag_dem"][i], "median_D_m": float(np.nanmedian(D)),
                         "field_mean": rec["field_mean"], "seconds_total": rec["seconds_total"]})
    rz = pd.DataFrame(rows)
    rz.to_csv(C.OUT / "item5_realisations.csv", index=False)
    R_done = rz.groupby(["class", "sigma_m"])["r"].nunique()
    out, first = [], {}
    for cls in CLASSES:
        for i, sec in enumerate(C.VALIDATED):
            base = s0[cls]
            out.append({"class": cls, "control": CLASSES[cls], "sector": sec, "sigma_m": 0.0, "R": 1,
                        "n_flagged": int(base["flag"][i]), "p_flag": float(base["flag"][i]), "ci95_lo": np.nan,
                        "ci95_hi": np.nan, "p_flag_dem": float(base["flag_dem"][i]),
                        "median_n_exceeding": base["n_exceeding"][i], "median_n_evaluable": base["n_evaluable"][i],
                        "median_D_m": np.nan, "c_median_m": base["c_m"], "c_p05_m": np.nan, "c_p95_m": np.nan})
            ps = []
            for s in SIGMAS:
                g = rz[(rz["class"] == cls) & (rz["sector"] == sec) & np.isclose(rz["sigma_m"], s)]
                k, R = int(g["flag"].sum()), len(g)
                lo, hi = oc.clopper_pearson(k, R)
                ps.append(k / R if R else np.nan)
                out.append({"class": cls, "control": CLASSES[cls], "sector": sec, "sigma_m": s, "R": R,
                            "n_flagged": k, "p_flag": k / R if R else np.nan, "ci95_lo": lo, "ci95_hi": hi,
                            "p_flag_dem": float(g["flag_dem"].mean()) if R else np.nan,
                            "median_n_exceeding": float(g["n_exceeding"].median()),
                            "median_n_evaluable": float(g["n_evaluable"].median()),
                            "median_D_m": float(g["median_D_m"].median()),
                            "c_median_m": float(g["c_m"].median()), "c_p05_m": float(g["c_m"].quantile(0.05)),
                            "c_p95_m": float(g["c_m"].quantile(0.95))})
            first[f"{cls}|{sec}"] = (0.0 if base["flag"][i] else oc.first_sigma_above(SIGMAS, ps, 0.5))
    pf = pd.DataFrame(out)
    pf.to_csv(C.OUT / "item5_pflag.csv", index=False)
    summ = {"rule": "A19 item 5 (D-148)", "classes": CLASSES, "sigmas_m": SIGMAS, "range_m": RANGE_M,
            "R_per_class_sigma": {f"{k[0]}|{k[1]}": int(v) for k, v in R_done.items()},
            "first_sigma_p_above_0.5": first, "sigma0": s0,
            "negative_eigenvalue_share_max": max(r["negative_eigenvalue_share"] for r in recs) if recs else None,
            "seconds_per_realisation_median": float(np.median([r["seconds_total"] for r in recs])) if recs else None}
    C.write_json(C.OUT / "item5_summary.json", summ)
    print(pf.to_string())
    print(json.dumps(first, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["check", "pilot", "run", "summarise"])
    ap.add_argument("--R", type=int, default=R_MAX)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    return {"check": cmd_check, "pilot": cmd_pilot, "run": cmd_run, "summarise": cmd_summarise}[a.command](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
