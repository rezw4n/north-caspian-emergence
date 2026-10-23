"""Amendment 19 item 9: systematic-budget Monte Carlo (D-148). Exploratory.

B = 500 draws. Each draw adds an offset delta(s, e) ~ N(0, sigma_budget(s, e)) per validated
sector s and epoch e (sigma_budget: the Amendment 18 item 7 budget, the median over the epoch's
level bins of results/revision/r/budget/levels.csv sigma_budget_m). The existing flood-level
map of each grid (the hindcast's cache, corrected heights) is thresholded at L + c + delta, with
c the grid's D-070 constant; the displacement D is recomputed per level (epochs pooled) in the
four validated sectors, and the H1 rule is applied with the primary floor Q unchanged.
delta is rounded to 0.01 m so each distinct threshold is computed once (D-148); the same draw
applies to all five grids. Check: delta = 0 must reproduce the primary D, n and verdicts.

Commands:
  draws      sigma table and the draws -> results/revision2/oc/item9_sigma.csv, item9_draws.csv
  run        per grid -> data/interim/revision2/oc/item9/<grid>.npz [--products a,b] [--workers 2]
             [--bins 30,31: pilot on these bins only, timed; output .../item9_pilot/]
  summarise  -> results/revision2/oc/item9_pmispredict.csv, item9_check.csv, item9_summary.json
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

from caspian import products, rev2_oc as oc  # noqa: E402
from caspian.config import RESULTS  # noqa: E402

B = 500
EPOCHS = ["E1", "E2", "E3"]
BUDGET = RESULTS / "revision" / "r" / "budget" / "levels.csv"
RUN = C.WORK / "item9"
TOL_M = 1e-9          # floating-point rounding of the median (D-148 addendum)


def cmd_draws(a) -> int:
    S = C.setup(load_static=False)
    lv = pd.read_csv(BUDGET)
    sig = oc.sector_epoch_sigma(lv, S["units"])
    rows = [{"sector": s, "epoch": e, "sigma_budget_m": sig[(s, e)],
             "n_bins": sum(1 for _, ep in S["units"] if ep == e)} for s in C.VALIDATED for e in EPOCHS]
    st = pd.DataFrame(rows)
    if not np.isfinite(st["sigma_budget_m"]).all():
        raise SystemExit("a sector-epoch budget sigma is missing")
    rng = np.random.default_rng(C.seeds()[1])
    sd = st["sigma_budget_m"].to_numpy(float)          # order: sector-major, epoch-minor
    delta = rng.standard_normal((B, sd.size)) * sd[None, :]
    dr = []
    for i in range(B):
        for k, r in enumerate(rows):
            dr.append({"draw": i + 1, "sector": r["sector"], "epoch": r["epoch"], "delta_m": float(delta[i, k]),
                       "delta_cm_rounded": int(np.rint(delta[i, k] * 100))})
    C.OUT.mkdir(parents=True, exist_ok=True)
    st.to_csv(C.OUT / "item9_sigma.csv", index=False)
    pd.DataFrame(dr).to_csv(C.OUT / "item9_draws.csv", index=False)
    C.log("draws written", st.to_dict(orient="records"))
    return 0


def load_draws(S) -> np.ndarray:
    """[B, sector (VALIDATED order), epoch (E1..E3)] offsets in whole cm."""
    d = pd.read_csv(C.OUT / "item9_draws.csv")
    arr = np.zeros((B, len(C.VALIDATED), len(EPOCHS)), np.int64)
    for r in d.itertuples():
        arr[r.draw - 1, C.VALIDATED.index(r.sector), EPOCHS.index(r.epoch)] = r.delta_cm_rounded
    return arr


def run_product(p: str, bins=None, out_dir: Path = RUN) -> dict:
    t0 = time.time()
    S = C.setup()
    sector = S["static"]["sector"]
    F, fin = C.product_flood(S, p)
    c = S["c"][p]
    dcm = load_draws(S)
    T0 = {b: float(products.flood_level(S["level"][b], 0.0, 0.0, c)) for b in S["bins"]}
    bins = S["bins"] if bins is None else [b for b in S["bins"] if b in bins]
    lo_cm, hi_cm = int(min(dcm.min(), 0)), int(max(dcm.max(), 0))
    t_lo = min(T0[b] for b in bins) + lo_cm / 100.0
    t_hi = max(T0[b] for b in bins) + hi_cm / 100.0
    cand = oc.waterline_candidates(F, fin, t_lo - 1e-9, t_hi + 1e-9)
    C.log(f"{p}: {cand['idx'].size} candidate cells for T in [{t_lo:.2f}, {t_hi:.2f}], {time.time() - t0:.0f} s")
    nb, ns = len(S["bins"]), len(C.VIDS)
    D = np.full((B, ns, nb), np.nan)
    n = np.zeros((B, ns, nb), np.int64)
    D0 = np.full((ns, nb), np.nan)
    n0 = np.zeros((ns, nb), np.int64)
    n_thr = 0
    for b in bins:
        j = S["bins"].index(b)
        store = {}
        for ep in S["epochs_of"][b]:
            e = EPOCHS.index(ep)
            ob = C.Obs(b, ep)
            fin_obs = np.asarray(fin[ob.rows, ob.cols], bool) if ob.rows.size else np.empty(0, bool)
            need = sorted(set(dcm[:, :, e].ravel().tolist()) | {0})
            cu = oc.restrict(cand, T0[b] + min(need) / 100.0 - 1e-9, T0[b] + max(need) / 100.0 + 1e-9)
            for d in need:
                store[(ep, d)] = ob.codes(cu, T0[b] + d / 100.0, sector, fin_obs)
            n_thr += len(need)
            del ob, cu
        eps = S["epochs_of"][b]
        for i, s in enumerate(C.VIDS):
            D0[i, j], n0[i, j] = oc.pooled_median([store[(ep, 0)][s] for ep in eps])
            memo = {}
            for k in range(B):
                key = tuple(int(dcm[k, i, EPOCHS.index(ep)]) for ep in eps)
                if key not in memo:
                    memo[key] = oc.pooled_median([store[(ep, d)][s] for ep, d in zip(eps, key)])
                D[k, i, j], n[k, i, j] = memo[key]
        del store
        C.log(f"{p}: bin {b} {eps} done, {n_thr} thresholds so far, {time.time() - t0:.0f} s")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{p}.npz", D=D, n=n, D0=D0, n0=n0, dcm=dcm, bins=np.array(S["bins"]),
             done_bins=np.array(bins), vids=np.array(C.VIDS), c=c)
    rec = {"product": p, "c_m": c, "candidates": int(cand["idx"].size), "t_range": [t_lo, t_hi],
           "thresholds": n_thr, "bins": bins, "seconds": round(time.time() - t0, 1)}
    C.write_json(out_dir / f"{p}.json", rec)
    C.log(f"{p}: done, {rec}")
    return rec


def _worker(args):
    p, bins, out_dir = args
    return run_product(p, bins, Path(out_dir))


def cmd_run(a) -> int:
    S = C.setup()
    C.prep_obs(S)
    names = a.products.split(",") if a.products else C.PRODUCTS
    bins = [int(x) for x in a.bins.split(",")] if a.bins else None
    out_dir = C.WORK / "item9_pilot" if bins else RUN
    todo = [p for p in names if bins or not (out_dir / f"{p}.npz").exists()]
    if a.workers <= 1 or len(todo) <= 1:
        for p in todo:
            run_product(p, bins, out_dir)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for rec in pool.imap_unordered(_worker, [(p, bins, str(out_dir)) for p in todo]):
                C.log("finished", rec["product"], rec["seconds"], "s")
    return 0


def cmd_summarise(a) -> int:
    S = C.setup(load_static=False)
    lv = C.level_frame(S, C.observed_levels(S))
    disp = pd.read_csv(RESULTS / "hindcast" / C.TAG / "displacement.csv")
    prim = disp[(disp["epoch"] == "pooled") & (disp["variant"] == "shape")]
    verd = pd.read_csv(RESULTS / "hindcast" / C.TAG / "h1_mispredicts.csv")
    verd = verd[verd["variant"] == "shape"].set_index(["product", "sector"])
    rows, checks = [], []
    max_diff = 0.0
    for p in C.PRODUCTS:
        z = np.load(RUN / f"{p}.npz")
        if list(z["done_bins"]) != S["bins"]:
            raise SystemExit(f"{p}: not all bins were run")
        D, n, D0, n0 = z["D"], z["n"], z["D0"], z["n0"]
        # check: delta = 0 reproduces the primary pooled rows and verdicts
        for i, s in enumerate(C.VIDS):
            for j, b in enumerate(S["bins"]):
                r = prim[(prim["product"] == p) & (prim["sector"] == C.NAME[s]) & (prim["bin"] == b)]
                ref_D = float(r["median_m"].iloc[0]) if len(r) else np.nan
                ref_n = int(r["n"].iloc[0]) if len(r) else 0
                same = (ref_n == int(n0[i, j])) and ((np.isnan(ref_D) and np.isnan(D0[i, j])) or ref_D == D0[i, j]
                                                       or abs(ref_D - D0[i, j]) <= TOL_M)
                if np.isfinite(ref_D) and np.isfinite(D0[i, j]):
                    max_diff = max(max_diff, abs(ref_D - D0[i, j]))
                checks.append({"product": p, "sector": C.NAME[s], "bin": b, "D_primary_m": ref_D, "D_delta0_m": D0[i, j],
                               "n_primary": ref_n, "n_delta0": int(n0[i, j]), "identical": bool(same)})
        f0, nex0, nev0 = C.flags(lv, D0, n0)
        f, nex, nev = C.flags(lv, D, n)
        fdem, _, _ = C.flags(lv, D, n, restrict_dem=True)
        for i, s in enumerate(C.VIDS):
            sec = C.NAME[s]
            prim_v = verd.loc[(p, sec), "mispredicts"]
            k = int(f[:, i].sum())
            lo, hi = oc.clopper_pearson(k, B)
            share = nex[:, i] / np.maximum(nev[:, i], 1)
            rows.append({"product": p, "sector": sec, "B": B, "n_flagged": k, "p_mispredict": k / B,
                         "ci95_lo": lo, "ci95_hi": hi, "primary_verdict": str(prim_v),
                         "delta0_verdict": bool(f0[i]), "delta0_n_exceeding": int(nex0[i]),
                         "delta0_n_evaluable": int(nev0[i]),
                         "median_share_exceeding": float(np.median(share)),
                         "p05_share_exceeding": float(np.percentile(share, 5)),
                         "p95_share_exceeding": float(np.percentile(share, 95)),
                         "median_n_evaluable": float(np.median(nev[:, i])),
                         "p_mispredict_dem_range": float(fdem[:, i].mean())})
    ck = pd.DataFrame(checks)
    res = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    ck.to_csv(C.OUT / "item9_check.csv", index=False)
    res.to_csv(C.OUT / "item9_pmispredict.csv", index=False)
    verdict_ok = bool((res["delta0_verdict"].astype(str) == res["primary_verdict"].astype(str)).all())
    summ = {"rule": "A19 item 9 (D-148)", "B": B, "seed": C.SEED, "stream": "SeedSequence(20261023).spawn(2)[1]",
            "check_tolerance_m": TOL_M, "check_all_identical": bool(ck["identical"].all()), "check_n_rows": int(len(ck)),
            "check_n_not_identical": int((~ck["identical"]).sum()), "check_max_abs_diff_m": max_diff,
            "check_verdicts_reproduced": verdict_ok,
            "sigma": pd.read_csv(C.OUT / "item9_sigma.csv").to_dict(orient="records"),
            "p_mispredict": {f"{r.product}|{r.sector}": r.p_mispredict for r in res.itertuples()},
            "runs": {p: json.loads((RUN / f"{p}.json").read_text(encoding="utf-8")) for p in C.PRODUCTS}}
    C.write_json(C.OUT / "item9_summary.json", summ)
    print(res.to_string())
    print(json.dumps({k: v for k, v in summ.items() if k.startswith("check")}, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["draws", "run", "summarise"])
    ap.add_argument("--products", default="")
    ap.add_argument("--bins", default="")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    return {"draws": cmd_draws, "run": cmd_run, "summarise": cmd_summarise}[a.command](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
