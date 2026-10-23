"""A19 item 3 (D-146): the five-grid agreement on the first metre and its reference from
perturbed copies of one grid, for the maps shown: the observed re-referenced baseline
(A19 item 1) and the shifted grids. Exploratory (post-review).

Simulator: D-125 as cut by its first addendum: 50 Gaussian fields per grid with
exponential covariance (sill = the grid's ATL08 test-block NMAD squared, practical range
150 km), on a 2 km grid by circulant embedding, bilinear to a 480 m flood grid (8 x 8 blocks
of the 60 m grid; block-mean total-variant heights; open-sea seed blocks); 10 disjoint
groups of five realisations. The realisations are D-125's own: same seed (20261022), the
same SeedSequence child per (range, product) and the same batches, so the shifted and
uncorrected statistics must reproduce results/revision/n/sim_groups.csv and
sim_realisations.csv exactly (checked; the run stops otherwise).

Observed re-referenced at 480 m: each unperturbed grid and each realisation gets its own
c26, by bisection on the 0.01 m grid, so that its domain sea area (blocks weighted by
their 60 m domain cells) equals |S_obs| (the composite observed sea, 60 m cells in the
same blocks). Its first metre is the blocks with T - 1 <= F < T, weighted by their S_obs
domain cells; agreement = all five over any with those weights.

Outputs (results/revision2/n/): sim2_realisations.csv, sim2_groups.csv, sim2_summary.csv,
sim2_observed_480m.csv, run_simulate2.json.

Usage: python scripts/rev2_n_simulate.py [--tile 2048]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as C  # noqa: E402
import rev_n_simulate as SIM  # noqa: E402

from caspian import hindcast_tiled as ht  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian import rev2_n as r2  # noqa: E402
from caspian.config import RESULTS  # noqa: E402
from caspian.geography import SECTOR_ORDER  # noqa: E402

OUT2 = RESULTS / "revision2" / "n"
N_FIELDS = 50                 # D-125 addendum
RANGE_M = 150000.0
BLOCK = 8                     # 480 m
DELTAS = (1.0, 2.0)
K_RANGE = 500


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=True, heights=True, fused_form="composite")
    names = S["names"]
    L26, cell, k = S["L26"], S["a7"].CELL_KM2, BLOCK
    tr = S["transform"]
    x0, y0 = float(tr[2]), float(tr[5])
    nm = SIM.nmad_table()
    sill = {p: nm[p]["matched"] ** 2 for p in names}
    lab = S["lab_in"]
    vidx = [SECTOR_ORDER.index(s) for s in C.VALIDATED]
    Wr = nt.block_region_counts(lab, k, S["K"]).astype(np.uint8)
    w_dom = Wr.sum(axis=0).astype(np.float64)
    w_val = Wr[vidx].sum(axis=0).astype(np.float64)
    del Wr
    Gc = S["Ffused"][names[0]]
    sobs = np.zeros(S["shape"], bool)
    for r0, r1, c0, c1 in ht.tiles(S["shape"], a.tile):
        sobs[r0:r1, c0:c1] = np.asarray(Gc[r0:r1, c0:c1], np.float64) < L26
    n_S60 = int(((np.asarray(lab) > 0) & sobs).sum())
    WS = nt.block_region_counts(lab, k, S["K"], mask=sobs).astype(np.uint8)
    wS_dom = WS.sum(axis=0).astype(np.float64)
    wS_val = WS[vidx].sum(axis=0).astype(np.float64)
    del WS, sobs
    target = float(wS_dom.sum())
    crop = {"S_obs_domain_cells_60m": n_S60, "S_obs_domain_cells_in_blocks": int(target)}
    seed_c = nt.block_all(S["static"]["sea_seed"], k)
    R, Cc = seed_c.shape
    res = float(S["a7"].RES)
    ys = y0 - k * res * (np.arange(R) + 0.5)
    xs = x0 + k * res * (np.arange(Cc) + 0.5)
    ny = int(np.ceil((y0 - ys[-1]) / SIM.GRF_DX)) + 2
    nx = int(np.ceil((xs[-1] - x0) / SIM.GRF_DX)) + 2
    zc, F0 = {}, {}
    for p in names:
        zc[p] = nt.block_mean(S["z"][p], k)
        F0[p] = nt.minimax_flood(zc[p], seed_c)
        C.log("480 m set-up", p)

    def measures(F, p):
        c = S["c"][p]
        out, bands = {}, {}
        A = r2.WeightedArea(F, w_dom)
        fit = r2.bisect_offset(lambda kk: float(A(round(L26 + c + kk * 0.01, 6))), target, -K_RANGE, K_RANGE)
        T = round(L26 + c + fit["k"] * 0.01, 6)
        out["c26_m"], out["T_m"] = fit["offset_m"], T
        out["area_error_cells"] = fit["error"]
        out["M_observed_reref_domain"] = float((~(F < T) * wS_dom).sum() * cell)
        for D in DELTAS:
            for surf, top in (("uncorrected", L26), ("shifted", L26 + c)):
                band = (F < top) & ~(F < top - D)
                bands[(surf, D)] = band
                out[f"E_{surf}_{D:g}m_domain"] = float((band * w_dom).sum() * cell)
                out[f"E_{surf}_{D:g}m_validated"] = float((band * w_val).sum() * cell)
            band = (F < T) & ~(F < T - D)
            bands[("observed_reref", D)] = band
            out[f"E_observed_reref_{D:g}m_domain"] = float((band * wS_dom).sum() * cell)
            out[f"E_observed_reref_{D:g}m_validated"] = float((band * wS_val).sum() * cell)
        return out, bands

    weight = {"uncorrected": w_dom, "shifted": w_dom, "observed_reref": wS_dom}
    obs_rows, obs_bands = [], {}
    for p in names:
        m, b = measures(F0[p], p)
        obs_rows.append({"product": p, **m})
        obs_bands[p] = b
    for key in obs_bands[names[0]]:
        stk = np.stack([obs_bands[p][key] for p in names])
        obs_rows.append({"product": "all_five_agreement", "surface": key[0], "delta_m": key[1],
                         "ratio": nt.agreement_ratio(stk, weight[key[0]])})
    del obs_bands
    obs = pd.DataFrame(obs_rows)
    OUT2.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT2 / "sim2_observed_480m.csv", index=False)
    C.log("unperturbed 480 m done")
    ranges = [RANGE_M]
    children = np.random.SeedSequence(SIM.SEED).spawn(len(ranges) * len(S["names"]))
    real_rows, group_rows = [], []
    for ir, Rr in enumerate(ranges):
        for ip, p in enumerate(S["names"]):
            rng = np.random.default_rng(children[ir * len(S["names"]) + ip])
            t1 = time.time()
            acc_all, acc_any, g, i_in = {}, {}, 0, 0
            fields = None
            for i in range(N_FIELDS):
                if fields is None or not len(fields):
                    fields, _ = nt.grf_exponential(ny, nx, SIM.GRF_DX, sill[p], Rr, rng, n=min(20, N_FIELDS - i))
                    fields = list(fields)
                f = fields.pop(0)
                eps = nt.bilinear_to(f, y0, x0, SIM.GRF_DX, ys, xs)
                F = nt.minimax_flood(zc[p] + eps, seed_c)
                m, bands = measures(F, p)
                real_rows.append({"range_m": Rr, "product": p, "realisation": i + 1,
                                  "eps_mean_m": float(eps.mean()), "eps_sd_m": float(eps.std()), **m})
                for key, b in bands.items():
                    if i_in == 0:
                        acc_all[key], acc_any[key] = b.copy(), b.copy()
                    else:
                        acc_all[key] &= b
                        acc_any[key] |= b
                i_in += 1
                if i_in == SIM.GROUP:
                    g += 1
                    for key in acc_all:
                        w = weight[key[0]]
                        an = float((acc_any[key] * w).sum())
                        group_rows.append({"range_m": Rr, "product": p, "group": g, "surface": key[0],
                                           "delta_m": key[1], "ratio": float((acc_all[key] * w).sum()) / an
                                           if an > 0 else np.nan})
                    i_in = 0
                del F, eps
            C.log("simulated", p, N_FIELDS, "fields", round(time.time() - t1, 1), "s")
    rz = pd.DataFrame(real_rows)
    gz = pd.DataFrame(group_rows)
    # check: the D-125 realisations are reproduced
    checks = {}
    old_g = pd.read_csv(C.OUT / "sim_groups.csv")
    old_r = pd.read_csv(C.OUT / "sim_realisations.csv")
    mg = gz[gz["surface"].isin(["uncorrected", "shifted"])].merge(
        old_g, on=["range_m", "product", "group", "surface", "delta_m"], suffixes=("", "_d125"), how="inner")
    checks["groups_rows_compared"] = int(len(mg))
    checks["groups_max_abs_diff"] = float(np.nanmax(np.abs(mg["ratio"] - mg["ratio_d125"])))
    cols = [c_ for c_ in old_r.columns if c_.startswith("E_uncorrected") or c_.startswith("E_shifted")]
    mr = rz.merge(old_r[["range_m", "product", "realisation"] + cols], on=["range_m", "product", "realisation"],
                  suffixes=("", "_d125"), how="inner")
    checks["realisations_rows_compared"] = int(len(mr))
    checks["realisations_max_abs_diff_km2"] = float(max(np.nanmax(np.abs(mr[c_] - mr[c_ + "_d125"])) for c_ in cols))
    old_o = pd.read_csv(C.OUT / "sim_observed_240m.csv")
    oa = old_o[old_o["product"] == "all_five_agreement"]
    na = obs[obs["product"] == "all_five_agreement"]
    mo = na.merge(oa[["surface", "delta_m", "ratio"]], on=["surface", "delta_m"], suffixes=("", "_d125"))
    checks["observed_agreement_max_abs_diff"] = float(np.abs(mo["ratio"] - mo["ratio_d125"]).max())
    checks["crop"] = crop
    C.log("checks", checks)
    if not (len(mg) == 2 * 2 * 10 * len(names) and len(mr) == N_FIELDS * len(names)) or \
            checks["groups_max_abs_diff"] > 1e-12 or checks["realisations_max_abs_diff_km2"] > 1e-6 or \
            checks["observed_agreement_max_abs_diff"] > 1e-12:
        rz.to_csv(OUT2 / "sim2_realisations_FAILED_CHECK.csv", index=False)
        raise SystemExit("the D-125 realisations are not reproduced")
    rz.to_csv(OUT2 / "sim2_realisations.csv", index=False)
    gz.to_csv(OUT2 / "sim2_groups.csv", index=False)
    # summary
    cur = pd.read_csv(OUT2 / "reref_curves.csv")
    srows = []
    for p, g in rz.groupby("product"):
        o = obs[obs["product"] == p].iloc[0]
        for q in [c_ for c_ in rz.columns if c_.startswith("E_observed_reref") or c_ in ("c26_m", "M_observed_reref_domain")]:
            v = g[q].to_numpy(float)
            rec = {"range_m": RANGE_M, "product": p, "quantity": q, "n": int(v.size), "mean": float(v.mean()),
                   "sd": float(v.std(ddof=1)), "p05": float(np.percentile(v, 5)), "p50": float(np.median(v)),
                   "p95": float(np.percentile(v, 95)), "unperturbed_480m": float(o[q])}
            rec["sd_rel"] = rec["sd"] / rec["unperturbed_480m"] if rec["unperturbed_480m"] > 0 else np.nan
            if q.startswith("E_"):
                D, reg = float(q.split("_")[-2][:-1]), q.split("_")[-1]
                r60 = cur[(cur["product"] == p) & (cur["variant"] == "area_match") & (cur["sensitivity"] == "main")
                          & (cur["region"] == reg) & (cur["quantity"] == "E") & np.isclose(cur["delta_m"], D)]
                rec["value_60m"] = float(r60["km2"].iloc[0]) if len(r60) else np.nan
            srows.append(rec)
        for (surf, D), gg in gz[gz["product"] == p].groupby(["surface", "delta_m"]):
            v = gg["ratio"].to_numpy(float)
            ob = float(obs[(obs["product"] == "all_five_agreement") & (obs["surface"] == surf)
                           & np.isclose(obs["delta_m"], D)]["ratio"].iloc[0])
            srows.append({"range_m": RANGE_M, "product": p, "quantity": f"agreement_{surf}_{D:g}m", "n": int(v.size),
                          "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
                          "p05": float(np.nanpercentile(v, 5)), "p50": float(np.nanmedian(v)),
                          "p95": float(np.nanpercentile(v, 95)), "observed_five_products_480m": ob,
                          "share_groups_at_or_below_observed": float(np.mean(v <= ob))})
    pd.DataFrame(srows).to_csv(OUT2 / "sim2_summary.csv", index=False)
    C.write_json(OUT2 / "run_simulate2.json", {
        "script": "scripts/rev2_n_simulate.py", "decision": "D-146 (D-125 settings, first addendum)",
        "amendment": "PREREG Amendment 19 item 3", "seed": SIM.SEED, "n_fields": N_FIELDS, "range_m": RANGE_M,
        "grf_dx_m": SIM.GRF_DX, "flood_grid_m": k * 60, "group_size": SIM.GROUP, "sill_m2": sill,
        "c26_target_cells": target, "level_2026_m": L26, "shape_constants_m": S["c"],
        "flood_grid_shape": [int(R), int(Cc)], "checks": checks, "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
