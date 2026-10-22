"""Revision R: diagnostics for reviewer comments (D-119).

  season       R2.M1(c): the share of May-July Landsat acquisitions per (epoch, level bin)
               of the main stage-2 plan, and its Spearman correlation with the per-unit
               displacement residuals (unit D minus the median D of the product's units in
               the sector; primary shape variant) in the Volga and Ural deltas, per product;
               Holm over the 10 tests. A per-level version (epochs pooled) is reported beside it.
  komsomolets  R1.M1 (last bullet): GEBCO_2026, GEBCO_2022 and SRTM15+ native cells in
               Komsomolets Bay (one 15" grid): identical values, differences; the shape
               constants; the shelf gradient; and, if the constant search has run, the
               sector median displacement against c for the three grids.
  nmad         R1.M4(d): ICESat-2 NMAD (test blocks, matched variant) per sector and grid,
               converted to an expected horizontal displacement with the sector gradient
               (NMAD / g), against the audited sector median displacement; rank agreement.
Outputs: results/revision/r/diagnostics/<command>_*.csv and <command>.json
Usage: python scripts/rev_r_diagnostics.py season|komsomolets|nmad
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import shapely
from scipy import stats

from caspian import products
from caspian.config import INTERIM, RESULTS
from caspian.geography import sectors

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rev_r_tables", ROOT / "scripts" / "rev_r_tables.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

OUT = RESULTS / "revision" / "r" / "diagnostics"
PLAN = INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv"
MAIN = RESULTS / "hindcast" / "main"
FLOOD_MONTHS = (5, 6, 7)
PRODUCTS = T.PRODUCTS


def _write(name: str, rec: dict, tables: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(OUT / f"{name}_{k}.csv", index=False)
    (OUT / f"{name}.json").write_text(T._json(rec), encoding="utf-8")
    print(T._json(rec)[:4000])


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def cmd_season() -> int:
    plan = pd.read_csv(PLAN, usecols=["scene", "acq", "t", "epoch", "bin"]).drop_duplicates("scene")
    plan["month"] = pd.to_datetime(plan["t"], format="ISO8601").dt.month
    acq = plan.drop_duplicates(["acq", "epoch", "bin"])
    sh = (acq.assign(flood=acq["month"].isin(FLOOD_MONTHS))
          .groupby(["epoch", "bin"]).agg(n_acq=("acq", "size"), n_flood=("flood", "sum")).reset_index())
    sh["share_may_jul"] = sh["n_flood"] / sh["n_acq"]
    shb = (acq.assign(flood=acq["month"].isin(FLOOD_MONTHS)).groupby("bin")
           .agg(n_acq=("acq", "size"), n_flood=("flood", "sum")).reset_index())
    shb["share_may_jul"] = shb["n_flood"] / shb["n_acq"]
    disp = pd.read_csv(MAIN / "displacement.csv")
    rows, rows_l = [], []
    for sct in ("volga", "ural"):
        u = disp[(disp["variant"] == "shape") & disp["epoch"].isin(["E1", "E2", "E3"]) & (disp["sector"] == sct)]
        lv = disp[(disp["variant"] == "shape") & (disp["epoch"] == "pooled") & (disp["sector"] == sct)]
        for p in PRODUCTS:
            g = u[(u["product"] == p) & np.isfinite(u["median_m"])].merge(sh, on=["epoch", "bin"])
            g = g.assign(resid_m=g["median_m"] - g["median_m"].median())
            r = stats.spearmanr(g["share_may_jul"], g["resid_m"])
            rows.append({"sector": sct, "product": p, "n_units": len(g), "rho": float(r.statistic),
                         "p": float(r.pvalue)})
            gl = lv[(lv["product"] == p) & np.isfinite(lv["median_m"])].merge(shb, on="bin")
            gl = gl.assign(resid_m=gl["median_m"] - gl["median_m"].median())
            rl = stats.spearmanr(gl["share_may_jul"], gl["resid_m"])
            rows_l.append({"sector": sct, "product": p, "n_levels": len(gl), "rho": float(rl.statistic),
                           "p": float(rl.pvalue)})
    res, resl = pd.DataFrame(rows), pd.DataFrame(rows_l)
    res["p_holm"] = holm(res["p"])
    resl["p_holm"] = holm(resl["p"])
    within = {}
    for ep, g in sh.groupby("epoch"):          # is the flood-season share confounded with level?
        r = stats.spearmanr(g["bin"], g["share_may_jul"])
        within[ep] = {"n_bins": len(g), "rho_share_vs_level": float(r.statistic), "p": float(r.pvalue)}
    rec_w = {"within_epoch_share_vs_level": within}
    rec = {**rec_w, "months": list(FLOOD_MONTHS), "unit": "Landsat acquisitions (acq) of the main stage-2 plan",
           "share_by_epoch": sh.groupby("epoch")["share_may_jul"].describe()[["min", "50%", "max"]].round(3)
           .to_dict(orient="index"),
           "units": res.round(4).to_dict(orient="records"), "levels": resl.round(4).to_dict(orient="records"),
           "n_p_below_0_05": int((res["p"] < 0.05).sum()), "n_holm_below_0_05": int((res["p_holm"] < 0.05).sum()),
           "rho_range": [float(res["rho"].min()), float(res["rho"].max())]}
    _write("season", rec, {"shares": sh, "shares_by_bin": shb, "spearman_units": res, "spearman_levels": resl})
    return 0


def cmd_komsomolets() -> int:
    poly = sectors()["komsomolets"][1]
    nat = {p: products.load_native(p) for p in ("gebco_2026", "gebco_2022", "srtm15p_v2.7")}
    a = nat["gebco_2026"]
    for p, n in nat.items():
        if not (np.allclose(n.lon, a.lon) and np.allclose(n.lat, a.lat)):
            raise SystemExit(f"{p}: not on GEBCO_2026's native grid")
    LON, LAT = np.meshgrid(a.lon, a.lat)
    inside = shapely.contains_xy(poly, LON, LAT)
    z = {p: n.z for p, n in nat.items()}
    fin = inside & np.isfinite(z["gebco_2026"]) & np.isfinite(z["gebco_2022"]) & np.isfinite(z["srtm15p_v2.7"])
    band = fin & (np.minimum.reduce([z[p] for p in z]) <= -24) & (np.maximum.reduce([z[p] for p in z]) >= -32)
    pairs = [("gebco_2026", "gebco_2022"), ("gebco_2026", "srtm15p_v2.7"), ("gebco_2022", "srtm15p_v2.7")]
    rows = []
    for which, m in (("sector", fin), ("band_-32_to_-24", band)):
        for x, y in pairs:
            d = (z[x] - z[y])[m].astype(float)
            rows.append({"cells": which, "pair": f"{x}|{y}", "n": int(m.sum()), "share_identical": float(np.mean(d == 0)),
                         "share_within_0p01": float(np.mean(np.abs(d) <= 0.01)),
                         "share_within_1m": float(np.mean(np.abs(d) <= 1.0)),
                         "median_diff_m": float(np.median(d)), "mean_abs_diff_m": float(np.mean(np.abs(d))),
                         "p05_diff_m": float(np.percentile(d, 5)), "p95_diff_m": float(np.percentile(d, 95))})
        all3 = (z["gebco_2026"] == z["gebco_2022"]) & (z["gebco_2026"] == z["srtm15p_v2.7"])
        rows.append({"cells": which, "pair": "all three identical", "n": int(m.sum()),
                     "share_identical": float(np.mean(all3[m]))})
    so = json.loads((MAIN / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    rec = {"shape_constants_m": {p: so[p]["offset_m"] for p in nat}, "cells": rows}
    grad = RESULTS / "revision" / "r" / "budget" / "gradients.csv"
    if grad.exists():
        g = pd.read_csv(grad).set_index("sector").loc["komsomolets"]
        rec["gradient_m_per_km"] = {"median": float(g["g_median_m_per_km"]), "mean": float(g["g_mean_m_per_km"])}
        rec["km_per_0p2m_at_median_gradient"] = float(0.2 / g["g_median_m_per_km"])
    det = pd.read_csv(MAIN / "h1_mispredicts_detail.csv")
    k = det[(det["sector"] == "komsomolets") & (det["variant"] == "shape") & det["product"].isin(list(nat))]
    rec["primary_sector_median_D_km"] = (k[k["evaluable"]].groupby("product")["median"].median() / 1000).round(2).to_dict()
    tabs = {"cells": pd.DataFrame(rows)}
    vk = RESULTS / "revision" / "r" / "constants" / "verdicts_by_k.csv"
    lvl = RESULTS / "revision" / "r" / "audit" / "main" / "levels_D.csv.gz"
    if lvl.exists():
        L = pd.read_csv(lvl)
        L = L[L["config"].isin(["replica", "cd"]) & (L["sector"] == "komsomolets") & L["product"].isin(list(nat))]
        curve = (L[L["observed_level"].astype(bool) & (L["n"] > 0)].groupby(["product", "k", "c_m"])["median_m"]
                 .median().div(1000).rename("sector_median_D_km").reset_index())
        tabs["D_vs_c"] = curve
        rec["D_vs_c_km"] = {p: g.set_index("c_m")["sector_median_D_km"].round(2).to_dict()
                            for p, g in curve.groupby("product")}
    if vk.exists():
        v = pd.read_csv(vk)
        v = v[(v["sector"] == "komsomolets") & v["product"].isin(list(nat))]
        tabs["exceeding_vs_c"] = v[["product", "k", "c_m", "n_evaluable", "n_exceeding", "mispredicts"]]
    _write("komsomolets", rec, tabs)
    return 0


def cmd_nmad() -> int:
    grad = pd.read_csv(RESULTS / "revision" / "r" / "budget" / "gradients.csv").set_index("sector")
    det = pd.read_csv(MAIN / "h1_mispredicts_detail.csv")
    d = det[(det["variant"] == "shape") & det["evaluable"].astype(bool)]
    Dmed = (d.groupby(["product", "sector"])["median"].median() / 1000).rename("D_sector_median_km")
    rows = []
    for src in ("ATL06", "ATL08"):
        acc = pd.read_csv(RESULTS / "validation" / "main" / f"{src}_accuracy.csv")
        acc = acc[acc["model"].str.endswith("|matched")].copy()
        acc["product"] = acc["model"].str.split("|").str[0]
        for r in acc.itertuples():
            if r.scope == "overall" or r.product not in PRODUCTS or not np.isfinite(r.nmad):
                continue
            g = float(grad.loc[r.scope, "g_median_m_per_km"])
            rows.append({"source": src, "product": r.product, "sector": r.scope, "n_points": int(r.n),
                         "nmad_m": float(r.nmad), "rmse_m": float(r.rmse), "bias_m": float(r.bias),
                         "g_m_per_km": g, "nmad_over_g_km": float(r.nmad) / g,
                         "abs_bias_over_g_km": abs(float(r.bias)) / g})
    t = pd.DataFrame(rows).merge(Dmed.reset_index(), on=["product", "sector"], how="left")
    t["D_over_nmad_equiv"] = t["D_sector_median_km"] / t["nmad_over_g_km"]
    ov = []
    for src in ("ATL06", "ATL08"):
        acc = pd.read_csv(RESULTS / "validation" / "main" / f"{src}_accuracy.csv")
        acc = acc[(acc["scope"] == "overall") & acc["model"].str.endswith("|matched")].copy()
        acc["product"] = acc["model"].str.split("|").str[0]
        for r in acc.itertuples():
            ov.append({"source": src, "product": r.product, "nmad_m": float(r.nmad), "rmse_m": float(r.rmse),
                       "bias_m": float(r.bias)})
    ov = pd.DataFrame(ov)
    mis = pd.read_csv(MAIN / "h1_mispredicts.csv")
    mis = mis[mis["variant"] == "shape"].groupby("product")["mispredicts"].sum().rename("n_sectors_mispredicted")
    dall = (d.groupby("product")["median"].median() / 1000).rename("D_all_levels_median_km")
    ov = ov.merge(mis.reset_index(), on="product").merge(dall.reset_index(), on="product")
    corr = []
    for (src, sct), g in t.groupby(["source", "sector"]):
        if len(g) >= 4 and g["D_sector_median_km"].notna().all():
            r = stats.spearmanr(g["nmad_m"], g["D_sector_median_km"])
            corr.append({"source": src, "sector": sct, "n_products": len(g), "rho_nmad_vs_D": float(r.statistic)})
    for src, g in ov.groupby("source"):
        r = stats.spearmanr(g["nmad_m"], g["D_all_levels_median_km"])
        corr.append({"source": src, "sector": "overall", "n_products": len(g), "rho_nmad_vs_D": float(r.statistic)})
    corr = pd.DataFrame(corr)
    rec = {"method": "expected horizontal displacement = vertical NMAD / sector median gradient g",
           "overall": ov.round(4).to_dict(orient="records"),
           "rank_correlations": corr.round(3).to_dict(orient="records"),
           "median_ratio_D_to_nmad_equiv": t.groupby("source")["D_over_nmad_equiv"].median().round(3).to_dict()}
    _write("nmad", rec, {"by_sector": t, "overall": ov, "rank_correlations": corr})
    return 0


def main(argv) -> int:
    cmd = argv[0] if argv else ""
    fn = {"season": cmd_season, "komsomolets": cmd_komsomolets, "nmad": cmd_nmad}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    return fn()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
