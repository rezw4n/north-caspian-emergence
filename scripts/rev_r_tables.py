"""Revision R: H1 verdicts from the audit-pass tables of scripts/rev_r_audit.py (A18 items
3 and 6).

Commands (each reads results/revision/r/audit/<name>/ and results/hindcast/main/):
  replica   the pass's primary against results/hindcast/main: per-unit and per-level
            medians and the floor must agree exactly (largest differences reported)
  static    verdicts with the static-contour filter on the observed waterlines, and the
            static shares of the audit waterline cells and of the DEM contour points
  res       verdicts with the resolution-matched floor (observed extents aggregated to
            each product's native cell)
  constants the displacement-fitted constant c_D per product and its verdicts; the
            best-case per-sector constant (fewest levels above the floor over the search)
  variant   verdicts of an observation variant against the primary (on the variant's
            footprint for the R-block variants): --name <audit name>
Outputs: results/revision/r/<command>/*.csv and summary.json.

Usage: python scripts/rev_r_tables.py replica|static|res|constants|variant [--name main]
"""

from __future__ import annotations

import argparse
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import h1
from caspian.config import INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

R = RESULTS / "revision" / "r"
AUDIT = R / "audit"
MAIN = RESULTS / "hindcast" / "main"
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]
NOT_VALIDATED = ["volga", "ural", "komsomolets"]
PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7"]
MISPREDICTING = ["gebco_2026", "gebco_2022", "etopo1", "srtm15p_v2.7"]


def _json(obj) -> str:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if o is pd.NA:
            return None
        return str(o)
    return json.dumps(obj, indent=1, default=conv)


def write(cmd: str, tables: dict, summary: dict):
    d = R / cmd
    d.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(d / f"{k}.csv", index=False)
    (d / "summary.json").write_text(_json(summary), encoding="utf-8")
    print(_json(summary)[:4000])


def main_levels():
    disp = pd.read_csv(MAIN / "displacement.csv")
    noise = pd.read_csv(MAIN / "noise_floor.csv")
    D = disp[(disp["epoch"] == "pooled") & (disp["sector"] != "all") & (disp["variant"] == "shape")]
    Q = noise[(noise["kind"] == "pooled_level") & (noise["sector"] != "all")]
    return D, Q


def verdicts(D: pd.DataFrame, Q: pd.DataFrame, variant: str):
    """caspian.h1.mispredicts on pass tables (D: product, sector, bin, observed_level, n,
    median_m; Q: sector, bin, n, p95_m)."""
    prod = D.assign(variant=variant).rename(columns={"median_m": "median"})
    nz = Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]]
    return h1.mispredicts(prod[["product", "variant", "sector", "bin", "level_m", "observed_level", "n", "median"]], nz)


def count_table(summ: pd.DataFrame, label: str) -> pd.DataFrame:
    s = summ.copy()
    s["config"] = label
    return s


def sector_counts(summ: pd.DataFrame) -> dict:
    out = {}
    for p, g in summ.groupby("product"):
        mis = g["mispredicts"].fillna(False).astype(bool)
        out[p] = {"all7": int(mis.sum()), "validated4": int(mis[g["sector"].isin(VALIDATED)].sum()),
                  "six_without_komsomolets": int(mis[g["sector"] != "komsomolets"].sum()),
                  "sectors": sorted(g.loc[mis, "sector"])}
    return out


def compare(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    k = ["product", "sector"]
    m = base[k + ["n_evaluable", "n_exceeding", "mispredicts"]].merge(
        new[k + ["n_evaluable", "n_exceeding", "mispredicts"]], on=k, suffixes=("_primary", "_new"))
    m["changed"] = m["mispredicts_primary"].astype("boolean") != m["mispredicts_new"].astype("boolean")
    m["validated"] = m["sector"].isin(VALIDATED)
    return m


# --- commands ----------------------------------------------------------------------------------

def cmd_replica(a) -> int:
    d = AUDIT / a.name
    LD = pd.read_csv(d / "levels_D.csv.gz")
    LQ = pd.read_csv(d / "levels_Q.csv")
    U = pd.read_csv(d / "units_D.csv.gz")
    D, Q = main_levels()
    disp = pd.read_csv(MAIN / "displacement.csv")
    rep = LD[LD["config"] == "replica"].merge(D, on=["product", "sector", "bin"], suffixes=("", "_main"))
    du = disp[(disp["variant"] == "shape") & disp["epoch"].isin(["E1", "E2", "E3"]) & (disp["sector"] != "all")]
    uu = U[U["config"] == "replica"].merge(du, on=["product", "epoch", "bin", "sector"], suffixes=("", "_main"))
    q = LQ[(LQ["config"] == "replica") & (LQ["kind"] == "pooled_level")].merge(Q, on=["sector", "bin"],
                                                                                suffixes=("", "_main"))

    def maxdiff(x, y):
        x, y = x.astype(float).to_numpy(), y.astype(float).to_numpy()
        both = np.isnan(x) & np.isnan(y)
        same_inf = np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y))
        dd = np.abs(x - y)
        dd[both | same_inf] = 0.0
        return float(np.nanmax(dd)) if dd.size else float("nan"), int(np.sum(np.isnan(x) != np.isnan(y)))

    out = {"levels_D_rows": len(rep), "levels_D_max_abs_diff_m": maxdiff(rep["median_m"], rep["median_m_main"]),
           "levels_D_n_max_abs_diff": maxdiff(rep["n"], rep["n_main"]),
           "units_rows": len(uu), "units_max_abs_diff_m": maxdiff(uu["median_m"], uu["median_m_main"]),
           "floor_rows": len(q), "floor_p95_max_abs_diff_m": maxdiff(q["p95_m"], q["p95_m_main"]),
           "floor_n_max_abs_diff": maxdiff(q["n"], q["n_main"])}
    write(f"replica_{a.name}", {}, out)
    return 0


def cmd_static(a) -> int:
    d = AUDIT / a.name
    LD = pd.read_csv(d / "levels_D.csv.gz")
    LQ = pd.read_csv(d / "levels_Q.csv")
    D, Q = main_levels()
    base, base_detail = h1.mispredicts(D.rename(columns={"median_m": "median"}),
                                       Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]])
    Ds = LD[LD["config"] == "static"]
    Qs = LQ[(LQ["config"] == "static") & (LQ["kind"] == "pooled_level")]
    summ, detail = verdicts(Ds, Qs, "shape_static")
    cmp = compare(base, summ)
    # floor and displacement change per sector: medians over levels
    qb = Q.groupby("sector")["p95_m"].median().rename("Q_median_primary_m")
    qs = Qs.groupby("sector")["p95_m"].median().rename("Q_median_static_m")
    db = D.groupby(["product", "sector"])["median_m"].median().rename("D_median_primary_m")
    ds = Ds.groupby(["product", "sector"])["median_m"].median().rename("D_median_static_m")
    eff = pd.concat([db, ds], axis=1).reset_index().merge(pd.concat([qb, qs], axis=1).reset_index(), on="sector")
    cells = pd.read_csv(d / "static_cells.csv")
    sh = cells.groupby(["sector", "epoch"])[["n_cells", "n_static"]].sum().reset_index()
    sh["share_static"] = sh["n_static"] / sh["n_cells"]
    sh_all = cells.groupby("sector")[["n_cells", "n_static"]].sum()
    sh_all["share_static"] = sh_all["n_static"] / sh_all["n_cells"]
    # DEM contour points (static-contour rule): share flagged static per sector, E2 + E3, all splits
    rule = json.loads((RESULTS / "levels" / "static_contour_rule.json").read_text(encoding="utf-8"))
    ex = pd.DataFrame(rule["excluded"])
    ex["n_static"] = ex["n"] * ex["static"]
    pts = ex.groupby("sector")[["n", "n_static"]].sum()
    pts["share_static"] = pts["n_static"] / pts["n"]
    summary = {"rho_star": rule["rho_star"], "counts_primary": sector_counts(base),
               "counts_static": sector_counts(summ),
               "n_changed": int(cmp["changed"].sum()),
               "changed": cmp.loc[cmp["changed"].astype(bool), ["product", "sector", "mispredicts_primary",
                                                                 "mispredicts_new"]].to_dict(orient="records"),
               "audit_cells_static_share_by_sector": sh_all["share_static"].round(4).to_dict(),
               "dem_points_static_share_by_sector": pts["share_static"].round(4).to_dict(),
               "floor_median_by_sector_m": pd.concat([qb, qs], axis=1).round(1).to_dict(orient="index")}
    write("static", {"verdicts": summ, "detail": detail, "compare": cmp, "effect": eff,
                     "audit_cells_static_share": sh, "dem_points_static_share": pts.reset_index()}, summary)
    return 0


def cmd_res(a) -> int:
    d = AUDIT / a.name
    LD = pd.read_csv(d / "levels_D.csv.gz")
    LQ = pd.read_csv(d / "levels_Q.csv")
    D, Q = main_levels()
    base, _ = h1.mispredicts(D.rename(columns={"median_m": "median"}),
                             Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]])
    grid_of = {"gebco_2026": "res_15s", "gebco_2022": "res_15s", "srtm15p_v2.7": "res_15s", "etopo_2022": "res_15s",
               "etopo1": "res_1m"}
    parts, det = [], []
    for p, g in grid_of.items():
        Dp = LD[(LD["config"] == "res") & (LD["product"] == p)]
        Qg = LQ[(LQ["config"] == g) & (LQ["kind"] == "pooled_level")]
        s, dt = verdicts(Dp, Qg, f"shape_{g}")
        parts.append(s)
        det.append(dt)
    summ, detail = pd.concat(parts, ignore_index=True), pd.concat(det, ignore_index=True)
    cmp = compare(base, summ)
    qrows = []
    for g in ("res_15s", "res_1m"):
        qg = LQ[(LQ["config"] == g) & (LQ["kind"] == "pooled_level")].groupby("sector")["p95_m"].median()
        for s_, v in qg.items():
            qrows.append({"grid": g, "sector": s_, "Q_median_m": v})
    qtab = pd.DataFrame(qrows).merge(Q.groupby("sector")["p95_m"].median().rename("Q_median_primary_m").reset_index(),
                                     on="sector")
    summary = {"counts_primary": sector_counts(base), "counts_resolution_matched": sector_counts(summ),
               "n_changed": int(cmp["changed"].sum()),
               "changed": cmp.loc[cmp["changed"].astype(bool), ["product", "sector", "mispredicts_primary",
                                                                 "mispredicts_new"]].to_dict(orient="records"),
               "floor_median_by_grid_sector_m": qtab.round(1).to_dict(orient="records")}
    write("res", {"verdicts": summ, "detail": detail, "compare": cmp, "floor": qtab}, summary)
    return 0


def cmd_constants(a) -> int:
    d = AUDIT / a.name
    LD = pd.read_csv(d / "levels_D.csv.gz")
    LQ = pd.read_csv(d / "levels_Q.csv")
    obj = pd.read_csv(d / "cd_objective.csv")
    D, Q = main_levels()
    Qr = LQ[(LQ["config"] == "replica") & (LQ["kind"] == "pooled_level")]
    base, _ = h1.mispredicts(D.rename(columns={"median_m": "median"}),
                             Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]])
    so = json.loads((MAIN / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    curves, fits, per_k = [], {}, []
    for p, g in obj.groupby("product"):
        g = g.copy()
        ok = g.groupby(["epoch", "bin"])["median_train_m"].apply(lambda x: bool(np.all(np.isfinite(x))))
        keep = set(ok[ok].index)
        g["use"] = [(e, b) in keep for e, b in zip(g["epoch"], g["bin"])]
        cur = g[g["use"]].groupby(["k", "c_m"])["median_train_m"].agg(["sum", "size"]).reset_index()
        cur["product"] = p
        curves.append(cur)
        best = cur["sum"].min()
        cand = cur[cur["sum"] == best].copy()
        cand["dist"] = np.abs(cand["k"])
        pick = cand.sort_values(["dist", "c_m"]).iloc[0]
        c_area = float(so[p]["offset_m"])
        fits[p] = {"c_area_m": c_area, "c_D_m": float(pick["c_m"]), "k": int(pick["k"]),
                   "c_D_minus_c_area_m": round(float(pick["c_m"]) - c_area, 2),
                   "objective_km_at_c_D": round(float(pick["sum"]) / 1000, 3),
                   "objective_km_at_c_area": round(float(cur.loc[cur["k"] == 0, "sum"].iloc[0]) / 1000, 3),
                   "n_units": int(pick["size"]), "n_units_dropped": int((~ok).sum()),
                   "at_search_edge": bool(abs(int(pick["k"])) == int(cur["k"].abs().max())),
                   "c_step_m": round(float(np.diff(np.sort(cur["c_m"].unique()))[0]), 2), "n_tied": int(len(cand))}
    for (p, k), g in LD[LD["config"].isin(["replica", "cd"])].groupby(["product", "k"]):
        s, _ = verdicts(g, Qr, f"k{k}")
        s["k"], s["c_m"] = int(k), float(g["c_m"].iloc[0])
        per_k.append(s)
    pk = pd.concat(per_k, ignore_index=True)
    at_cd = pd.concat([pk[(pk["product"] == p) & (pk["k"] == f["k"])] for p, f in fits.items()], ignore_index=True)
    cmp = compare(base, at_cd)
    # best case per sector: the fewest levels above the floor over the search
    pk["frac"] = pk["n_exceeding"] / pk["n_evaluable"].replace(0, np.nan)
    rows = []
    for (p, s_), g in pk.groupby(["product", "sector"]):
        g = g.sort_values(["n_exceeding", "frac"])
        r = g.iloc[0]
        rows.append({"product": p, "sector": s_, "min_n_exceeding": int(r["n_exceeding"]),
                     "n_evaluable_at_min": int(r["n_evaluable"]), "c_at_min_m": float(r["c_m"]), "k_at_min": int(r["k"]),
                     "min_is_majority": bool(r["n_exceeding"] > r["n_evaluable"] / 2),
                     "min_frac_exceeding": float(r["frac"]),
                     "n_k_mispredicting": int(g["mispredicts"].fillna(False).astype(bool).sum()),
                     "validated": s_ in VALIDATED})
    best = pd.DataFrame(rows)
    summary = {"fits": fits, "counts_primary": sector_counts(base), "counts_at_c_D": sector_counts(at_cd),
               "n_changed_at_c_D": int(cmp["changed"].sum()),
               "changed_at_c_D": cmp.loc[cmp["changed"].astype(bool), ["product", "sector", "mispredicts_primary",
                                                                       "mispredicts_new"]].to_dict(orient="records"),
               "best_case": {p: {"no_single_constant_sectors": sorted(g.loc[g["min_is_majority"], "sector"]),
                                 "n_no_single_constant": int(g["min_is_majority"].sum()),
                                 "n_no_single_constant_validated": int(g.loc[g["validated"], "min_is_majority"].sum())}
                             for p, g in best.groupby("product")}}
    write("constants", {"objective": pd.concat(curves, ignore_index=True), "verdicts_by_k": pk,
                        "verdicts_at_c_D": at_cd, "compare_at_c_D": cmp, "best_case": best}, summary)
    return 0


def cmd_variant(a) -> int:
    d = AUDIT / a.name
    LD = pd.read_csv(d / "levels_D.csv.gz")
    LQ = pd.read_csv(d / "levels_Q.csv")
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    res = {}
    tabs = {}
    ref_cfg = "primary_fp" if run["hybrid"] else None
    Dv = LD[LD["config"] == "variant"]
    Qv = LQ[(LQ["config"] == "variant") & (LQ["kind"] == "pooled_level")]
    sv, _ = verdicts(Dv, Qv, "variant")
    if ref_cfg:
        Dp = LD[LD["config"] == ref_cfg]
        Qp = LQ[(LQ["config"] == ref_cfg) & (LQ["kind"] == "pooled_level")]
        sp, _ = verdicts(Dp, Qp, "primary_fp")
    else:
        D, Q = main_levels()
        sp, _ = h1.mispredicts(D.rename(columns={"median_m": "median"}),
                               Q.rename(columns={"p95_m": "p95"})[["sector", "bin", "n", "p95"]])
        Dp = D
    cmp = compare(sp, sv)
    med = (Dv.groupby(["product", "sector"])["median_m"].median().rename("D_median_variant_m").to_frame()
           .join(Dp.groupby(["product", "sector"])["median_m"].median().rename("D_median_reference_m")))
    res = {"name": a.name, "stage2": run["stage2"], "hybrid": run["hybrid"],
           "reference": "primary on the same footprint" if ref_cfg else "primary (results/hindcast/main)",
           "counts_reference": sector_counts(sp), "counts_variant": sector_counts(sv),
           "n_changed": int(cmp["changed"].sum()),
           "n_changed_validated": int((cmp["changed"].astype(bool) & cmp["validated"]).sum()),
           "changed": cmp.loc[cmp["changed"].astype(bool), ["product", "sector", "mispredicts_primary",
                                                             "mispredicts_new", "n_exceeding_primary",
                                                             "n_exceeding_new", "n_evaluable_new"]].to_dict(orient="records")}
    tabs = {f"verdicts_{a.name}": sv, f"reference_{a.name}": sp, f"compare_{a.name}": cmp,
            f"medians_{a.name}": med.reset_index()}
    d2 = R / "variant"
    d2.mkdir(parents=True, exist_ok=True)
    for k, t in tabs.items():
        t.to_csv(d2 / f"{k}.csv", index=False)
    (d2 / f"summary_{a.name}.json").write_text(_json(res), encoding="utf-8")
    print(_json(res)[:3000])
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["replica", "static", "res", "constants", "variant"])
    ap.add_argument("--name", default="main")
    a = ap.parse_args(argv)
    return {"replica": cmd_replica, "static": cmd_static, "res": cmd_res, "constants": cmd_constants,
            "variant": cmd_variant}[a.command](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
