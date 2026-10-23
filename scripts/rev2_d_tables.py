"""Second-review analyses D, table stages (PREREG Amendment 19 items 4, 6b, 12 and the
lineage table of item 14). Exploratory, post-review; the registered verdicts stand. Defaults
in D-149.

Stages (each reads committed results only and writes under results/revision2/d/):
  vertical   A19 item 4: the A18 item 7 vertical equivalents
             (results/revision/r/budget/vertical_equivalents.csv: per product and sector,
             the median over evaluable levels of D*g and Q*g) as a table with the validated
             sectors first, checked against the per-level rows of levels.csv, with the
             per-level IQR, the ratio of the medians and the ranges over the four
             mispredicting grids.
  correl     A19 item 6b: Pearson and Spearman correlations over levels of the per-level
             signed median displacement (results/hindcast/main/h1_mispredicts_detail.csv,
             shape variant, epochs pooled, evaluable levels) for every pair of products in
             each validated sector; the spread across products per level.
  changepoint A19 item 12: one mean shift in ETOPO 2022's per-level ln(D/Q), pooled over
             the seven sectors, by least squares over the bin edges (caspian.rev2_d.
             single_changepoint); the validated sectors alone as a sensitivity.
  lineage    A19 item 14 (with item 2's lineages): the existing verdicts by lineage.

Usage: python scripts/rev2_d_tables.py [vertical|correl|changepoint|lineage|all]
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import rev2_d
from caspian.config import RESULTS
from caspian.geography import SECTOR_ORDER

OUT = RESULTS / "revision2" / "d"
DETAIL = RESULTS / "hindcast" / "main" / "h1_mispredicts_detail.csv"
BUDGET = RESULTS / "revision" / "r" / "budget"
VALIDATED = ["west", "interfluve", "northeast", "open_shelf"]          # A18 item 1
OTHERS = [s for s in SECTOR_ORDER if s not in VALIDATED]
PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "srtm15p_v2.7", "etopo1"]
MISPREDICTING = ["gebco_2026", "gebco_2022", "srtm15p_v2.7", "etopo1"]   # shape verdicts in the validated sectors
LINEAGE = {"gebco_2026": "gebco", "gebco_2022": "gebco", "etopo_2022": "gebco", "etopo1": "etopo1",
           "srtm15p_v2.7": "srtm15p"}                                    # A19 item 2
MIN_LEVELS_CP = 3


def _write_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def _detail_shape() -> pd.DataFrame:
    d = pd.read_csv(DETAIL)
    d = d[(d["variant"] == "shape") & (d["epoch"] == "pooled")]
    if d.duplicated(["product", "sector", "bin"]).any():
        raise SystemExit("duplicate (product, sector, bin) rows in the detail table")
    return d


# --- item 4 -----------------------------------------------------------------------------------

def stage_vertical() -> int:
    ve = pd.read_csv(BUDGET / "vertical_equivalents.csv")
    lv = pd.read_csv(BUDGET / "levels.csv")
    lv = lv[lv["evaluable"].astype(bool)]
    agg = (lv.groupby(["product", "sector"])
           .agg(n_levels=("D_vertical_m", "size"), D_g_p25_m=("D_vertical_m", lambda v: np.percentile(v, 25)),
                D_g_p75_m=("D_vertical_m", lambda v: np.percentile(v, 75)), D_g_check_m=("D_vertical_m", "median"),
                Q_g_check_m=("Q_vertical_m", "median"), g_m_per_km=("g_median_m_per_km", "first"))
           .reset_index())
    t = ve.merge(agg, on=["product", "sector"], how="left")
    bad = (np.abs(t["D_vertical_median_m"] - t["D_g_check_m"]) > 1e-9) | (np.abs(t["Q_vertical_median_m"] - t["Q_g_check_m"]) > 1e-9)
    if bad.any():
        raise SystemExit(f"vertical_equivalents.csv does not match levels.csv in {int(bad.sum())} rows")
    t = t.drop(columns=["D_g_check_m", "Q_g_check_m"]).rename(
        columns={"D_vertical_median_m": "D_g_median_m", "Q_vertical_median_m": "Q_g_median_m"})
    t["ratio_D_to_Q"] = t["D_g_median_m"] / t["Q_g_median_m"]
    t["validated"] = t["sector"].isin(VALIDATED)
    t["sector"] = pd.Categorical(t["sector"], VALIDATED + OTHERS, ordered=True)
    t["product"] = pd.Categorical(t["product"], PRODUCTS, ordered=True)
    t = t.sort_values(["sector", "product"]).reset_index(drop=True)
    cols = ["sector", "validated", "product", "n_levels", "g_m_per_km", "D_g_median_m", "D_g_p25_m", "D_g_p75_m",
            "Q_g_median_m", "ratio_D_to_Q"]
    OUT.mkdir(parents=True, exist_ok=True)
    t[cols].to_csv(OUT / "vertical_equivalents_table.csv", index=False, float_format="%.4f")
    s = {"rule": "A19 item 4 (D-149): medians over evaluable levels of D*g and Q*g (A18 item 7, shape variant, "
                 "epochs pooled; g = sector median |grad z| of the waterline DEM in the band); reporting only",
         "source": "results/revision/r/budget/vertical_equivalents.csv (checked against levels.csv)",
         "by_sector": {}}
    for sec in VALIDATED + OTHERS:
        g = t[t["sector"] == sec].set_index("product")
        mis = g.loc[MISPREDICTING]
        s["by_sector"][sec] = {
            "validated": sec in VALIDATED,
            "Q_g_median_m": float(g["Q_g_median_m"].iloc[0]),
            "D_g_median_m": {p: float(g.loc[p, "D_g_median_m"]) for p in PRODUCTS},
            "mispredicting_four_min_m": float(mis["D_g_median_m"].min()),
            "mispredicting_four_max_m": float(mis["D_g_median_m"].max()),
            "ratio_D_to_Q_four_min": float(mis["ratio_D_to_Q"].min()),
            "ratio_D_to_Q_four_max": float(mis["ratio_D_to_Q"].max())}
    wi = t[t["sector"].isin(["west", "interfluve"])]
    m4 = wi[wi["product"].isin(MISPREDICTING)]
    s["west_interfluve"] = {"mispredicting_four_D_g_min_m": float(m4["D_g_median_m"].min()),
                            "mispredicting_four_D_g_max_m": float(m4["D_g_median_m"].max()),
                            "Q_g_min_m": float(wi["Q_g_median_m"].min()), "Q_g_max_m": float(wi["Q_g_median_m"].max()),
                            "etopo2022_D_g_min_m": float(wi.loc[wi["product"] == "etopo_2022", "D_g_median_m"].min()),
                            "etopo2022_D_g_max_m": float(wi.loc[wi["product"] == "etopo_2022", "D_g_median_m"].max())}
    va = t[t["validated"]]
    m4v = va[va["product"].isin(MISPREDICTING)]
    s["validated_four_sectors"] = {"mispredicting_four_D_g_min_m": float(m4v["D_g_median_m"].min()),
                                   "mispredicting_four_D_g_max_m": float(m4v["D_g_median_m"].max()),
                                   "Q_g_min_m": float(va["Q_g_median_m"].min()), "Q_g_max_m": float(va["Q_g_median_m"].max())}
    _write_json("vertical_equivalents_summary.json", s)
    print(t[cols].round(3).to_string(index=False))
    print(json.dumps({k: s[k] for k in ("west_interfluve", "validated_four_sectors")}, indent=1))
    return 0


# --- item 6b ----------------------------------------------------------------------------------

def stage_correl() -> int:
    d = _detail_shape()
    d = d[d["evaluable"].astype(bool) & d["sector"].isin(VALIDATED)]
    rows, spread, summ = [], [], {}
    for sec in VALIDATED:
        w = d[d["sector"] == sec].pivot(index="level_m", columns="product", values="signed_median_m")[PRODUCTS]
        w = w.replace([np.inf, -np.inf], np.nan)
        n_nonfinite = int(w.isna().sum().sum())
        c = rev2_d.pairwise_correlations(w)
        c.insert(0, "sector", sec)
        c["same_lineage"] = [LINEAGE[a] == LINEAGE[b] for a, b in zip(c["a"], c["b"])]
        c["both_mispredicting"] = [a in MISPREDICTING and b in MISPREDICTING for a, b in zip(c["a"], c["b"])]
        rows.append(c)
        full = w.dropna()
        sd = full.std(axis=1, ddof=1)
        rng = full.max(axis=1) - full.min(axis=1)
        sign = np.sign(full)
        same = (sign.nunique(axis=1) == 1)
        for lev in full.index:
            spread.append({"sector": sec, "level_m": lev, "mean_m": float(full.loc[lev].mean()),
                           "sd_m": float(sd.loc[lev]), "range_m": float(rng.loc[lev]),
                           "all_same_sign": bool(same.loc[lev]), "n_negative": int((full.loc[lev] < 0).sum())})
        m4 = full[MISPREDICTING]
        summ[sec] = {
            "n_levels": int(len(w)), "n_levels_all_finite": int(len(full)), "n_nonfinite_values": n_nonfinite,
            "pearson_median": float(c["pearson_r"].median()), "pearson_min": float(c["pearson_r"].min()),
            "pearson_max": float(c["pearson_r"].max()),
            "spearman_median": float(c["spearman_rho"].median()), "spearman_min": float(c["spearman_rho"].min()),
            "spearman_max": float(c["spearman_rho"].max()),
            "pearson_cross_lineage_median": float(c.loc[~c["same_lineage"], "pearson_r"].median()),
            "spearman_cross_lineage_median": float(c.loc[~c["same_lineage"], "spearman_rho"].median()),
            "pearson_mispredicting_pairs_median": float(c.loc[c["both_mispredicting"], "pearson_r"].median()),
            "spearman_mispredicting_pairs_median": float(c.loc[c["both_mispredicting"], "spearman_rho"].median()),
            "n_pairs": int(len(c)),
            "spread_sd_median_m": float(sd.median()), "spread_range_median_m": float(rng.median()),
            "spread_sd_mispredicting_four_median_m": float(m4.std(axis=1, ddof=1).median()),
            "share_levels_all_same_sign": float(same.mean()),
            "share_levels_all_negative": float((full < 0).all(axis=1).mean()),
            "mean_signed_median_m_median_over_levels": float(full.mean(axis=1).median()),
        }
    corr = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT / "signed_displacement_correlations.csv", index=False, float_format="%.6g")
    pd.DataFrame(spread).to_csv(OUT / "signed_displacement_spread_by_level.csv", index=False, float_format="%.3f")
    _write_json("signed_displacement_correlations.json", {
        "rule": "A19 item 6b (D-149): signed_median_m of results/hindcast/main/h1_mispredicts_detail.csv, shape "
                "variant, epochs pooled, evaluable levels; non-finite values dropped pairwise; positive = the grid "
                "predicts too much water (caspian.hindcast)",
        "lineages": LINEAGE, "mispredicting": MISPREDICTING, "by_sector": summ})
    print(corr.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))
    return 0


# --- item 12 ----------------------------------------------------------------------------------

def _changepoint(rows: pd.DataFrame, label: str) -> tuple[dict, pd.DataFrame]:
    lev = rows["level_m"].to_numpy(float)
    y = np.log(rows["D"].to_numpy(float) / rows["Q"].to_numpy(float))
    # every 0.1 m bin edge (x.x0 m) between the lowest and highest level; where a level has no
    # evaluable row, two adjacent edges give the same partition (reported as equivalent)
    edges = np.round(np.arange(np.round(lev.min(), 2) + 0.05, lev.max(), 0.1), 2)
    r = rev2_d.single_changepoint(lev, y, edges, MIN_LEVELS_CP)
    t = r["edge"]
    dq = rows["D"].to_numpy(float) / rows["Q"].to_numpy(float)
    below, above = lev < t, lev > t
    prof = pd.DataFrame(r.pop("profile"))
    equiv = prof.loc[prof["n_below"] == r["n_below"], "edge"].tolist()
    at277 = prof.loc[np.isclose(prof["edge"], -27.7)]
    out = {"label": label, "edge_m": t, "equivalent_edges_m": equiv, "rss": r["rss"],
           "rss_no_break": r["rss_no_break"], "r2": r["r2"],
           "n": r["n"], "n_below": r["n_below"], "n_above": r["n_above"],
           "n_levels_below": int(np.unique(lev[below]).size), "n_levels_above": int(np.unique(lev[above]).size),
           "mean_ln_DQ_below": r["mean_below"], "mean_ln_DQ_above": r["mean_above"],
           "DQ_geomean_below": float(np.exp(r["mean_below"])), "DQ_geomean_above": float(np.exp(r["mean_above"])),
           "DQ_median_below": float(np.median(dq[below])), "DQ_median_above": float(np.median(dq[above])),
           "share_D_gt_Q_below": float(np.mean(dq[below] > 1)), "share_D_gt_Q_above": float(np.mean(dq[above] > 1)),
           "min_levels_each_side": MIN_LEVELS_CP,
           "rss_at_m27p7": float(at277["rss"].iloc[0]) if len(at277) else None,
           "level_range_m": [float(lev.min()), float(lev.max())]}
    prof.insert(0, "label", label)
    return out, prof


def stage_changepoint() -> int:
    d = _detail_shape()
    d = d[(d["product"] == "etopo_2022") & d["evaluable"].astype(bool)].rename(columns={"median": "D", "p95_noise": "Q"})
    ok = np.isfinite(d["D"]) & np.isfinite(d["Q"]) & (d["D"] > 0) & (d["Q"] > 0)
    n_drop = int((~ok).sum())
    d = d[ok]
    res, profs = {}, []
    for label, rows in (("all_sectors", d), ("validated_sectors", d[d["sector"].isin(VALIDATED)])):
        r, p = _changepoint(rows, label)
        res[label] = r
        profs.append(p)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(profs, ignore_index=True).to_csv(OUT / "etopo2022_changepoint_profile.csv", index=False,
                                               float_format="%.6g")
    _write_json("etopo2022_changepoint.json", {
        "rule": "A19 item 12 (D-149): ETOPO 2022, shape variant, evaluable levels of results/hindcast/main/"
                "h1_mispredicts_detail.csv (D = median, Q = p95_noise), y = ln(D/Q), sectors pooled; one mean "
                "shift at a bin edge (midpoints of consecutive levels) minimising the residual sum of squares; "
                f"at least {MIN_LEVELS_CP} levels each side; ties to the lowest edge. Descriptive.",
        "n_rows_dropped_nonpositive_or_nonfinite": n_drop, **res})
    print(json.dumps(res, indent=1))
    return 0


# --- item 14: verdicts by lineage -------------------------------------------------------------

def stage_lineage() -> int:
    h1 = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts.csv")
    h1 = h1[h1["variant"] == "shape"]
    dec = json.loads((RESULTS / "hindcast" / "main" / "h1_decision.json").read_text(encoding="utf-8"))
    gsl = json.loads((RESULTS / "generality" / "great_salt_lake" / "h1_decision.json").read_text(encoding="utf-8"))
    rows = []
    for p in PRODUCTS:
        g = h1[h1["product"] == p].set_index("sector")["mispredicts"].astype(bool)
        bt = dec["block_test"][f"{p}|shape"]
        gp = gsl["products"].get(p, {}).get("shape", {})
        rows.append({"lineage": LINEAGE[p], "product": p,
                     "n_mispredict_all7": int(g.sum()), "n_mispredict_validated4": int(g[VALIDATED].sum()),
                     "n_mispredict_six": int(g.drop("komsomolets").sum()),
                     **{f"mispredicts_{s}": bool(g[s]) for s in VALIDATED},
                     "block_test_p": float(bt["p"]), "block_test_significant": bool(bt["significant"]),
                     "gsl_arms_mispredict": gp.get("n_sectors_mispredict"), "gsl_arms_evaluable": gp.get("n_sectors_evaluable")})
    t = pd.DataFrame(rows)
    lin = []
    for name, g in t.groupby("lineage", sort=False):
        lin.append({"lineage": name, "products": ",".join(g["product"]), "n_products": int(len(g)),
                    "validated_sectors_all_members_mispredict": int(sum(g[f"mispredicts_{s}"].all() for s in VALIDATED)),
                    "validated_sectors_any_member_mispredicts": int(sum(g[f"mispredicts_{s}"].any() for s in VALIDATED)),
                    "members_block_test_significant": int(g["block_test_significant"].sum()),
                    "members_mispredicting_in_all_validated": int((g["n_mispredict_validated4"] == 4).sum())})
    lt = pd.DataFrame(lin)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "verdicts_by_product_lineage.csv", index=False)
    lt.to_csv(OUT / "verdicts_by_lineage.csv", index=False)
    _write_json("verdicts_by_lineage.json", {
        "rule": "A19 items 2 and 14 (D-149): existing verdicts only (results/hindcast/main/h1_mispredicts.csv shape "
                "variant; h1_decision.json block test; results/generality/great_salt_lake/h1_decision.json); "
                "lineages GEBCO (GEBCO_2026, GEBCO_2022, ETOPO 2022), ETOPO1, SRTM15+",
        "products": rows, "lineages": lin})
    print(t.to_string(index=False))
    print(lt.to_string(index=False))
    return 0


def main(argv) -> int:
    stage = argv[0] if argv else "all"
    stages = {"vertical": stage_vertical, "correl": stage_correl, "changepoint": stage_changepoint,
              "lineage": stage_lineage}
    if stage == "all":
        return max(f() for f in stages.values())
    return stages[stage]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
