"""Generality test, the decision (PREREG Amendment 16 items 8 and 9), applied
once to the frozen tables of scripts/g1_hindcast.py.

Per arm and product (global products and, for GSL, the 3DEP substitute):
- caspian.h1.mispredicts on the pooled-per-level displacement rows and the per-level
  noise floor (shape variant decides; total variant and the pooled-floor
  sensitivity reported);
- the 3DEP substitute only on its bins (lower edge at or above the arm's hydro-flattened
  2016 surface; scripts/g1_hindcast.py already restricts its run to them).
Per basin: a product "mispredicts in the basin" in a majority of evaluable arms
(caspian.generality.in_basin); the verdict (caspian.generality.verdict).
Descriptive (not deciding): median of the per-level displacement medians and of the
per-level noise-floor 95th percentiles; ETOPO 2022's displacement above and below the
arm's level on 2013-07-01 (the level-band mechanism of scripts/a7_level_bands.py).

Outputs: results/generality/<basin>/{h1_mispredicts.csv, h1_mispredicts_detail.csv,
h1_decision.json}
Usage: python scripts/g1_decide.py <basin>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import generality as G
from caspian import h1
from caspian.config import RESULTS


def arm_decision(disp: pd.DataFrame, noise: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(summary, detail, pooled-floor summary) of the H1 rule for one arm's tables."""
    prod = disp[(disp["epoch"] == "pooled") & (disp["sector"] == arm)].rename(columns={"median_m": "median"})
    nz = noise[(noise["kind"] == "pooled_level") & (noise["sector"] == arm)].rename(columns={"p95_m": "p95"})
    nz_all = noise[(noise["kind"] == "pooled_all_levels") & (noise["sector"] == arm)].rename(columns={"p95_m": "p95"})
    s, d = h1.mispredicts(prod, nz[["sector", "bin", "n", "p95"]])
    s_all, _ = h1.mispredicts(prod, nz_all[["sector", "n", "p95"]])
    return s, d, s_all


def finite_json(x):
    """Strict JSON: an infinite median (no predicted waterline at most levels) becomes the
    string "inf" (or "-inf"); NaN becomes null."""
    if isinstance(x, dict):
        return {k: finite_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_json(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def _flag(v):
    return None if v is None or pd.isna(v) else bool(v)


def basin_decision(tables: dict, products: list, control: str | None, level_split: dict | None = None,
                   offset_m: float = 0.0) -> dict:
    """``tables``: {arm: {"disp", "noise", "disp_control", "noise_control"}} (the control
    entries may be absent). Returns the decision record (see the module docstring)."""
    rec = {"products": {}, "arms": {}, "positive_control": None}
    flags = {p: {"shape": [], "total": []} for p in products}
    pc_flags = []
    details = []
    for arm, t in tables.items():
        s, d, s_all = arm_decision(t["disp"], t["noise"], arm)
        details.append(d.assign(arm=arm))
        nz = t["noise"][(t["noise"]["kind"] == "pooled_level") & (t["noise"]["sector"] == arm)]
        rec["arms"][arm] = {"noise_floor_p95_median_m": float(np.nanmedian(nz["p95_m"])) if len(nz) else None,
                            "n_noise_levels": int((nz["n"] > 0).sum())}
        for p in products:
            for var in ("shape", "total"):
                r = s[(s["product"] == p) & (s["variant"] == var)]
                ra = s_all[(s_all["product"] == p) & (s_all["variant"] == var)]
                if r.empty:
                    flags[p][var].append(None)
                    continue
                r = r.iloc[0]
                dd = d[(d["product"] == p) & (d["variant"] == var) & d["evaluable"]]
                flags[p][var].append(_flag(r["mispredicts"]))
                e = rec["products"].setdefault(p, {}).setdefault(var, {}).setdefault("arms", {})
                e[arm] = {"mispredicts": _flag(r["mispredicts"]), "n_observed_levels": int(r["n_observed_levels"]),
                          "n_evaluable": int(r["n_evaluable"]), "n_exceeding": int(r["n_exceeding"]),
                          "mispredicts_pooled_floor": _flag(ra.iloc[0]["mispredicts"]) if len(ra) else None,
                          "median_displacement_m": float(np.median(dd["median"])) if len(dd) else None,
                          "median_noise_p95_m": float(np.median(dd["p95_noise"])) if len(dd) else None}
                if p == "etopo_2022" and var == "shape" and level_split and arm in level_split and len(dd):
                    lv = dd["level_m"].astype(float) + offset_m
                    e[arm]["median_displacement_above_2013_m"] = (float(np.median(dd.loc[lv >= level_split[arm], "median"]))
                                                                 if (lv >= level_split[arm]).any() else None)
                    e[arm]["median_displacement_below_2013_m"] = (float(np.median(dd.loc[lv < level_split[arm], "median"]))
                                                                 if (lv < level_split[arm]).any() else None)
        if control and "disp_control" in t:
            sc, dc, _ = arm_decision(t["disp_control"], t["noise_control"], arm)
            r = sc[(sc["product"] == control) & (sc["variant"] == "shape")]
            f = _flag(r.iloc[0]["mispredicts"]) if len(r) else None
            pc_flags.append(f)
            ddc = dc[(dc["product"] == control) & (dc["variant"] == "shape") & dc["evaluable"]]
            rec.setdefault("control_arms", {})[arm] = {
                "mispredicts": f, "n_evaluable": int(r.iloc[0]["n_evaluable"]) if len(r) else 0,
                "n_exceeding": int(r.iloc[0]["n_exceeding"]) if len(r) else 0,
                "median_displacement_m": float(np.median(ddc["median"])) if len(ddc) else None,
                "median_noise_p95_m": float(np.median(ddc["p95_noise"])) if len(ddc) else None}
    for p in products:
        for var in ("shape", "total"):
            e = rec["products"].setdefault(p, {}).setdefault(var, {})
            e["n_sectors_mispredict"] = int(sum(bool(v) for v in flags[p][var] if v is not None))
            e["n_sectors_evaluable"] = int(sum(v is not None for v in flags[p][var]))
            e["in_basin"] = G.in_basin(flags[p][var])
    pc = G.in_basin(pc_flags) if control else None
    rec["positive_control"] = {"product": control, "mispredicts_in_basin": pc}
    rec["verdict"] = G.verdict({p: rec["products"][p]["shape"]["in_basin"] for p in products}, pc)
    rec["verdict_total_variant"] = G.verdict({p: rec["products"][p]["total"]["in_basin"] for p in products}, pc)
    rec["detail"] = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return rec


def main(basin: str) -> int:
    B = G.BASINS[basin]
    d = RESULTS / "generality" / basin
    tables = {}
    for arm in B["sectors"]:
        h = d / arm / "hindcast"
        t = {"disp": pd.read_csv(h / "displacement.csv"), "noise": pd.read_csv(h / "noise_floor.csv")}
        c = d / arm / "hindcast_dep3"
        if (c / "displacement.csv").exists():
            t["disp_control"], t["noise_control"] = pd.read_csv(c / "displacement.csv"), pd.read_csv(c / "noise_floor.csv")
        tables[arm] = t
    split = None
    lv_path = Path(RESULTS.parent / "data" / "raw" / "generality" / basin / "levels" / "gsl_daily_levels.csv")
    if basin == "great_salt_lake" and lv_path.exists():
        lv = pd.read_csv(lv_path, parse_dates=["date"])
        split = {a: float(g.loc[g["date"] == "2013-07-01", "level_navd88_m"].iloc[0]) for a, g in lv.groupby("arm")}
    rec = basin_decision(tables, G.GLOBAL_PRODUCTS, "dep3" if basin == "great_salt_lake" else None, split,
                         B["level_offset_m"])
    det = rec.pop("detail")
    det.to_csv(d / "h1_mispredicts_detail.csv", index=False)
    rows = [{"product": p, "variant": v, "arm": a, **x} for p, vv in rec["products"].items() for v, e in vv.items()
            for a, x in e.get("arms", {}).items()]
    pd.DataFrame(rows).to_csv(d / "h1_mispredicts.csv", index=False)
    offsets = {}
    for arm in B["sectors"]:
        for sub in ("hindcast", "hindcast_dep3"):
            f = d / arm / sub / "shape_offsets.json"
            if f.exists():
                so = json.loads(f.read_text(encoding="utf-8"))
                for p, r in so["products"].items():
                    offsets.setdefault(arm, {})[p] = {"offset_m": r["offset_m"], "loeo_m": {e: x["offset_m"] for e, x in
                                                                                         r["leave_one_epoch_out"].items()}}
    rec.update(basin=basin, deciding_variant="shape (D-072)", rule="PREREG Amendment 16 items 8 and 9",
               level_2013_07_01_m=split, shape_offsets=offsets)
    (d / "h1_decision.json").write_text(json.dumps(finite_json(rec), indent=1, default=str, allow_nan=False),
                                        encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("verdict", "verdict_total_variant", "positive_control")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
