"""A18 items 10(j) and 10(k): emerged area at 2035 and 2050 for every pathway x product on
the common baseline (a) and the shifted products (b), the shares of the sum of squares
from pathway, product and interaction, the range pathways, each pathway's
2021-2026 change against the observed fall, and the observed-trend pathway over three
windows. Exploratory.

Inputs: results/revision/n/curves_fine.csv.gz (scripts/rev_n_curves.py; with the
composite and constant-form baselines when present), the pathways as
scripts/a8_emergence.py reads and dates them (read_pathways, exposure.pathway_curves),
the DAHITI series, results/emergence/main/run.json (the 2026 level) and
results/emergence/main/scenario_dates.csv (the range pathways' dates).

Outputs (results/revision/n/): pathways_matrix.csv, pathways_variance.csv,
pathways_anchoring.csv, trend_windows.csv, run_pathways.json.

Usage: python scripts/rev_n_pathways.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as C  # noqa: E402

from caspian import exposure as ex  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian.config import RESULTS  # noqa: E402

YEARS = (2035.0, 2050.0)
BASE = 2026.0
SURFACES = ("common_area", "common_location", "shifted", "shifted_common_location", "uncorrected",
            "common_area_composite", "common_location_composite", "shifted_common_location_composite",
            "common_location_constfused")
EXTRA = {"_composite": ("common_area", "common_location", "shifted_common_location"),      # composite baseline
         "_constfused": ("common_location",)}                                               # constant-form
WINDOWS = ((1996, 2021), (2006, 2025), (2016, 2025))
HORIZON = 2100.0


def e_at(curve: pd.DataFrame, fall: float) -> tuple:
    """E (km2) at a fall from a fine curve (delta_m, E_km2): E(max(fall, 0)); nan beyond the
    curve's largest decline. Returns (E, note)."""
    d, e = curve["delta_m"].to_numpy(float), curve["E_km2"].to_numpy(float)
    if not np.isfinite(fall):
        return np.nan, "no fall (outside the pathway's years)"
    if fall > d.max() + 1e-9:
        return np.nan, f"fall beyond {d.max():g} m"
    note = "rise or no fall: E at 0 m" if fall <= 0 else ""
    return float(np.interp(max(fall, 0.0), d, e)), note


def main(argv) -> int:
    t0 = time.time()
    a8 = C.load_script("a8_emergence")
    fine = pd.read_csv(C.OUT / "curves_fine.csv.gz")
    for sfx, surfs in EXTRA.items():
        f2 = C.OUT / f"curves_fine{sfx}.csv.gz"
        if f2.exists():
            x = pd.read_csv(f2)
            x = x[x["surface"].isin(surfs)]
            fine = pd.concat([fine, x.assign(surface=x["surface"] + sfx)], ignore_index=True)
    erun = json.loads((RESULTS / "emergence" / "main" / "run.json").read_text(encoding="utf-8"))
    L26 = round(float(erun["level_2026"]["level_m"]), 4)
    pw = a8.read_pathways(a8.PATHWAYS)
    curves = ex.pathway_curves(pw)
    role = dict(zip(pw["pathway_id"], pw["role"]))
    primary = [k for k in curves if role.get(k) == "primary"]
    rng_ids = [k for k in curves if role.get(k) == "range"]
    products = sorted(fine["product"].unique(), key=lambda p: list(C.LABEL).index(p))
    rows = []
    for pid in primary + rng_ids:
        for y in YEARS:
            fall = nt.pathway_fall(curves[pid], y, BASE)
            for surf in SURFACES:
                for reg in ("domain", "validated"):
                    for p in products:
                        cur = fine[(fine["surface"] == surf) & (fine["product"] == p) & (fine["region"] == reg)]
                        E, note = e_at(cur, fall)
                        rows.append({"pathway_id": pid, "role": role[pid], "year": int(y), "fall_m": fall,
                                     "surface": surf, "region": reg, "product": p, "E_km2": E, "note": note})
    mat = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    mat.to_csv(C.OUT / "pathways_matrix.csv", index=False)
    # variance shares, primary pathways only
    vrows = []
    for (surf, reg, y), g in mat[mat["role"] == "primary"].groupby(["surface", "region", "year"]):
        tab = g.pivot(index="pathway_id", columns="product", values="E_km2").loc[primary, products]
        y_area = tab.to_numpy(float)
        if surf.startswith("common_area"):
            y_area = np.maximum(y_area, 0.0)              # clipped at 0 for the decomposition
        for scale, yy in (("area_km2", y_area), ("log10_area_plus_1", np.log10(y_area + 1.0))):
            if not np.isfinite(yy).all():
                vrows.append({"surface": surf, "region": reg, "year": int(y), "scale": scale, "note": "non-finite cell"})
                continue
            sh = nt.two_way_shares(yy)
            vrows.append({"surface": surf, "region": reg, "year": int(y), "scale": scale,
                          "share_pathway": sh["row"], "share_product": sh["column"],
                          "share_interaction": sh["interaction"], "ss_total": sh["ss_total"],
                          "range_over_pathways_median_product": float(np.median(np.ptp(yy, axis=0))),
                          "range_over_products_median_pathway": float(np.median(np.ptp(yy, axis=1))),
                          "n_pathways": len(primary), "n_products": len(products)})
    pd.DataFrame(vrows).to_csv(C.OUT / "pathways_variance.csv", index=False)
    # (k) anchoring
    d = pd.read_csv(C.DAHITI, sep=";")
    t = pd.to_datetime(d["datetime"])
    lev = d["wse"].to_numpy(float)
    in26 = (t.dt.year == 2026).to_numpy()
    first26, last26 = t[in26].min(), t[in26].max()
    y21 = (t.dt.year == 2021).to_numpy()
    md = t.dt.strftime("%m%d")
    win = y21 & (md >= first26.strftime("%m%d")).to_numpy() & (md <= last26.strftime("%m%d")).to_numpy()
    obs_full = float(lev[y21].mean() - L26)
    obs_win = float(lev[win].mean() - L26)
    arows = []
    for pid in primary + rng_ids:
        arows.append({"pathway_id": pid, "role": role[pid],
                      "pathway_fall_2021_2026_m": nt.pathway_fall(curves[pid], 2026.0, 2021.0),
                      "observed_fall_2021_2026_m": obs_full, "observed_fall_same_window_m": obs_win,
                      "n_records_2021": int(y21.sum()), "n_records_2021_window": int(win.sum()),
                      "n_records_2026": int(in26.sum())})
    an = pd.DataFrame(arows)
    an["pathway_minus_observed_m"] = an["pathway_fall_2021_2026_m"] - an["observed_fall_2021_2026_m"]
    an.to_csv(C.OUT / "pathways_anchoring.csv", index=False)
    # (k) trend windows
    ty = nt.decimal_year(d["datetime"])
    trows = []
    for a, b in WINDOWS:
        m = (t.dt.year >= a).to_numpy() & (t.dt.year <= b).to_numpy()
        r = nt.ols_rate(ty[m], lev[m])
        rec = {"window": f"{a}-{b}", "start_year": a, "end_year": b, **r,
               "n_years_with_records": int(t[m].dt.year.nunique())}
        for D in (1.0, 2.0):
            yr = BASE + D / -r["rate_m_per_yr"] if r["rate_m_per_yr"] < 0 else np.inf
            rec[f"year_reaches_{D:g}m"] = float(yr) if yr <= HORIZON else np.nan
            rec[f"note_{D:g}m"] = "" if yr <= HORIZON else f"not by {HORIZON:.0f}"
        trows.append(rec)
    pd.DataFrame(trows).to_csv(C.OUT / "trend_windows.csv", index=False)
    C.write_json(C.OUT / "run_pathways.json", {
        "script": "scripts/rev_n_pathways.py", "decision": "D-127", "years": list(YEARS), "base_year": BASE,
        "level_2026_m": L26, "primary": primary, "range": rng_ids, "products": products,
        "observed_fall_2021_2026_m": obs_full, "observed_fall_same_window_m": obs_win,
        "window_2026_records": [str(first26.date()), str(last26.date())],
        "anchoring_rule": "exposure.scenario_dates anchor 'base': fall after 2026 = c(2026) - c(t); calendar time "
                          "kept (not re-anchored); the pathway's change before 2026 is dropped",
        "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
