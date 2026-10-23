"""A19 item 2 (D-146): the pathway × grid partition of emerged area at 2035 and 2050 for
every baseline (as distributed, shifted, observed re-referenced; the superseded A18
observed-shoreline estimate carried for the Supplementary deviation), on the linear and
the log scale, with the grid factor as five products and as three depth-source lineages
(GEBCO = GEBCO_2026, GEBCO_2022, ETOPO 2022; ETOPO1; SRTM15+), and the ratio of the
largest to the smallest emerged area across products (and lineages) at 0.5, 1 and 2 m.
Exploratory (post-review).

Pathways, falls and interpolation exactly as scripts/rev_n_pathways.py (D-127). A
lineage's value is the mean of its members' emerged areas (km2); sensitivity: GEBCO_2022
alone for the GEBCO lineage.

Inputs: results/revision/n/curves_fine.csv.gz (uncorrected, shifted),
results/revision/n/curves_fine_composite.csv.gz (common_location, superseded),
results/revision2/n/reref_curves_fine.csv.gz (scripts/rev2_n_reref.py).

Outputs (results/revision2/n/): pathways2_matrix.csv, pathways2_variance.csv,
product_ratios.csv, run_pathways2.json.

Usage: python scripts/rev2_n_pathways.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as C  # noqa: E402
import rev_n_pathways as P  # noqa: E402

from caspian import exposure as ex  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian import rev2_n as r2  # noqa: E402
from caspian.config import RESULTS  # noqa: E402

OUT2 = RESULTS / "revision2" / "n"
BASELINES = {   # name: (file, surface in the file, the A18 surface name in pathways_variance.csv)
    "as_distributed": (C.OUT / "curves_fine.csv.gz", "uncorrected", "uncorrected"),
    "shifted": (C.OUT / "curves_fine.csv.gz", "shifted", "shifted"),
    "observed_reref": (OUT2 / "reref_curves_fine.csv.gz", "observed_reref", None),
    "observed_a18_superseded": (C.OUT / "curves_fine_composite.csv.gz", "common_location", "common_location_composite"),
}
NEAR = (0.5, 1.0, 2.0)
REP = {"GEBCO": "gebco_2022"}


def main(argv) -> int:
    t0 = time.time()
    a8 = C.load_script("a8_emergence")
    fine = []
    for name, (f, surf, _) in BASELINES.items():
        x = pd.read_csv(f)
        x = x[x["surface"] == surf]
        if x.empty:
            raise SystemExit(f"{f.name}: no surface {surf}")
        fine.append(x.assign(baseline=name))
    fine = pd.concat(fine, ignore_index=True)
    pw = a8.read_pathways(a8.PATHWAYS)
    curves = ex.pathway_curves(pw)
    role = dict(zip(pw["pathway_id"], pw["role"]))
    primary = [k for k in curves if role.get(k) == "primary"]
    products = sorted(fine["product"].unique(), key=lambda p: list(C.LABEL).index(p))
    rows = []
    for pid in primary:
        for y in P.YEARS:
            fall = nt.pathway_fall(curves[pid], y, P.BASE)
            for (b, reg, p), cur in fine.groupby(["baseline", "region", "product"]):
                E, note = P.e_at(cur.sort_values("delta_m"), fall)
                rows.append({"pathway_id": pid, "year": int(y), "fall_m": fall, "baseline": b, "region": reg,
                             "product": p, "E_km2": E, "note": note})
    mat = pd.DataFrame(rows)
    OUT2.mkdir(parents=True, exist_ok=True)
    mat.to_csv(OUT2 / "pathways2_matrix.csv", index=False)
    vrows = []
    for (b, reg, y), g in mat.groupby(["baseline", "region", "year"]):
        tab = g.pivot(index="pathway_id", columns="product", values="E_km2").loc[primary, products].to_numpy(float)
        if (tab < 0).any():
            raise SystemExit(f"negative emerged area on {b}")
        grids = {"products": (tab, products)}
        lt, ln = r2.lineage_table(tab, products)
        grids["lineages_mean"] = (lt, ln)
        lr, _ = r2.lineage_table(tab, products, representative=REP)
        grids["lineages_gebco2022"] = (lr, ln)
        for gname, (t, cols) in grids.items():
            for scale, yy in (("area_km2", t), ("log10_area_plus_1", np.log10(t + 1.0))):
                if not np.isfinite(yy).all():
                    vrows.append({"baseline": b, "region": reg, "year": int(y), "grid_factor": gname, "scale": scale,
                                  "note": "non-finite cell"})
                    continue
                sh = nt.two_way_shares(yy)
                vrows.append({"baseline": b, "region": reg, "year": int(y), "grid_factor": gname, "scale": scale,
                              "share_pathway": sh["row"], "share_grid": sh["column"],
                              "share_interaction": sh["interaction"], "ss_total": sh["ss_total"],
                              "n_pathways": len(primary), "n_grid_levels": len(cols), "grid_levels": ";".join(cols),
                              "max_fall_m": float(g["fall_m"].max())})
    var = pd.DataFrame(vrows)
    var.to_csv(OUT2 / "pathways2_variance.csv", index=False)
    # check: the 7 x 5 shares reproduce rev_n_pathways (A18) where both exist
    old = pd.read_csv(C.OUT / "pathways_variance.csv")
    worst, n_cmp = 0.0, 0
    for name, (_, _, oldsurf) in BASELINES.items():
        if oldsurf is None:
            continue
        for r in var[(var["baseline"] == name) & (var["grid_factor"] == "products")].itertuples():
            o = old[(old["surface"] == oldsurf) & (old["region"] == r.region) & (old["year"] == r.year)
                    & (old["scale"] == r.scale)]
            if len(o) != 1:
                raise SystemExit(f"check: no single A18 row for {oldsurf} {r.region} {r.year} {r.scale}")
            for a, bcol in (("share_pathway", "share_pathway"), ("share_grid", "share_product"),
                            ("share_interaction", "share_interaction")):
                worst = max(worst, abs(getattr(r, a) - float(o[bcol].iloc[0])))
            n_cmp += 1
    if worst > 1e-9:
        raise SystemExit(f"check failed: the A18 shares are not reproduced ({worst})")
    C.log("A18 shares reproduced", n_cmp, "rows, max abs diff", worst)
    # ratios at fixed declines
    rrows = []
    for (b, reg), g in fine.groupby(["baseline", "region"]):
        for D in NEAR:
            v = g[np.isclose(g["delta_m"], D)].set_index("product")["E_km2"].reindex(products)
            if v.isna().any():
                raise SystemExit(f"no value at {D} m on {b}")
            lt, ln = r2.lineage_table(v.to_numpy(float)[None, :], products)
            rrows.append({"baseline": b, "region": reg, "delta_m": D,
                          "ratio_max_min_products": r2.max_min_ratio(v.to_numpy(float)),
                          "max_product": v.idxmax(), "max_km2": float(v.max()),
                          "min_product": v.idxmin(), "min_km2": float(v.min()),
                          "ratio_max_min_lineages": r2.max_min_ratio(lt[0]),
                          **{f"E_{C.KEY[p]}_km2": float(v[p]) for p in products},
                          **{f"E_lineage_{n}_km2": float(x) for n, x in zip(ln, lt[0])}})
    pd.DataFrame(rrows).to_csv(OUT2 / "product_ratios.csv", index=False)
    C.write_json(OUT2 / "run_pathways2.json", {
        "script": "scripts/rev2_n_pathways.py", "decision": "D-146", "amendment": "PREREG Amendment 19 item 2",
        "years": list(P.YEARS), "base_year": P.BASE, "primary": primary, "products": products,
        "lineages": {k: list(v) for k, v in r2.LINEAGES.items()}, "lineage_value": "mean of the members' km2",
        "lineage_sensitivity": REP, "baselines": {k: [str(v[0].relative_to(C.ROOT)), v[1]] for k, v in BASELINES.items()},
        "checks": {"a18_products_shares_max_abs_diff": worst, "n_rows_compared": n_cmp},
        "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
