"""Robustness: one table of the headline numbers of every completed variant against the
primary, from results/robustness/verdicts.json (scripts/r20_verdicts.py), for the G3 memo
and the plan's status section. Writes results/robustness/headline_table.csv.

Usage: python scripts/r22_report_table.py
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import pandas as pd

from caspian.config import RESULTS

ROB = RESULTS / "robustness"


def main() -> int:
    v = json.loads((ROB / "verdicts.json").read_text(encoding="utf-8"))
    rows = []
    items = [(c, n, r) for c, rec in v["checks"].items() for n, r in rec["variants"].items()]
    items += [("named", n, r) for n, r in v["named"].items()]
    for check, name, r in sorted(items):
        m = r["metrics"]
        row = {"check": check, "variant": name, "verdict": r["verdict"]}
        for src in ("atl06", "atl08"):
            for q in ("dem_nmad_m", "dem_rmse_m", "dem_bias_m", "coverage"):
                if f"{src}.{q}" in m:
                    row[f"{src}.{q}"] = f"{m[f'{src}.{q}_primary']:.3f} -> {m[f'{src}.{q}']:.3f}"
            for var in ("matched", "total"):
                k = f"{src}.h2_{var}_supported"
                if k in m:
                    row[f"{src}.h2_{var}"] = f"{m[k + '_primary']} -> {m[k]}"
            if f"{src}.min_product_nmad_m" in m:
                row[f"{src}.min_product_nmad_m"] = round(m[f"{src}.min_product_nmad_m"], 3)
        h1 = {k.split(".")[0]: (m[k.replace("n_sectors_mispredict", "n_sectors_mispredict_primary")], m[k])
              for k in m if k.endswith(".n_sectors_mispredict")}
        if h1:
            row["h1_sectors_mispredict"] = "; ".join(f"{p} {a}->{b}" for p, (a, b) in sorted(h1.items()))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(ROB / "headline_table.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 80):
        print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
