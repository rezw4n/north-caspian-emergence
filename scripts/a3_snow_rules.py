"""PREREG Amendments 3 and 9: evaluate the seasonal snow/ice rules on stage-1 run 2, and
tabulate the monthly snow/ice shares used for the ice season (D-062).

For each test (the NDSI test; the product QA flag: Landsat CFMask snow bit or
Sentinel-2 SCL class 11) and each sensor family, the share of clear in-zone pixels
flagged is pooled by calendar month (cloud-masked diagnostics zone_snow_ndsi and
zone_snow_qa; denominator zone_clear + zone_snow). A test applies year-round only if
the share is at most design.SNOW_RULE_MAX_PCT in every May-September month.

Output (committed): results/qc/snow_rules_run2.json
"""

import glob
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design
from caspian.config import INTERIM, RESULTS

COLS = ["zone_clear", "zone_snow", "zone_snow_ndsi", "zone_snow_qa", "zone_px"]


def main() -> int:
    d = pd.concat([pd.read_csv(p, usecols=["scene", "sensor", "time"] + COLS)
                   for p in sorted(glob.glob(str(INTERIM / "stage1_v2" / "*.csv"))) if "thresholds" not in p],
                  ignore_index=True)
    for c in COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["month"] = pd.to_datetime(d["time"], unit="ms").dt.month
    d["family"] = np.where(d["sensor"] == "MSI_S2", "sentinel2", "landsat")
    d["den"] = d["zone_clear"] + d["zone_snow"]
    d = d[d["den"] > 0]
    out = {"rule_max_pct": design.SNOW_RULE_MAX_PCT, "tests": {}, "monthly": {}}
    for fam, g in d.groupby("family"):
        gm = g.groupby("month")
        for test, col in (("ndsi", "zone_snow_ndsi"), ("qa", "zone_snow_qa")):
            pct = (100 * gm[col].sum() / gm["den"].sum()).round(3)
            summer = pct.loc[[m for m in range(5, 10) if m in pct.index]]
            out["tests"][f"{fam}_{test}"] = {
                "may_sep_pct": summer.to_dict(), "max_may_sep_pct": float(summer.max()),
                "applies": "year-round" if summer.max() <= design.SNOW_RULE_MAX_PCT else "november-april"}
        share = np.maximum(g["zone_snow_ndsi"], g["zone_snow_qa"]) / g["den"]
        g = g.assign(share=share)
        gm = g.groupby("month")
        out["monthly"][fam] = {int(m): {"n_scenes": int(len(x)), "median_scene_pct": round(100 * float(x["share"].median()), 3),
                                        "p75_scene_pct": round(100 * float(x["share"].quantile(0.75)), 3),
                                        "pooled_pct": round(100 * float(np.maximum(x["zone_snow_ndsi"], x["zone_snow_qa"]).sum() / x["den"].sum()), 3)}
                               for m, x in gm}
    (RESULTS / "qc").mkdir(parents=True, exist_ok=True)
    (RESULTS / "qc" / "snow_rules_run2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["tests"], indent=1))
    for fam, mm in out["monthly"].items():
        print(fam, {m: (v["median_scene_pct"], v["pooled_pct"]) for m, v in mm.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
