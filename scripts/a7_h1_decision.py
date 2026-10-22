"""The PREREG section 9 H1 decisions from the frozen hindcast tables
(caspian.h1), applied once after the hindcast run.

1. "Mispredicts" per product, variant and sector: the product's median absolute
   shoreline displacement (epochs pooled per level) exceeds the sector's
   noise-floor 95th percentile (pooled per level: both halves of every epoch, plus
   E1 against E2/E3) at the majority of the sector's evaluable observed levels.
   Sensitivity: the sector-wide pooled floor (all levels).
2. "Largest in deltaic and vegetated sectors": one-sided block-level permutation
   test (10,000 permutations) of the median block displacement, deltaic-or-vegetated
   sectors (results/levels/sector_labels.csv) against the rest.
The shape variant decides; the total variant is reported beside it.

Outputs (committed): results/hindcast/<tag>/h1_mispredicts.csv,
h1_mispredicts_pooled_floor.csv, h1_mispredicts_detail.csv, h1_block_test.csv, h1_decision.json
Usage: python scripts/a7_h1_decision.py <tag>
"""

import json
import sys

import caspian  # noqa: F401
import pandas as pd

from caspian import h1
from caspian.config import RESULTS


def main(tag: str) -> int:
    d = RESULTS / "hindcast" / tag
    disp = pd.read_csv(d / "displacement.csv")
    noise = pd.read_csv(d / "noise_floor.csv")
    prod = disp[(disp["epoch"] == "pooled") & (disp["sector"] != "all")].rename(columns={"median_m": "median"})
    nz_level = noise[(noise["kind"] == "pooled_level") & (noise["sector"] != "all")].rename(columns={"p95_m": "p95"})
    nz_all = noise[(noise["kind"] == "pooled_all_levels") & (noise["sector"] != "all")].rename(columns={"p95_m": "p95"})
    summ, detail = h1.mispredicts(prod, nz_level[["sector", "bin", "n", "p95"]])
    summ_all, _ = h1.mispredicts(prod, nz_all[["sector", "n", "p95"]])
    blocks = pd.read_csv(d / "block_h1.csv")
    labels = pd.read_csv(RESULTS / "levels" / "sector_labels.csv")
    bt = h1.block_test(blocks, labels)
    summ.to_csv(d / "h1_mispredicts.csv", index=False)
    summ_all.to_csv(d / "h1_mispredicts_pooled_floor.csv", index=False)
    detail.to_csv(d / "h1_mispredicts_detail.csv", index=False)
    bt.to_csv(d / "h1_block_test.csv", index=False)
    s = summ.reset_index() if "product" not in summ.columns else summ
    dec = {}
    for (p, v), g in s.groupby(["product", "variant"]):
        dec.setdefault(v, {})[p] = {"mispredicts_in": sorted(g.loc[g["mispredicts"] == True, "sector"]),
                                    "not_in": sorted(g.loc[g["mispredicts"] == False, "sector"]),
                                    "not_evaluable": sorted(g.loc[g["mispredicts"].isna(), "sector"])}
    btd = {f"{r.product}|{r.variant}": {"statistic_m": r.statistic, "p": r.p_value, "significant": r.significant}
           for r in bt.itertuples()}
    out = {"tag": tag, "deciding_variant": "shape (D-072)", "mispredicts": dec, "block_test": btd,
           "group_sectors": bt["group_sectors"].iloc[0] if len(bt) else None}
    (d / "h1_decision.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
