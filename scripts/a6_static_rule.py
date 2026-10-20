"""A6: the static-contour rule (D-034, fixed in D-075) on training blocks.

Contour points whose location does not move with level (channel banks, ponds) are
flagged as static and excluded from the DEM (scripts/a6_dem.py reads the `static`
column). The statistic is rho_level (scripts/a6_waterlines.py): the Spearman
correlation, at the point's pixel, between water frequency and bin level over the
epoch's valid bins. Sea-controlled shorelines get wetter as the level rises (rho
near 1); channel banks do not (rho near 0).

Rule (D-075): the threshold rho* is Otsu's split of the rho_level histogram (256 bins
on [-1, 1], caspian.thresholds.diagnostics) of the E2 and E3 contour points in
TRAINING blocks only. It is used if the split is valid by the D-061 rules (eta >=
0.70, valley ratio <= 0.50) and lies in [0.2, 0.8]; otherwise rho* = 0.5. A point
is static if rho_level < rho*. Points without rho_level (fewer than 4 valid bins)
are kept and counted.

Outputs: the points table rewritten with `static` and `block_split` columns;
results/levels/static_contour_rule.json (rho*, the diagnostics, the training
histogram, and the excluded fraction per sector, split and epoch).

Usage: python scripts/a6_static_rule.py <tag>
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import blocking, waterlines
from caspian.config import INTERIM, RESULTS

EPOCHS = ("E2", "E3")                  # the DEM epochs
STEEP_M = 120.0                        # two 60 m cells (D-075 steep-slope check)


def main(argv) -> int:
    tag = argv[0]
    path = INTERIM / "waterlines" / tag / "points.csv.gz"
    p = pd.read_csv(path)
    blocks = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    p["block_split"] = blocking.split_of_points(p["x"], p["y"], blocks, blocks["split"])
    p["sector"] = blocking.split_of_points(p["x"], p["y"], blocks, blocks["sector"])
    train = p[(p["block_split"] == "train") & p["epoch"].isin(EPOCHS)]
    rule = waterlines.static_threshold(train["rho_level"])
    p["static"] = p["rho_level"] < rule["rho_star"]          # NaN compares False: kept
    summ = (p.assign(no_rho=~np.isfinite(p["rho_level"]))
            .groupby(["sector", "block_split", "epoch"], dropna=False)
            .agg(n=("static", "size"), static=("static", "mean"), no_rho=("no_rho", "mean")).reset_index())
    # D-034 check: contours on real steep slopes must be kept. Steep = a training point
    # within STEEP_M of the contour of the next level bin (same epoch).
    from scipy.spatial import cKDTree
    steep = np.zeros(len(train), bool)
    tv = train.reset_index(drop=True)
    for (ep, b), g in tv.groupby(["epoch", "bin"]):
        nxt = tv[(tv["epoch"] == ep) & (tv["bin"] == b + 1)]
        if len(nxt):
            d, _ = cKDTree(nxt[["x", "y"]].to_numpy()).query(g[["x", "y"]].to_numpy())
            steep[g.index.to_numpy()] = d < STEEP_M
    st_train = tv["rho_level"].to_numpy() < rule["rho_star"]
    rule["steep_check"] = {"steep_m": STEEP_M, "n_steep": int(steep.sum()),
                           "static_share_steep": float(st_train[steep].mean()) if steep.any() else None,
                           "static_share_train": float(st_train.mean()) if len(st_train) else None}
    rule.update(tag=tag, epochs=list(EPOCHS), excluded=summ.to_dict(orient="records"))
    (RESULTS / "levels").mkdir(parents=True, exist_ok=True)
    (RESULTS / "levels" / "static_contour_rule.json").write_text(json.dumps(rule, indent=1, default=str),
                                                                   encoding="utf-8")
    p.to_csv(path, index=False, float_format="%.6f")
    print(f"rho* = {rule['rho_star']:.3f} ({rule['source']}; {rule['n_train_points']} training points); "
          f"static share overall {p['static'].mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
