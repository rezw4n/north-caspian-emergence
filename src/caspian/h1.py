"""A7: the pre-registered H1 decision rule, applied to hindcast tables (PREREG
section 9 with Amendments 6 and 7; D-030, D-032).

These functions only read tables written by scripts/a7_hindcast.py (or synthetic
ones); the driver never calls them, so no H1 outcome is computed as a side effect
of producing the metrics.

"Mispredicts" in a sector (``mispredicts``): at each observed level of the sector
(a level at which the observed extent has a waterline cell in it, in any epoch), the
product's median absolute shoreline displacement in the sector, with the epochs of
that level pooled, is compared with the 95th percentile of the absolute noise-floor
displacements in the sector at the same level (the odd/even halves of every epoch,
and E1 against E2/E3 where the level was crossed twice, pooled). A level is
evaluable when at least one product distance and one noise-floor distance were
measured in the sector; other observed levels are reported as not evaluable, and no
pooled floor is substituted. A product median of +inf (no predicted waterline)
exceeds the floor. The product mispredicts when the levels that exceed are more than
half of the evaluable observed levels. Sensitivity (reported, not deciding): one
floor per sector, pooled over all its levels and epochs (pass a noise table without
a ``bin`` column).

"Largest in deltaic and vegetated sectors" (``block_test``, ``permutation_test``): a
one-sided block-level permutation test of the difference in median displacement
between blocks of the deltaic or vegetated sectors and the other blocks. The block
value (D-070) is, per block and product, the median of the absolute displacement
distances pooled over all its (epoch, level) units (scripts/a7_hindcast.py writes it
to block_h1.csv); blocks with fewer than 20 distances are left out and counted. A
block belongs to the sector that contains its centre (results/study_area/
blocks_25km.csv). The deltaic and vegetated labels are an input table: the vegetated
set needs the A3 vegetation fraction, fixed before any hindcast metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import design

KEYS = ["product", "variant", "sector"]


def mispredicts(product_levels: pd.DataFrame, noise_levels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, detail).

    ``product_levels``: one row per product, variant, sector and bin, epochs pooled,
    with ``observed_level`` (bool), ``n`` (distances measured) and ``median`` (median
    absolute displacement, m). ``noise_levels``: one row per sector and bin with ``n``
    and ``p95``; without a ``bin`` column, one row per sector (the pooled-floor
    sensitivity).

    ``detail`` adds per level: n_noise, p95_noise, evaluable, exceeds. ``summary`` per
    product, variant and sector: n_observed_levels, n_evaluable, n_not_evaluable,
    n_exceeding and ``mispredicts`` (True / False, or NA when no observed level is
    evaluable)."""
    on = ["sector", "bin"] if "bin" in noise_levels.columns else ["sector"]
    nz = noise_levels[on + ["n", "p95"]].rename(columns={"n": "n_noise", "p95": "p95_noise"})
    if nz.duplicated(on).any():
        raise ValueError(f"noise_levels must have one row per {on}")
    d = product_levels.merge(nz, on=on, how="left")
    d["n_noise"] = d["n_noise"].fillna(0).astype(int)
    obs = d["observed_level"].astype(bool)
    d["evaluable"] = obs & (d["n"] > 0) & (d["n_noise"] > 0)
    med = d["median"].astype(float)
    d["exceeds"] = d["evaluable"] & ((med == np.inf) | (med > d["p95_noise"].astype(float)))
    g = d.groupby(KEYS, sort=True)
    s = pd.DataFrame({"n_observed_levels": g["observed_level"].apply(lambda x: int(x.astype(bool).sum())),
                      "n_evaluable": g["evaluable"].sum().astype(int),
                      "n_exceeding": g["exceeds"].sum().astype(int)}).reset_index()
    s["n_not_evaluable"] = s["n_observed_levels"] - s["n_evaluable"]
    s["mispredicts"] = pd.Series(pd.NA, index=s.index, dtype="boolean")
    ev = s["n_evaluable"] > 0
    s.loc[ev, "mispredicts"] = s.loc[ev, "n_exceeding"] > s.loc[ev, "n_evaluable"] / 2
    return s, d


def permutation_test(blocks: pd.DataFrame, group_sectors, *, value: str = "value",
                     n_perm: int = design.PERMUTATIONS_N, seed: int = design.SEED,
                     alpha: float = design.ALPHA) -> dict:
    """One-sided block-level permutation test (PREREG section 9).

    ``blocks``: one row per block with ``sector`` and ``value`` (the block's
    displacement statistic). T = median(value of blocks in ``group_sectors``) -
    median(value of the other blocks). The group labels are permuted across blocks
    ``n_perm`` times (``seed``); p = (1 + #{T_perm >= T}) / (1 + n_perm), and the
    difference is significant when p <= alpha. Blocks with a NaN value are left out
    and counted."""
    v = blocks[value].astype(float).to_numpy()
    in_g = blocks["sector"].isin(list(group_sectors)).to_numpy()
    keep = ~np.isnan(v)
    v, in_g = v[keep], in_g[keep]
    out = {"n_group": int(in_g.sum()), "n_rest": int((~in_g).sum()), "n_excluded_nan": int((~keep).sum()),
           "n_perm": int(n_perm), "seed": int(seed), "alpha": float(alpha)}
    if out["n_group"] == 0 or out["n_rest"] == 0:
        return {**out, "statistic": float("nan"), "p_value": float("nan"), "significant": None}
    t = float(np.median(v[in_g]) - np.median(v[~in_g]))
    rng = np.random.default_rng(seed)
    k = int(in_g.sum())
    perm = np.empty(n_perm)
    for i in range(n_perm):
        g = np.zeros(v.size, bool)
        g[rng.permutation(v.size)[:k]] = True
        perm[i] = np.median(v[g]) - np.median(v[~g])
    if np.isnan(t):
        return {**out, "statistic": t, "p_value": float("nan"), "significant": None}
    p = (1 + int(np.sum(perm >= t))) / (1 + n_perm)
    return {**out, "statistic": t, "p_value": p, "significant": bool(p <= alpha)}


MIN_BLOCK_N = 20      # D-070


def block_test(blocks: pd.DataFrame, sector_labels: pd.DataFrame, *, value: str = "median_abs_m",
               min_n: int = MIN_BLOCK_N, n_perm: int = design.PERMUTATIONS_N, seed: int = design.SEED,
               alpha: float = design.ALPHA) -> pd.DataFrame:
    """``permutation_test`` per product and variant on the block table (D-070).

    ``blocks``: product, variant, block_sector, n (distances), and ``value``.
    ``sector_labels``: one row per sector with boolean ``deltaic`` and ``vegetated``;
    the group is deltaic or vegetated. Every sector of an included block must be
    labelled, with no missing value. Blocks with n < ``min_n`` and blocks without a
    sector are left out and counted."""
    lab = sector_labels.set_index("sector")[["deltaic", "vegetated"]]
    if lab.index.duplicated().any() or lab.isna().any().any():
        raise ValueError("sector_labels needs one row per sector and no missing label")
    group = set(lab.index[lab["deltaic"].astype(bool) | lab["vegetated"].astype(bool)])
    out = []
    for (prod, var), b in blocks.groupby(["product", "variant"], sort=True):
        few = b["n"] < min_n
        no_sector = b["block_sector"].isna()
        keep = b[~few & ~no_sector]
        missing = set(keep["block_sector"]) - set(lab.index)
        if missing:
            raise ValueError(f"no deltaic/vegetated label for sectors {sorted(missing)}")
        r = permutation_test(keep.rename(columns={"block_sector": "sector"}), group, value=value,
                             n_perm=n_perm, seed=seed, alpha=alpha)
        out.append({"product": prod, "variant": var, "n_blocks": len(b),
                    "n_excluded_few": int((few & ~no_sector).sum()), "n_excluded_no_sector": int(no_sector.sum()),
                    "min_n": int(min_n), "group_sectors": "+".join(sorted(group)), **r})
    return pd.DataFrame(out)
