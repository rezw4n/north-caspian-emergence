"""A8: the pre-registered H3 decision rule, applied to emergence tables (PREREG
section 10).

These functions only read tables written by scripts/a8_emergence.py (or synthetic
ones); the driver never calls them, so no H3 outcome is computed as a side effect of
producing the areas and intervals (as caspian.h1 for H1).

Rule (PREREG section 10): "the correction is decision-relevant for a scenario if the
corrected emerged area lies outside the 90% interval of the uncorrected estimate, or
the reverse." ``decision`` applies it per product, region, functional and scenario:
the corrected estimate (the fused surface) against the uncorrected interval, and the
uncorrected estimate against the corrected interval. An infinite interval (too few
calibration scores) contains every value, so that direction cannot be decision-
relevant; this is reported, not hidden.

Summary (proposed; the pre-registration does not say how scenarios combine): per
product, region and functional, over the near-term scenarios (design.NEAR_TERM_RANGE_M,
0.5-2 m), "supported" if the correction is decision-relevant at every near-term
scenario, "partial" if at some, "not supported" if at none, NA if no near-term scenario
is present. The headline is the domain and the primary functional ("emergence"); every
other row is reported beside it. Products are never pooled. Exposure is reported with
intervals and is not part of the rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import design

KEYS = ["product", "region", "functional"]
CORRECTED, UNCORRECTED = "fused", "uncorrected"


def _outside(x, lo, hi):
    x, lo, hi = (np.asarray(v, float) for v in (x, lo, hi))
    return ~np.isnan(x) & ~np.isnan(lo) & ~np.isnan(hi) & ((x < lo) | (x > hi))


def decision(emergence: pd.DataFrame, *, corrected: str = CORRECTED, uncorrected: str = UNCORRECTED) -> pd.DataFrame:
    """One row per product, region, functional and scenario.

    ``emergence``: rows with product, surface, region, functional, scenario_m,
    estimate_km2, lo_km2 and hi_km2 (the emergence.csv table of scripts/a8_emergence.py),
    with one ``corrected`` and one ``uncorrected`` row per key and scenario. Adds the
    two directions (``corrected_outside_uncorrected``, ``uncorrected_outside_corrected``),
    ``decision_relevant`` (either), the difference corrected - uncorrected, and whether
    an interval was infinite."""
    need = {*KEYS, "surface", "scenario_m", "estimate_km2", "lo_km2", "hi_km2"}
    missing = need - set(emergence.columns)
    if missing:
        raise ValueError(f"emergence table lacks columns {sorted(missing)}")
    on = [*KEYS, "scenario_m"]
    cols = ["estimate_km2", "lo_km2", "hi_km2"]
    c = emergence[emergence["surface"] == corrected][on + cols]
    u = emergence[emergence["surface"] == uncorrected][on + cols]
    for name, t in ((corrected, c), (uncorrected, u)):
        if t.duplicated(on).any():
            raise ValueError(f"more than one {name} row per {on}")
    d = c.merge(u, on=on, how="outer", suffixes=("_corrected", "_uncorrected"), indicator=True)
    if (d["_merge"] != "both").any():
        bad = d.loc[d["_merge"] != "both", on].to_dict("records")
        raise ValueError(f"rows without both surfaces: {bad[:3]}")
    d = d.drop(columns="_merge")
    d["corrected_outside_uncorrected"] = _outside(d["estimate_km2_corrected"], d["lo_km2_uncorrected"],
                                                  d["hi_km2_uncorrected"])
    d["uncorrected_outside_corrected"] = _outside(d["estimate_km2_uncorrected"], d["lo_km2_corrected"],
                                                  d["hi_km2_corrected"])
    d["decision_relevant"] = d["corrected_outside_uncorrected"] | d["uncorrected_outside_corrected"]
    d["difference_km2"] = d["estimate_km2_corrected"] - d["estimate_km2_uncorrected"]
    for s in ("corrected", "uncorrected"):
        d[f"interval_infinite_{s}"] = ~np.isfinite(d[f"lo_km2_{s}"].astype(float)) | ~np.isfinite(
            d[f"hi_km2_{s}"].astype(float))
    return d.sort_values(on).reset_index(drop=True)


def summary(dec: pd.DataFrame, near_term=design.NEAR_TERM_RANGE_M) -> pd.DataFrame:
    """Per product, region and functional: the near-term scenarios evaluated, how many
    are decision-relevant, and the verdict of the module docstring."""
    lo, hi = near_term
    nt = dec[(dec["scenario_m"] >= lo - 1e-9) & (dec["scenario_m"] <= hi + 1e-9)]
    g = nt.groupby(KEYS, sort=True)
    s = pd.DataFrame({"n_near_term": g.size(), "n_relevant": g["decision_relevant"].sum().astype(int),
                      "scenarios_relevant": g.apply(lambda x: ",".join(f"{v:g}" for v in
                                                                        x.loc[x["decision_relevant"], "scenario_m"]),
                                                    include_groups=False)}).reset_index()
    s["verdict"] = np.where(s["n_relevant"] == s["n_near_term"], "supported",
                            np.where(s["n_relevant"] > 0, "partial", "not supported"))
    allk = dec[KEYS].drop_duplicates()
    s = allk.merge(s, on=KEYS, how="left")
    s["n_near_term"] = s["n_near_term"].fillna(0).astype(int)
    s["n_relevant"] = s["n_relevant"].fillna(0).astype(int)
    s.loc[s["n_near_term"] == 0, "verdict"] = pd.NA
    return s.sort_values(KEYS).reset_index(drop=True)
