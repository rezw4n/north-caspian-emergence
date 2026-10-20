"""caspian.h3: the pre-registered H3 decision rule (PREREG section 10), on synthetic
tables only."""

import numpy as np
import pandas as pd
import pytest

from caspian import h3


def table(rows):
    """rows: (product, region, scenario, corrected (est, lo, hi), uncorrected (est, lo, hi))."""
    out = []
    for prod, reg, sc, c, u in rows:
        for surf, (e, lo, hi) in (("fused", c), ("uncorrected", u)):
            out.append({"product": prod, "surface": surf, "region": reg, "functional": "emergence",
                        "scenario_m": sc, "estimate_km2": e, "lo_km2": lo, "hi_km2": hi})
    return pd.DataFrame(out)


def test_both_directions_and_infinite_intervals():
    t = table([
        ("p", "domain", 0.5, (100, 95, 105), (130, 90, 170)),     # corrected inside unc; unc outside corrected
        ("p", "domain", 1.0, (100, 60, 140), (130, 125, 135)),    # corrected outside unc
        ("p", "domain", 2.0, (100, 80, 120), (110, 90, 130)),     # neither
        ("p", "domain", 3.0, (100, -np.inf, np.inf), (300, 290, 310)),   # infinite corrected interval
        ("p", "domain", 5.0, (100, 99, 101), (100, 99, 101)),
    ])
    d = h3.decision(t).set_index("scenario_m")
    assert not d.loc[0.5, "corrected_outside_uncorrected"] and d.loc[0.5, "uncorrected_outside_corrected"]
    assert d.loc[1.0, "corrected_outside_uncorrected"] and not d.loc[1.0, "uncorrected_outside_corrected"]
    assert d["decision_relevant"].tolist() == [True, True, False, True, False]
    # with an infinite corrected interval only the other direction can count
    assert d.loc[3.0, "interval_infinite_corrected"] and d.loc[3.0, "corrected_outside_uncorrected"]
    assert not d.loc[3.0, "uncorrected_outside_corrected"]
    assert d.loc[1.0, "difference_km2"] == -30


def test_boundary_values_are_inside():
    d = h3.decision(table([("p", "domain", 1.0, (100, 90, 110), (110, 100, 120))]))
    assert not d.loc[0, "decision_relevant"]


def test_nan_is_never_relevant():
    d = h3.decision(table([("p", "domain", 1.0, (np.nan, 90, 110), (110, np.nan, 120))]))
    assert not d.loc[0, "decision_relevant"]


def test_missing_or_duplicated_surface_raises():
    t = table([("p", "domain", 1.0, (100, 90, 110), (130, 125, 135))])
    with pytest.raises(ValueError, match="both surfaces"):
        h3.decision(t[t["surface"] == "fused"].assign(scenario_m=1.0).pipe(
            lambda x: pd.concat([x, t[t["surface"] == "uncorrected"].assign(scenario_m=2.0)])))
    with pytest.raises(ValueError, match="more than one"):
        h3.decision(pd.concat([t, t[t["surface"] == "fused"]]))
    with pytest.raises(ValueError, match="lacks columns"):
        h3.decision(t.drop(columns="lo_km2"))


def test_summary_verdicts_over_the_near_term_scenarios():
    rows = []
    for prod, rel in (("all", (True, True, True)), ("some", (True, False, False)), ("none", (False,) * 3)):
        for sc, r in zip((0.5, 1.0, 2.0), rel):
            rows.append((prod, "domain", sc, (100, 95, 105), (130, 125, 135) if r else (101, 90, 110)))
        rows.append((prod, "domain", 5.0, (100, 95, 105), (300, 290, 310)))   # beyond the near term: ignored
    rows.append(("far", "domain", 5.0, (100, 95, 105), (300, 290, 310)))
    s = h3.summary(h3.decision(table(rows))).set_index("product")
    assert s.loc["all", "verdict"] == "supported" and s.loc["all", "n_near_term"] == 3
    assert s.loc["some", "verdict"] == "partial" and s.loc["some", "scenarios_relevant"] == "0.5"
    assert s.loc["none", "verdict"] == "not supported" and s.loc["none", "n_relevant"] == 0
    assert pd.isna(s.loc["far", "verdict"]) and s.loc["far", "n_near_term"] == 0
