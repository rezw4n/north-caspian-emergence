"""scripts/g1_decide.py on synthetic hindcast tables (no real data): the Amendment 16
rule per arm, the basin aggregation, the positive control and the verdict."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("g1_decide", ROOT / "scripts" / "g1_decide.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7"]
BINS = [80, 81, 82, 83]


def _tables(arm, medians: dict, p95=500.0, control_median=None):
    rows = []
    for p, med in medians.items():
        for var in ("shape", "total"):
            for b in BINS:
                rows.append({"product": p, "variant": var, "epoch": "pooled", "sector": arm, "bin": b,
                             "level_m": -30 + 0.1 * (b + 0.5), "median_m": med, "n": 50, "observed_level": True})
    disp = pd.DataFrame(rows)
    noise = pd.DataFrame([{"kind": "pooled_level", "sector": arm, "bin": b, "n": 40, "p95_m": p95} for b in BINS]
                         + [{"kind": "pooled_all_levels", "sector": arm, "bin": np.nan, "n": 160, "p95_m": p95}])
    t = {"disp": disp, "noise": noise}
    if control_median is not None:
        t["disp_control"] = pd.DataFrame([{"product": "dep3", "variant": v, "epoch": "pooled", "sector": arm, "bin": b,
                                           "level_m": -30 + 0.1 * (b + 0.5), "median_m": control_median, "n": 50,
                                           "observed_level": True} for v in ("shape", "total") for b in BINS])
        t["noise_control"] = noise
    return t


def test_recurs_when_global_products_exceed_and_control_passes():
    med = {p: 3000.0 for p in PRODUCTS}
    med["etopo_2022"] = 100.0                     # one product within the floor
    tables = {"north": _tables("north", med, control_median=60.0), "south": _tables("south", med, control_median=60.0)}
    rec = D.basin_decision(tables, PRODUCTS, "dep3")
    assert rec["verdict"]["verdict"] == "recurs"
    assert rec["verdict"]["n_mispredict"] == 4
    assert rec["positive_control"]["mispredicts_in_basin"] is False
    assert rec["products"]["gebco_2026"]["shape"]["n_sectors_mispredict"] == 2
    assert rec["products"]["etopo_2022"]["shape"]["in_basin"] is False
    assert rec["products"]["gebco_2026"]["shape"]["arms"]["north"]["median_displacement_m"] == 3000.0


def test_failed_control_gives_partial():
    med = {p: 3000.0 for p in PRODUCTS}
    tables = {"south": _tables("south", med, control_median=2000.0)}
    rec = D.basin_decision(tables, PRODUCTS, "dep3")
    assert rec["positive_control"]["mispredicts_in_basin"] is True
    assert rec["verdict"]["verdict"] == "partial"


def test_split_arms_do_not_make_a_basin_majority():
    tables = {"north": _tables("north", {p: 3000.0 for p in PRODUCTS}),
              "south": _tables("south", {p: 100.0 for p in PRODUCTS})}
    rec = D.basin_decision(tables, PRODUCTS, None)
    assert all(rec["products"][p]["shape"]["in_basin"] is False for p in PRODUCTS)   # 1 of 2 is not a majority
    assert rec["verdict"]["verdict"] == "does not recur"
    assert rec["verdict"]["positive_control"] == "not evaluable"


def test_etopo_split_by_the_2013_level():
    med = {p: 100.0 for p in PRODUCTS}
    t = _tables("south", med)
    m = (t["disp"]["product"] == "etopo_2022") & (t["disp"]["bin"] <= 81)
    t["disp"].loc[m, "median_m"] = 4000.0          # large errors at the two lower bins
    split = {"south": 1300 - 30 + 8.2}             # 2013 level between bins 81 and 82
    rec = D.basin_decision({"south": t}, PRODUCTS, None, split, 1300.0)
    e = rec["products"]["etopo_2022"]["shape"]["arms"]["south"]
    assert e["median_displacement_below_2013_m"] == 4000.0
    assert e["median_displacement_above_2013_m"] == 100.0
