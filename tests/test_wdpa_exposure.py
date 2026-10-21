"""scripts/a8_wdpa_exposure.py on synthetic inputs (no WDPA data is read): the record
selection, the dissolve (overlapping designations counted once), the interval bounds read
back from an emergence table, the exposure cross-check, and the member-name guard."""

import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


w = _load("a8_wdpa_exposure")
a8 = _load("a8_emergence")


def test_select_rules():
    g = pd.DataFrame({
        "SITE_TYPE": ["PA", "PA", "PA", "OECM", "PA", "PA"],
        "STATUS": ["Designated", "Proposed", "Designated", "Designated", "Not Reported", "Inscribed"],
        "DESIG_ENG": ["Zakaznik", "Zakaznik", "Wetland of International Importance (Ramsar Site)", "Other",
                      "Zakaznik", "UNESCO-MAB Biosphere Reserve"]})
    s = w.select(g)
    assert s["wdpa_protected_areas"].tolist() == [True, False, True, False, False, True]
    assert s["wdpa_ramsar"].tolist() == [False, False, True, False, False, False]
    assert s["_oecm"].tolist() == [False, False, False, True, False, False]
    assert s["_mab"].tolist() == [False, False, False, False, False, True]
    assert s["_not_designated"].tolist() == [False, True, False, False, True, False]


def test_select_reads_the_older_field_name():
    g = pd.DataFrame({"PA_DEF": ["1", "0"], "STATUS": ["Designated", "Designated"],
                      "DESIG_ENG": ["Zakaznik", "Zakaznik"]})
    assert w.select(g)["wdpa_protected_areas"].tolist() == [True, False]


def test_dissolve_counts_overlaps_once_and_splits_parts():
    a, b, c = box(0, 0, 2, 2), box(1, 1, 3, 3), box(10, 10, 11, 11)
    parts = w.dissolve([a, b, c, None])
    assert len(parts) == 2
    assert sum(p.area for p in parts) == pytest.approx(4 + 4 - 1 + 1)
    assert w.dissolve([]) == []


def test_dissolved_layer_area_exposure():
    # 10 x 10 grid of 60 m cells; flood level F rises eastward 0.1 m per column from -1.0 m;
    # L26 = 0 (every cell sea), scenario level -0.5 m: columns 5-9 emerge
    F = np.tile((np.arange(10) * 0.1 - 1.0).astype(np.float32) + 0.05, (10, 1))
    tr = [60.0, 0.0, 0.0, 0.0, -60.0, 600.0]
    lab = np.ones((10, 10), np.uint8)
    # two overlapping squares over columns 4-9: cells 5..9 emerge in both
    p1, p2 = box(240, 0, 600, 600), box(300, 0, 600, 600)
    layers = {"undissolved": {"geoms": [p1, p2], "rule": "area", "threshold_m": None},
              "dissolved": {"geoms": w.dissolve([p1, p2]), "rule": "area", "threshold_m": None}}
    x = a8.exposure_table(layers, {"p": {"uncorrected": F, "fused": F}}, {}, lab, ["s"], tr, (10, 10), 0.0, [0.5])
    dom = x[(x["region"] == "domain") & (x["surface"] == "uncorrected")].set_index("layer")
    cell_km2 = 0.06 * 0.06
    assert dom.loc["dissolved", "amount"] == pytest.approx(5 * 10 * cell_km2)
    assert dom.loc["undissolved", "amount"] == pytest.approx(2 * 5 * 10 * cell_km2)   # double counted
    assert dom.loc["dissolved", "n_features"] == 1


def test_bounds_of_reads_the_equivalent_levels():
    em = pd.DataFrame({"product": ["p", "p"], "surface": ["fused", "fused"], "region": ["domain", "domain"],
                       "functional": ["emergence", "sea_area"], "scenario_m": [1.0, 1.0],
                       "level_lo_bound": [-29.3294, -1.0], "lo_beyond_grid": [False, False],
                       "level_hi_bound": [-30.3194, -2.0], "hi_beyond_grid": [True, False]})
    b = w.bounds_of(em)
    assert list(b) == [("p", "fused", "domain", 1.0)]
    assert b[("p", "fused", "domain", 1.0)] == {"level_lo_bound": -29.3294, "lo_beyond_grid": False,
                                                 "level_hi_bound": -30.3194, "hi_beyond_grid": True}


def _rows(amount):
    return pd.DataFrame({"layer": ["osm_wetlands"], "product": ["p"], "surface": ["fused"], "region": ["domain"],
                         "scenario_m": [1.0], "n_features": [3], "n_considered": [2], "amount": [amount],
                         "amount_lo": [np.nan], "amount_hi": [1.0], "n_changed": [1], "n_changed_lo": [np.nan],
                         "n_changed_hi": [1], "level_lo_bound": [-29.0], "level_hi_bound": [-30.0]})


def test_compare_exposure_passes_on_equal_and_stops_on_a_difference():
    assert w.compare_exposure(_rows(0.5), _rows(0.5))["max_abs_diff"] == 0.0
    with pytest.raises(SystemExit):
        w.compare_exposure(_rows(0.5), _rows(0.6))


def test_safe_extract_refuses_parent_steps(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ok/a.shp", b"x")
        zf.writestr("../evil.shp", b"x")
    with zipfile.ZipFile(buf) as zf:
        assert w.safe_extract(zf, "ok/a.shp", tmp_path).name == "a.shp"
        with pytest.raises(ValueError):
            w.safe_extract(zf, "../evil.shp", tmp_path)
