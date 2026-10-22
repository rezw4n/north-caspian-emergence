"""Display helpers of the ERL figures: the reductions of
scripts/f1_erl_display.py (water share per display block, observed and predicted states,
grid windows, the Great Salt Lake unit rule), the number formatting of figures/_erl.py
(the text's macros, in Unicode), the report-card grades and the contour smoothing of the
then-and-now figure. Synthetic inputs only."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def f1():
    return _load(ROOT / "scripts" / "f1_erl_display.py", "f1_erl_display")


@pytest.fixture(scope="module")
def figs():
    sys.path.insert(0, str(ROOT / "figures"))
    try:
        yield {"erl": _load(ROOT / "figures" / "_erl.py", "_erl"),
               "card": _load(ROOT / "figures" / "tab_report_card.py", "tab_report_card"),
               "then": _load(ROOT / "figures" / "erl_fig_then_now.py", "erl_fig_then_now")}
    finally:
        sys.path.remove(str(ROOT / "figures"))


def test_block_fraction_ignores_unknown_and_drops_partial_blocks(f1):
    s = np.full((9, 9), -1, np.int8)
    s[:4, :4] = 1                      # block (0, 0): all water
    s[:4, 4:8] = 0                     # block (0, 1): all land
    s[4:8, :4] = [[1, 0, -1, -1]] * 4  # block (1, 0): half of the known cells water
    out = f1.block_fraction(s, 4)
    assert out.shape == (2, 2)         # the ninth row and column do not fill a block
    assert out[0, 0] == 1.0 and out[0, 1] == 0.0 and out[1, 0] == 0.5
    assert np.isnan(out[1, 1])         # no known cell


def test_obs_state_uses_outside_states_and_the_domain(f1):
    code = np.array([[1, 0, -1, -2, -2, -2]], np.int8)
    outside = np.array([[0, 1, 1, 1, 0, -1]], np.int8)
    dom = np.ones((1, 6), bool)
    assert f1.obs_state(code, outside, dom, 4).tolist() == [[1, 0, -1, 1, 0, -1]]
    dom[0, 0] = False
    assert f1.obs_state(code, outside, dom, 4)[0, 0] == -1


def test_pred_state_is_the_strict_flood_threshold(f1):
    F = np.array([[-29.0, -28.0, -27.99, np.inf, np.nan]], np.float32)
    dom = np.ones((1, 5), bool)
    # F < level floods; +inf (no path from the sea) over a finite height is land; NaN unknown
    assert f1.pred_state(F, -28.0, dom, 4).tolist() == [[1, 0, 0, 0, -1]]
    heights_finite = np.array([[True, True, True, False, False]])
    st = f1.pred_state(F, -28.0, dom, 4, lambda r0, r1, c0, c1: heights_finite[r0:r1, c0:c1])
    assert st.tolist() == [[1, 0, 0, -1, -1]]
    dom[0, 0] = False
    assert f1.pred_state(F, -28.0, dom, 4)[0, 0] == -1


def test_window_covers_the_box(f1):
    tr = [60.0, 0.0, 0.0, 0.0, -60.0, 600.0]           # 10 x 10 cells from (0, 0) to (600, 600)
    rs, cs = f1.window(tr, (120.0, 60.0, 300.0, 330.0))
    assert (cs.start, cs.stop) == (2, 5)
    assert (rs.start, rs.stop) == (4, 9)


def test_gsl_unit_picks_the_most_typical_unit(f1, tmp_path):
    arms = ("north", "south")
    prods = ["p1", "p2", "p3", "p4", "p5"]
    dec = {"products": {p: {"shape": {"arms": {a: {"median_displacement_m": 1000.0} for a in arms}}} for p in prods},
           "control_arms": {a: {"median_displacement_m": 100.0} for a in arms}}
    (tmp_path / "h1_decision.json").write_text(json.dumps(dec), encoding="utf-8")
    for arm in arms:
        for sub, names in (("hindcast", prods), ("hindcast_dep3", ["dep3"])):
            rows = []
            for ep, b, scale in (("E2", 101, 1.0), ("E3", 90, 5.0), ("E3", 95, 1.0)):
                for p in names:
                    base = 100.0 if p == "dep3" else 1000.0
                    rows.append({"product": p, "variant": "shape", "epoch": ep, "bin": b, "sector": arm,
                                 "median_m": base * scale, "observed_level": True})
            d = tmp_path / arm / sub
            d.mkdir(parents=True)
            pd.DataFrame(rows).to_csv(d / "displacement.csv", index=False)
    (ep, b), sel = f1.gsl_unit(tmp_path, list(arms))
    assert (ep, b) == ("E3", 95)       # ties with E2 101 go to the later epoch
    assert sel["n_candidates"] == 3 and sel["score"] == 0.0


def test_num_formats_as_the_text_does(figs):
    E = figs["erl"]
    m = {"a": {"value": -28.05, "unit": "m", "decimals": 2}, "b": {"value": 38263.6, "unit": "km2"},
         "c": {"value": "2021", "unit": "year"}, "d": {"value": 9.960723, "unit": "km", "decimals": 1}}
    assert E.num(m, "a") == "−28.05"
    assert E.num(m, "b", "km²") == "38 264 km²"
    assert E.num(m, "c") == "2021"
    assert E.num(m, "d") == "10.0"
    with pytest.raises(SystemExit):
        E.num(m, "missing")


# The report card's grade shading was removed, and figure 1 now
# draws observed waterlines instead of smoothed emergence dates, so their tests went too.
def test_iso_lines_keeps_long_lines_and_drops_short_pieces(figs):
    T = figs["then"]
    field = np.tile(np.linspace(0.0, 1.0, 20), (10, 1))   # values increase eastward
    field[:, :2] = np.nan
    ext = [0.0, 20 * 600.0, 0.0, 10 * 600.0]
    long_ = T.iso_lines(field, ext, 0.5, min_len_m=1000.0)
    assert len(long_) == 1 and np.ptp(long_[0][:, 1]) > 4000.0     # one north-south line
    assert T.iso_lines(field, ext, 0.5, min_len_m=1e6) == []
