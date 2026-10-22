"""ATL24 check of the near-term emergence surfaces (PREREG Amendment 17):
caspian.atl24 on synthetic photons and a synthetic ATL24 file."""

import json

import numpy as np
import pandas as pd
import pytest

import caspian  # noqa: F401
from caspian import atl24, validation


# --- reading -----------------------------------------------------------------------------------

def test_extent_polygon_reads_lat_lon_pairs():
    ext = json.dumps({"polygon": "27.0 52.0 27.5 52.1 27.6 52.4 27.0 52.0", "begin_time": "x"})
    p = atl24.extent_polygon(ext.encode())
    w, s, e, n = p.bounds
    assert (w, s, e, n) == (52.0, 27.0, 52.4, 27.6)          # lon from the second of each pair
    assert atl24.extent_polygon(json.dumps({"polygon": ""})) is None
    assert atl24.extent_polygon(json.dumps({"polygon": "1 2 3"})) is None


def test_box_mask_includes_edges():
    m = atl24.box_mask([44.0, 47.35, 47.36, 45.0], [46.4, 54.6, 50.0, 46.39], (46.4, 43.9677, 54.6, 47.35))
    assert m.tolist() == [True, True, False, False]


def _synthetic_file(path, sc_orient=0):
    import h5py
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as h:
        h.create_dataset("orbit_info/sc_orient", data=np.array([sc_orient], np.int8))
        h.create_dataset("ancillary_data/release", data=np.bytes_(b"02"))
        h.create_dataset("ancillary_data/resource", data=np.bytes_(b"ATL24_x_006_01_001_01.h5"))
        h.create_dataset("metadata/extent", data=np.bytes_(json.dumps({"polygon": "44 50 45 50 45 51 44 50"})))
        for gt in ("gt1l", "gt2r"):
            n = 50
            lat = np.r_[np.linspace(27.0, 28.0, 20), np.linspace(44.0, 46.0, 20), np.linspace(48.0, 49.0, 10)]
            lon = np.r_[np.full(20, 50.0), np.full(10, 50.0), np.full(10, 56.0), np.full(10, 50.0)]
            cls = np.r_[np.zeros(20), np.tile([40, 41], 10), np.zeros(10)].astype(np.int8)
            g = h.create_group(gt)
            g["lat_ph"], g["lon_ph"], g["class_ph"] = lat, lon, cls
            for v in atl24.PHOTON_VARS:
                if v not in g:
                    g[v] = rng.normal(size=n).astype(np.float32)
    return path


def test_read_granule_keeps_class40_inside_the_box(tmp_path):
    import h5py
    p = _synthetic_file(tmp_path / "g.h5")
    with h5py.File(p, "r") as h:
        arr, rec = atl24.read_granule(h, (46.4, 43.9677, 54.6, 47.35))
    # per beam: 20 photons in the latitude band, 10 of them in the lon box; 5 of those are class 40
    assert rec["n_saved"] == 10
    assert rec["n_box_by_class"] == {"40": 10, "41": 10}
    assert set(arr["class_ph"].tolist()) == {40}
    assert (arr["lon_ph"] <= 54.6).all() and (arr["lat_ph"] >= 43.9677).all()
    assert sorted(set(arr["beam"].tolist())) == [atl24.BEAMS.index("gt1l"), atl24.BEAMS.index("gt2r")]
    assert rec["sc_orient"] == 0 and rec["release"] == "02"
    for v in atl24.PHOTON_VARS:
        assert arr[v].size == 10


def test_photon_time_epoch():
    t = atl24.photon_time([0.0, 86400.5])
    assert t[0] == np.datetime64("2018-01-01T00:00:00")
    assert t[1] == np.datetime64("2018-01-02T00:00:00.500")


# --- selection and datum -----------------------------------------------------------------------

def _photons(**over):
    d = pd.DataFrame({"class_ph": [40, 40, 41, 40, 40, 40, 40, 40],
                      "ellipse_h": [-50.0] * 8, "ortho_h": [-30.0] * 8,
                      "surface_h": [-28.0, -28.0, -28.0, -28.0, -28.0, -31.0, -20.0, -28.0],
                      "low_confidence_flag": [0, 1, 0, 0, 0, 0, 0, 0],
                      "confidence": [0.9, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                      "sensor_depth_exceeded": [0, 0, 0, 1, 0, 0, 0, 0],
                      "beam": ["gt1l"] * 4 + ["gt1r"] + ["gt1l"] * 3, "sc_orient": [0] * 8,
                      "geoid": [-20.0] * 7 + [np.nan]})
    for k, v in over.items():
        d[k] = v
    return d


def test_photon_steps_order_and_rules():
    d = _photons()
    steps = atl24.photon_steps(d)
    assert list(steps) == ["class_40", "finite_heights", "confidence", "sensor_depth_ok", "strong_beam",
                           "depth_range", "geoid"]
    keep, casc = validation.cascade(steps, len(d))
    # row 0 passes; 1 low confidence; 2 class 41; 3 depth exceeded; 4 weak beam (sc_orient 0: left strong);
    # 5 depth negative (surface below the photon); 6 depth 10 m > 6.5; 7 no geoid
    assert keep.tolist() == [True] + [False] * 7
    assert casc["n_remaining"].tolist() == [8, 7, 7, 6, 5, 4, 2, 1]
    assert atl24.photon_steps(d, strong_only=False).keys() == steps.keys() - {"strong_beam"}


def test_confidence_needs_flag_and_value():
    d = _photons(low_confidence_flag=[0] * 8, confidence=[0.59] + [0.9] * 7)
    assert not atl24.photon_steps(d)["confidence"][0]


def test_reference_and_surface_heights():
    # ellipse_h - N + c; surface_h + (ellipse_h - ortho_h) - N + c
    assert atl24.reference_height(-50.0, -20.0, 0.7) == pytest.approx(-29.3)
    assert atl24.surface_working(-28.0, -50.0, -30.0, -20.0, 0.7) == pytest.approx(-28.0 - 20.0 + 20.0 + 0.7)
    assert atl24.depth(-28.0, -30.0) == pytest.approx(2.0)


# --- aggregation ---------------------------------------------------------------------------------

def test_aggregate_median_per_cell_and_track_with_minimum():
    tr = [60.0, 0.0, 0.0, 0.0, -60.0, 600.0]
    ph = pd.DataFrame({"x": [5, 10, 20, 30, 100, 5, 6, 7], "y": [590, 580, 570, 560, 590, 590, 590, 590],
                       "track": ["a"] * 5 + ["b"] * 3, "y_ref": [1.0, 2.0, 9.0, 3.0, 5.0, 4.0, 4.0, 4.0],
                       "depth": [1.0] * 8, "surface": [0.0] * 8, "night": [1, 1, 0, 0, 1, 0, 0, 0],
                       "time": np.datetime64("2020-01-01") + np.arange(8).astype("timedelta64[s]")})
    a = atl24.aggregate(ph, tr, min_photons=3)
    # cell (0, 0) holds 4 photons of track a (median 2.5) and 3 of track b; cell (0, 1) holds 1 (dropped)
    assert len(a) == 2
    ra = a[a["track"] == "a"].iloc[0]
    assert (ra["row"], ra["col"], ra["n_photons"], ra["y_ref"]) == (0, 0, 4, 2.5)
    assert ra["night_share"] == pytest.approx(0.5)
    assert a[a["track"] == "b"].iloc[0]["y_ref"] == 4.0
    assert atl24.aggregate(ph.iloc[:0], tr).empty
    x, y = atl24.cell_centres([0], [1], tr)
    assert (x[0], y[0]) == (90.0, 570.0)


def test_water_surface_constant_median_of_overpasses():
    surf = np.array([-29.0, -29.2, -28.0, -28.1, -30.0])
    level = np.array([-28.5, -28.5, -27.7, -27.7, -29.4])
    over = np.array(["g1", "g1", "g2", "g2", "g3"])
    r = atl24.water_surface_constant(surf, level, over, n_boot=50)
    # per overpass: g1 median(0.5, 0.7) = 0.6, g2 median(0.3, 0.4) = 0.35, g3 0.6 -> 0.6
    assert r["constant_m"] == pytest.approx(0.6)
    assert r["n_overpasses"] == 3


# --- comparison and verdict ----------------------------------------------------------------------

def _table(n_blocks=8, per_block=30, err_u=1.0, err_f=0.2, bias_u=0.5, bias_f=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = n_blocks * per_block
    y = rng.normal(-30, 1, n)
    return pd.DataFrame({"y_ref": y, "block": np.repeat([f"b{k}" for k in range(n_blocks)], per_block),
                         "sector": np.where(np.arange(n) < n // 2, "west", "volga"),
                         "u": y + bias_u + rng.normal(0, err_u, n), "f": y + bias_f + rng.normal(0, err_f, n)})


def test_compare_and_verdict_improves():
    acc, dif = atl24.compare(_table(), {"p": ("u", "f")}, ["west", "volga"], n_boot=200)
    o = dif[(dif["scope"] == "overall")]
    assert set(o["metric"]) == {"nmad", "abs_bias", "rmse"}
    assert (o.set_index("metric").loc["nmad", "ci_high"]) < 0
    assert atl24.verdict(o)["verdict"] == "correction improves"
    a = acc[(acc["scope"] == "overall")].set_index("surface")
    assert a.loc["uncorrected", "nmad"] > a.loc["fused", "nmad"]
    assert a.loc["uncorrected", "n"] == 240 and a.loc["fused", "n_blocks"] == 8


def test_verdict_worsens_and_partial_and_not_evaluable():
    _, dif = atl24.compare(_table(err_u=0.2, err_f=1.0, bias_u=0.1, bias_f=0.6), {"p": ("u", "f")}, [], n_boot=200)
    assert atl24.verdict(dif)["verdict"] == "correction worsens"
    # NMAD clearly lower but |bias| higher: no clear difference, with what held stated
    _, dif = atl24.compare(_table(bias_u=0.0, bias_f=0.8), {"p": ("u", "f")}, [], n_boot=200)
    v = atl24.verdict(dif)
    assert v["verdict"] == "no clear difference"
    assert "NMAD lower" in v["partial"] and "|bias| higher" in v["partial"]
    _, dif = atl24.compare(_table(n_blocks=3), {"p": ("u", "f")}, [], n_boot=50)
    assert atl24.verdict(dif)["verdict"] == "not evaluable"


def test_compare_pairs_only_rows_with_both_surfaces():
    t = _table()
    t.loc[:9, "f"] = np.nan
    acc, dif = atl24.compare(t, {"p": ("u", "f")}, [], n_boot=20)
    assert (acc["n"] == len(t) - 10).all()
    assert (dif["n"] == len(t) - 10).all()


# --- driver helpers (scripts/a8_atl24_check.py) ----------------------------------------------------

def _driver():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "a8_atl24_check.py"
    spec = importlib.util.spec_from_file_location("a8_atl24_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sample_reads_cells_across_tiles():
    drv = _driver()
    a = np.arange(100, dtype=float).reshape(10, 10)
    rows, cols = np.array([0, 9, 4, 5, -1, 3]), np.array([0, 9, 5, 4, 2, 10])
    v = drv.sample(a, rows, cols, a.shape, tile=4)
    assert v[:4].tolist() == [0.0, 99.0, 45.0, 54.0]
    assert np.isnan(v[4:]).all()                 # outside the grid


def test_population_cuts_on_the_reference_only():
    drv = _driver()
    ct = pd.DataFrame({"y_ref": [-29.0, -30.0, -33.7, -28.0, -25.0, -29.5],
                       "domain": [True, True, True, True, True, False]})
    near = drv.population(ct, 0.5, -28.6, -25.8, "near_term")      # y_ref + c in [-33.6, -28.6)
    assert np.allclose(near["y_ref"].tolist(), [-29.5, -33.2])  # -28.5 is at or above L2026; -29.0 outside the domain
    band = drv.population(ct, 0.5, -28.6, -25.8, "emerged_band")   # [-28.6, -25.8]
    assert np.allclose(band["y_ref"].tolist(), [-28.5, -27.5])  # -24.5 is above the record high
