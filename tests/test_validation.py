"""A6 validation (caspian.validation) on synthetic data: datum chain, ICESat-2 and GEDI
steps, dryness, raster sampling, the training-only fits, the block bootstrap, conformal
calibration and coverage, and the H2 decision rule (PREREG sections 2 and 5-8)."""

import numpy as np
import pandas as pd
import pytest

from caspian import blocking, design, validation
from caspian.config import CRS_EQUAL_AREA

SIZE = 25_000
SECS = ("west", "volga")


def _blocks(n=10, size=SIZE):
    b = blocking.block_grid((0, 0, n * size, n * size), size)
    b["sector"] = np.where(b["i"] < n // 2, "west", "volga")
    b["split"] = blocking.assign_splits(b)
    return b


def _points(n_pts=6000, seed=0, c_true=-1.25, n=10):
    """DEM error with σ = 0.03 (1 + d)^0.4 (log-linear in log(1 + d)); ICESat-2 on EGM2008
    is the truth minus c_true; etopo_2022 is on EGM2008, gebco_2026 0.7 m above the DEM
    reference; both noisier than the DEM."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, n * SIZE, n_pts), rng.uniform(0, n * SIZE, n_pts)
    truth = -27.0 + 2e-5 * x - 1e-5 * y
    d = rng.uniform(0, 3000, n_pts)
    sigma = 0.03 * np.exp(0.4 * np.log1p(d))
    pts = pd.DataFrame({
        "x": x, "y": y, "h_orth": truth - c_true + rng.normal(0, 0.01, n_pts),
        "dem_h": truth + rng.normal(0, sigma), "interp_dist": d,
        "contour_density": rng.uniform(0, 50, n_pts), "slope": rng.uniform(0, 0.01, n_pts),
        "prod:etopo_2022": truth - c_true + rng.normal(0, 0.8, n_pts),
        "prod:gebco_2026": truth + 0.7 + rng.normal(0, 0.8, n_pts)})
    return validation.attach_blocks(pts, _blocks(n), SIZE)


PRODUCTS = {"etopo_2022": "prod:etopo_2022", "gebco_2026": "prod:gebco_2026"}
FITTED_COLS = ["h_orth", "dem_h", "interp_dist", "contour_density", "slope", *PRODUCTS.values()]


COVARIATE_RANGE = {"interp_dist": 20_000.0, "contour_density": 200.0, "slope": 0.05}


def _perturb(pts, splits, cols=FITTED_COLS, seed=99):
    """Replace the values of the given splits: heights by wild values, scale covariates by
    other values in a plausible range; some of each NaN."""
    rng = np.random.default_rng(seed)
    p = pts.copy()
    m = p["split"].isin(splits).to_numpy()
    for c in cols:
        v = (rng.uniform(0, COVARIATE_RANGE[c], m.sum()) if c in COVARIATE_RANGE
             else rng.normal(1000, 500, m.sum()))
        v[::7] = np.nan
        p.loc[m, c] = v
    return p


# --- datum chain ---------------------------------------------------------------------------

def test_datum_chain_ellipsoid_to_dem_reference():
    h_ell, n_geoid, c = np.array([-42.6, -40.0]), np.array([-15.5, -15.6]), -0.8
    h = validation.orthometric(h_ell, n_geoid)
    assert np.allclose(h, [-27.1, -24.4])
    pts = pd.DataFrame({"h_orth": h})
    y = validation.apply_datum_constant(pts, {"constant_m": c})["y_ref"].to_numpy()
    assert np.allclose(y, h_ell - n_geoid + c)


def test_datum_constant_is_median_of_training_points_only():
    pts = _points()
    rec = validation.fit_datum_constant(pts, n_boot=200)
    tr = pts[pts["split"] == "train"]
    assert rec["constant_m"] == pytest.approx(np.median(tr["dem_h"] - tr["h_orth"]))
    assert rec["constant_m"] == pytest.approx(-1.25, abs=0.02)
    assert rec["split"] == "train" and rec["n_points"] == len(tr) and rec["sigma_boot_m"] > 0
    assert rec["n_blocks"] == tr["block"].nunique()
    # calibration and test points cannot move it, whatever their values
    assert validation.fit_datum_constant(_perturb(pts, ["calib", "test"]), n_boot=200) == rec
    # training points do (the check above is not vacuous)
    assert validation.fit_datum_constant(_perturb(pts, ["train"]), n_boot=200)["constant_m"] != rec["constant_m"]


def test_datum_constant_defaults_are_preregistered():
    import inspect
    sig = inspect.signature(validation.fit_datum_constant).parameters
    assert sig["n_boot"].default == design.BOOTSTRAP_N == 2000 and sig["seed"].default == design.SEED
    sig = inspect.signature(validation.paired_block_bootstrap).parameters
    assert sig["n_boot"].default == design.BOOTSTRAP_N and sig["seed"].default == design.SEED


def test_documented_egm2008_products_match_the_product_table():
    from caspian import products
    for name in validation.DOCUMENTED_EGM2008:
        assert name in products.PRODUCTS and "EGM2008" in products.PRODUCTS[name]["reference"]


def test_product_offsets_training_only_and_variants():
    pts = validation.apply_datum_constant(_points(), {"constant_m": -1.25})
    off = validation.fit_product_offsets(pts, PRODUCTS, n_boot=100)
    assert off["gebco_2026"]["offset_m"] == pytest.approx(-0.7, abs=0.06)
    assert off["etopo_2022"]["offset_m"] == pytest.approx(-1.25, abs=0.06)
    p2 = _perturb(pts, ["calib", "test"], cols=list(PRODUCTS.values()) + ["y_ref"])
    assert validation.fit_product_offsets(p2, PRODUCTS, n_boot=100) == off
    const = {"constant_m": -1.25}
    v = validation.product_variants(pts, PRODUCTS, off, const)
    # documented EGM2008 (ETOPO 2022): total = z + c; undocumented (GEBCO): total = matched
    assert np.allclose(v["etopo_2022|total"], pts["prod:etopo_2022"] - 1.25)
    assert np.allclose(v["etopo_2022|matched"], pts["prod:etopo_2022"] + off["etopo_2022"]["offset_m"])
    assert np.array_equal(v["gebco_2026|total"], v["gebco_2026|matched"])


# --- splits and blocks ---------------------------------------------------------------------

def test_attach_blocks_and_split_rows():
    b = _blocks(4)
    pts = pd.DataFrame({"x": [10.0, 30_000.0, 1e7], "y": [10.0, 10.0, 10.0]})
    out = validation.attach_blocks(pts, b, SIZE)
    assert out["block"].tolist()[:2] == ["0_0", "1_0"]
    assert out["split"].iloc[0] == b.loc[(b.i == 0) & (b.j == 0), "split"].item()
    assert pd.isna(out["split"].iloc[2]) and pd.isna(out["sector"].iloc[2])
    for s in validation.SPLITS:   # a point outside every block enters no split
        assert 2 not in validation.split_rows(out, s).index
    with pytest.raises(ValueError):
        validation.attach_blocks(pts, b, 50_000)        # table is not on 50 km blocks
    bad = out.assign(split=["train", "holdout", np.nan])
    with pytest.raises(ValueError):
        validation.split_rows(bad, "train")
    with pytest.raises(ValueError):
        validation.split_rows(out, "validation")


# --- ICESat-2, GEDI and dryness -------------------------------------------------------------

def test_granule_key_shared_by_atl06_and_atl08():
    assert (validation.granule_key("ATL08_20181015000939_02500106_007_01.h5")
            == validation.granule_key("ATL06_20181015000939_02500106_007_02.h5.csv.gz")
            == "20181015000939_02500106")


def test_strong_beams():
    s = validation.strong_beam(["gt1l", "gt1r", "gt2l", "gt2r", "gt3l"], [0, 0, 1, 1, 2])
    assert s.tolist() == [True, False, False, True, False]


def _t(seconds):
    return np.datetime64("2020-01-01T00:00:00", "ns") + (np.asarray(seconds) * 1e9).astype("timedelta64[ns]")


def test_geoid_along_track_interpolation():
    rt = np.arange(0, 1.0, 0.003)                    # ATL06 segments, about 20 m apart
    rg = -15.5 + 0.2 * rt                            # geoid linear in time along the track
    rt = np.r_[rt, 2.0, 2.5]                         # then a 0.5 s gap
    rg = np.r_[rg, -15.0, -14.9]
    ref_b = np.full(rt.size, "gt1l")
    t = np.array([0.0015, 0.6, rt[10], 2.2, 3.0, 0.5])
    b = np.array(["gt1l", "gt1l", "gt1l", "gt1l", "gt1l", "gt2l"])
    g = validation.geoid_along_track(_t(t), b, _t(rt), ref_b, rg)
    assert g[0] == pytest.approx(-15.5 + 0.2 * 0.0015, abs=1e-6)
    assert g[1] == pytest.approx(-15.5 + 0.2 * 0.6, abs=1e-6)
    assert g[2] == pytest.approx(rg[10])                         # exact time
    assert np.isnan(g[3]) and np.isnan(g[4]) and np.isnan(g[5])  # gap, past the end, other beam


def _atl06_frame():
    """Six rows; row k (k >= 1) fails exactly one step, in step order."""
    return pd.DataFrame({
        "lat": 46.0, "lon": 50.0,
        "h": [-40.0, np.nan, -40.0, -40.0, -40.0, -40.0],
        "h_sigma": 0.05, "quality": [0, 0, 0, 1, 0, 0],
        "geoid_h": [-15.0, -15.0, -15.0, -15.0, np.nan, -15.0],
        "dem_h": [-41.0, -41.0, -41.0, -41.0, -41.0, 20.0],
        "time": _t(np.arange(6.0)), "beam": ["gt1l", "gt1l", "gt1r", "gt1l", "gt1l", "gt1l"], "sc_orient": 0})


def test_icesat2_steps_cascade_atl06():
    d = _atl06_frame()
    kept, casc = validation.prepare_icesat2(d, "ATL06")
    assert casc["step"].tolist() == ["input", "height", "strong_beam", "quality_summary_0", "geoid", "gross_error"]
    assert casc["n_remaining"].tolist() == [6, 5, 4, 3, 2, 1]
    assert casc["n_fail_alone"].tolist()[1:] == [1, 1, 1, 1, 2]   # a NaN height fails the gross-error test too
    assert len(kept) == 1 and kept["h_orth"].item() == pytest.approx(-25.0) and kept["source"].item() == "ATL06"
    assert "dem_h" not in kept and kept["is2_dem_h"].item() == -41.0     # dem_h is kept for the waterline DEM


def test_icesat2_steps_atl08_flags_and_geoid():
    n = 5
    d = pd.DataFrame({"lat": 46.0, "lon": 50.0, "h": -40.0, "h_sigma": 0.3, "watermask": [0, 1, 0, 0, 0],
                      "landcover": 80, "cloud": [0, 0, 1, 0, 0], "msw": [0, 0, 0, 2, 0], "night": 1,
                      "dem_h": -41.0, "time": _t(np.arange(n) * 0.01), "beam": "gt2r", "sc_orient": 1})
    ref = pd.DataFrame({"time": _t(np.arange(-0.03, 0.1, 0.003)), "beam": "gt2r", "geoid_h": -15.2})
    kept, casc = validation.prepare_icesat2(d, "ATL08", ref)
    steps = casc.set_index("step")["n_fail_alone"]
    assert "land_watermask" not in steps.index         # D-069: the static water flag is not a filter
    assert steps[["cloud_free", "no_multiple_scattering"]].tolist() == [1, 1]
    assert len(kept) == 3 and np.allclose(kept["h_orth"], -40.0 + 15.2)   # the watermask = 1 row is kept
    kept0, casc0 = validation.prepare_icesat2(d, "ATL08", None)    # no ATL06 of the same orbit
    assert kept0.empty and casc0.set_index("step").loc["geoid", "n_fail_alone"] == n


def test_read_icesat2_csv_format(tmp_path):
    p = tmp_path / "ATL06_x.h5.csv.gz"
    pd.DataFrame({"lat": [47.05], "lon": [52.14], "h": [-40.7], "h_sigma": [0.02], "quality": [0],
                  "geoid_h": [-15.53], "dem_h": [-40.70], "n_photons": [40], "snr_sig": [3.4e38],
                  "time": ["2018-10-15 00:12:54.744847780"], "beam": ["gt1l"], "sc_orient": [1],
                  "granule": ["ATL06_x.h5"]}).to_csv(p, index=False)
    d = validation.read_icesat2_csv(p, "ATL06")
    assert list(d.columns) == validation.ICESAT2_COLUMNS["ATL06"]
    assert d["time"].dtype == "datetime64[ns]" and d["time"].iloc[0] == pd.Timestamp("2018-10-15 00:12:54.744847780")


def test_gedi_steps():
    d = pd.DataFrame({"elev_lowestmode": [-40.0, -40.0, -40.0, -40.0, -40.0, -40.0],
                      "digital_elevation_model": [-41.0, -41.0, -41.0, -41.0, -41.0, 30.0],
                      "quality_flag": [1, 0, 1, 1, 1, 1], "degrade_flag": [0, 0, 1, 0, 0, 0],
                      "beam": ["BEAM0101", "BEAM0101", "BEAM0101", "BEAM0000", "BEAM0001", "BEAM0110"],
                      "time": pd.to_datetime(["2019-06-01", "2019-06-01", "2019-06-01", "2019-06-01",
                                              "2020-03-01", "2019-06-01"])})
    keep, casc = validation.cascade(validation.gedi_steps(d), len(d))
    assert keep.tolist() == [True, False, False, False, True, False]


def test_dry_at_acquisition_every_pixel_strictly_after_lag():
    day = np.datetime64("2020-01-01", "ns")
    e0 = float(day.astype("int64") / 86_400e9)                # 2020-01-01 in days since 1970
    t = np.array([day + np.timedelta64(31, "D"), day + np.timedelta64(30, "D"), day + np.timedelta64(31, "D"),
                  day + np.timedelta64(400, "D")])
    em = np.array([[e0, e0 - 5, e0], [e0, e0, e0], [e0, np.nan, e0], [e0, e0 + 380, e0]])
    assert validation.dry_at_acquisition(t, em).tolist() == [True, False, False, False]
    assert design.EMERGENCE_LAG_DAYS == 30


# --- rasters ---------------------------------------------------------------------------------

def _write_tile(path, x0, ytop, nx, ny, res=30.0):
    import rasterio
    from rasterio.transform import Affine

    xs = x0 + (np.arange(nx) + 0.5) * res
    ys = ytop - (np.arange(ny) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=2, dtype="float32",
                       crs=CRS_EQUAL_AREA, transform=Affine(res, 0, x0, 0, -res, ytop), nodata=np.nan) as dst:
        dst.write((0.01 * gx + 0.02 * gy).astype("float32"), 1)
        dst.write(np.full((ny, nx), 7.0, "float32"), 2)


def test_tileset_sampling_window_slope_and_gaps(tmp_path):
    _write_tile(tmp_path / "dem_0_0.tif", 0, 300, 10, 10)
    _write_tile(tmp_path / "dem_1_0.tif", 300, 300, 10, 10)
    _write_tile(tmp_path / "dem_0_1.tif", 0, 600, 9, 9)       # stops 30 m short of the tile edge
    ts = validation.TileSet(tmp_path, tile_m=300, res=30.0)
    v = ts.sample([15.0, 305.0, -5.0, 280.0, 15.0], [285.0, 15.0, 100.0, 450.0, 590.0])
    assert v[0] == pytest.approx(0.01 * 15 + 0.02 * 285, abs=1e-5)
    assert v[1] == pytest.approx(0.01 * 315 + 0.02 * 15, abs=1e-5)
    assert np.isnan(v[2]) and np.isnan(v[3])                  # no tile; strip the tile does not reach
    assert v[4] == pytest.approx(0.01 * 15 + 0.02 * 585, abs=1e-5)
    assert ts.sample([15.0], [285.0], band=2)[0] == 7.0
    w = ts.sample_window(np.array([45.0]), np.array([255.0]), radius_px=1)
    assert w.shape == (1, 3, 3)
    assert w[0, 0, 1] == pytest.approx(0.01 * 45 + 0.02 * 285, abs=1e-5)    # north
    assert w[0, 1, 2] == pytest.approx(0.01 * 75 + 0.02 * 255, abs=1e-5)    # east
    s = ts.slope(np.array([295.0, 150.0, 5.0]), np.array([150.0, 150.0, 150.0]))
    assert s[0] == pytest.approx(np.hypot(0.01, 0.02), abs=1e-5)             # across the tile edge
    assert s[1] == pytest.approx(np.hypot(0.01, 0.02), abs=1e-5)
    assert np.isnan(s[2])                                                    # west neighbour missing


def test_contour_density():
    cx, cy = np.meshgrid(np.arange(-1000, 1001, 100.0), np.arange(-1000, 1001, 100.0))
    d = validation.contour_density([0.0, np.nan, 5000.0], [0.0, 0.0, 5000.0], cx.ravel(), cy.ravel(), radius_m=150)
    assert d[0] == pytest.approx(9 / (np.pi * 0.15 ** 2))     # 3 x 3 points within 150 m
    assert np.isnan(d[1]) and d[2] == 0


# --- statistics and bootstrap ----------------------------------------------------------------

def test_error_metrics_known_values():
    m = validation.error_metrics([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    assert m["n"] == 5 and m["bias"] == 0 and m["median"] == 0
    assert m["rmse"] == pytest.approx(np.sqrt(2)) and m["nmad"] == pytest.approx(1.4826)
    assert m["p05"] == pytest.approx(-1.8) and m["p95"] == pytest.approx(1.8)
    e = np.random.default_rng(3).normal(0, 0.5, 200_000)
    assert validation.nmad(e) == pytest.approx(0.5, rel=0.01)
    assert np.isnan(validation.error_metrics([])["nmad"])


def test_accuracy_tables_paired_set_pooled_and_per_block():
    pts = pd.DataFrame({"split": ["test"] * 6 + ["train"], "block": ["a", "a", "a", "b", "b", "b", "c"],
                        "sector": ["west"] * 3 + ["volga"] * 3 + ["west"],
                        "y_ref": [0.0] * 7, "dem_h": [1.0, -1.0, 0.0, 2.0, 2.0, 2.0, 9.0],
                        "p": [0.0, 0.0, np.nan, 1.0, 1.0, 1.0, 9.0]})
    s, pb = validation.accuracy_tables(pts, {"dem": "dem_h", "p": "p"}, sectors=SECS)
    s = s.set_index(["scope", "model"])
    assert s.loc[("overall", "dem"), "n"] == 5            # the point without a product value is out for both
    assert s.loc[("overall", "dem"), "n_test_points"] == 6 and s.loc[("overall", "dem"), "n_blocks"] == 2
    assert s.loc[("overall", "dem"), "bias"] == pytest.approx(6 / 5)
    assert s.loc[("west", "dem"), "rmse"] == pytest.approx(1.0) and s.loc[("volga", "dem"), "nmad"] == 0
    assert s.loc[("overall", "dem"), "block_median_bias"] == pytest.approx(1.0)   # blocks: 0 and 2
    pb = pb.set_index(["model", "block"])
    assert pb.loc[("dem", "b"), "bias"] == 2 and pb.loc[("dem", "b"), "sector"] == "volga"
    assert "c" not in pb.index.get_level_values("block")                          # training block


def test_block_bootstrap_draws_whole_blocks():
    lab = np.array(["b"] * 2 + ["a"] * 1 + ["c"] * 3)
    code = {"a": 0, "b": 1, "c": 2}
    sizes = np.array([1, 2, 3])
    ints = np.array([code[k] for k in lab])
    boot = validation.block_bootstrap(lab, lambda ix: np.bincount(ints[ix], minlength=3) / sizes, n_boot=300, seed=1)
    assert boot.shape == (300, 3)
    assert np.allclose(boot, np.round(boot))              # each block enters whole, 0..3 times
    assert np.allclose(boot.sum(axis=1), 3)               # three blocks per replicate
    perm = np.random.default_rng(2).permutation(lab.size)
    first = validation.block_bootstrap(lab, lambda ix: np.bincount(ints[ix], minlength=3), n_boot=50, seed=1)
    again = validation.block_bootstrap(lab[perm], lambda ix: np.bincount(ints[perm][ix], minlength=3), n_boot=50, seed=1)
    assert np.array_equal(again, first)                   # row order does not matter


def test_paired_bootstrap_identical_and_better():
    pts = validation.apply_datum_constant(_points(), {"constant_m": -1.25})
    pts["same"] = pts["dem_h"]
    pts["worse"] = pts["y_ref"] + 3 * (pts["dem_h"] - pts["y_ref"]) + 0.5
    b = validation.paired_block_bootstrap(pts, "dem_h", {"same": "same", "worse": "worse"}, n_boot=200, sectors=SECS)
    same = b[b["product"] == "same"]
    assert np.allclose(same[["diff", "ci_low", "ci_high"]].to_numpy(), 0)
    worse = b[(b["product"] == "worse") & (b["scope"] == "overall")].set_index("metric")
    assert (worse["ci_high"] < 0).all() and (worse["diff"] < 0).all()
    te = pts[pts["split"] == "test"]
    assert worse.loc["nmad", "n_points"] == len(te) and worse.loc["nmad", "n_blocks"] == te["block"].nunique()
    # test-block statistics only: training and calibration rows do not enter
    p2 = _perturb(pts, ["train", "calib"], cols=["dem_h", "same", "worse", "y_ref"])
    b2 = validation.paired_block_bootstrap(p2, "dem_h", {"same": "same", "worse": "worse"}, n_boot=200, sectors=SECS)
    pd.testing.assert_frame_equal(b, b2)


# --- scale model, conformal calibration and coverage -------------------------------------------

def test_scale_model_training_only_and_recovers_slope_on_log_distance():
    pts = validation.apply_datum_constant(_points(), {"constant_m": -1.25})
    m = validation.fit_scale_model(pts)
    assert m["terms"] == ["intercept", "log1p_interp_dist", "contour_density", "slope"]
    assert m["beta"][1] == pytest.approx(0.4, abs=0.05)
    assert abs(m["beta"][2]) < 0.01
    p2 = _perturb(pts, ["calib", "test"], cols=FITTED_COLS + ["y_ref"])
    assert validation.fit_scale_model(p2) == m


def test_calibration_uses_calibration_blocks_only_and_records_every_sector():
    pts = validation.apply_datum_constant(_points(), {"constant_m": -1.25})
    m = validation.fit_scale_model(pts)
    sectors = ("west", "volga", "ural")
    cal = validation.calibrate(pts, m, sectors=sectors)
    assert cal["split"] == "calib" and cal["level"] == design.CONFORMAL_LEVEL
    assert cal["alpha"] == pytest.approx(0.10) and cal["min_n"] == design.MONDRIAN_MIN_N
    assert cal["groups"]["ural"] == {"q": cal["pooled"]["q"], "n": 0, "used_pooled": True}
    assert not cal["groups"]["west"]["used_pooled"]
    n_cal = (pts["split"] == "calib").sum()
    assert cal["pooled"]["n"] == n_cal and sum(cal["n_excluded"].values()) == 0
    assert validation.calibrate(_perturb(pts, ["train", "test"], cols=["y_ref", "dem_h"]), m, sectors=sectors) == cal
    # a calibration point without a prediction is excluded and counted
    p3 = pts.copy()
    k = p3.index[(p3["split"] == "calib") & (p3["sector"] == "west")][:4]
    p3.loc[k, "dem_h"] = np.nan
    cal3 = validation.calibrate(p3, m, sectors=sectors)
    assert cal3["n_excluded"]["west"] == 4 and cal3["pooled"]["n"] == n_cal - 4


def test_coverage_near_nominal_and_counts_missing_predictions():
    pts = validation.apply_datum_constant(_points(n_pts=8000, seed=4), {"constant_m": -1.25})
    k = pts.index[pts["split"] == "test"][:10]
    pts.loc[k, "dem_h"] = np.nan                   # outside the TIN hull
    m = validation.fit_scale_model(pts)
    cal = validation.calibrate(pts, m, sectors=SECS)
    cov = validation.coverage_on_test(pts, m, cal, n_boot=200, sectors=SECS).set_index("scope")
    n_test = (pts["split"] == "test").sum()
    assert cov.loc["overall", "n"] == n_test - 10 and cov.loc["overall", "n_excluded"] == 10
    assert cov.loc["overall", "n_no_prediction"] == 10
    assert cov.loc["overall", "frac_no_prediction"] == pytest.approx(10 / n_test)
    assert abs(cov.loc["overall", "coverage"] - 0.90) < 0.04
    assert cov.loc["overall", "boot_low"] <= cov.loc["overall", "coverage"] <= cov.loc["overall", "boot_high"]
    assert cov.loc["west", "q"] == cal["groups"]["west"]["q"]
    assert cov.loc[["west", "volga"], "n"].sum() == cov.loc["overall", "n"]


def test_block_size_check_uses_training_blocks_only():
    rng = np.random.default_rng(5)
    n = 900
    pts = pd.DataFrame({"x": rng.uniform(0, 250_000, n), "y": rng.uniform(0, 250_000, n)})
    pts["h_orth"] = -26.0 + rng.normal(0, 0.02, n)
    pts["dem_h"] = pts["h_orth"] - 1.0 + 0.3 * np.sin(pts["x"] / 8_000) + rng.normal(0, 0.05, n)
    pts = validation.attach_blocks(pts, _blocks(), SIZE)
    rec, vario = validation.block_size_check(pts)
    assert rec["split"] == "train" and rec["n_points"] == (pts["split"] == "train").sum()
    assert rec["final_block_size_m"] == blocking.block_size_for_range(rec["fit"]["practical_range"])
    assert len(vario) > 0
    rec2, vario2 = validation.block_size_check(_perturb(pts, ["calib", "test"], cols=["h_orth", "dem_h"]))
    assert rec2 == rec
    pd.testing.assert_frame_equal(vario, vario2)


def test_block_size_check_flags_a_range_beyond_the_fitted_lags():
    """A residual trend gives a semivariogram that keeps rising over every fitted lag; the
    exponential fit then extrapolates a range far beyond the data, which is flagged rather
    than silently turned into a block size."""
    rng = np.random.default_rng(6)
    n = 900
    pts = pd.DataFrame({"x": rng.uniform(0, 250_000, n), "y": rng.uniform(0, 250_000, n),
                        "h_orth": rng.normal(-26.0, 0.02, n)})
    pts["dem_h"] = pts["h_orth"] - 1.0 + 2e-5 * pts["x"] + rng.normal(0, 0.05, n)
    rec, vario = validation.block_size_check(validation.attach_blocks(pts, _blocks(), SIZE), max_lag=30_000)
    assert not rec["range_within_max_lag"]
    assert rec["fit"]["practical_range"] > vario["lag"].max()
    assert 0 <= rec["partial_sill_share"] <= 1


# --- end to end ---------------------------------------------------------------------------------

def test_run_end_to_end_and_test_rows_cannot_change_any_fit():
    pts = _points()
    r = validation.run(pts, PRODUCTS, n_boot=100, sectors=SECS)
    assert r["constant"]["constant_m"] == pytest.approx(-1.25, abs=0.02)
    acc = r["accuracy"].set_index(["scope", "model"])
    assert set(acc.index.get_level_values("model")) == {"dem", "etopo_2022|total", "etopo_2022|matched",
                                                        "gebco_2026|total", "gebco_2026|matched"}
    assert acc.loc[("overall", "dem"), "nmad"] < acc.loc[("overall", "gebco_2026|matched"), "nmad"]
    assert (acc.xs("overall")["n"] == acc.loc[("overall", "dem"), "n"]).all()     # one paired set
    assert set(r["paired_bootstrap"]["variant"]) == {"total", "matched"}
    assert not r["accuracy_blocks"].empty and set(r["accuracy_blocks"]["block"]) <= set(pts.loc[pts["split"] == "test", "block"])
    fits = ("constant", "product_offsets", "scale_model", "calibration")
    r_test = validation.run(_perturb(pts, ["test"]), PRODUCTS, n_boot=100, sectors=SECS)
    for k in fits:
        assert r_test[k] == r[k], k
    r_cal = validation.run(_perturb(pts, ["calib"]), PRODUCTS, n_boot=100, sectors=SECS)
    for k in ("constant", "product_offsets", "scale_model"):
        assert r_cal[k] == r[k], k
    assert r_cal["calibration"] != r["calibration"]


# --- H2 decision rule (synthetic only) ----------------------------------------------------------

def _comparisons(rows):
    return pd.DataFrame(rows, columns=["scope", "product", "metric", "diff", "ci_low", "ci_high"])


def _all_better(scopes=("overall", "west"), products=("a", "b")):
    return [(s, p, m, -0.2, -0.3, -0.1) for s in scopes for p in products for m in ("nmad", "rmse")]


def test_h2_supported_when_every_product_beaten_on_both_metrics():
    d = validation.h2_decision(_comparisons(_all_better()), ["a", "b"])
    assert d["overall"] == "supported" and d["pivot_brief_16"] is False
    assert d["sectors_supported"] == ["west"] and not d["scopes"]["overall"]["partial"]


def test_h2_not_supported_if_interval_includes_zero_and_partial_reported():
    rows = _all_better()
    rows[2] = ("overall", "b", "nmad", -0.2, -0.3, 0.05)       # better, but the interval includes zero
    d = validation.h2_decision(_comparisons(rows), ["a", "b"])
    assert d["overall"] == "not supported" and d["pivot_brief_16"] is True
    assert d["scopes"]["overall"]["dem_better_than"] == ["a"] and d["scopes"]["overall"]["partial"]
    assert d["scopes"]["west"]["status"] == "supported"        # partial support by sector


def test_h2_dem_worse_is_not_supported():
    rows = _all_better()
    rows[1] = ("overall", "a", "rmse", 0.2, 0.1, 0.3)
    assert validation.h2_decision(_comparisons(rows), ["a", "b"])["overall"] == "not supported"


def test_h2_missing_product_is_not_evaluable_never_silently_dropped():
    rows = [r for r in _all_better() if r[1] != "b"]
    d = validation.h2_decision(_comparisons(rows), ["a", "b"])
    assert d["overall"] == "not evaluable" and d["pivot_brief_16"] is None
    rows = _all_better()
    rows[0] = ("overall", "a", "nmad", np.nan, np.nan, np.nan)
    assert validation.h2_decision(_comparisons(rows), ["a", "b"])["overall"] == "not evaluable"


def test_eigen_geoid_used_for_both_products(tmp_path):
    lon = np.array([49.0, 50.0, 51.0])
    lat = np.array([45.0, 46.0])
    N = np.array([[-20.0, -21.0, -22.0], [-19.0, -20.0, -21.0]])
    p = tmp_path / "g.npz"
    np.savez(p, lon=lon, lat=lat, N_eigen6c4_icgem_m=N)
    g = validation.EigenGeoid(p)
    assert np.allclose(g([49.5], [45.5]), [-20.0])          # mean of the four corners
    d = pd.DataFrame({"lat": 45.5, "lon": 49.5, "h": -40.0, "h_sigma": 0.3, "watermask": 0, "landcover": 80,
                      "cloud": 0, "msw": 0, "night": 1, "dem_h": -41.0, "time": _t(np.zeros(1)), "beam": "gt2r",
                      "sc_orient": 1})
    kept, _ = validation.prepare_icesat2(d, "ATL08", None, geoid=g)    # no ATL06 needed
    assert len(kept) == 1 and np.allclose(kept["h_orth"], -40.0 + 20.0)


def test_block_size_check_sector_detrend_removes_offsets():
    rng = np.random.default_rng(5)
    n = 4000
    x, y = rng.uniform(0, 200_000, n), rng.uniform(0, 100_000, n)
    sector = np.where(x < 100_000, "a", "b")
    r = np.where(sector == "a", 0.0, 2.0) + rng.normal(0, 0.2, n)     # a sector offset, white noise inside
    pts = pd.DataFrame({"x": x, "y": y, "dem_h": r, "h_orth": 0.0, "split": "train", "sector": sector,
                        "block": (x // 25_000).astype(int)})
    raw, _ = validation.block_size_check(pts, max_lag=150_000)
    det, _ = validation.block_size_check(pts, max_lag=150_000, detrend="sector")
    assert raw["structure_test"]["detected"]                 # the offset looks like long-range structure
    assert not det["structure_test"]["detected"] and det["final_block_size_m"] == 25_000.0
