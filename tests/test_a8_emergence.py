"""scripts/a8_emergence.py end to end on a synthetic world (no real stage-2 data,
product, DEM or projection is read), followed by caspian.h3 on its tables.

The world is that of tests/test_a7_hindcast.py with more units: a 100 x 140 grid of
60 m cells, seabed rising eastward 0.05 m per cell with a north-south wobble and an
inland pit, open sea west of a north-south stage-2 cover, land east of it, two
sectors (north: rows 0-49, south: rows 50-99). The hindcast driver runs first and its
tables are the H3 driver's inputs. Products: "perfect" (the truth) and "offset" (0.3 m
too high in the north, 0.2 m too low in the south). The DEM is the truth inside the
observed range of the cover.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from caspian import design, h3, hindcast, levels

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a7 = _load("a7_hindcast")
a8 = _load("a8_emergence")

H, W = 100, 140
TR = [60, 0, 0, 0, -60, 6000]
UNITS = {"E1": [18, 19, 20, 21, 22, 23], "E2": [16, 17, 18, 19, 20, 21, 22], "E3": [8, 9, 10, 11, 12, 13, 14]}
SECTORS = ["north", "south"]
L26 = -28.95
SCEN = (0.5, 1.0, 2.0)


def lvl(b):
    return float(levels.bin_center(b, design.BIN_WIDTH_M, design.BIN_ORIGIN_M))


def truth():
    r, c = np.mgrid[:H, :W]
    z = -32 + 0.05 * c + 0.3 * np.sin(r / 7.0 + 0.3)
    z[20:25, 95:100] = -40.0
    return z


def write_stage2(path: Path, z):
    path.mkdir(parents=True, exist_ok=True)
    for name, c0, c1 in (("0_0", 30, 71), ("1_0", 70, 110)):
        zt = z[2:95, c0:c1]
        bands = {}
        for ep, bins in UNITS.items():
            for b in bins:
                for p in (0, 1):
                    v = np.full(zt.shape, 6, np.uint16)
                    bands[f"w_{ep}_b{b:03d}_p{p}"] = ((zt < lvl(b)) * v).astype(np.uint16)
                    bands[f"v_{ep}_b{b:03d}_p{p}"] = v
        np.savez_compressed(path / f"tile_{name}.npz", transform=np.array([60, 0, 60 * c0, 0, -60, 6000 - 120]),
                            **bands)


def static(tile):
    sea = np.zeros((20, 28), np.uint8)
    sea[:, :5] = 1
    beyond = np.full((25, 35), -1, np.int8)
    beyond[:, 29:] = 0
    return a7.static_layers(
        TR, (H, W), domain=box(120, 300, 8280, 5880),
        sector_polys={"north": box(120, 3000, 8280, 5880), "south": box(120, 300, 8280, 3000)},
        cover=box(1800, 300, 6600, 5880),
        sea={"mask": sea, "transform": [300, 0, 0, 0, -300, 6000]},
        beyond={"state": beyond, "transform": [240, 0, 0, 0, -240, 6000]}, tile=tile)


def heights():
    z = truth().astype(np.float32)
    z[:2, :10] = np.nan
    off = z.copy()
    off[:50] += np.float32(0.3)
    off[50:] -= np.float32(0.2)
    return {"perfect": z, "offset": off}


def dem_grid(lowest):
    z = truth().astype(np.float32)
    d = np.full(z.shape, np.nan, np.float32)
    band = np.zeros(z.shape, bool)
    band[2:95, 30:110] = True
    band &= (z >= lowest) & (z <= lvl(23) + 0.2)
    d[band] = z[band]
    return d


def layers():
    z = truth()
    # a platform at a cell between the scenario levels -29.45 and L26, and one deeper
    r1 = 30
    c1 = int(np.flatnonzero((z[r1] > -29.40) & (z[r1] < -29.0))[0])
    c2 = int(np.flatnonzero((z[r1] > -30.4) & (z[r1] < -30.0))[0])
    return {"platforms": {"geoms": [Point(60 * c1 + 30, 6000 - 60 * r1 - 30), Point(60 * c2 + 30, 6000 - 60 * r1 - 30)],
                          "rule": "onshore", "threshold_m": None},
            "pipelines": {"geoms": [LineString([(1500, 4000), (4500, 4000)])], "rule": "length", "threshold_m": None},
            "ports": {"geoms": [Point(60 * 58 + 30, 6000 - 60 * 70 - 30)], "rule": "access", "threshold_m": 300.0,
                      "radius_m": 3000.0},
            "reserve": {"geoms": [box(2400, 600, 3600, 2400)], "rule": "area", "threshold_m": None},
            "habitat": {"samples": a8.ex.fraction_samples(habitat_share(), TR), "rule": "area"}}, (r1, c1, c2)


def habitat_share():
    f = np.full((H, W), np.nan, np.float32)
    f[60:90, 40:80] = 0.5                                     # half-covered cells in the south
    f[60:90, 60:70] = 0.0
    return f


def pathways():
    return pd.DataFrame({"pathway_id": ["lin", "trend"], "kind": ["projection", "rate"], "ref_year": [2020, 2021],
                         "year": [2100, 2100], "value_m": [-8.0, -0.057], "source": ["synthetic", "synthetic"]})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a8")
    write_stage2(tmp / "stage2", truth())
    s = static(16)
    blocks = pd.DataFrame([{"i": i, "j": j, "sector": "north" if j == 1 else "south", "split": "train"}
                           for i in range(3) for j in range(2)])
    hts = heights()
    t7 = a7.run(tmp / "stage2", tmp / "hindcast", s, hts, TR, sector_names=SECTORS, blocks=blocks, block_m=3000,
                tile=16, fit_region=s["domain"] & a7.train_mask(blocks, TR, (H, W), 3000), log=lambda *a, **k: None)
    lowest = lvl(UNITS["E3"][0])
    lay, pts = layers()
    extra = {"quad": {"curve": a8.ex.trend_plus_quadratic_curve(-4.0, -0.05, 2020.0, 2100.0), "role": "sensitivity",
                      "source": "synthetic"}}
    out = a8.run(tmp / "stage2", tmp / "emergence", s, hts, TR, dem=dem_grid(lowest), level_2026=L26,
                 hindcast_tables={"extent": t7["extent"], "displacement": t7["displacement"]},
                 sector_names=SECTORS, scenarios=SCEN, layers=lay, pathways=pathways(), extra_curves=extra, tile=16,
                 log=lambda *a, **k: None)
    return {"tmp": tmp, "static": s, "out": out, "pts": pts, "lowest": lowest}


def test_outputs_written(world):
    d = world["tmp"] / "emergence"
    for f in ("shifts.csv", "error_models.json", "loeo_errors.csv", "transfer.csv", "emergence.csv",
              "fused_summary.csv", "exposure.csv", "scenario_dates.csv", "run.json"):
        assert (d / f).exists(), f
    rec = world["out"]["run"]
    assert rec["observed_level_source"].startswith("hindcast displacement.csv")
    assert rec["folds"] == {"E1": ["E2", "E3"], "E2": ["E1", "E3"], "E3": ["E1", "E2"]}


def test_area_bias_cross_check_against_the_hindcast(world):
    chk = world["out"]["run"]["cross_check_extent_domain"]
    assert chk["n_rows"] == 2 * sum(len(v) for v in UNITS.values())
    assert chk["max_abs_diff_cells"] == 0.0 and chk["max_abs_diff_km2"] < 1e-9


def test_shifts_recover_the_product_errors(world):
    s = world["out"]["shifts"]
    s = s[s["usable"]]
    perfect = s[s["product"] == "perfect"]
    assert len(perfect) and (perfect["shift_m"] == 0).all()
    off = s[s["product"] == "offset"]
    north, south = off[off["region"] == "north"]["shift_m"], off[off["region"] == "south"]["shift_m"]
    assert len(north) >= 10 and len(south) >= 10
    assert np.all(np.abs(north - 0.3) <= 0.03) and np.all(np.abs(south + 0.2) <= 0.03)


def test_error_models(world):
    m = world["out"]["models"]["offset"][a8.ex.PRIMARY_FORM]
    fin = m["final"]
    assert fin["epochs"] == ["E2", "E3"] and fin["lowest_level_m"] == pytest.approx(world["lowest"])
    assert fin["models"]["north"]["a"] == pytest.approx(0.3, abs=0.03)
    assert fin["models"]["south"]["a"] == pytest.approx(-0.2, abs=0.03)
    assert abs(fin["models"]["north"]["b"]) < 0.1 and not fin["models"]["north"]["used_domain"]
    assert set(m["folds"]) == {"E1", "E2", "E3"} and m["folds"]["E3"]["epochs"] == ["E1", "E2"]


def test_leave_one_epoch_out_errors(world):
    e = world["out"]["loeo_errors"]
    assert set(e["fold"]) == {"E1", "E2", "E3"} and e["score"].any()
    sc = e[e["score"] & (e["product"] == "offset") & (e["functional"] == "sea_area") & (e["region"] != "domain")]
    corr = sc[sc["surface"] == "corrected"]["error_km2"].abs()
    unc = sc[sc["surface"] == "uncorrected"]["error_km2"].abs()
    assert len(corr) == len(unc) > 0
    assert corr.median() < 0.2 * unc.median()
    pe = e[(e["product"] == "perfect") & e["score"]]
    assert (pe[pe["surface"] == "uncorrected"]["error_km2"] == 0).all()
    anchors = e[e["is_anchor"]]
    assert not anchors["score"].any()
    t = world["out"]["transfer"]
    row = t[(t["product"] == "offset") & (t["fold"] == "E3") & (t["region"] == "north")].iloc[0]
    assert abs(row["shift_transfer_median_m"]) <= 0.03 and row["share_extrapolated"] > 0.5


def _brute_emergence(F_like, seed, region, L):
    a = hindcast.flood_fill(F_like, L26, seed) & region
    b = hindcast.flood_fill(F_like, L, seed) & region
    return (a.sum() - b.sum()) * a8.CELL_KM2


def test_emergence_matches_a_brute_force_flood_fill(world):
    s = world["static"]
    em = world["out"]["emergence"]
    z = truth().astype(np.float32)
    z[:2, :10] = np.nan
    for k, reg in enumerate(SECTORS):
        region = s["domain"] & (s["sector"] == k + 1)
        for D in SCEN:
            r = em[(em["product"] == "perfect") & (em["surface"] == "uncorrected") & (em["region"] == reg)
                   & (em["functional"] == "emergence") & (em["scenario_m"] == D)].iloc[0]
            assert r["estimate_km2"] == pytest.approx(_brute_emergence(z, s["sea_seed"], region, L26 - D))
    pe = em[em["product"] == "perfect"].pivot_table(index=["region", "functional", "scenario_m"], columns="surface",
                                                    values="estimate_km2")
    assert np.allclose(pe["fused"], pe["uncorrected"]) and np.allclose(pe["corrected"], pe["uncorrected"])
    fu = em[(em["surface"] == "fused") & (em["functional"] == "emergence")]
    assert (fu["lo_km2"] <= fu["estimate_km2"]).all() and (fu["estimate_km2"] <= fu["hi_km2"]).all()
    assert (fu["extrapolated_share"].dropna().between(0, 1)).all()
    assert np.isfinite(fu["q_km2"]).all() and (fu["n_scores"] >= 9).all()
    assert (fu["extrapolating_fold"] == "E3").all() and (fu["n_scores_extrap"] < fu["n_scores"]).all()


def test_offset_cancels_in_emergence_but_not_in_sea_area(world):
    """A constant offset moves the product's 2026 shoreline too, so it cancels in the
    self-referenced emergence; against the observed baseline it does not: 0.3 m / 0.05 m
    per cell = 6 columns of 48 rows too little sea in the north, 4 columns too much in the
    south."""
    em = world["out"]["emergence"]
    for reg, sign, cells in (("north", 1, 6 * 48), ("south", -1, 4 * 48)):
        o = em[(em["product"] == "offset") & (em["region"] == reg)].pivot_table(
            index=["functional", "scenario_m"], columns="surface", values="estimate_km2")
        emg, sea = o.loc["emergence"], o.loc["sea_area"]
        assert np.all(np.abs(emg["uncorrected"] - emg["fused"]) <= 0.15)
        diff = sea["uncorrected"] - sea["fused"]
        assert np.all(sign * diff > 0.7 * cells * a8.CELL_KM2) and np.all(sign * diff < 1.3 * cells * a8.CELL_KM2)


def test_h3_rule_applied_to_the_driver_tables(world):
    d = h3.decision(world["out"]["emergence"])
    assert not d[d["product"] == "perfect"]["decision_relevant"].any()
    s = h3.summary(d)
    assert set(s["verdict"].dropna()) <= {"supported", "partial", "not supported"}
    assert (s[s["product"] == "perfect"]["verdict"] == "not supported").all()


def test_exposure(world):
    x = world["out"]["exposure"]
    assert set(x["layer"]) == {"platforms", "pipelines", "ports", "reserve", "habitat"}
    hb = x[(x["layer"] == "habitat") & (x["surface"] == "fused") & (x["product"] == "perfect")
           & (x["region"] == "domain")].sort_values("scenario_m")
    assert (hb["n_features"] == 2).all() and np.all(np.diff(hb["amount"].to_numpy()) >= 0)
    assert hb["amount"].max() <= 0.5 * 30 * 30 * a8.CELL_KM2 + 1e-9
    p = x[(x["layer"] == "platforms") & (x["product"] == "perfect") & (x["surface"] == "fused")
          & (x["region"] == "domain")].set_index("scenario_m")
    assert p.loc[0.5, "n_changed"] == 1 and p.loc[1.0, "n_changed"] == 1 and p.loc[2.0, "n_changed"] == 2
    assert (p["n_changed_lo"] <= p["n_changed"]).all() and (p["n_changed"] <= p["n_changed_hi"]).all()
    ln = x[(x["layer"] == "pipelines") & (x["surface"] == "fused") & (x["product"] == "perfect")
           & (x["region"] == "domain")].sort_values("scenario_m")
    assert np.all(np.diff(ln["amount"].to_numpy()) >= 0) and ln["amount"].iloc[-1] > 0
    assert (ln["amount_lo"] <= ln["amount"]).all() and (ln["amount"] <= ln["amount_hi"]).all()
    rs = x[(x["layer"] == "reserve") & (x["surface"] == "fused") & (x["product"] == "perfect")
           & (x["region"] == "south")]
    assert (rs["n_features"] == 1).all()
    po = x[(x["layer"] == "ports") & (x["surface"] == "fused") & (x["product"] == "perfect")
           & (x["region"] == "domain")].sort_values("scenario_m")
    assert (po["radius_m"] == 3000.0).all() and np.all(np.diff(po["median_distance_increase_m"]) >= 0)


def test_scenario_dates(world):
    d = world["out"]["scenario_dates"].set_index(["pathway_id", "delta_m"])
    assert d.loc[("lin", 0.5), "date"] == pytest.approx(2031.0)
    assert d.loc[("trend", 2.0), "date"] == pytest.approx(2026 + 2 / 0.057, abs=1e-3)
    assert (d["source"] == "synthetic").all()
    assert d.loc[("quad", 0.5), "reached"] and d.loc[("quad", 0.5), "role"] == "sensitivity"


def test_fused_summary_and_rejections(world, tmp_path):
    fs = world["out"]["fused_summary"]
    assert (fs["n_dem"] > 0).all() and (fs["n_product"] + fs["n_extrapolated"] > 0).all()
    perfect = fs[fs["product"] == "perfect"].set_index("region")
    # the only clamp of a perfect product: the 5 x 5 inland pit, observed "land" (water not
    # connected to the sea), raised to L_min, which leaves every flood fill from the sea unchanged
    assert (perfect["n_clamped_sea"] == 0).all() and perfect.loc["north", "n_clamped_land"] == 25
    assert perfect.loc["south", "n_clamped_land"] == 0
    with pytest.raises(ValueError, match="primary"):
        a8.run(world["tmp"] / "stage2", tmp_path, world["static"], heights(), TR, dem=dem_grid(world["lowest"]),
               level_2026=L26, sector_names=SECTORS, forms=("constant",), tile=16, log=lambda *a, **k: None)
    with pytest.raises(ValueError, match="fit epochs"):
        a8.run(world["tmp"] / "stage2", tmp_path, world["static"], heights(), TR, dem=dem_grid(world["lowest"]),
               level_2026=L26, sector_names=SECTORS, fit_epochs=("E4",), tile=16, log=lambda *a, **k: None)


def test_dem_to_grid_aggregates_and_checks_alignment(tmp_path):
    import rasterio
    from rasterio.transform import Affine
    import pyproj
    from caspian.config import CRS_EQUAL_AREA
    z = np.arange(16, dtype=np.float32).reshape(4, 4)
    z[0, 0] = np.nan
    p = tmp_path / "dem_0_0.tif"
    with rasterio.open(p, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32",
                       crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(), transform=Affine(30, 0, 120, 0, -30, 5880),
                       nodata=np.nan) as w:
        w.write(z, 1)
    g = a8.dem_to_grid([p], TR, (H, W))
    assert g[2, 2] == pytest.approx(np.mean([1, 4, 5])) and g[2, 3] == pytest.approx(np.mean([2, 3, 6, 7]))
    assert np.isfinite(g).sum() == 4
    q = tmp_path / "dem_bad.tif"
    with rasterio.open(q, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32",
                       crs=pyproj.CRS(CRS_EQUAL_AREA).to_wkt(), transform=Affine(30, 0, 150, 0, -30, 5880),
                       nodata=np.nan) as w:
        w.write(z, 1)
    with pytest.raises(ValueError, match="aligned"):
        a8.dem_to_grid([q], TR, (H, W))


def test_read_pathways_selects_and_completes_rows(tmp_path):
    f = tmp_path / "p.csv"
    pd.DataFrame({"pathway_id": ["a", "b", "c", "d"], "kind": ["projection", "projection", "rate", "projection"],
                  "reference": ["2020 (window start)", "absolute level (BHS)", "per year", "2021"],
                  "year": [2100, 2100, np.nan, 2100], "value_m": [-8, -33, -0.057, 0.3],
                  "source": ["x", "y", "z", "w"], "scenario": ["s"] * 4, "statistic": ["m"] * 4}).to_csv(f, index=False)
    p = a8.read_pathways(f, selection=[("a", "primary", None, None), ("c", "primary", 2021, 2100),
                                       ("d", "range", None, None)])
    assert p["pathway_id"].tolist() == ["a", "c", "d"] and p["ref_year"].tolist() == [2020, 2021, 2021]
    assert p["year"].tolist() == [2100, 2100, 2100] and p["role"].tolist() == ["primary", "primary", "range"]
    d = a8.ex.scenario_dates(a8.ex.pathway_curves(p), [1.0]).set_index("pathway_id")
    assert d.loc["a", "date"] == pytest.approx(2036.0) and not d.loc["d", "reached"]
    with pytest.raises(ValueError, match="missing"):
        a8.read_pathways(f, selection=[("zz", "primary", None, None)])
    with pytest.raises(ValueError, match="no reference year"):
        a8.read_pathways(f, selection=[("b", "primary", None, None)])
    assert a8.read_pathways(tmp_path / "missing.csv") is None


def test_eq1_curves_from_the_project_table():
    if not a8.PATHWAYS.exists():
        pytest.skip("no projection_pathways.csv")
    c = a8.eq1_curves(a8.PATHWAYS)
    assert len(c) == 2
    for k, v in c.items():
        t, ch = v["curve"]
        assert t[0] == 2020 and t[-1] == 2100 and v["role"] == "sensitivity"
    totals = sorted(round(-v["curve"][1][-1], 2) for v in c.values())
    assert totals == [8.35, 14.11]
    assert a8.eq1_curves(Path("missing.csv")) == {}


def test_the_project_pathway_selection_resolves():
    """The selected ids exist in results/levels/projection_pathways.csv (if present) and
    every one has a reference year, a target year and a value."""
    if not a8.PATHWAYS.exists():
        pytest.skip("no projection_pathways.csv")
    p = a8.read_pathways(a8.PATHWAYS)
    assert len(p) == len(a8.DATE_PATHWAYS) and (p["year"] > p["ref_year"]).all()


def test_results_do_not_depend_on_the_tiling(world, tmp_path):
    lay, _ = layers()
    lowest = world["lowest"]
    t7 = {"extent": pd.read_csv(world["tmp"] / "hindcast" / "extent.csv"),
          "displacement": pd.read_csv(world["tmp"] / "hindcast" / "displacement.csv")}
    out = a8.run(world["tmp"] / "stage2", tmp_path, static(64), heights(), TR, dem=dem_grid(lowest), level_2026=L26,
                 hindcast_tables=t7, sector_names=SECTORS, scenarios=SCEN, layers=lay, pathways=pathways(), tile=64,
                 log=lambda *a, **k: None)
    for k in ("shifts", "loeo_errors", "emergence", "exposure"):
        a, b = world["out"][k], out[k]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=False,
                                      rtol=1e-12, atol=1e-12)


def test_fit_mask_restricts_the_shift_counts(world, tmp_path):
    s = world["static"]
    mask = np.zeros((H, W), bool)
    mask[:, :85] = True
    out = a8.run(world["tmp"] / "stage2", tmp_path, s, heights(), TR, dem=dem_grid(world["lowest"]), level_2026=L26,
                 sector_names=SECTORS, scenarios=(0.5,), fit_mask=mask, tile=16, log=lambda *a, **k: None)
    assert out["run"]["cross_check_extent_domain"].startswith("not run")
    assert out["run"]["observed_level_source"].startswith("fallback")
    full, part = world["out"]["shifts"], out["shifts"]
    assert (part["n_known"] <= full["n_known"].to_numpy()).all() and (part["n_known"] < full["n_known"].to_numpy()).any()


def test_raster_layers_read_a_habitat_product(tmp_path):
    import rasterio
    from rasterio.transform import Affine
    d = tmp_path / "glwd_v2"
    d.mkdir()
    data = np.zeros((40, 60), np.uint8)
    data[:, 30:] = 13                                       # a wetland class
    data[:5, :5] = 1                                        # a lake class: not wetland
    with rasterio.open(d / "GLWD_v2_0_main_class.tif", "w", driver="GTiff", width=60, height=40, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=Affine(0.004, 0, 49.95, 0, -0.004, 45.6),
                       nodata=255) as w:
        w.write(data, 1)
    lay = a8.raster_layers(TR, (H, W), root=tmp_path)
    assert list(lay) == ["glwd_v2_wetland"]
    smp = lay["glwd_v2_wetland"]["samples"]
    assert len(smp) > 0 and np.allclose(smp["weight"], a8.CELL_KM2 * 1e6)   # coarse source: shares 0 or 1
    assert lay["glwd_v2_wetland"]["classes"] == list(range(8, 33))
