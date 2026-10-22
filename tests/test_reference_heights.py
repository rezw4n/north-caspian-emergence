"""Additional independent heights: the GLO-30 and GEDI helpers in
caspian.validation, the point stages of scripts/a6_glo30.py and scripts/a6_gedi.py on
synthetic files, and the additional-source handling of scripts/a6_validate_dem.py (the
H2 block size inherited, results in their own folders, never pooled with ICESat-2)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import caspian  # noqa: F401
from caspian import blocking, design, validation
from caspian.config import CRS_EQUAL_AREA

ROOT = Path(__file__).resolve().parents[1]
DAY0 = np.datetime64("1970-01-01")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _days(s):
    return float((np.datetime64(s) - DAY0).astype(int))


# --- caspian.validation helpers ---------------------------------------------------------

def test_glo30_names_and_excluded_cells():
    assert validation.glo30_cell_name(45, 49) == "N45E049"
    assert validation.glo30_cell_name(-1, -2) == "S01W002"
    assert validation.glo30_tile_id("N45E049") == "Copernicus_DSM_COG_10_N45_00_E049_00_DEM"
    with pytest.raises(ValueError):
        validation.glo30_tile_id("N45_E049")
    assert validation.GLO30_EXCLUDED_TILES == ("N44E048", "N44E049")
    assert str(validation.GLO30_WINDOW[0])[:10] == "2010-12-01"
    assert validation.DRY_RADIUS_PX["GLO30"] == validation.DRY_RADIUS_PX["GEDI"] == 1


def test_thin_index_centred_every_step():
    r = validation.thin_index(3600, 3)
    assert r[0] == 1 and np.all(np.diff(r) == 3) and r.size == 1200
    c = validation.thin_index(3600, 4)
    assert c[0] == 2 and c.size == 900 and c[-1] == 3598
    assert np.array_equal(validation.thin_index(5, 1), np.arange(5))
    with pytest.raises(ValueError):
        validation.thin_index(10, 0)


def test_glo30_steps_follow_the_handbook_codes():
    h = np.array([-27.0, np.nan, -27.0, -27.0, -27.0, -27.0, -32767.0])
    wbm = np.array([0, 0, 2, 0, 0, 0, 0])
    edm = np.array([1, 1, 1, 9, 1, 1, 1])
    flm = np.array([2, 2, 2, 2, 1, 2, 2])
    dN = np.array([0.1, 0.1, 0.1, 0.1, 0.1, np.nan, 0.1])
    steps = validation.glo30_steps(h, wbm, edm, flm, dN)
    assert list(steps) == ["height", "wbm_no_water", "edm_not_edited", "flm_not_filled", "geoid"]
    keep, casc = validation.cascade(steps, len(h))
    assert keep.tolist() == [True, False, False, False, False, False, False]
    assert casc.set_index("step").loc["height", "n_fail_alone"] == 2          # NaN and the -32767 fill


def test_grid_lookup_is_bilinear_and_nan_outside(tmp_path):
    lat, lon = np.arange(44.0, 47.01, 0.5), np.arange(46.0, 54.01, 0.5)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    np.savez(tmp_path / "g.npz", lat=lat, lon=lon, dN=(0.1 + 0.02 * (lo - 50) - 0.03 * (la - 45)).astype("float32"))
    f = validation.GridLookup(tmp_path / "g.npz", "dN")
    v = f(np.array([50.25, 47.1, 60.0]), np.array([45.25, 46.6, 45.0]))
    assert v[0] == pytest.approx(0.1 + 0.005 - 0.0075, abs=1e-6)
    assert v[1] == pytest.approx(0.1 + 0.02 * -2.9 - 0.03 * 1.6, abs=1e-6)
    assert np.isnan(v[2])


def test_gedi_time_from_delta_time():
    t = validation.gedi_time([0.0, 86_400.5, 108_802_800.25])
    assert t[0] == np.datetime64("2018-01-01T00:00:00", "ns")
    assert t[1] == np.datetime64("2018-01-02T00:00:00.5", "ns")
    assert str(t[2])[:10] == "2021-06-13"


def test_dry_throughout_the_glo30_window():
    t0 = validation.GLO30_WINDOW[0]
    e = np.array([[_days("2010-10-31"), _days("2010-10-31")],     # 31 days before: dry
                  [_days("2010-11-01"), _days("2010-10-01")],     # exactly 30 days before one pixel: not dry
                  [_days("2005-01-01"), np.nan]])                 # one pixel never emerged
    assert validation.dry_at_acquisition(np.full(3, t0), e).tolist() == [True, False, False]


# --- scripts/a6_gedi.py helpers ---------------------------------------------------------

def test_gedi_beam_names_month_cut_and_row_merge():
    g = _load("a6_gedi")
    assert g.beam_name([0, 1, 5, 11, np.nan]).tolist() == ["BEAM0000", "BEAM0001", "BEAM0101", "BEAM1011", ""]
    assert g.month_cut("LARSE/GEDI/GEDI02_A_002_MONTHLY/202012_048E_042N") == _days("2021-01-01") - 30
    assert g.month_cut("202106_042E_042N") == _days("2021-07-01") - 30
    cells = [[0.0, 0.0, 0.005, 0.005], [0.005, 0.0, 0.01, 0.005], [0.02, 0.0, 0.025, 0.005], [0.0, 0.005, 0.005, 0.01]]
    assert g._merge_rows(cells) == [[0.0, 0.0, 0.01, 0.005], [0.02, 0.0, 0.025, 0.005], [0.0, 0.005, 0.005, 0.01]]


def test_glo30_src_dates():
    g = _load("a6_glo30")
    kml = "<when>2011-03-10T05:00:00</when> 2012-11-26 x 2011-03-10 <name>TDM1_SAR__COS_BIST_SM_S_SRA_20110310</name>"
    assert g.src_dates(kml) == {"first": "2011-03-10", "last": "2012-11-26", "n_distinct_dates": 2}


# --- synthetic world near the equal-area origin (45.5 N, 50 E) ----------------------------

def _geotiff(path, bands, transform, crs, dtype="float32", nodata=None):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = bands[0].shape
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=len(bands), dtype=dtype, crs=crs,
                       transform=transform, nodata=nodata) as dst:
        for k, b in enumerate(bands, 1):
            dst.write(b.astype(dtype), k)


def _world(tmp_path):
    """DEM and emergence tiles (equal-area, 30 m) over x, y in [0, 12 km]; emergence
    2000-01-01 except x > 6 km (2012-06-01, inside the GLO-30 window) and y > 10 km
    (never emerged); contour points; the two geoid grids."""
    from rasterio.transform import Affine

    n = 400
    xs = (np.arange(n) + 0.5) * 30.0
    gx, gy = np.meshgrid(xs, 12_000 - xs)
    tr = Affine(30.0, 0, 0, 0, -30.0, 12_000)
    dem = -27.0 + 1e-4 * gx
    _geotiff(tmp_path / "processed" / "dem" / "syn" / "dem_0_0.tif", [dem, np.full_like(dem, 250.0)], tr, CRS_EQUAL_AREA,
             nodata=np.nan)
    em = np.full(gx.shape, _days("2000-01-01"))
    em[gx > 6_000] = _days("2012-06-01")
    em[gy > 10_000] = np.nan
    _geotiff(tmp_path / "emergence" / "emergence_0_0.tif", [em], tr, CRS_EQUAL_AREA, nodata=np.nan)
    wl = tmp_path / "interim" / "waterlines" / "syn"
    wl.mkdir(parents=True)
    pd.DataFrame({"x": np.arange(0, 12_000, 30.0), "y": 5_000.0, "epoch": "E2"}).to_csv(wl / "points.csv.gz", index=False)
    study = tmp_path / "results" / "study_area"
    study.mkdir(parents=True)
    lat, lon = np.arange(45.0, 46.01, 0.05), np.arange(49.5, 50.51, 0.05)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    np.savez(study / "geoid_correction_primary.npz", lat=lat, lon=lon, dN=(-0.4 + 0.1 * (lo - 50)).astype("float32"))
    return dem


def _patched_driver(tmp_path, monkeypatch):
    drv = _load("a6_validate_dem")
    for name, sub in (("INTERIM", "interim"), ("PROCESSED", "processed"), ("RESULTS", "results")):
        monkeypatch.setattr(drv, name, tmp_path / sub)
    return drv


def test_glo30_points_stage_on_synthetic_tiles(tmp_path, monkeypatch):
    from rasterio.transform import Affine

    _world(tmp_path)
    drv = _patched_driver(tmp_path, monkeypatch)
    g = _load("a6_glo30")
    monkeypatch.setattr(g, "RAW_DIR", tmp_path / "raw" / "glo30")
    monkeypatch.setattr(g, "PROV", tmp_path / "results" / "provenance" / "copernicus_glo30.json")
    monkeypatch.setattr(g, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(g, "_drv", lambda: drv)
    # a 1" tile with pixel (0, 0) centred on (50.0 E, 45.6 N), RasterPixelIsPoint as on AWS
    n = 360
    d = 1 / 3600
    tr = Affine(d, 0, 50.0 - d / 2, 0, -d, 45.6 + d / 2)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lon, lat = 50.0 + cols * d, 45.6 - rows * d
    h = -27.3 + 0.5 * (lon - 50.0)
    wbm = np.where(cols < 30, 2, 0)                      # a lake strip on the west edge
    edm = np.where(rows < 30, 9, 1)                      # edited (lake) rows on the north edge
    flm = np.where((rows >= 30) & (cols >= 30), 2, 1)
    files = {}
    for role, arr, dt in (("dem", h, "float32"), ("wbm", wbm, "uint8"), ("edm", edm, "uint8"), ("flm", flm, "uint8")):
        key = f"Copernicus_DSM_COG_10_N45_00_E050_00_DEM/{role}.tif"
        _geotiff(g.RAW_DIR / "N45E050" / f"{role}.tif", [arr], tr, "EPSG:4326", dtype=dt)
        files[role] = {"key": key}
    g.PROV.parent.mkdir(parents=True)
    g.PROV.write_text(json.dumps({"tiles": {"N45E050": {"files": files}, "N44E049": {"files": files}}}))
    em = str(tmp_path / "emergence")
    assert g.main(["points", "syn", "--emergence", em, "--no-products"]) == 0

    res = tmp_path / "results" / "validation" / "syn_glo30"
    meta = json.loads((res / "prepare.json").read_text())
    assert meta["tiles"] == ["N45E050"]                                       # the excluded cell is never read
    casc = pd.read_csv(res / "GLO30_filter_cascade.csv").set_index("step")
    assert casc.loc["input", "n_remaining"] == (n // 3) * (n // 4)            # thinned 3 x 4
    assert list(casc.index) == ["input", "dry_centre", "height", "wbm_no_water", "edm_not_edited", "flm_not_filled",
                                "geoid", "dry_throughout_window"]
    pts = pd.read_csv(tmp_path / "interim" / "validation" / "syn" / "GLO30_points.csv.gz")
    assert len(pts) == casc.loc["dry_throughout_window", "n_remaining"] > 0
    import pyproj

    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    x, y = fwd.transform(pts["lon"].to_numpy(), pts["lat"].to_numpy())
    assert np.allclose(x, pts["x"], atol=0.05) and np.allclose(y, pts["y"], atol=0.05)   # lon/lat written to 9 digits
    assert (pts["x"] < 6_000 - 30).all() and (pts["y"] < 10_000 - 30).all()   # dry throughout, with the 1-pixel ring
    assert (pts["wbm"] == 0).all() and (pts["edm"] == 1).all() and (pts["flm"] == 2).all()
    # pixel centres on whole arc-seconds, from the thinned rows and columns
    col = np.round((pts["lon"] - 50.0) * 3600).astype(int)
    row = np.round((45.6 - pts["lat"]) * 3600).astype(int)
    assert set(col % 4) == {2} and set(row % 3) == {1}
    assert np.allclose(pts["h"], -27.3 + 0.5 * (pts["lon"] - 50.0), atol=1e-5)
    assert np.allclose(pts["h_orth"], pts["h"] + (-0.4 + 0.1 * (pts["lon"] - 50.0)), atol=1e-5)   # + dN
    assert (pts["time"] == "2010-12-01").all() and (pts["source"] == "GLO30").all()
    assert np.allclose(pts["dem_h"], -27.0 + 1e-4 * (np.floor(pts["x"] / 30) * 30 + 15), atol=1e-4)
    assert {"interp_dist", "slope", "contour_density"} <= set(pts)


def test_gedi_points_stage_on_synthetic_shots(tmp_path, monkeypatch):
    _world(tmp_path)
    drv = _patched_driver(tmp_path, monkeypatch)
    g = _load("a6_gedi")
    monkeypatch.setattr(g, "RAW_DIR", tmp_path / "interim" / "gedi")
    monkeypatch.setattr(g, "STATE", tmp_path / "results" / "inventory" / "gedi_export.json")
    monkeypatch.setattr(g, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(g, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(g, "_drv", lambda: drv)
    import pyproj

    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    rng = np.random.default_rng(1)
    k = 400
    x, y = rng.uniform(100, 11_900, k), rng.uniform(100, 11_900, k)
    lon, lat = inv.transform(x, y)
    N = validation.EigenGeoid()(lon, lat)
    H = -27.0 + 1e-4 * x
    dt = np.where(np.arange(k) % 2 == 0, (np.datetime64("2021-06-01") - np.datetime64("2018-01-01")).astype(int),
                  (np.datetime64("2019-06-01") - np.datetime64("2018-01-01")).astype(int)) * 86_400.0 + np.arange(k)
    raw = pd.DataFrame({"elev_lowestmode": H + N, "digital_elevation_model": H + N + 1.0,
                        "quality_flag": 1.0, "degrade_flag": 0.0, "beam": 5.0,
                        "solar_elevation": np.where(np.arange(k) % 4 == 0, -10.0, 30.0), "delta_time": dt,
                        "lon": lon, "lat": lat})
    raw.loc[0:9, "quality_flag"] = 0.0
    raw.loc[10:19, "degrade_flag"] = 80.0
    raw.loc[20:29, "digital_elevation_model"] = -999999.0                   # gross-error screen
    raw.loc[30:39, "beam"] = 0.0                                             # BEAM0000 in 2019: dropped
    raw.loc[40:49, "solar_elevation"] = -9999.0                              # unmasked fill: not a night shot
    g.RAW_DIR.mkdir(parents=True)
    raw.to_csv(g.RAW_DIR / "202106_048E_042N_45N050E.csv.gz", index=False, float_format="%.17g")
    raw.iloc[:50].to_csv(g.RAW_DIR / "202106_048E_042N_45N049E.csv.gz", index=False, float_format="%.17g")  # duplicates
    g.STATE.parent.mkdir(parents=True)
    g.STATE.write_text(json.dumps({"export": {"stopped_by_budget": True}}))
    em = str(tmp_path / "emergence")
    with pytest.raises(SystemExit, match="budget"):
        g.main(["points", "syn", "--emergence", em, "--no-products"])
    g.STATE.write_text(json.dumps({"export": {"stopped_by_budget": False}}))
    assert g.main(["points", "syn", "--emergence", em, "--no-products"]) == 0

    res = tmp_path / "results" / "validation" / "syn_gedi"
    casc = pd.read_csv(res / "GEDI_filter_cascade.csv").set_index("step")
    assert casc.loc["input", "n_remaining"] == k + 50
    assert casc.loc["quality_flag_1", "n_fail_alone"] == 20 and casc.loc["degrade_flag_0", "n_fail_alone"] == 20
    assert casc.loc["gross_error", "n_fail_alone"] == 20
    assert casc.loc["beam_ranging", "n_fail_alone"] == 10                    # shot 30-39: odd ones are 2019 shots
    assert casc.loc["unique_across_images", "n_fail_alone"] > 0
    pts = pd.read_csv(tmp_path / "interim" / "validation" / "syn" / "GEDI_points.csv.gz")
    assert len(pts) == casc.loc["dry_at_acquisition", "n_remaining"] > 0
    assert not pts.duplicated(["delta_time", "beam"]).any()
    t = pd.to_datetime(pts["time"]).to_numpy()
    # dry at acquisition: x > 6 km emerged 2012-06 (dry for every shot), y > 10 km never emerged
    assert (pts["y"] < 10_000 - 30).all()
    assert np.allclose(pts["h_orth"], -27.0 + 1e-4 * pts["x"], atol=1e-6)                 # h - N(EIGEN-6C4)
    assert ((pts["night"] == 1) == (pts["solar_elevation"] < 0)).all()
    assert not ((pts["solar_elevation"].isna()) & (pts["night"] == 1)).any()
    assert (t >= np.datetime64("2019-06-01")).all()
    meta = json.loads((res / "prepare.json").read_text())
    assert meta["n_shots_exported"] == k + 50 and meta["n_night"] == int(pts["night"].sum())


# --- scripts/a6_validate_dem.py with an additional source ---------------------------------

def _hashes(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*"))
            if p.is_file()}


def test_additional_sources_inherit_the_h2_blocks_and_leave_icesat2_untouched(tmp_path, monkeypatch):
    drv = _patched_driver(tmp_path, monkeypatch)
    size = 5_000
    study = tmp_path / "results" / "study_area"
    study.mkdir(parents=True)
    b = blocking.block_grid((0, 0, 50_000, 50_000), size)
    b["sector"] = np.where(b["xc"] < 25_000, "west", "volga")
    b["split"] = blocking.assign_splits(b)
    b.to_csv(study / "blocks_5km.csv", index=False)
    h2 = tmp_path / "results" / "validation" / "syn_all"
    h2.mkdir(parents=True)
    (h2 / "block_size_check.json").write_text(json.dumps({"status": "resolved", "base_block_size_m": 2500.0,
                                                         "final_block_size_m": float(size),
                                                         "sources": {"ATL06": {}, "ATL08": {}}}))
    (h2 / "ATL06_accuracy.csv").write_text("scope,model\noverall,dem\n")
    before = _hashes(tmp_path / "results" / "validation")
    rng = np.random.default_rng(3)
    n = 4000
    x, y = rng.uniform(0, 50_000, n), rng.uniform(0, 50_000, n)
    truth = -27.0 + 2e-5 * x
    pts = pd.DataFrame({"x": x, "y": y, "lon": 50.0, "lat": 45.5, "time": "2021-06-01T00:00:00",
                        "h_orth": truth + 0.4 + rng.normal(0, 0.05, n), "dem_h": truth + rng.normal(0, 0.2, n),
                        "interp_dist": rng.uniform(0, 3000, n), "contour_density": rng.uniform(0, 50, n),
                        "slope": rng.uniform(0, 0.01, n), "prod:etopo_2022": truth + 0.4 + rng.normal(0, 0.8, n),
                        "prod:gebco_2026": truth + 1.0 + rng.normal(0, 0.8, n), "night": (rng.uniform(size=n) < 0.4).astype(int)})
    cache = tmp_path / "interim" / "validation" / "syn"
    cache.mkdir(parents=True)
    for s in ("GLO30", "GEDI"):
        pts.to_csv(cache / f"{s}_points.csv.gz", index=False)

    with pytest.raises(SystemExit):                                         # never pooled with ICESat-2
        drv.main(["syn", "evaluate", "--sources", "ATL06,GLO30", "--population", "all"])
    with pytest.raises(SystemExit):                                         # built by its own script
        drv.main(["syn", "prepare", "--sources", "GLO30"])
    with pytest.raises(SystemExit, match="blocksize"):
        drv.main(["syn", "evaluate", "--sources", "GLO30", "--population", "all"])
    assert drv.main(["syn", "blocksize", "--sources", "GLO30", "--population", "all"]) == 0
    out = tmp_path / "results" / "validation" / "syn_glo30_all"
    chk = json.loads((out / "block_size_check.json").read_text())
    assert chk["final_block_size_m"] == size and chk["status"] == "resolved"
    assert chk["sources"]["GLO30"]["inherited"] and not chk["sources"]["GLO30"]["refitted"]
    assert chk["inherited_sources"] == ["ATL06", "ATL08"]
    assert not (out / "GLO30_semivariogram.csv").exists()                  # nothing refitted
    assert drv.main(["syn", "evaluate", "--sources", "GLO30", "--population", "all"]) == 0
    c = json.loads((out / "GLO30_datum_constant.json").read_text())["constant"]
    assert c["constant_m"] == pytest.approx(-0.4, abs=0.03) and c["split"] == "train"
    for f in ("accuracy", "accuracy_blocks", "paired_bootstrap", "coverage"):
        assert (out / f"GLO30_{f}.csv").exists()
    # GEDI: the night-only sensitivity of Amendment 2 in its own subfolder
    assert drv.main(["syn", "blocksize", "--sources", "GEDI", "--population", "all"]) == 0
    assert drv.main(["syn", "evaluate", "--sources", "GEDI", "--population", "all", "--subset", "night"]) == 0
    night = tmp_path / "results" / "validation" / "syn_gedi_all" / "subset_night"
    cnt = json.loads((night / "GEDI_counts.json").read_text())
    assert cnt["n_points"] == int(pts["night"].sum()) and cnt["subset"] == "night"
    with pytest.raises(SystemExit, match="no such column"):
        drv.main(["syn", "evaluate", "--sources", "GEDI", "--population", "all", "--subset", "nocturnal"])
    # the ICESat-2 results are unchanged
    after = _hashes(tmp_path / "results" / "validation")
    assert {k: v for k, v in after.items() if k.startswith("syn_all")} == before
    assert design.BLOCK_SIZE_M == 25_000
