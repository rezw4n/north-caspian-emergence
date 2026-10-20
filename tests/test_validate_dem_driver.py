"""scripts/a6_validate_dem.py on synthetic files: the three stages, their order, and the
plumbing from granule files and tiles to the output tables (no real data)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import caspian
from caspian import blocking, design
from caspian.config import CRS_EQUAL_AREA

ROOT = Path(__file__).resolve().parents[1]
C_TRUE = -1.1
T0 = np.datetime64("2023-06-01T10:00:00", "ns")
BEAMS = {"gt1l": 0, "gt1r": 90, "gt2l": 3300, "gt2r": 3390, "gt3l": 6600, "gt3r": 6690}


@pytest.fixture
def drv():
    spec = importlib.util.spec_from_file_location("a6_validate_dem", ROOT / "scripts" / "a6_validate_dem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _truth(x, y):
    return -27.0 + 1e-5 * np.asarray(x) - 2e-6 * np.asarray(y)


def _geoid(y):
    return -15.5 + 1e-6 * np.asarray(y)


def _tile(path, bands, res=30.0):
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = bands[0].shape
    with rasterio.open(path, "w", driver="GTiff", width=nx, height=ny, count=len(bands), dtype="float32",
                       crs=CRS_EQUAL_AREA, transform=Affine(res, 0, 0, 0, -res, 50_000), nodata=np.nan) as dst:
        for k, b in enumerate(bands, 1):
            dst.write(b.astype("float32"), k)


def _blocks(size):
    b = blocking.block_grid((0, 0, 50_000, 50_000), size)
    b["sector"] = np.where(b["xc"] < 25_000, "west", "volga")
    b["split"] = blocking.assign_splits(b)
    return b


def _track(product, x0, t0, rng, inv):
    step = 100.0 if product == "ATL06" else 300.0
    rows = []
    for beam, dx in BEAMS.items():
        y = np.arange(1_000.0, 49_000.0, step) + (0 if product == "ATL06" else 50)
        x = np.full(y.size, x0 + dx)
        lon, lat = inv.transform(x, y)
        h = _truth(x, y) - C_TRUE + _geoid(y) + rng.normal(0, 0.01, y.size)
        t = t0 + ((y - 1_000) / 7_000 * 1e9).astype("timedelta64[ns]")
        d = pd.DataFrame({"lat": lat, "lon": lon, "h": h, "h_sigma": 0.03, "dem_h": h + 1.0,
                          "time": pd.Series(t).astype(str), "beam": beam, "sc_orient": 0})
        if product == "ATL06":
            d = d.assign(quality=0, geoid_h=_geoid(y), n_photons=50, snr_sig=0.0)
        else:
            d = d.assign(h_std=0.1, n_photons=40, watermask=0, landcover=80, cloud=0, msw=0, night=1,
                         terrain_flg=1)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _inputs(tmp_path, monkeypatch, drv):
    import pyproj

    for name, sub in (("INTERIM", "interim"), ("PROCESSED", "processed"), ("RESULTS", "results")):
        monkeypatch.setattr(drv, name, tmp_path / sub)
    monkeypatch.setattr(design, "BLOCK_SIZE_M", 5_000)       # 100 blocks in one 50 km tile
    rng = np.random.default_rng(0)
    n = 1666
    xs = (np.arange(n) + 0.5) * 30.0
    gx, gy = np.meshgrid(xs, 50_000 - (np.arange(n) + 0.5) * 30.0)
    dist = np.abs((gx % 2_000) - 1_000) / 2
    # DEM error: white noise growing with the interpolation distance, plus a smooth field with a
    # practical range of a few km (so the block-size check resolves at the 5 km base)
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(rng.normal(size=(556, 556)), 8)
    smooth = np.repeat(np.repeat(0.3 * smooth / smooth.std(), 3, 0), 3, 1)[:n, :n]
    _tile(tmp_path / "processed" / "dem" / "syn" / "dem_0_0.tif",
          [_truth(gx, gy) + smooth + rng.normal(0, 0.03 * (1 + dist) ** 0.4), dist])
    em = np.full(gx.shape, float(np.datetime64("2020-01-01", "D").astype(int)))
    em[gx > 45_000] = np.nan                                  # never emerged
    _tile(tmp_path / "emergence" / "emergence_0_0.tif", [em])
    study = tmp_path / "results" / "study_area"
    study.mkdir(parents=True)
    _blocks(5_000).to_csv(study / "blocks_5km.csv", index=False)
    wl = tmp_path / "interim" / "waterlines" / "syn"
    wl.mkdir(parents=True)
    cy = np.arange(0, 50_000, 30.0)
    pd.DataFrame({"x": np.repeat(np.arange(0, 50_001, 2_000.0), cy.size), "y": np.tile(cy, 26),
                  "z_m": -27.0, "epoch": "E2"}).to_csv(wl / "points.csv.gz", index=False)
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    prov = tmp_path / "results" / "provenance"
    prov.mkdir()
    for product in ("ATL06", "ATL08"):
        rec = {}
        folder = tmp_path / "interim" / "icesat2" / product
        folder.mkdir(parents=True)
        for g, (x0, t0) in enumerate([(2_000, T0), (14_000, T0), (26_000, T0),
                                      (36_000, np.datetime64("2019-12-15T10:00:00", "ns"))]):
            name = f"{product}_2023060{g}100000_0{g}000106_007_01.h5"
            d = _track(product, x0, t0, rng, inv)
            d.to_csv(folder / f"{name}.csv.gz", index=False, float_format="%.9g")
            rec[name] = len(d)
        rec[f"{product}_20230609100000_09000106_007_01.h5"] = 0      # no points in the cover
        rec[f"{product}_20230608100000_08000106_007_01.h5"] = 999    # file still being written
        (folder / f"{product}_20230608100000_08000106_007_01.h5.csv.gz").write_text("lat,lon\n1,2\n")
        (prov / f"icesat2_{product.lower()}.json").write_text(json.dumps({"granules": rec}))
    fwd = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)

    def sample(native, lon, lat, method="bilinear"):     # mirrors caspian.products.sample
        x, y = fwd.transform(lon, lat)
        noise = np.random.default_rng(len(native)).normal(0, 0.6, np.size(x))
        return (_truth(x, y) - C_TRUE + noise if native == "etopo_2022" else _truth(x, y) + 0.5 + noise).astype("float32")

    fake = types.ModuleType("caspian.products")
    fake.PRODUCTS = {"etopo_2022": {}, "gebco_2026": {}}
    fake.load_native = lambda name: name                  # the fake's native grid is its name
    fake.sample = sample
    return fake


def test_driver_stages(tmp_path, monkeypatch, drv):
    fake = _inputs(tmp_path, monkeypatch, drv)
    em = str(tmp_path / "emergence")
    with pytest.raises(SystemExit, match="emergence"):
        drv.main(["syn", "prepare"])
    monkeypatch.setitem(sys.modules, "caspian.products", None)        # module not importable
    monkeypatch.delattr(caspian, "products", raising=False)
    with pytest.raises(SystemExit, match="--no-products"):
        drv.main(["syn", "prepare", "--emergence", em])
    monkeypatch.setitem(sys.modules, "caspian.products", fake)
    monkeypatch.setattr(caspian, "products", fake, raising=False)
    with pytest.raises(SystemExit, match="unknown products"):
        drv.main(["syn", "prepare", "--emergence", em, "--products", "gebco_2026,etopo_2023"])
    assert drv.main(["syn", "prepare", "--emergence", em]) == 0

    out = tmp_path / "results" / "validation" / "syn"
    meta = json.loads((out / "prepare.json").read_text())
    for product in ("ATL06", "ATL08"):
        g = meta["sources"][product]["granules"]
        assert g["read"] == 4 and g["incomplete_file"] == 1 and g["recorded_with_points"] == 5
        casc = pd.read_csv(out / f"{product}_filter_cascade.csv").set_index("step")
        n_in = casc.loc["input", "n_remaining"]
        assert casc.loc["strong_beam", "n_fail_alone"] == n_in // 2                  # weak beams
        assert casc.loc["dry_at_acquisition", "n_remaining"] < casc.iloc[-2]["n_remaining"]
        pts = pd.read_csv(tmp_path / "interim" / "validation" / "syn" / f"{product}_points.csv.gz")
        assert len(pts) == casc.loc["dry_at_acquisition", "n_remaining"]
        assert (pts["x"] < 45_000).all() and set(pts["beam"]) == {"gt1l", "gt2l", "gt3l"}
        assert not pts["granule"].str.contains("20230603").any()                      # acquired before emergence
        # ellipsoid -> EGM2008 (ATL08: geoid interpolated from the ATL06 of the same orbit)
        assert np.abs(pts["h_orth"] - (_truth(pts["x"], pts["y"]) - C_TRUE)).max() < 0.06
        assert {"dem_h", "interp_dist", "slope", "contour_density", "prod:etopo_2022", "prod:gebco_2026"} <= set(pts)

    with pytest.raises(SystemExit, match="blocksize"):
        drv.main(["syn", "evaluate"])
    assert drv.main(["syn", "blocksize"]) == 0
    chk_path = out / "block_size_check.json"
    chk = json.loads(chk_path.read_text())
    final = chk["final_block_size_m"]
    assert chk["status"] == "resolved"
    assert final == max(chk["sources"][s]["final_block_size_m"] for s in ("ATL06", "ATL08")) == 10_000
    assert chk["sources"]["ATL06"]["split"] == "train" and chk["sources"]["ATL06"]["range_within_max_lag"]
    # an unresolved check (range beyond the largest lag) blocks the evaluation
    chk_path.write_text(json.dumps({**chk, "status": "unresolved", "final_block_size_m": None}))
    with pytest.raises(SystemExit, match="unresolved"):
        drv.main(["syn", "evaluate"])
    chk_path.write_text(json.dumps(chk))
    # the enlarged size needs its block table, made by scripts/a6_blocks.py
    with pytest.raises(SystemExit, match="a6_blocks.py"):
        drv.main(["syn", "evaluate"])
    _blocks(10_000).to_csv(tmp_path / "results" / "study_area" / "blocks_10km.csv", index=False)
    assert drv.main(["syn", "evaluate"]) == 0
    assert drv.main(["syn", "evaluate", "--block-size", "5000", "--sources", "ATL06"]) == 0   # sensitivity
    assert (out / "sensitivity_5000m" / "ATL06_accuracy.csv").exists()
    for product in ("ATL06", "ATL08"):
        c = json.loads((out / f"{product}_datum_constant.json").read_text())["constant"]
        assert c["constant_m"] == pytest.approx(C_TRUE, abs=0.03) and c["split"] == "train"
        acc = pd.read_csv(out / f"{product}_accuracy.csv")
        assert {"dem", "etopo_2022|total", "gebco_2026|matched"} <= set(acc["model"])
        boot = pd.read_csv(out / f"{product}_paired_bootstrap.csv")
        assert set(boot["variant"]) == {"total", "matched"} and (boot["n_boot"] == design.BOOTSTRAP_N).all()
        cov = pd.read_csv(out / f"{product}_coverage.csv")
        assert set(cov["scope"]) >= {"overall", "west", "volga"}
        cal = json.loads((out / f"{product}_calibration.json").read_text())["calibration"]
        assert set(cal["groups"]) >= set(caspian.geography.SECTOR_ORDER)
