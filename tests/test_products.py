"""caspian.products: loading the audited bathymetry products, resampling them to the
equal-area hindcast grid, and the vertical-offset convention (PREREG section 5).

Synthetic grids only, except one smoke test that the five downloaded subsets load
(skipped where data/ is absent); it reads their axes, never a hindcast quantity.
"""

import numpy as np
import pyproj
import pytest

from caspian import hindcast, products
from caspian.config import CRS_EQUAL_AREA

FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
STEP = 15 / 3600                                   # 15 arc-seconds


def _native(fn, lon0=49.9, lat0=45.4, n_lon=60, n_lat=50, step=STEP):
    lon = lon0 + step * np.arange(n_lon)
    lat = lat0 + step * np.arange(n_lat)
    return products.Native(lon, lat, fn(*np.meshgrid(lon, lat)).astype(np.float32))


def _plane(lon, lat):
    return -30.0 + 4.0 * (lon - 50.0) - 3.0 * (lat - 45.5)


# --- sampling -------------------------------------------------------------------------------

def test_bilinear_reproduces_a_plane_and_nearest_takes_the_nearest_centre():
    nat = _native(_plane)
    rng = np.random.default_rng(1)
    lon = rng.uniform(nat.lon[0], nat.lon[-1], 500)
    lat = rng.uniform(nat.lat[0], nat.lat[-1], 500)
    assert np.allclose(products.sample(nat, lon, lat), _plane(lon, lat), atol=1e-4)
    nn = products.sample(nat, lon, lat, "nearest")
    j = np.argmin(np.abs(lon[:, None] - nat.lon[None, :]), axis=1)
    i = np.argmin(np.abs(lat[:, None] - nat.lat[None, :]), axis=1)
    assert np.array_equal(nn, nat.z[i, j])
    with pytest.raises(ValueError, match="method"):
        products.sample(nat, lon, lat, "cubic")


def test_points_beyond_the_grid_and_next_to_nan_are_nan():
    nat = _native(_plane)
    lon_c, lat_c = nat.lon[10] + STEP / 3, nat.lat[10] + STEP / 3
    # bilinear: NaN beyond the outermost centres; nearest: NaN beyond half a cell
    out_b = products.sample(nat, [nat.lon[0] - STEP / 4, nat.lon[-1] + STEP / 4, lon_c], [lat_c] * 3)
    assert np.isnan(out_b[:2]).all() and np.isfinite(out_b[2])
    out_n = products.sample(nat, [nat.lon[0] - STEP / 4, nat.lon[0] - STEP * 0.6], [lat_c] * 2, "nearest")
    assert np.isfinite(out_n[0]) and np.isnan(out_n[1])
    nat.z[11, 11] = np.nan                         # one of the four neighbours of the point
    assert np.isnan(products.sample(nat, [lon_c], [lat_c])[0])
    assert np.isfinite(products.sample(nat, [nat.lon[30] + STEP / 3], [nat.lat[30] + STEP / 3])[0])


def test_irregular_axis_is_rejected():
    nat = _native(_plane)
    nat.lon[5] += STEP / 10
    with pytest.raises(ValueError, match="regular"):
        products.sample(nat, [50.0], [45.5])


def test_to_grid_samples_cell_centres_of_the_equal_area_grid():
    nat = _native(_plane)
    x0, y1 = FWD.transform(nat.lon[5], nat.lat[-5])
    x0, y1 = np.floor(x0 / 60) * 60, np.floor(y1 / 60) * 60
    tr, shape = [60, 0, x0, 0, -60, y1], (40, 50)
    z = products.to_grid(nat, tr, shape, rows_per_block=7)
    assert z.dtype == np.float32 and z.shape == shape
    xs, ys = np.meshgrid(x0 + 60 * (np.arange(50) + 0.5), y1 - 60 * (np.arange(40) + 0.5))
    lon, lat = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform(xs, ys)
    assert np.allclose(z, _plane(lon, lat), atol=1e-4)
    out = np.zeros(shape, np.float32)
    products.to_grid(nat, tr, shape, "nearest", out=out)
    assert np.array_equal(out, products.sample(nat, lon, lat, "nearest"))


# --- loading ----------------------------------------------------------------------------------

def test_load_native_netcdf_and_geotiff(tmp_path):
    import rasterio
    import xarray as xr
    from rasterio.transform import Affine
    lon = 50.0 + STEP * np.arange(6)
    lat = 45.5 + STEP * np.arange(4)
    z = np.arange(24, dtype=np.int16).reshape(4, 6) - 40
    # stored north-up (descending lat), as some products are
    xr.Dataset({"elevation": (("lat", "lon"), z[::-1])}, coords={"lat": lat[::-1], "lon": lon}).to_netcdf(
        tmp_path / "gebco_2026_ncaspian.nc")
    nat = products.load_native("gebco_2026", tmp_path)
    assert np.all(np.diff(nat.lat) > 0) and np.array_equal(nat.z, z.astype(np.float32))
    assert np.allclose(nat.lat, lat) and np.allclose(nat.lon, lon)
    # ETOPO1-style GeoTIFF: 1' pixels centred on whole minutes, band named "bedrock"
    m = 1 / 60
    tif = tmp_path / "etopo1_ncaspian.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=6, height=4, count=2, dtype="int16", crs="EPSG:4326",
                       transform=Affine(m, 0, 50.0 - m / 2, 0, -m, 45.6 + m / 2)) as dst:
        dst.write(z, 1)
        dst.write(z + 1000, 2)
        dst.set_band_description(1, "bedrock")
        dst.set_band_description(2, "ice_surface")
    nat = products.load_native("etopo1", tmp_path)
    assert np.allclose(nat.lon, 50.0 + m * np.arange(6)) and np.allclose(nat.lat, 45.6 - m * np.arange(4)[::-1])
    assert np.array_equal(nat.z, z[::-1].astype(np.float32))      # band 1, flipped to ascending lat


def test_heights_cache_is_keyed_by_grid_and_method(tmp_path):
    import xarray as xr
    nat = _native(_plane)
    xr.Dataset({"elevation": (("lat", "lon"), nat.z)}, coords={"lat": nat.lat, "lon": nat.lon}).to_netcdf(
        tmp_path / "srtm15p_v2.7_ncaspian.nc")
    x0, y1 = FWD.transform(nat.lon[5], nat.lat[-5])
    tr, shape = [60, 0, np.floor(x0 / 60) * 60, 0, -60, np.floor(y1 / 60) * 60], (20, 30)
    cache = tmp_path / "cache"
    h = products.heights("srtm15p_v2.7", tr, shape, root=tmp_path, cache=cache)
    assert isinstance(h, np.memmap) and h.shape == shape
    assert np.array_equal(h, products.to_grid(products.load_native("srtm15p_v2.7", tmp_path), tr, shape))
    files = sorted(cache.glob("*.npy"))
    products.heights("srtm15p_v2.7", tr, shape, root=tmp_path, cache=cache)
    assert sorted(cache.glob("*.npy")) == files                     # reused
    products.heights("srtm15p_v2.7", tr, shape, "nearest", root=tmp_path, cache=cache)
    assert len(list(cache.glob("*.npy"))) == 2
    assert len(list(cache.glob("*.json"))) == 2


# --- vertical offsets ------------------------------------------------------------------------

def test_flood_level_sign_convention():
    rng = np.random.default_rng(7)
    truth = -32 + 0.05 * np.arange(80)[None, :] + rng.normal(0, 0.02, (30, 80))   # analysis datum
    seed = np.zeros(truth.shape, bool)
    seed[:, 0] = True
    h, d_level = -28.0, -0.4                    # DAHITI level and DAHITI -> analysis datum
    expect = hindcast.flood_fill(truth, h + d_level, seed)
    # a product on another reference: z = truth - d_p
    d_p = 0.25
    z = truth - d_p
    assert np.array_equal(hindcast.flood_fill(z, products.flood_level(h, d_level, d_p), seed), expect)
    # a product uniformly 0.3 m too high is matched by the shape constant +0.3
    high = truth + 0.3
    assert not np.array_equal(hindcast.flood_fill(high, products.flood_level(h, d_level), seed), expect)
    assert np.array_equal(hindcast.flood_fill(high, products.flood_level(h, d_level, 0.0, 0.3), seed), expect)
    assert products.flood_level(np.array([-28.0, -27.0]), 0.1).tolist() == pytest.approx([-27.9, -26.9])


# --- the downloaded subsets (smoke test; skipped without data/) -------------------------------

@pytest.mark.parametrize("name", list(products.PRODUCTS))
def test_downloaded_subsets_load_on_regular_grids(name):
    path = products.BATHY / products.PRODUCTS[name]["file"]
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded")
    nat = products.load_native(name)
    step = 1 / 60 if name == "etopo1" else STEP
    assert nat.z.dtype == np.float32 and nat.z.shape == (nat.lat.size, nat.lon.size)
    assert np.allclose(np.diff(nat.lon), step, atol=1e-6) and np.allclose(np.diff(nat.lat), step, atol=1e-6)
    assert nat.lon[0] < 46.41 and nat.lon[-1] > 54.59 and nat.lat[0] < 43.98 and nat.lat[-1] > 47.34


# --- the geoid-difference correction (total error, D-070) -------------------------------------

def _dN(lon, lat):
    return 0.2 + 0.05 * (lon - 50.0) - 0.04 * (lat - 45.5)


def _grid_near(nat, shape=(20, 30)):
    x0, y1 = FWD.transform(nat.lon[5], nat.lat[-5])
    return [60, 0, np.floor(x0 / 60) * 60, 0, -60, np.floor(y1 / 60) * 60], shape


def test_correction_from_npz_and_geotiff_is_resampled_bilinearly(tmp_path):
    import rasterio
    from rasterio.transform import Affine
    nat = _native(_dN, step=1 / 60, n_lon=40, n_lat=30, lon0=49.7, lat0=45.2)
    np.savez(tmp_path / "dn.npz", lon=nat.lon, lat=nat.lat[::-1], dN=nat.z[::-1],
             definition="N(EGM2008) - N(EIGEN-6C4)")
    tr, shape = _grid_near(nat)
    c1, rec1 = products.correction(tmp_path / "dn.npz", tr, shape, cache=tmp_path / "c")
    assert rec1["definition"] == "N(EGM2008) - N(EIGEN-6C4)" and len(rec1["sha256"]) == 64
    xs, ys = np.meshgrid(tr[2] + 60 * (np.arange(30) + 0.5), tr[5] - 60 * (np.arange(20) + 0.5))
    lon, lat = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform(xs, ys)
    assert np.allclose(c1, _dN(lon, lat), atol=1e-5)
    m = 1 / 60
    with rasterio.open(tmp_path / "dn.tif", "w", driver="GTiff", width=40, height=30, count=1, dtype="float32",
                       crs="EPSG:4326", transform=Affine(m, 0, nat.lon[0] - m / 2, 0, -m, nat.lat[-1] + m / 2)) as d:
        d.write(nat.z[::-1], 1)
    c2, _ = products.correction(tmp_path / "dn.tif", tr, shape, cache=tmp_path / "c")
    assert np.allclose(c1, c2, atol=1e-6)
    with rasterio.open(tmp_path / "utm.tif", "w", driver="GTiff", width=4, height=3, count=1, dtype="float32",
                       crs="EPSG:32639", transform=Affine(100, 0, 0, 0, -100, 0)) as d:
        d.write(np.zeros((3, 4), np.float32), 1)
    with pytest.raises(ValueError, match="EPSG:4326"):
        products.load_correction(tmp_path / "utm.tif")


def test_corrected_heights_add_the_geoid_difference():
    # a product given as heights above EGM2008: adding N_EGM2008 - N_EIGEN gives heights above EIGEN-6C4
    rng = np.random.default_rng(2)
    h_ell = rng.normal(-5, 1, (6, 7))
    n_egm, n_eigen = rng.normal(-14, 0.2, (6, 7)), rng.normal(-14, 0.2, (6, 7))
    z_egm = (h_ell - n_egm).astype(np.float32)
    dn = (n_egm - n_eigen).astype(np.float32)
    v = products.Corrected(z_egm, dn)
    assert v.shape == (6, 7) and v[1:3, 2:5].dtype == np.float32
    assert np.allclose(v[:, :], h_ell - n_eigen, atol=1e-5)
    with pytest.raises(ValueError, match="same grid"):
        products.Corrected(z_egm, dn[1:])
    assert products.flood_level(-28.0, shape_offset_m=np.array([-0.01, 0.0, 0.3])).tolist() == pytest.approx(
        [-28.01, -28.0, -27.7])
    assert float(products.flood_level(-28.05, 0.0, 0.0, 0.3)) == products.flood_level(-28.05, 0.0, 0.0,
                                                                                         np.array([0.3]))[0]
