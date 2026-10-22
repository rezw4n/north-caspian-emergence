"""The audited bathymetry products as heights on the hindcast grid (PREREG
sections 2, 5 and 9; Amendment 1).

Products, as subset by scripts/a7_get_*.py into data/raw/bathymetry/ (provenance in
results/provenance/<name>.json):

  name           file                       layer      native grid
  gebco_2026     gebco_2026_ncaspian.nc     elevation  15", int16 m, cell-centre lon/lat
  etopo_2022     etopo_2022_ncaspian.nc     elevation  15", float32 m, cell-centre lon/lat
  srtm15p_v2.7   srtm15p_v2.7_ncaspian.nc   elevation  15", float32 m, cell-centre lon/lat
  gebco_2022     gebco_2022_ncaspian.nc     elevation  15", int16 m, cell-centre lon/lat
  etopo1         etopo1_ncaspian.tif        bedrock    1', int16 m, grid-registered (the
                                                       GeoTIFF's pixels are centred on the
                                                       whole arc-minutes)

The first three are the current-product audit, the last two the grids of published
Caspian studies (Amendment 1). GEBCO and ETOPO1 are distributed in whole metres.

Vertical handling (PREREG section 5, docs/DATUMS.md). The working datum is
the DAHITI reference: the level bins, the waterlines and the hindcast all live on
it (BHS-77 is the reporting datum; a constant DAHITI -> BHS offset cancels in every
hindcast metric). Heights are read as distributed: metres, positive up, no shift.
ETOPO 2022's `geoid_height` layer is not applied (it converts the EGM2008 heights to
ellipsoidal heights).
- Total error (PREREG 5a): the products are read as heights above EGM2008 (ETOPO
  2022 documents it; GEBCO, SRTM15+ and ETOPO1 are assumed so) and brought to
  the DAHITI reference cell by cell by adding N(EGM2008) - N(EIGEN-6C4)
  (``correction``, ``Corrected``). Constant offsets act on the level instead
  (``flood_level``): ``level_datum_offset_m`` (default 0 on the DAHITI working
  datum) and ``product_offset_m`` (default 0).
- Shape error (PREREG 5b): after removing one constant c per product from its
  (corrected) heights; c is fitted by scripts/a7_hindcast.py on the flood-level map
  and enters here as ``shape_offset_m``.

Correction raster (the input that another process provides). The value to ADD to a
product height above EGM2008 to obtain the height on the DAHITI reference, i.e.
N(EGM2008) - N(EIGEN-6C4) in metres (H_EIGEN = h - N_EIGEN = H_EGM + N_EGM - N_EIGEN),
on a regular lon/lat (EPSG:4326) grid that covers the product subsets (DOMAIN
bounds +- 0.1 deg), values at cell centres. Either
- a GeoTIFF: EPSG:4326, north-up, one band (or a band described "dN"), metres,
  pixel-is-area georeferencing (centres at the corner plus half a pixel), nodata
  allowed; or
- an .npz with `lon` (1-D, cell centres), `lat` (1-D, cell centres) and `dN` (2-D,
  [lat, lon], metres); an optional `definition` string is recorded.
It is resampled to the hindcast grid bilinearly, like the products.

Resampling. Each hindcast cell centre (equal-area CRS, config.CRS_EQUAL_AREA) is
transformed to lon/lat, and the product is interpolated there, bilinearly between
the four surrounding native cell centres (default), or by nearest neighbour.
- Bilinear is the primary method. The 60 m hindcast cell is 5 to 8 times smaller
  than the 15" cells (about 320 x 460 m here) and 20 to 30 times smaller than ETOPO1's
  1' cells. Nearest neighbour would give every product a staircase shoreline with
  steps one native cell wide, adding up to half a native cell of displacement
  (about 230 m at 15", 930 m for ETOPO1) whatever the product's accuracy, and its
  shoreline would depend on how the 60 m grid falls on the native one. Bilinear
  treats the values as samples at the cell centres, which is how the products
  document their registration, and adds no structure beyond them (bicubic can
  overshoot between samples).
- Interpolation is in the native lon/lat index space, so that each native cell is
  read with the weights of its own grid.
- A target cell whose four neighbours include a NaN, or that lies beyond the
  outermost native cell centres, is NaN (no product state; pass ``isfinite`` as
  ``pred_valid``). For nearest neighbour, NaN beyond half a cell outside.
- Nearest neighbour is kept as a sensitivity (``method="nearest"``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyproj

from .config import CRS_EQUAL_AREA, INTERIM, RAW

BATHY = RAW / "bathymetry"

# docs/DATUMS.md and results/provenance/<name>.json; "reference" is what the product's
# own documentation or file says, not an estimate
PRODUCTS = {
    "gebco_2026": {"file": "gebco_2026_ncaspian.nc", "layer": "elevation",
                   "reference": "mean sea level as stated by GEBCO; unresolved in the Caspian (docs/DATUMS.md)"},
    "etopo_2022": {"file": "etopo_2022_ncaspian.nc", "layer": "elevation",
                   "reference": "EGM2008 (EPSG:3855) per the file attributes; GEBCO 2022 base layer in the Caspian"},
    "srtm15p_v2.7": {"file": "srtm15p_v2.7_ncaspian.nc", "layer": "elevation",
                     "reference": "not stated by the distribution (docs/DATUMS.md)"},
    "gebco_2022": {"file": "gebco_2022_ncaspian.nc", "layer": "elevation",
                   "reference": "mean sea level as stated by GEBCO; unresolved in the Caspian (docs/DATUMS.md)"},
    "etopo1": {"file": "etopo1_ncaspian.tif", "layer": "bedrock",
               "reference": "not recorded in docs/DATUMS.md"},
}
METHODS = ("bilinear", "nearest")


@dataclass
class Native:
    """A product on its own regular lon/lat grid: ``z[i, j]`` at (lat[i], lon[j]),
    cell centres, both axes ascending; float32 metres, positive up, as distributed."""
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray


def _ascending(lon, lat, z):
    lon, lat, z = np.asarray(lon, float), np.asarray(lat, float), np.asarray(z, np.float32)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    if lon[0] > lon[-1]:
        lon, z = lon[::-1], z[:, ::-1]
    return Native(lon, lat, np.ascontiguousarray(z))


def load_native(name: str, root: Path = BATHY) -> Native:
    """A product's heights on its native grid, as distributed (float32 m)."""
    spec = PRODUCTS[name]
    path = Path(root) / spec["file"]
    if path.suffix == ".tif":
        import rasterio
        with rasterio.open(path) as r:
            band = list(r.descriptions).index(spec["layer"]) + 1
            z = r.read(band, masked=True).astype(np.float32).filled(np.nan)
            t = r.transform
            if t.b or t.d:
                raise ValueError(f"{path.name}: rotated grid")
            lon = t.c + t.a * (np.arange(r.width) + 0.5)     # pixel centres
            lat = t.f + t.e * (np.arange(r.height) + 0.5)
        return _ascending(lon, lat, z)
    import xarray as xr
    with xr.open_dataset(path) as ds:   # mask_and_scale: a declared _FillValue becomes NaN
        da = ds[spec["layer"]].transpose("lat", "lon")
        return _ascending(ds["lon"].values, ds["lat"].values, da.values)


def _axis(v: np.ndarray, what: str) -> tuple[float, float]:
    """(first value, step) of a regular ascending axis."""
    step = (v[-1] - v[0]) / (v.size - 1)
    if not np.allclose(np.diff(v), step, rtol=0, atol=1e-6 * abs(step)):
        raise ValueError(f"{what} axis is not regular")
    return float(v[0]), float(step)


def sample(native: Native, lon, lat, method: str = "bilinear") -> np.ndarray:
    """The product at lon/lat points (any shape), float32; see the module docstring
    for the NaN rules."""
    lon0, dlon = _axis(native.lon, "lon")
    lat0, dlat = _axis(native.lat, "lat")
    fj = (np.asarray(lon, float) - lon0) / dlon
    fi = (np.asarray(lat, float) - lat0) / dlat
    ni, nj = native.z.shape
    z = native.z
    if method == "nearest":
        i, j = np.floor(fi + 0.5).astype(np.int64), np.floor(fj + 0.5).astype(np.int64)
        inside = (i >= 0) & (i < ni) & (j >= 0) & (j < nj)
        out = z[np.clip(i, 0, ni - 1), np.clip(j, 0, nj - 1)].astype(np.float32)
    elif method == "bilinear":
        inside = (fi >= 0) & (fi <= ni - 1) & (fj >= 0) & (fj <= nj - 1)
        i0 = np.clip(np.floor(fi), 0, max(ni - 2, 0)).astype(np.int64)
        j0 = np.clip(np.floor(fj), 0, max(nj - 2, 0)).astype(np.int64)
        ti, tj = fi - i0, fj - j0
        i1, j1 = np.minimum(i0 + 1, ni - 1), np.minimum(j0 + 1, nj - 1)
        zz = z.astype(np.float64)
        out = ((1 - ti) * (1 - tj) * zz[i0, j0] + (1 - ti) * tj * zz[i0, j1]
               + ti * (1 - tj) * zz[i1, j0] + ti * tj * zz[i1, j1]).astype(np.float32)
    else:
        raise ValueError(f"method must be one of {METHODS}")
    out[~inside] = np.nan
    return out


def to_grid(native: Native, transform, shape, method: str = "bilinear", crs: str = CRS_EQUAL_AREA,
            out=None, rows_per_block: int = 256) -> np.ndarray:
    """The product on a north-up grid (``transform`` [a, 0, c, 0, e, f] in ``crs``,
    ``shape`` (rows, cols)), sampled at the cell centres; float32. ``out`` may be a
    preallocated array or memory map. Works in blocks of rows, so that the lon/lat of
    only one block is held at a time."""
    a, b, c, d, e, f = (float(v) for v in transform[:6])
    if b or d:
        raise ValueError("rotated grids are not supported")
    rows, cols = shape
    out = np.empty(shape, np.float32) if out is None else out
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs = c + a * (np.arange(cols) + 0.5)
    for r0 in range(0, rows, rows_per_block):
        r1 = min(r0 + rows_per_block, rows)
        ys = f + e * (np.arange(r0, r1) + 0.5)
        lon, lat = inv.transform(*np.meshgrid(xs, ys))
        out[r0:r1] = sample(native, lon, lat, method)
    return out


def heights(name: str, transform, shape, method: str = "bilinear", root: Path = BATHY,
            cache: Path = INTERIM / "hindcast" / "heights") -> np.ndarray:
    """The product on the hindcast grid, read-only memory map of a cached float32 .npy
    (git-ignored data/). The cache key covers the raw file's sha256, the grid and the
    method, so a changed input never reuses a stale grid."""
    src = Path(root) / PRODUCTS[name]["file"]
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    meta = {"product": name, "source": src.name, "source_sha256": sha, "method": method,
            "transform": [float(v) for v in transform[:6]], "shape": [int(s) for s in shape],
            "crs": CRS_EQUAL_AREA, "vertical": "as distributed (no offset applied)",
            "reference": PRODUCTS[name]["reference"]}
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}_{method}_{key}.npy"
    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
        to_grid(load_native(name, root), transform, shape, method, out=mm)
        mm.flush()
        del mm
        tmp.replace(path)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return np.load(path, mmap_mode="r")


def load_correction(path) -> tuple[Native, dict]:
    """The correction raster (see the module docstring) as a Native grid, and a record
    (file, sha256, definition)."""
    path = Path(path)
    rec = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "meaning": "N(EGM2008) - N(EIGEN-6C4), m, added to product heights"}
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as r:
            if r.crs is None or r.crs.to_epsg() != 4326:
                raise ValueError(f"{path.name}: the correction raster must be in EPSG:4326")
            desc = list(r.descriptions)
            if "dN" in desc:
                band = desc.index("dN") + 1
            elif r.count == 1:
                band = 1
            else:
                raise ValueError(f"{path.name}: several bands and none described 'dN'")
            t = r.transform
            if t.b or t.d:
                raise ValueError(f"{path.name}: rotated grid")
            z = r.read(band, masked=True).astype(np.float32).filled(np.nan)
            lon = t.c + t.a * (np.arange(r.width) + 0.5)
            lat = t.f + t.e * (np.arange(r.height) + 0.5)
        return _ascending(lon, lat, z), rec
    with np.load(path) as d:
        if "definition" in d.files:
            rec["definition"] = str(d["definition"])
        return _ascending(d["lon"], d["lat"], d["dN"]), rec


def correction(path, transform, shape, cache: Path = INTERIM / "hindcast" / "correction") -> tuple[np.ndarray, dict]:
    """The correction raster on the hindcast grid (bilinear, float32 memory map of a
    cached .npy keyed by the file's sha256 and the grid), and its record."""
    native, rec = load_correction(path)
    key = hashlib.sha256(json.dumps({**rec, "transform": [float(v) for v in transform[:6]],
                                     "shape": [int(s) for s in shape]}, sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"correction_{key}.npy"
    if not out.exists():
        tmp = out.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
        to_grid(native, transform, shape, "bilinear", out=mm)
        mm.flush()
        del mm
        tmp.replace(out)
    return np.load(out, mmap_mode="r"), {**rec, "cache": out.name}


class Corrected:
    """Product heights plus the correction raster, read window by window (both on the
    hindcast grid, float32): what the flood fill, the flood-level map and the
    finite-height mask all read, so they see the same values."""

    def __init__(self, heights, corr):
        if tuple(np.shape(heights)) != tuple(np.shape(corr)):
            raise ValueError("heights and correction must be on the same grid")
        self.h, self.c = heights, corr
        self.shape, self.dtype = tuple(np.shape(heights)), np.dtype(np.float32)

    def __getitem__(self, key):
        return np.asarray(self.h[key], np.float32) + np.asarray(self.c[key], np.float32)


def flood_level(bin_level_m, level_datum_offset_m: float = 0.0, product_offset_m: float = 0.0,
                shape_offset_m: float = 0.0):
    """Level on the product's own vertical scale at which to flood its heights.

    With h the observed level on the DAHITI scale (the stage-2 bin centres),
    delta_l = DAHITI -> working datum (``level_datum_offset_m``; 0 on the DAHITI
    working datum), z the product height as read (as distributed, or
    ``Corrected`` to DAHITI cell by cell), delta_p = any further constant from the
    product's reference to the working datum (``product_offset_m``, default 0) and
    c the constant removed from the product's heights for the shape variant
    (``shape_offset_m``), a cell is below the level when

        z + delta_p - c < h + delta_l   <=>   z < h + delta_l - delta_p + c.

    So a product that is uniformly 0.3 m too high is matched by c = +0.3. ``shape_offset_m``
    may be an array (the search grid); each element is computed in the same order
    of operations as the scalar case, so a fitted c gives bit-identical levels."""
    return (np.asarray(bin_level_m, float) + float(level_datum_offset_m) - float(product_offset_m)
            + np.asarray(shape_offset_m, float))
