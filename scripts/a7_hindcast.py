"""A7: hindcast audit driver, per-unit metrics for H1 (PREREG section 9 with
Amendments 6, 7 and 10; D-030, D-032, D-033, D-050, D-067, D-070).

For each level bin in the stage-2 counts (data/interim/stage2/<tag>/tile_*.npz,
scripts/a6_stage2_export.py):

1. Observed extents. For each epoch the two acquisition-parity halves are summed
   (caspian.frequency.combine); a cell is valid with at least design.MIN_VALID_OBS
   valid acquisitions and water where its frequency is >= design.CONTOUR_LEVEL
   (PREREG section 4). Each half alone, and E2 + E3 summed at bins shared with E1,
   are built the same way, each with its own valid mask (Amendment 7).
2. Sea connection (Amendment 7 item 2, D-033; hindcast_tiled.sea_classes): observed
   water counts as water only if it is 8-connected through known water to the
   open-sea seed. Unconnected water is land if bordered only by known land, unknown
   if it touches an unknown cell. Class areas per sector go to observed.csv.
3. Predicted extents. Each product's flood-level map F (hindcast_tiled.
   flood_level_map: the minimax-path level from the open-sea seed, so that the flood
   fill at level L is exactly F < L) is thresholded at the bin centre on the
   product's scale (caspian.products.flood_level). Variant "total" (PREREG 5a): the
   heights brought to the DAHITI working datum by the per-cell geoid correction
   (--vertical-correction; D-070), no constant. Variant "shape" (PREREG 5b): with the
   constant c of D-070 (step 4), or one supplied with --shape-offsets.
4. Shape offset (D-070, fit_shape): per product, c on a 0.01 m grid over +-5 m that
   minimises the sum over all (epoch, level) units of |area bias| (km2) over known
   cells inside the domain and in training blocks (blocks_25km.csv, split == train;
   a cell takes the 25 km block that contains its centre); ties go to the smallest
   |c|, then to the smaller c. Each unit's bias at every c is a count of F values
   below a threshold, so the search needs no flood fill. The fit, its full objective
   curve and the refits without each epoch (leave-one-epoch-out, PREREG section 10)
   go to shape_offsets.json; --fit-exclude-epoch runs the shape variant with such a
   refit.
5. Metrics per sector, for the whole grid ("all"), and for the area metrics also for
   the domain only ("domain") and the fit region ("domain_train"): area bias, IoU
   and F1 within the valid mask and with the known outside states (extent.csv; every
   cell count kept), and the symmetric shoreline displacement with every count
   (displacement.csv); per 25 km block and unit (blocks.csv.gz).
6. Noise floor (noise_floor.csv): the two parity halves per epoch and bin (A =
   parity 0), and E1 vs E2 + E3 at shared bins (A = E1). A positive signed median
   means more water in A.
7. Pooled rows (D-032): per bin, the epochs' distances pooled for each product, and
   the noise-floor distances (halves of every epoch, and E1 vs E2 + E3) pooled; per
   sector, the noise floor pooled over every level and epoch (the sensitivity).
8. H1 block values (D-070; block_h1.csv): per product, variant and block, the median
   of the absolute displacement distances pooled over all units, with their number
   (caspian.h1.block_test applies the 20-distance minimum and the permutation test).

Static layers, on the 60 m equal-area grid that covers the product subsets (DOMAIN
bounds +- 0.1 deg; aligned to multiples of 60 m like the stage-2 tiles):
- domain: DOMAIN with its lon/lat edges densified to 0.01 deg (D-039), rasterised
  by cell centre; zone: the buffered stage-2 cover (D-060; straight edges in the
  equal-area CRS, as scripts/a6_stage2_export.py reads it), clipped to the domain;
- sectors: caspian.geography, densified likewise, extended beyond the domain to
  the nearest sector (nearest_region, D-050);
- open-sea seed (D-070): the D-029 open-sea mask (results/study_area/
  sea_mask_1km.npz), nearest neighbour, outside the zone inside the domain less its
  one-cell rim;
- beyond-domain state: results/study_area/beyond_state_250m.npz, nearest neighbour,
  -1 -> unknown (D-067);
- land seeds (D-070): the beyond-domain state's land cells inside the domain (its
  5 km inner band), outside the zone and the rim;
- outside states: hindcast_tiled.outside_state (= hindcast.outside_state_from_zone),
  which stops with an error if the open sea and a land seed connect outside the zone.

Memory: the grid has about 75 million cells. Whole-grid arrays are kept only as
1-byte masks and codes; heights, corrections and flood-level maps are read from
memory-mapped caches under data/interim/hindcast/; connectivity, waterlines and
counts are computed tile by tile (caspian.hindcast_tiled, exact: labels joined
across tile seams, nearest-waterline distances by k-d tree over the whole grid, no
margins). Measured on a synthetic grid of the full size (2048-cell tiles): a flood
fill about 4 s, a sea connection about 8 s, the area counts and the waterline
distances about 5 s each, with about 0.5 GB of temporary arrays.

Not computed here: any H1 outcome (caspian.h1 applies the rule to these tables).

Usage:
  python scripts/a7_hindcast.py <tag> [--vertical-correction <tif|npz>]
      [--level-datum-offset <m> --level-datum-source <text>]
      [--products gebco_2026,etopo_2022,...] [--product-offsets <json>]
      [--shape-offsets <json> | --no-shape] [--fit-exclude-epoch E1|E2|E3]
      [--resampling bilinear|nearest] [--tile 2048]

Offsets JSON: {"<product>": {"offset_m": <float>, "source": "<where it comes from>"}}.
Outputs: results/hindcast/<tag>/{extent,displacement,noise_floor,observed,block_h1}.csv,
blocks.csv.gz, shape_offsets.json (when fitted) and run.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import shapely
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape as as_shape

from caspian import design, frequency, levels, products
from caspian import hindcast_tiled as ht
from caspian.config import CRS_EQUAL_AREA, INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER, sectors

BAND = re.compile(r"([wv])_(E[123])_b(\d{3})_p([01])$")
RES = 60                  # stage-2 grid (scripts/a6_stage2_export.py)
MARGIN_DEG = 0.1          # the product subsets' margin (scripts/a7_get_*.py)
DENSIFY_DEG = 0.01        # D-039
CELL_KM2 = RES * RES / 1e6
C_GRID = np.round(np.arange(-500, 501) / 100.0, 2)    # D-070: shape offsets, 0.01 m over +-5 m
DATUM_SOURCE = "DAHITI working datum (D-070): no level offset"
STUDY = RESULTS / "study_area"
COVER = STUDY / "zone_v6_cover_buffered.geojson"
SEA = STUDY / "sea_mask_1km.npz"
BEYOND = STUDY / "beyond_state_250m.npz"
BLOCKS = STUDY / "blocks_25km.csv"
FWD = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)


# --- grid and static layers ----------------------------------------------------------------

def grid(bounds_lonlat, res: float = RES):
    """Transform [res, 0, x0, 0, -res, y1] and shape (rows, cols) of the equal-area grid
    aligned to multiples of ``res`` that covers a lon/lat box (as
    pipeline_ee.laea_transform)."""
    w, s, e, n = bounds_lonlat
    xs, ys = FWD.transform([w, e, w, e], [s, s, n, n])
    x0, x1 = math.floor(min(xs) / res) * res, math.ceil(max(xs) / res) * res
    y0, y1 = math.floor(min(ys) / res) * res, math.ceil(max(ys) / res) * res
    return [res, 0, x0, 0, -res, y1], (int(round((y1 - y0) / res)), int(round((x1 - x0) / res)))


def to_laea(geom_lonlat, densify_deg: float | None = DENSIFY_DEG):
    """A lon/lat geometry in the equal-area CRS. Densified first when its edges are
    straight in lon/lat (DOMAIN and sectors, D-039); not when they are straight in the
    equal-area CRS (the stage-2 cover)."""
    g = geom_lonlat.segmentize(densify_deg) if densify_deg else geom_lonlat
    return shapely.transform(g, lambda xy: np.column_stack(FWD.transform(xy[:, 0], xy[:, 1])))


def burn(shapes, transform, shape) -> np.ndarray:
    """Rasterise (geometry, value) pairs by cell centre, later shapes on top; uint8."""
    return rasterize(list(shapes), out_shape=shape, transform=Affine(*transform[:6]), fill=0, dtype="uint8")


def nearest(arr, src_transform, transform, shape, fill) -> np.ndarray:
    """Nearest-neighbour resampling between north-up grids of the same CRS: each target
    cell takes the source cell that contains its centre; ``fill`` outside the source."""
    a, _, c, _, e, f = (float(v) for v in src_transform[:6])
    ta, _, tc, _, te, tf = (float(v) for v in transform[:6])
    rows, cols = shape
    j = np.floor((tc + ta * (np.arange(cols) + 0.5) - c) / a).astype(np.int64)
    i = np.floor((tf + te * (np.arange(rows) + 0.5) - f) / e).astype(np.int64)
    oi, oj = (i >= 0) & (i < arr.shape[0]), (j >= 0) & (j < arr.shape[1])
    out = np.array(np.asarray(arr)[np.clip(i, 0, arr.shape[0] - 1)[:, None],
                                   np.clip(j, 0, arr.shape[1] - 1)[None, :]])
    out[~(oi[:, None] & oj[None, :])] = fill
    return out


def check_crs(wkt: str, what: str):
    if not pyproj.CRS(str(wkt)).equals(pyproj.CRS(CRS_EQUAL_AREA), ignore_axis_order=True):
        raise ValueError(f"{what}: CRS differs from config.CRS_EQUAL_AREA")


def block_index(transform, shape, block_m: float):
    """Block indices (i of each column, j of each row) of the ``block_m`` squares on the
    equal-area grid that contain the cell centres (caspian.blocking.block_index)."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    rows, cols = shape
    return (np.floor((c + a * (np.arange(cols) + 0.5)) / block_m).astype(np.int64),
            np.floor((f + e * (np.arange(rows) + 0.5)) / block_m).astype(np.int64))


def train_mask(blocks: pd.DataFrame, transform, shape, block_m: float = design.BLOCK_SIZE_M) -> np.ndarray:
    """Cells whose block is a training block (``split`` == "train"); a cell takes the
    block that contains its centre. Cells in blocks absent from the table are not."""
    bi, bj = block_index(transform, shape, block_m)
    lut = np.zeros((bj.max() - bj.min() + 1, bi.max() - bi.min() + 1), bool)
    t = blocks[blocks["split"] == "train"]
    ii, jj = t["i"].to_numpy() - bi.min(), t["j"].to_numpy() - bj.min()
    ok = (ii >= 0) & (ii < lut.shape[1]) & (jj >= 0) & (jj < lut.shape[0])
    lut[jj[ok], ii[ok]] = True
    return lut[(bj - bj.min())[:, None], (bi - bi.min())[None, :]]


COAST_BAND_M = 2000     # D-083: shorelines of the zone-v6 extents dilated by the zone dilation
RIM_M = 2000            # D-083: domain rim whose cells take the beyond-domain state


def coast_band(extents: dict, transform, shape, band_m: float = COAST_BAND_M) -> np.ndarray:
    """Cells within ``band_m`` of a shoreline of the 1995-96 or the 2025-26 zone-v6
    extent (D-083). Added to the zone for the outside states only: where the sea meets
    land outside the stage-2 cover (steep coasts whose shoreline barely moved, the
    cover's ends at the domain edge), the state is unknown rather than leaking sea into
    the hinterland. ``extents`` = {"high", "low", "transform"} (250 m grid)."""
    from scipy import ndimage
    four = ndimage.generate_binary_structure(2, 1)

    def shore(m):
        return m & ~ndimage.binary_erosion(m, structure=four, border_value=1)
    hi, lo = np.asarray(extents["high"]) == 1, np.asarray(extents["low"]) == 1
    res = abs(float(extents["transform"][0]))
    band = ndimage.binary_dilation(shore(hi) | shore(lo) | shore(~hi) | shore(~lo),
                                   iterations=max(1, int(round(band_m / res))))
    return nearest(band.astype(np.int8), extents["transform"], transform, shape, 0) > 0


def static_layers(transform, shape, *, domain, sector_polys: dict, cover, sea: dict, beyond: dict,
                  tile: int = ht.TILE, coast=None, rim_cells: int = 1) -> dict:
    """Static layers on the grid (see the module docstring). Geometries in the equal-area
    CRS; ``sea`` = {"mask", "transform"}, ``beyond`` = {"state" (0 land, -1 unknown),
    "transform"}. Sector codes are 1, 2, ... in the order of ``sector_polys``."""
    dom = burn([(domain, 1)], transform, shape) > 0
    zone = (burn([(cover, 1)], transform, shape) > 0) & dom
    if coast is not None:                     # D-083: the coastal band joins the zone (outside states only)
        zone |= np.asarray(coast, bool) & dom
    lab = burn([(g, k + 1) for k, g in enumerate(sector_polys.values())], transform, shape)
    lab[~dom] = 0
    gaps = dom & (lab == 0)                   # cell centres on a shared edge, if any
    if gaps.any():
        lab = np.where(gaps, ht.nearest_region(lab, lab > 0, tile=tile), lab)
    sector = ht.nearest_region(lab, dom, tile=tile)
    rim_in = ht.eroded(dom, rim_cells, tile=tile)
    sea_seed = (nearest(sea["mask"], sea["transform"], transform, shape, 0) > 0) & ~zone & rim_in
    b = nearest(beyond["state"], beyond["transform"], transform, shape, -1).astype(np.int8)
    land_seed = (b == 0) & ~zone & rim_in
    outside = ht.outside_state(zone, sea_seed, land_seed, domain=dom, beyond_state=b, tile=tile,
                               domain_margin=rim_cells)
    del rim_in, b
    return {"domain": dom, "zone": zone, "sector": sector, "sea_seed": sea_seed, "land_seed": land_seed,
            "outside": outside,
            "counts": {"domain": int(dom.sum()), "zone": int(zone.sum()), "sector_gaps_filled": int(gaps.sum()),
                       "sea_seed": int(sea_seed.sum()), "land_seed": int(land_seed.sum()),
                       "outside_water": int((outside == 1).sum()), "outside_land": int((outside == 0).sum()),
                       "outside_unknown": int((outside == -1).sum())}}


def production_static(transform, shape, tile: int) -> dict:
    # the committed cover is an invalid MultiPolygon: make each part valid first (D-080)
    cover = shapely.union_all([shapely.make_valid(as_shape(f["geometry"])) for f in
                               json.loads(COVER.read_text(encoding="utf-8"))["features"]])
    sea_npz, beyond_npz = np.load(SEA), np.load(BEYOND)
    check_crs(sea_npz["crs"], SEA.name)
    check_crs(beyond_npz["crs"], BEYOND.name)
    return static_layers(
        transform, shape, domain=to_laea(DOMAIN),
        sector_polys={k: to_laea(p) for k, (_, p) in sectors().items()},
        cover=to_laea(cover, None),
        sea={"mask": sea_npz["mask"], "transform": sea_npz["transform"]},
        beyond={"state": beyond_npz["state"], "transform": beyond_npz["transform"]}, tile=tile,
        coast=coast_band(np.load(STUDY / "extents_v6_250m.npz"), transform, shape),
        rim_cells=int(round(RIM_M / RES)))


# --- stage-2 counts --------------------------------------------------------------------------

def stage2_index(src: Path) -> dict:
    """{bin: {epoch: {parity: [tile files]}}} from the band names (main-export groups)."""
    units: dict = {}
    for f in sorted(Path(src).glob("tile_*.npz")):
        with np.load(f) as z:
            names = set(z.files)
        for k in names:
            m = BAND.match(k)
            if m and m.group(1) == "w" and f"v{k[1:]}" in names:
                ep, b, p = m.group(2), int(m.group(3)), int(m.group(4))
                units.setdefault(b, {}).setdefault(ep, {}).setdefault(p, []).append(f)
    return units


ENVELOPE = None          # record-low water (D-082); set by main unless --no-envelope
ENVELOPE_LOW_BINS = 3    # the lowest E3 bins pooled for the record-low extent


def record_low_water(units: dict, transform, shape) -> np.ndarray:
    """Cells that are water at the record low (D-082): the observed extent of the
    ENVELOPE_LOW_BINS lowest E3 level bins, counts pooled over the bins and both halves.
    A cell under water at the lowest level of the record is under water at every higher
    level, so such a cell observed dry at a higher bin is an observation error (dark,
    clear deep water whose MNDWI scatters around the scene threshold)."""
    e3 = sorted(b for b, eps in units.items() if "E3" in eps)
    groups: dict = {}
    for b in e3[:ENVELOPE_LOW_BINS]:
        for f, g in groups_for(units, b, ["E3"]).items():
            groups.setdefault(f, []).extend(g)
    code, _ = observed_raw(groups, transform, shape, envelope=False)
    return code == 1


def observed_raw(groups_by_tile: dict, transform, shape, envelope: bool = True) -> tuple[np.ndarray, dict]:
    """Observed extent of a set of groups, mosaicked on the grid: int8 -2 not valid,
    0 dry, 1 water. Per stage-2 tile the groups' counts are summed (frequency.combine).
    Adjacent tiles can share a column or row of cells; where two tiles both have a
    valid cell, the first is kept and the overlap is counted (and how often the two
    disagree), never summed."""
    code = np.full(shape, -2, np.int8)
    info = {"overlap_cells": 0, "overlap_disagree": 0, "max_valid_count": 0}
    x0, y1 = float(transform[2]), float(transform[5])
    for f, groups in groups_by_tile.items():
        with np.load(f) as z:
            tr = z["transform"]
            counts = {g: (z[f"w_{g}"], z[f"v_{g}"]) for g in groups}
        w, v = frequency.combine(counts, groups)
        c0, r0 = (float(tr[2]) - x0) / RES, (y1 - float(tr[5])) / RES
        if float(tr[0]) != RES or float(tr[4]) != -RES or c0 != round(c0) or r0 != round(r0):
            raise ValueError(f"{Path(f).name}: not on the {RES} m grid of the hindcast")
        r0, c0 = int(round(r0)), int(round(c0))
        if r0 < 0 or c0 < 0 or r0 + v.shape[0] > shape[0] or c0 + v.shape[1] > shape[1]:
            raise ValueError(f"{Path(f).name}: outside the hindcast grid")
        valid = v >= design.MIN_VALID_OBS
        water = valid & (np.nan_to_num(frequency.water_frequency(w, v), nan=-1.0) >= design.CONTOUR_LEVEL)
        dst = code[r0:r0 + v.shape[0], c0:c0 + v.shape[1]]
        both = valid & (dst > -2)
        info["overlap_cells"] += int(both.sum())
        info["overlap_disagree"] += int((both & ((dst == 1) != water)).sum())
        new = valid & (dst == -2)
        dst[new] = water[new]
        info["max_valid_count"] = max(info["max_valid_count"], int(v.max()) if v.size else 0)
    if envelope and ENVELOPE is not None:
        fix = (code == 0) & ENVELOPE
        info["envelope_cells"] = int(fix.sum())
        code[fix] = 1
    return code, info


def groups_for(units: dict, b: int, epochs, parities=(0, 1)) -> dict:
    """{tile file: [group names]} for these epochs and parities of bin b."""
    out: dict = {}
    for ep in epochs:
        for p in parities:
            for f in units[b].get(ep, {}).get(p, []):
                out.setdefault(f, []).append(f"{ep}_b{b:03d}_p{p}")
    return out


def _bin_level(b: int) -> float:
    return float(levels.bin_center(b, design.BIN_WIDTH_M, design.BIN_ORIGIN_M))


# --- per-unit rows ---------------------------------------------------------------------------

def _valid(code):
    return lambda r0, r1, c0, c1: code[r0:r1, c0:c1] > -2


def _water(code):
    return lambda r0, r1, c0, c1: code[r0:r1, c0:c1] == 1


def classify(raw, static, tile):
    return ht.sea_classes(_water(raw), _valid(raw), static["outside"], static["sea_seed"],
                          shape=raw.shape, tile=tile)


def class_areas(raw, code, sector, ids, tile) -> dict:
    """Areas (km2) of the observed classes per sector id (None = all)."""
    keys = ("valid", "water", "sea", "unconnected_land", "unconnected_unknown")
    out = {k: dict.fromkeys(keys, 0) for k in ids}
    for r0, r1, c0, c1 in ht.tiles(raw.shape, tile):
        r, c, lab = raw[r0:r1, c0:c1], code[r0:r1, c0:c1], sector[r0:r1, c0:c1]
        m = {"valid": r > -2, "water": r == 1, "sea": c == 1, "unconnected_land": (r == 1) & (c == 0),
             "unconnected_unknown": c == -1}
        for k in ids:
            reg = None if k is None else lab == k
            for name, mm in m.items():
                out[k][name] += int(mm.sum() if reg is None else (mm & reg).sum())
    return {k: {f"area_{n}_km2": v * CELL_KM2 for n, v in d.items()} for k, d in out.items()}


def _disp_row(res: dict) -> dict:
    return {"median_m": res["median"], "p95_m": res["p95"], "signed_median_m": res["signed_median"],
            "n": res["n"], "n_pred": res["n_pred"], "n_obs": res["n_obs"],
            "n_pred_unknown": res["n_pred_unknown"], "n_obs_unknown": res["n_obs_unknown"],
            "observed_level": res["observed_level"]}


_INF = np.iinfo(np.int32).max


def encode(d, pixel: float = RES) -> np.ndarray:
    """Signed distances between cell centres as int32 codes, sign x (d / pixel)^2 (an
    integer on the grid; +-INF for infinite), so that a pool over every level holds 4
    bytes per distance and decodes to the same float64 values."""
    d = np.asarray(d, float)
    if np.isnan(d).any():   # NaN would cast to INT32_MIN and decode as a finite -2,780 km (D-089)
        raise ValueError("NaN distance passed to encode(); drop undefined distances first")
    k = np.where(np.isinf(d), _INF, np.rint((np.where(np.isinf(d), 0, d) / pixel) ** 2))
    if np.any(k > _INF):
        raise ValueError("distance too large for the int32 code")
    return (np.sign(d) * k).astype(np.int32)


def decode(code, pixel: float = RES) -> np.ndarray:
    c = np.asarray(code, np.int64)
    a = np.where(np.abs(c) == _INF, np.inf, np.sqrt(np.abs(c).astype(float)) * pixel)
    return np.sign(c) * a


class Pool:
    """Signed distances (encoded) and counts per key, for pooled summaries (D-032)."""

    def __init__(self, pixel: float = RES):
        self.d: dict = {}
        self.pixel = pixel

    def add(self, key, res: dict):
        e = self.d.setdefault(key, {"codes": [], "n_pred": 0, "n_obs": 0, "n_pred_unknown": 0,
                                    "n_obs_unknown": 0, "observed_level": False})
        dist = np.asarray(res.get("distances", np.empty(0)), float)
        e["n_nan_dropped"] = e.get("n_nan_dropped", 0) + int(np.isnan(dist).sum())
        e["codes"].append(encode(dist[~np.isnan(dist)], self.pixel))
        for k in ("n_pred", "n_obs", "n_pred_unknown", "n_obs_unknown"):
            e[k] += res[k]
        e["observed_level"] = e["observed_level"] or bool(res["observed_level"])

    def summary(self, key) -> dict:
        e = self.d[key]
        return ht.pooled([decode(c, self.pixel) for c in e["codes"]], e["n_pred"], e["n_obs"],
                         e["n_pred_unknown"], e["n_obs_unknown"], e["observed_level"])


def _measured(cells: dict, transform, block_m: float):
    """Block indices and signed distances of the measured waterline cells (both
    directions)."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    p, o = cells["pred"], cells["obs"]
    rows = np.r_[p["rows"][p["known"]], o["rows"][o["known"]]]
    cols = np.r_[p["cols"][p["known"]], o["cols"][o["known"]]]
    d = np.r_[cells["signed_p"][p["known"]], cells["signed_o"][o["known"]]]
    ok = ~np.isnan(d)                      # undefined distances are left out (D-089); +-inf kept
    rows, cols, d = rows[ok], cols[ok], d[ok]
    return (np.floor((c + a * (cols + 0.5)) / block_m).astype(np.int64),
            np.floor((f + e * (rows + 0.5)) / block_m).astype(np.int64), d)


def block_rows(cells: dict, transform, block_m: float) -> pd.DataFrame:
    """Measured waterline cells summarised per block for one unit: n, median |d|,
    signed median."""
    bi, bj, d = _measured(cells, transform, block_m)
    if d.size == 0:
        return pd.DataFrame(columns=["block_i", "block_j", "n", "median_abs_m", "signed_median_m"])
    df = pd.DataFrame({"block_i": bi, "block_j": bj, "d": d, "a": np.abs(d)})
    return (df.groupby(["block_i", "block_j"])
            .agg(n=("a", "size"), median_abs_m=("a", "median"), signed_median_m=("d", "median")).reset_index())


def median_from_counts(k, counts, pixel: float = RES) -> float:
    """np.median of the distances sqrt(k) x pixel (k sorted ascending, ``_INF`` for an
    infinite distance), each repeated ``counts`` times; the two middle values are
    averaged as np.median does."""
    cum = np.cumsum(counts)
    n = int(cum[-1])
    lo, hi = k[np.searchsorted(cum, (n - 1) // 2, side="right")], k[np.searchsorted(cum, n // 2, side="right")]
    a, b = decode([lo, hi], pixel)
    return float((a + b) / 2)


class BlockHist:
    """Per key (product, variant) and block, how often each distance occurs (as its
    squared lattice length), summed over units: the exact pooled median of D-070
    without holding every distance."""

    def __init__(self, transform, block_m: float, pixel: float = RES):
        self.s: dict = {}
        self.transform, self.block_m, self.pixel = transform, block_m, pixel

    def add(self, key, cells: dict):
        bi, bj, d = _measured(cells, self.transform, self.block_m)
        if d.size == 0:
            return
        vc = pd.DataFrame({"block_i": bi, "block_j": bj,
                           "k": np.abs(encode(d, self.pixel)).astype(np.int64)}).value_counts()
        self.s[key] = vc if key not in self.s else self.s[key].add(vc, fill_value=0)

    def table(self) -> pd.DataFrame:
        rows = []
        for (prod, var), s in sorted(self.s.items()):
            df = s.astype(np.int64).rename("count").reset_index().sort_values(["block_i", "block_j", "k"])
            for (i, j), g in df.groupby(["block_i", "block_j"], sort=True):
                rows.append({"product": prod, "variant": var, "block_i": i, "block_j": j,
                             "n": int(g["count"].sum()),
                             "median_abs_m": median_from_counts(g["k"].to_numpy(), g["count"].to_numpy(), self.pixel)})
        return pd.DataFrame(rows, columns=["product", "variant", "block_i", "block_j", "n", "median_abs_m"])


# --- shape offset (PREREG 5b, D-070) -------------------------------------------------------

def unit_counts(F, code, outside, finite, region, thresholds, *, tile: int = ht.TILE):
    """For one unit and product: the number of known cells in ``region`` with F below
    each threshold (strictly increasing; F compared in float64), and the number of
    those cells observed as water. Known cells are those of hindcast.extent_metrics'
    total area bias: inside the valid mask with a known observed state, or outside it
    with a known outside state, and a finite product height. So count - water is that
    area bias (cells) at each threshold."""
    T = np.asarray(thresholds, float)
    if np.any(np.diff(T) <= 0):
        raise ValueError("thresholds must increase strictly")
    hist = np.zeros(T.size + 1, np.int64)
    water = 0
    for r0, r1, c0, c1 in ht.tiles(code.shape, tile):
        c, st = code[r0:r1, c0:c1], outside[r0:r1, c0:c1]
        ok = np.asarray(ht._get(finite, r0, r1, c0, c1), bool) & np.asarray(region[r0:r1, c0:c1], bool)
        k_in, k_out = (c >= 0) & ok, (c == -2) & (st >= 0) & ok
        water += int(((c == 1) & k_in).sum() + ((st == 1) & k_out).sum())
        v = np.asarray(F[r0:r1, c0:c1], float)[k_in | k_out]
        hist += np.bincount(np.searchsorted(T, v, side="right"), minlength=T.size + 1)
    return np.cumsum(hist)[:T.size], water


def fit_offset(bias, epochs, exclude: str | None = None) -> dict:
    """D-070: the c of C_GRID that minimises the sum over units of |area bias| (cells;
    ``bias`` [unit, c]), leaving out the units of epoch ``exclude``. Ties go to the
    smallest |c|, then to the smaller c."""
    use = np.array([e != exclude for e in epochs])
    if not use.any():
        raise ValueError("no unit left to fit")
    obj = np.abs(bias[use]).sum(axis=0)
    cand = np.flatnonzero(obj == obj.min())
    k = cand[np.lexsort((C_GRID[cand], np.abs(C_GRID[cand])))[0]]
    return {"offset_m": float(C_GRID[k]), "objective_km2_at_offset": round(float(obj[k]) * CELL_KM2, 4),
            "n_units": int(use.sum()), "n_tied_minima": int(cand.size),
            "objective_km2": [round(float(v) * CELL_KM2, 4) for v in obj]}


def fit_shape(units, transform, static, flood_maps, finite, fit_region, *, level_datum_offset: float,
              product_offsets: dict, tile: int, log=print) -> tuple[dict, dict]:
    """Per product, the signed area bias (cells) of every (epoch, level) unit at every
    c of C_GRID over the known cells of ``fit_region``, and the D-070 fit with its
    leave-one-epoch-out refits. Returns (record, {product: bias array})."""
    shape = static["sector"].shape
    ulist = [(b, ep) for b in sorted(units) for ep in sorted(units[b])]
    bias = {p: np.zeros((len(ulist), C_GRID.size), np.int64) for p in flood_maps}
    for i, (b, ep) in enumerate(ulist):
        raw, _ = observed_raw(groups_for(units, b, [ep]), transform, shape)
        code = classify(raw, static, tile)
        del raw
        for p, F in flood_maps.items():
            T = products.flood_level(_bin_level(b), level_datum_offset,
                                     float(product_offsets.get(p, {}).get("offset_m", 0.0)), C_GRID)
            cnt, water = unit_counts(F, code, static["outside"], finite[p], fit_region, T, tile=tile)
            bias[p][i] = cnt - water
        log(f"shape fit: unit {ep} bin {b} ({i + 1}/{len(ulist)})", flush=True)
    epochs = [ep for _, ep in ulist]
    rec = {"rule": "D-070: per product, the c (m) on a 0.01 m grid over [-5, 5] minimising the sum over all "
                   "(epoch, level) units of |area bias| (km2, known cells inside the domain and in training "
                   "blocks); ties to the smallest |c|, then the smaller c. Positive c: product heights too high.",
           "c_grid_m": {"min": float(C_GRID[0]), "max": float(C_GRID[-1]), "step": 0.01, "n": int(C_GRID.size)},
           "fit_region_cells": int(sum(np.asarray(fit_region[r0:r1, c0:c1], bool).sum()
                                       for r0, r1, c0, c1 in ht.tiles(shape, tile))),
           "level_datum_offset_m": float(level_datum_offset),
           "units": [{"epoch": ep, "bin": b, "level_m": round(_bin_level(b), 4)} for b, ep in ulist],
           "products": {}}
    for p in flood_maps:
        r = fit_offset(bias[p], epochs)
        r["leave_one_epoch_out"] = {e: fit_offset(bias[p], epochs, exclude=e) for e in sorted(set(epochs))
                                    if any(x != e for x in epochs)}
        rec["products"][p] = r
    return rec, bias


# --- driver ---------------------------------------------------------------------------------

def run(stage2_dir, out_dir, static: dict, heights: dict, transform, *, level_datum_offset: float = 0.0,
        level_datum_source: str = DATUM_SOURCE, product_offsets: dict | None = None, shape_offsets="fit",
        fit_region=None, fit_exclude_epoch: str | None = None, flood_maps: dict | None = None,
        sector_names=SECTOR_ORDER, blocks: pd.DataFrame | None = None, block_m: float = design.BLOCK_SIZE_M,
        tile: int = ht.TILE, meta: dict | None = None, log=print) -> dict:
    """Compute and write the per-unit tables (see the module docstring).

    ``heights``: {product: array, memory map or products.Corrected on the grid}.
    ``flood_maps``: their flood-level maps (computed here if None). ``product_offsets``:
    {product: {"offset_m", "source"}}. ``shape_offsets``: "fit" (D-070, needs
    ``fit_region``, the domain and training-block mask; with ``fit_exclude_epoch``
    the shape variant uses the refit without that epoch), a dict like
    ``product_offsets``, or None (no shape variant)."""
    t_start = time.time()
    product_offsets = product_offsets or {}
    sector, outside, seed, domain = static["sector"], static["outside"], static["sea_seed"], static["domain"]
    shape = sector.shape
    ids = list(range(1, len(sector_names) + 1)) + [None]
    name = {**{k + 1: s for k, s in enumerate(sector_names)}, None: "all"}
    regions = {"domain": domain} if fit_region is None else {"domain": domain, "domain_train": fit_region}
    units = stage2_index(stage2_dir)
    if not units:
        raise ValueError(f"no stage-2 tiles with main-export groups in {stage2_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    finite = {p: (lambda h: lambda r0, r1, c0, c1: np.isfinite(np.asarray(h[r0:r1, c0:c1], float)))(h)
              for p, h in heights.items()}
    if flood_maps is None:
        flood_maps = {p: ht.flood_level_map(h, seed, tile=tile) for p, h in heights.items()}
    shape_record = None
    if isinstance(shape_offsets, str):
        if shape_offsets != "fit" or fit_region is None:
            raise ValueError('shape_offsets must be "fit" (with fit_region), a dict or None')
        shape_record, _ = fit_shape(units, transform, static, flood_maps, finite, fit_region,
                                    level_datum_offset=level_datum_offset, product_offsets=product_offsets,
                                    tile=tile, log=log)
        src = "fitted in this run (D-070)"
        if fit_exclude_epoch:
            src += f", without epoch {fit_exclude_epoch} (leave-one-epoch-out)"
        shape_offsets = {p: {"offset_m": (r["leave_one_epoch_out"][fit_exclude_epoch] if fit_exclude_epoch
                                          else r)["offset_m"], "source": src}
                         for p, r in shape_record["products"].items()}
        shape_record["used_in_this_run"] = shape_offsets
        (out_dir / "shape_offsets.json").write_text(json.dumps(shape_record, indent=1), encoding="utf-8")
    shape_offsets = shape_offsets or {}
    variants = [(p, "total", float(product_offsets.get(p, {}).get("offset_m", 0.0)), 0.0) for p in heights]
    variants += [(p, "shape", float(product_offsets.get(p, {}).get("offset_m", 0.0)), float(s["offset_m"]))
                 for p, s in shape_offsets.items() if p in heights]
    rows = {k: [] for k in ("extent", "displacement", "noise_floor", "observed", "blocks")}
    nf_all, bh = Pool(), BlockHist(transform, block_m)
    unit_info = []
    for b in sorted(units):
        level = _bin_level(b)
        epochs = sorted(units[b])
        base = {"bin": b, "level_m": round(level, 4)}
        full, nf_bin = {}, Pool()
        for ep in epochs:
            raw, info = observed_raw(groups_for(units, b, [ep]), transform, shape)
            full[ep] = classify(raw, static, tile)
            unit_info.append({"epoch": ep, "sample": "full", **base, **info})
            for k, a in class_areas(raw, full[ep], sector, ids, tile).items():
                rows["observed"].append({"epoch": ep, "sample": "full", **base, "sector": name[k], **a})
            del raw
            if set(units[b][ep]) != {0, 1}:
                rows["noise_floor"].append({"kind": "halves", "epoch": ep, **base, "sector": "all",
                                            "note": "one parity half has no group"})
                continue
            half = []
            for p in (0, 1):
                raw, info = observed_raw(groups_for(units, b, [ep], (p,)), transform, shape)
                half.append(classify(raw, static, tile))
                unit_info.append({"epoch": ep, "sample": f"p{p}", **base, **info})
                for k, a in class_areas(raw, half[-1], sector, ids, tile).items():
                    rows["observed"].append({"epoch": ep, "sample": f"p{p}", **base, "sector": name[k], **a})
                del raw
            cells = ht.noise_floor_cells(half[0], _valid(half[0]), half[1], _valid(half[1]), sector, RES,
                                         outside_state=outside, shape=shape, tile=tile)
            for k, res in ht.summarise(cells, ids, return_distances=True).items():
                rows["noise_floor"].append({"kind": "halves", "epoch": ep, **base, "sector": name[k],
                                            **_disp_row(res)})
                if k is not None:
                    nf_bin.add(k, res)
                    nf_all.add(k, res)
            del half, cells
        e23 = [ep for ep in epochs if ep in ("E2", "E3")]
        if "E1" in epochs and e23:
            raw, info = observed_raw(groups_for(units, b, e23), transform, shape)
            later = classify(raw, static, tile)
            unit_info.append({"epoch": "+".join(e23), "sample": "for_e1", **base, **info})
            for k, a in class_areas(raw, later, sector, ids, tile).items():
                rows["observed"].append({"epoch": "+".join(e23), "sample": "for_e1", **base, "sector": name[k],
                                         **a})
            cells = ht.noise_floor_cells(full["E1"], _valid(full["E1"]), later, _valid(later), sector, RES,
                                         outside_state=outside, shape=shape, tile=tile)
            for k, res in ht.summarise(cells, ids, return_distances=True).items():
                rows["noise_floor"].append({"kind": "e1_vs_e23", "epoch": "E1|" + "+".join(e23), **base,
                                            "sector": name[k], **_disp_row(res)})
                if k is not None:
                    nf_bin.add(k, res)
                    nf_all.add(k, res)
            del raw, later, cells
        for k in ids[:-1]:
            if k in nf_bin.d:
                rows["noise_floor"].append({"kind": "pooled_level", "epoch": "+".join(epochs), **base,
                                            "sector": name[k], **_disp_row(nf_bin.summary(k))})
        for prod, var, off_p, off_s in variants:
            fl = float(products.flood_level(level, level_datum_offset, off_p, off_s))
            pred = ht.flooded(flood_maps[prod], fl)
            key = {"product": prod, "variant": var}
            pool = Pool()
            for ep in epochs:
                obs, val = full[ep], _valid(full[ep])
                cnt = ht.extent_counts(pred, obs, val, sector, ids, regions=regions, pred_valid=finite[prod],
                                       outside_state=outside, shape=shape, tile=tile)
                for k, c in cnt.items():
                    m = ht.metrics_from_counts(c, CELL_KM2)
                    rows["extent"].append({**key, "epoch": ep, **base, "flood_level_m": fl,
                                           "sector": name.get(k, k), **{f"n_{n}": v for n, v in c.items()},
                                           **{(f"{n}_km2" if n.startswith("area") and "rel" not in n else n): v
                                              for n, v in m.items()}})
                cells = ht.displacement_cells(pred, obs, val, sector, RES, pred_valid=finite[prod],
                                              outside_state=outside, shape=shape, tile=tile)
                for k, res in ht.summarise(cells, ids, return_distances=True).items():
                    rows["displacement"].append({**key, "epoch": ep, **base, "flood_level_m": fl,
                                                 "sector": name[k], **_disp_row(res)})
                    if k is not None:
                        pool.add(k, res)
                br = block_rows(cells, transform, block_m)
                if len(br):
                    rows["blocks"].append(br.assign(**key, epoch=ep, bin=b, level_m=base["level_m"]))
                bh.add((prod, var), cells)
                del cells
            for k in ids[:-1]:
                rows["displacement"].append({**key, "epoch": "pooled", "epochs": "+".join(epochs), **base,
                                             "flood_level_m": fl, "sector": name[k], **_disp_row(pool.summary(k))})
        log(f"bin {b} ({level:.2f} m) epochs {epochs} done, {time.time() - t_start:.0f} s", flush=True)
        del full
    for k in ids[:-1]:
        if k in nf_all.d:
            rows["noise_floor"].append({"kind": "pooled_all_levels", "epoch": "all", "bin": np.nan,
                                        "level_m": np.nan, "sector": name[k], **_disp_row(nf_all.summary(k))})
    tables = {k: pd.DataFrame(v) for k, v in rows.items() if k != "blocks"}
    bl = pd.concat(rows["blocks"], ignore_index=True) if rows["blocks"] else pd.DataFrame()
    h1b = bh.table()
    if blocks is not None:
        lab = blocks.rename(columns={"i": "block_i", "j": "block_j", "sector": "block_sector"})[
            ["block_i", "block_j", "block_sector", "split"]]
        if len(bl):
            bl = bl.merge(lab, on=["block_i", "block_j"], how="left")
        h1b = h1b.merge(lab, on=["block_i", "block_j"], how="left")
    tables["blocks"], tables["block_h1"] = bl, h1b
    for k, df in tables.items():   # the per-unit block table is the largest: gzip
        df.to_csv(out_dir / (f"{k}.csv.gz" if k == "blocks" else f"{k}.csv"), index=False)
    record = {**(meta or {}), "stage2_dir": str(stage2_dir),
              "level_datum": {"offset_m": float(level_datum_offset), "source": level_datum_source,
                              "meaning": "added to the DAHITI-scale bin centres (working datum: DAHITI, D-070)"},
              "product_offsets": product_offsets or "none: no constant offset",
              "shape_offsets": shape_offsets or "none: shape variant not computed",
              "shape_fit": "shape_offsets.json" if shape_record else "not fitted in this run",
              "products": list(heights), "grid": {"transform": [float(v) for v in transform[:6]], "shape": list(shape),
                                                  "crs": CRS_EQUAL_AREA},
              "static_counts": static.get("counts", {}), "units": unit_info,
              "block_h1": {"block_m": block_m, "n_rows": len(h1b)},
              "seconds": round(time.time() - t_start, 1)}
    (out_dir / "run.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    if shape_record:
        tables["shape_offsets"] = shape_record
    return tables


# --- production entry point -------------------------------------------------------------------

def _offsets(path):
    if not path:
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for k, v in d.items():
        if k not in products.PRODUCTS or "offset_m" not in v or not v.get("source"):
            raise ValueError(f"{path}: {k} needs a known product, offset_m and source")
    return d


def flood_map_cached(name: str, h, seed, key_meta: dict, tile: int, cache: Path = INTERIM / "hindcast" / "flood"):
    """The flood-level map of ``h`` from ``seed``, as a float32 memory map of a cached
    .npy keyed by the heights cache, the correction and the seed."""
    key = hashlib.sha256(json.dumps({**key_meta, "seed": hashlib.sha256(np.packbits(seed)).hexdigest(),
                                     "connectivity": 8}, sort_keys=True).encode()).hexdigest()[:12]
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}_{key}.npy"
    if not path.exists():
        tmp = path.with_suffix(".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(np.shape(h)))
        ht.flood_level_map(h, seed, tile=tile, out=mm)
        mm.flush()
        del mm
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("--vertical-correction", default=None,
                    help="N(EGM2008) - N(EIGEN-6C4) raster (GeoTIFF or npz; caspian.products), added to the heights")
    ap.add_argument("--level-datum-offset", type=float, default=0.0,
                    help="m added to the DAHITI-scale bin centres (default 0: DAHITI working datum, D-070)")
    ap.add_argument("--level-datum-source", default=DATUM_SOURCE)
    ap.add_argument("--products", default=",".join(products.PRODUCTS))
    ap.add_argument("--product-offsets", default=None, help="JSON: constant product offsets (default none)")
    ap.add_argument("--shape-offsets", default=None, help="JSON: shape constants instead of the D-070 fit")
    ap.add_argument("--no-shape", action="store_true", help="skip the shape variant")
    ap.add_argument("--fit-exclude-epoch", default=None, choices=["E1", "E2", "E3"],
                    help="shape variant with the D-070 refit without this epoch")
    ap.add_argument("--resampling", default="bilinear", choices=products.METHODS)
    ap.add_argument("--tile", type=int, default=ht.TILE)
    ap.add_argument("--no-envelope", action="store_true",
                    help="sensitivity: no record-low water envelope on the observed extents (D-082)")
    ap.add_argument("--out-suffix", default="", help="write to results/hindcast/<tag><suffix> (sensitivity runs)")
    a = ap.parse_args(argv)
    w, s, e, n = DOMAIN.bounds
    transform, shape = grid((w - MARGIN_DEG, s - MARGIN_DEG, e + MARGIN_DEG, n + MARGIN_DEG))
    names = [p for p in a.products.split(",") if p]
    unknown = set(names) - set(products.PRODUCTS)
    if unknown:
        raise SystemExit(f"unknown products: {sorted(unknown)}")
    print("grid", transform, shape, flush=True)
    static = production_static(transform, shape, a.tile)
    for k in ("zone", "land_seed"):
        static.pop(k)
    print("static layers", static["counts"], flush=True)
    blocks = pd.read_csv(BLOCKS)
    fit_region = static["domain"] & train_mask(blocks, transform, shape)
    corr, corr_rec = (None, None)
    if a.vertical_correction:
        corr, corr_rec = products.correction(a.vertical_correction, transform, shape)
        gaps = sum(int((~np.isfinite(np.asarray(corr[r0:r1, c0:c1])) & static["domain"][r0:r1, c0:c1]).sum())
                   for r0, r1, c0, c1 in ht.tiles(shape, a.tile))
        if gaps:
            raise SystemExit(f"the vertical correction leaves {gaps} domain cells without a value")
    heights, flood_maps = {}, {}
    for p in names:
        h = products.heights(p, transform, shape, a.resampling)
        heights[p] = h if corr is None else products.Corrected(h, corr)
        flood_maps[p] = flood_map_cached(p, heights[p], static["sea_seed"],
                                         {"heights": Path(h.filename).name,
                                          "correction": corr_rec["sha256"] if corr_rec else None}, a.tile)
        print("flood-level map", p, flush=True)
    if a.no_shape:
        shape_offsets = None
    elif a.shape_offsets:
        shape_offsets = _offsets(a.shape_offsets)
    else:
        shape_offsets = "fit"
    global ENVELOPE
    ENVELOPE = None if a.no_envelope else record_low_water(stage2_index(INTERIM / "stage2" / a.tag), transform, shape)
    env = ("none (sensitivity)" if ENVELOPE is None else
           {"rule": f"D-082: cells water at the record low ({ENVELOPE_LOW_BINS} lowest E3 bins) are water at every level",
            "cells": int(ENVELOPE.sum())})
    print("envelope", env, flush=True)
    meta = {"tag": a.tag, "resampling": a.resampling, "tile": a.tile, "envelope": env,
            "references": {p: products.PRODUCTS[p]["reference"] for p in names},
            "vertical_correction": corr_rec or "none: products as distributed (read as heights above EGM2008)",
            "seeds": {"sea": "D-029 open-sea mask outside the zone, inside the domain less its rim (D-070)",
                      "land": "beyond-domain land in the domain's 5 km inner band, outside zone and rim (D-070)"}}
    run(INTERIM / "stage2" / a.tag, RESULTS / "hindcast" / (a.tag + a.out_suffix), static, heights, transform,
        level_datum_offset=a.level_datum_offset, level_datum_source=a.level_datum_source,
        product_offsets=_offsets(a.product_offsets), shape_offsets=shape_offsets, fit_region=fit_region,
        fit_exclude_epoch=a.fit_exclude_epoch, flood_maps=flood_maps, blocks=blocks, tile=a.tile, meta=meta)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
