"""A9 generality test (brief section 11; PREREG Amendment 16; D-094): the Caspian
hindcast audit applied, lighter-weight, to other shrinking closed basins.

Only what the Caspian code does not already provide lives here: the basin
definitions, level assignment from daily gauge series, the local zone and tile
cover built from the observed extremes, the arm geometry of Great Salt Lake, and the
recurrence verdict of Amendment 16 item 9. Everything else (water classification,
stage-2 counts, flood fill, displacement, noise floor, the H1 rule) is the Caspian
code, called unchanged by scripts/g1_*.py.

Conventions: grids are north-up, row 0 at the top; ``transform`` is
[a, 0, c, 0, e, f] (x = c + a * (col + 0.5) at cell centres).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage

FT_M = 0.3048

# Amendment 16 items 2-5. ``level_offset_m`` is subtracted from working-datum levels so
# that the Caspian bin machinery (0.10 m bins from -30.00 m, three-digit bin numbers)
# applies unchanged; bin edges stay at whole metres + 0.1 k on the working datum.
BASINS = {
    "great_salt_lake": {
        "label": "Great Salt Lake",
        "bbox": (-113.17, 40.56, -111.82, 41.80),
        "crs": "+proj=laea +lat_0=41.2 +lon_0=-112.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        "working_datum": "NAVD 88 (m)",
        "level_offset_m": 1300.0,
        "gauges": {"south": "10010000", "north": "10010100"},
        "etopo_tiles": ["N45W120"],
        "sectors": ["north", "south"],
    },
    "lake_urmia": {
        "label": "Lake Urmia",
        "bbox": (44.85, 36.95, 46.15, 38.40),
        "crs": "+proj=laea +lat_0=37.7 +lon_0=45.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs",
        "working_datum": "DAHITI (EIGEN-6C4 normal heights, m)",
        "level_offset_m": 1290.0,
        "dahiti_id": 104,
        "etopo_tiles": ["N45E030", "N45E045"],
        "sectors": ["lake"],
    },
}
GLOBAL_PRODUCTS = ["gebco_2026", "gebco_2022", "etopo_2022", "etopo1", "srtm15p_v2.7"]
OPEN_WATER_RADIUS_M = 5000       # Amendment 16 item 6 (Caspian D-029: 20 km)
OPEN_WATER_MIN_FRAC = 0.80       # D-029, unchanged
MAX_GAP_DAYS = 16                # Amendment 16 item 2 and addendum: gauge gaps bridged linearly


# --- grid ----------------------------------------------------------------------------------

def grid(bbox, crs: str, res: float):
    """Transform [res, 0, x0, 0, -res, y1] and shape (rows, cols) of the equal-area grid,
    aligned to multiples of ``res``, that covers a lon/lat box (w, s, e, n); the box's
    edges are densified so that its curved image in the projection is covered."""
    import pyproj
    w, s, e, n = bbox
    t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon = np.r_[np.linspace(w, e, 50), np.full(50, e), np.linspace(e, w, 50), np.full(50, w)]
    lat = np.r_[np.full(50, s), np.linspace(s, n, 50), np.full(50, n), np.linspace(n, s, 50)]
    xs, ys = t.transform(lon, lat)
    x0, x1 = math.floor(min(xs) / res) * res, math.ceil(max(xs) / res) * res
    y0, y1 = math.floor(min(ys) / res) * res, math.ceil(max(ys) / res) * res
    return [float(res), 0.0, float(x0), 0.0, -float(res), float(y1)], (int(round((y1 - y0) / res)), int(round((x1 - x0) / res)))


def cell_centres(transform, shape):
    """x of each column and y of each row at cell centres."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    return c + a * (np.arange(shape[1]) + 0.5), f + e * (np.arange(shape[0]) + 0.5)


# --- levels --------------------------------------------------------------------------------

def ngvd29_ft_to_navd88_m(ft, delta_m: float):
    """Feet above NGVD 29 to metres above NAVD 88: 0.3048 ft + delta (delta = NAVD 88 -
    NGVD 29 at the gauge, VERTCON 3.0)."""
    return np.asarray(ft, float) * FT_M + float(delta_m)


def level_on_dates(series: pd.Series, dates, max_gap_days: float = MAX_GAP_DAYS) -> np.ndarray:
    """Daily level at each date (Amendment 16 item 2): the value of that day if present,
    else linear interpolation between the nearest earlier and later daily values when
    they are at most ``max_gap_days`` apart; NaN otherwise. ``series``: float values on
    a DatetimeIndex of calendar days (NaN values are ignored)."""
    s = series.dropna().sort_index()
    s = s[~s.index.duplicated()]
    t = s.index.values.astype("datetime64[D]").astype(np.int64)
    v = s.to_numpy(float)
    q = pd.to_datetime(pd.Index(dates)).values.astype("datetime64[D]").astype(np.int64)
    out = np.full(q.size, np.nan)
    if t.size == 0:
        return out
    k = np.searchsorted(t, q)
    exact = (k < t.size) & (t[np.minimum(k, t.size - 1)] == q)
    out[exact] = v[k[exact]]
    lo, hi = k - 1, k
    ok = ~exact & (lo >= 0) & (hi < t.size)
    lo_, hi_ = lo[ok], hi[ok]
    gap = t[hi_] - t[lo_]
    w = (q[ok] - t[lo_]) / np.where(gap > 0, gap, 1)
    val = v[lo_] + w * (v[hi_] - v[lo_])
    out[np.flatnonzero(ok)[gap <= max_gap_days]] = val[gap <= max_gap_days]
    return out


def epochs(times, e1_end, e2_end: str = "2010-12-31") -> np.ndarray:
    """E1 up to and including ``e1_end`` (the date of the record maximum), E2 to
    ``e2_end``, E3 after (PREREG section 4, Amendment 4)."""
    t = pd.to_datetime(pd.Index(times))
    e1, e2 = pd.Timestamp(e1_end), pd.Timestamp(e2_end) + pd.Timedelta(days=1)
    return np.where(t <= e1, "E1", np.where(t < e2, "E2", "E3"))


def parity(df: pd.DataFrame, by=("epoch", "bin"), acq: str = "acq", time: str = "t") -> pd.Series:
    """Acquisition parity within each (epoch, bin): acquisitions alternate 0, 1, 0, ...
    in time order (scripts/a6_stage2_plan.py; Amendment 3 item 3). Returns one value
    per row of ``df`` (all scenes of an acquisition share it)."""
    by = list(by)
    a = df.groupby(by + [acq], as_index=False)[time].min().sort_values(by + [time, acq])
    a["parity"] = a.groupby(by).cumcount() % 2
    return df.merge(a[by + [acq, "parity"]], on=by + [acq], how="left")["parity"].set_axis(df.index)


# --- zone, open water and cover (zone v6 rules on the basin grid) ------------------------------

def disk(radius_px: int) -> np.ndarray:
    r = int(radius_px)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def open_water(low, radius_px: int, min_frac: float = OPEN_WATER_MIN_FRAC) -> np.ndarray:
    """D-029 open-water mask: cells of the near-minimum extent whose surroundings
    within ``radius_px`` are at least ``min_frac`` water (cells beyond the array count
    as not water)."""
    lo = np.asarray(low, bool)
    k = disk(radius_px).astype(float)
    frac = ndimage.convolve(lo.astype(float), k / k.sum(), mode="constant", cval=0.0)
    return lo & (frac >= min_frac - 1e-9)


def largest_component(mask) -> np.ndarray:
    """The largest 8-connected component of ``mask`` (empty if ``mask`` is empty)."""
    m = np.asarray(mask, bool)
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m.copy()
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (int(np.argmax(sizes)) + 1)


def lake_connected(high, seed) -> np.ndarray:
    """Cells of the near-maximum extent (or the seed) 8-connected to the seed through
    near-maximum water: the sea connection of zone v6 (D-029), at the zone grid."""
    m = np.asarray(high, bool) | np.asarray(seed, bool)
    lab, _ = ndimage.label(m, structure=np.ones((3, 3), bool))
    ids = np.unique(lab[np.asarray(seed, bool)])
    return np.isin(lab, ids[ids > 0])


def zone_v6(high, low, seed, dilate_px: int) -> np.ndarray:
    """Zone v6 (D-029) on a local grid: the band in the near-maximum but not the
    near-minimum extent that is connected to open water, dilated by ``dilate_px`` (disk)."""
    band = np.asarray(high, bool) & ~np.asarray(low, bool) & lake_connected(high, seed)
    return ndimage.binary_dilation(band, structure=disk(dilate_px)) if band.any() else band


def tile_cover(zone, transform, tile_m: float, buffer_tiles: int = 1):
    """Equal-area squares of ``tile_m`` (aligned to multiples of tile_m) that contain a
    zone cell, plus ``buffer_tiles`` rings of neighbours (D-060). Returns a sorted
    list of (i, j) tile indices (x0 = i * tile_m, y0 = j * tile_m)."""
    a, _, c, _, e, f = (float(v) for v in transform[:6])
    rows, cols = np.nonzero(np.asarray(zone, bool))
    if rows.size == 0:
        return []
    i = np.floor((c + a * (cols + 0.5)) / tile_m).astype(int)
    j = np.floor((f + e * (rows + 0.5)) / tile_m).astype(int)
    cells = set(zip(i.tolist(), j.tolist()))
    out = set()
    for ii, jj in cells:
        for di in range(-buffer_tiles, buffer_tiles + 1):
            for dj in range(-buffer_tiles, buffer_tiles + 1):
                out.add((ii + di, jj + dj))
    return sorted(out)


# --- arm geometry (Great Salt Lake; Amendment 16 item 4) ----------------------------------------

def medial_line(land, start_rc, direction: int = -1, max_shift: int = 3, max_width: int | None = None):
    """Per-row midpoints of a land strip (a peninsula or an island) traced from
    ``start_rc`` = (row, col) row by row in ``direction`` (-1 = north, up the array):
    in each row, the land run that contains or is nearest to the previous midpoint
    (within ``max_shift`` cells of it); stops where no such run exists, or where the run
    is wider than ``max_width`` cells (the strip joins the mainland). Returns a list of
    (row, col) with fractional columns."""
    land = np.asarray(land, bool)
    r, c = int(start_rc[0]), float(start_rc[1])
    out = []
    while 0 <= r < land.shape[0]:
        row = land[r]
        edges = np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0]))
        runs = list(zip(edges[::2], edges[1::2] - 1))
        best, dist = None, None
        for s, t in runs:
            d = 0.0 if s <= c <= t else min(abs(c - s), abs(c - t))
            if d <= max_shift and (dist is None or d < dist):
                best, dist = (s, t), d
        if best is None or (max_width is not None and best[1] - best[0] + 1 > max_width):
            break
        c = 0.5 * (best[0] + best[1])
        out.append((r, c))
        r += direction
    return out


def split_box(line_xy, box):
    """Split a rectangle (x0, y0, x1, y1) by a polyline that crosses it from one edge
    to another; the line is extended along its end segments to reach the box edge.
    Returns the parts as shapely polygons, largest first."""
    from shapely.geometry import LineString, box as sbox
    from shapely.ops import split

    pts = [tuple(p) for p in line_xy]
    x0, y0, x1, y1 = box
    big = 4 * max(x1 - x0, y1 - y0)

    def extend(p, q):          # from q through p, far beyond p
        dx, dy = p[0] - q[0], p[1] - q[1]
        n = math.hypot(dx, dy) or 1.0
        return (p[0] + dx / n * big, p[1] + dy / n * big)
    line = LineString([extend(pts[0], pts[1])] + pts + [extend(pts[-1], pts[-2])])
    parts = list(split(sbox(*box), line).geoms)
    return sorted(parts, key=lambda g: -g.area)


# --- verdict (Amendment 16 item 9) --------------------------------------------------------------

def in_basin(sector_flags) -> bool | None:
    """A product mispredicts in the basin when it mispredicts in a majority (strictly
    more than half) of the basin's evaluable sectors. ``sector_flags``: True / False per
    evaluable sector (None or NA = not evaluable, left out). None if none evaluable."""
    ev = [bool(v) for v in sector_flags if v is not None and not (isinstance(v, float) and math.isnan(v))
          and v is not pd.NA]
    if not ev:
        return None
    return sum(ev) > len(ev) / 2


def verdict(product_in_basin: dict, positive_control: bool | None, n_global: int = 5) -> dict:
    """Amendment 16 item 9. ``product_in_basin``: {product: True/False/None}, the
    global products' "mispredicts in the basin" (shape variant). ``positive_control``:
    True if it mispredicts in the basin (fails), False if it passes, None if it could
    not be evaluated.

    recurs: >= 3 of the 5 mispredict, control passing or not evaluable; does not recur:
    <= 1; partial: 2, or the control fails. "not run" when no product is evaluable."""
    vals = [v for v in product_in_basin.values() if v is not None]
    n_mis = sum(bool(v) for v in vals)
    pc = ("not evaluable" if positive_control is None else ("fail" if positive_control else "pass"))
    if not vals:
        return {"verdict": "not run: no evaluable sector", "n_mispredict": 0, "n_evaluable": 0,
                "positive_control": pc}
    if positive_control:
        v = "partial"
    elif n_mis >= 3:
        v = "recurs"
    elif n_mis <= 1:
        v = "does not recur"
    else:
        v = "partial"
    return {"verdict": v, "n_mispredict": int(n_mis), "n_evaluable": len(vals), "n_global": int(n_global),
            "positive_control": pc}
