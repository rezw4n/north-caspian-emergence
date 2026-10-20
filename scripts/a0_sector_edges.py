"""Where the frozen reporting sectors (caspian.geography, D-014) lie relative to the
present shoreline, measured in each locally held 15-arc-second relief grid.

Replaces the hand-typed "25-60 km offshore" of the first Methods draft (review
finding 18): the sector edges were drawn by eye, so their distance from the coast
is a measured property of the polygons and must come from a script.

Definitions (all fixed here, before any hindcast metric is computed):
  * levels are read from the SMOOTH_K-point running median of the DAHITI series
    (caspian.levels.running_median; the D-021 correction, so that one noisy record
    cannot set a level). No datum offset is applied: the product-to-DAHITI offset is
    unknown and is what PREREG section 5 estimates.
  * present level  L_now: the running median at the most recent DAHITI record.
  * record maximum L_max: the maximum of the running median (mid-July 1995).
  * sea at level L: grid cells with elevation < L that are 4-connected to the deepest
    cell of the grid (the Middle Caspian at the southern edge). Cells below L that are
    not connected (closed depressions such as sub-sea-level salt flats) count as land.
  * present coast: land cells with a 4-neighbour in the sea at L_now.
  * offshore edge of a coastal sector: the part of its boundary shared with the open
    shelf, sampled every EDGE_STEP_KM in the equal-area projection. Each sample gets a
    signed distance to the present coast: positive in the sea, negative on land.
  * emerged band of a grid: cells that are sea at L_max but not at L_now.
  * two readings of the polygon edges: straight in lon/lat (our code and every area in
    the paper) and geodesic (Earth Engine's default for lon/lat GeoJSON; see
    scripts/a0_study_area.py). Per grid, the band cells whose domain or sector
    membership differs between the readings, and the cells that are sea at L_max
    between the two readings of the domain's northern edge.

The grids are the products the paper audits, and their vertical reference in the
Caspian is undocumented (docs/DATUMS.md), so every number here is reported per grid
and is indicative of the sector layout, not a measurement of the true coastline.
ETOPO1 is not analysed here: it is a 1-arc-minute grid, coarser than the others. The
script only reads the local subset's pixel size and writes the reason for skipping it
to the output (on 2026-10-20 the subset had 1-degree pixels, i.e. a faulty export).
Output (committed): results/study_area/sector_edges.json
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401  (Windows DLL-path fix)
import numpy as np
import pyproj
import shapely
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

from caspian import config
from caspian.geography import DOMAIN, SECTOR_ORDER, sectors
from caspian.io import find_dahiti_file, read_dahiti
from caspian.levels import running_median

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import a0_study_area  # noqa: E402  (geodesic reading of the same polygons)

GRIDS = {
    "gebco_2022": "gebco_2022_ncaspian.nc",
    "gebco_2026": "gebco_2026_ncaspian.nc",
    "etopo_2022": "etopo_2022_ncaspian.nc",
    "srtm15p_v2.7": "srtm15p_v2.7_ncaspian.nc",
}
EDGE_STEP_KM = 1.0
DENSIFY_DEG = 0.01
SMOOTH_K = 3  # as in scripts/p0_dahiti_qc.py and scripts/a4_scene_levels.py (D-021 correction)

_to_laea = pyproj.Transformer.from_crs(config.CRS_EXCHANGE, config.CRS_EQUAL_AREA, always_xy=True)


def sea_mask(z: np.ndarray, level: float) -> np.ndarray:
    """Cells below ``level`` that are 4-connected to the deepest cell of the grid."""
    below = np.isfinite(z) & (z < level)
    lab, _ = ndimage.label(below)  # default structure: 4-connectivity
    seed = np.unravel_index(np.nanargmin(np.where(np.isfinite(z), z, np.inf)), z.shape)
    if not below[seed]:
        raise ValueError("deepest cell is not below the level")
    return lab == lab[seed]


def coast_cells(sea: np.ndarray) -> np.ndarray:
    """Land cells with a 4-neighbour in the sea."""
    grown = ndimage.binary_dilation(sea, structure=ndimage.generate_binary_structure(2, 1))
    return grown & ~sea


def edge_points(sector_geom, open_shelf_geom, step_km: float = EDGE_STEP_KM) -> np.ndarray:
    """(lon, lat) points every ``step_km`` along the boundary a sector shares with the
    open shelf. Returns an (n, 2) array (possibly empty)."""
    shared = shapely.intersection(sector_geom.boundary, open_shelf_geom.boundary)
    lines = [g for g in getattr(shared, "geoms", [shared]) if g.geom_type in ("LineString", "LinearRing")]
    if not lines:
        merged = shapely.line_merge(shared) if not shared.is_empty else shared
        lines = [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "LineString"]
    pts = []
    for ln in lines:
        ln = shapely.segmentize(ln, DENSIFY_DEG)
        lp = shapely.transform(ln, lambda x, y: _to_laea.transform(x, y), interleaved=False)
        n = max(int(lp.length / (step_km * 1e3)), 1)
        for d in np.linspace(0, lp.length, n + 1):
            p = lp.interpolate(d)
            pts.append(_to_laea.transform(p.x, p.y, direction="INVERSE"))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def cell_areas_km2(lat: np.ndarray, dlon: float, dlat: float) -> np.ndarray:
    """Ellipsoidal area of one grid cell per row (km2), from pyproj.Geod."""
    geod = pyproj.Geod(ellps="WGS84")
    out = np.empty(lat.size)
    for i, la in enumerate(lat):
        lo = 50.0
        a, _ = geod.polygon_area_perimeter([lo, lo + dlon, lo + dlon, lo],
                                           [la - dlat / 2, la - dlat / 2, la + dlat / 2, la + dlat / 2])
        out[i] = abs(a) / 1e6
    return out


def summarise(v: np.ndarray) -> dict:
    v = np.asarray(v, float)
    if v.size == 0:
        return {}
    q = np.percentile(v, [5, 50, 95])
    return {"n": int(v.size), "min_km": round(float(v.min()), 1), "p05_km": round(float(q[0]), 1),
            "p50_km": round(float(q[1]), 1), "p95_km": round(float(q[2]), 1), "max_km": round(float(v.max()), 1),
            "share_on_land": round(float((v <= 0).mean()), 3)}


def geodesic_polygons(secs: dict) -> dict:
    """The domain ("domain") and every sector with geodesic edges between their vertices,
    traced in lon/lat (scripts/a0_study_area.geodesic_polygon)."""
    out = {"domain": a0_study_area.geodesic_polygon(DOMAIN)}
    for k, (_, g) in secs.items():
        out[k] = a0_study_area.geodesic_polygon(g)
    for g in out.values():
        shapely.prepare(g)
    return out


def reading_differences(lon2d, lat2d, area, sea_max, band_all, member, in_domain, geo: dict,
                        north_lat: float) -> dict:
    """Area (km2) of the band and of the sea at L_max that the two edge readings assign
    differently. ``member``/``in_domain`` are the lon/lat reading; ``geo`` holds the
    geodesic polygons. A band cell counts as changed if it is in the domain under one
    reading only, or if under the geodesic reading it is not in exactly one sector or
    in a sector other than its lon/lat one."""
    in_geo = shapely.contains_xy(geo["domain"], lon2d, lat2d)
    count = np.zeros(lon2d.shape, dtype=np.int16)
    key_geo = np.full(lon2d.shape, "", dtype=object)
    for key in SECTOR_ORDER:
        c = shapely.contains_xy(geo[key], lon2d, lat2d)
        count += c
        key_geo[c] = key
    changed_domain = band_all & (in_domain != in_geo)
    changed_sector = band_all & (in_domain | in_geo) & ((count != 1) | (key_geo != member))
    between = in_domain != in_geo
    north = between & (lat2d > north_lat - 0.05)
    return {"band_changed_km2": round(float(area[changed_domain | changed_sector].sum()), 1),
            "band_changed_domain_km2": round(float(area[changed_domain].sum()), 1),
            "band_changed_sector_km2": round(float(area[changed_sector].sum()), 1),
            "sea_at_max_between_readings_km2": round(float(area[sea_max & between].sum()), 1),
            "north_edge_between_km2": round(float(area[north].sum()), 1),
            "north_edge_sea_at_max_km2": round(float(area[north & sea_max].sum()), 1)}


def analyse_grid(path, l_now: float, l_max: float, secs: dict, geo: dict | None = None,
                 north_lat: float | None = None) -> dict:
    ds = xr.open_dataset(path)
    z = ds["elevation"].values.astype(float)
    lat, lon = ds["lat"].values, ds["lon"].values
    dlon, dlat = float(np.median(np.diff(lon))), float(np.median(np.diff(lat)))
    LON, LAT = np.meshgrid(lon, lat)
    X, Y = _to_laea.transform(LON, LAT)

    sea_now = sea_mask(z, l_now)
    coast = coast_cells(sea_now)
    tree = cKDTree(np.column_stack([X[coast], Y[coast]]))

    def signed_km(lonlat: np.ndarray) -> np.ndarray:
        if lonlat.size == 0:
            return np.empty(0)
        x, y = _to_laea.transform(lonlat[:, 0], lonlat[:, 1])
        d, _ = tree.query(np.column_stack([x, y]))
        i = np.clip(np.rint((lonlat[:, 1] - lat[0]) / dlat).astype(int), 0, lat.size - 1)
        j = np.clip(np.rint((lonlat[:, 0] - lon[0]) / dlon).astype(int), 0, lon.size - 1)
        return np.where(sea_now[i, j], d, -d) / 1e3

    # sector membership of each cell centre
    member = np.full(z.shape, "", dtype=object)
    in_domain = shapely.contains_xy(DOMAIN, LON, LAT)
    for key, (_, g) in secs.items():
        member[shapely.contains_xy(g, LON, LAT) & in_domain] = key

    sea_max = sea_mask(z, l_max)
    band = sea_max & ~sea_now & in_domain
    area = np.broadcast_to(cell_areas_km2(lat, dlon, dlat)[:, None], z.shape)
    band_total = float(area[band].sum())
    open_geom = secs["open_shelf"][1]

    out = {"present_coast_cells": int(coast.sum()), "band_km2_total": round(band_total, 0), "sectors": {}}
    for key in SECTOR_ORDER:
        geom = secs[key][1]
        rec = {"band_km2": round(float(area[band & (member == key)].sum()), 0)}
        rec["band_share"] = round(rec["band_km2"] / band_total, 3) if band_total else None
        if key != "open_shelf":
            rec["offshore_edge"] = summarise(signed_km(edge_points(geom, open_geom)))
            sec_sea = sea_now & (member == key)
            if sec_sea.any():
                d, _ = tree.query(np.column_stack([X[sec_sea], Y[sec_sea]]))
                rec["max_offshore_km"] = round(float(d.max()) / 1e3, 1)
        out["sectors"][key] = rec
    coastal = [k for k in SECTOR_ORDER if k != "open_shelf"]
    out["band_share_coastal_sectors"] = round(sum(out["sectors"][k]["band_km2"] for k in coastal) / band_total, 3) \
        if band_total else None
    if geo is not None:
        out["edge_readings"] = reading_differences(LON, LAT, area, sea_max, sea_max & ~sea_now, member,
                                                   in_domain, geo, north_lat)
    return out


def main() -> int:
    series = read_dahiti(find_dahiti_file(config.RAW / "dahiti")).data.sort_values("time")
    smooth = running_median(series["level_m"].to_numpy(float), SMOOTH_K)
    l_now = round(float(smooth[-1]), 3)
    t_now = str(series["time"].iloc[-1])
    i_max = int(np.nanargmax(smooth))
    l_max = round(float(smooth[i_max]), 3)
    t_max = str(series["time"].iloc[i_max])
    secs = sectors()
    geo = geodesic_polygons(secs)
    north_lat = max(DOMAIN.exterior.coords, key=lambda c: c[1])[1]

    res = {"definition": __doc__.split("Definitions")[1].split("The grids are")[0].strip(),
           "present_level_m": l_now, "present_level_time": t_now,
           "record_max_level_m": l_max, "record_max_time": t_max,
           "level_reference": f"DAHITI (normal heights), {SMOOTH_K}-point running median; no offset applied to the grids",
           "edge_step_km": EDGE_STEP_KM, "grids": {}}
    bathy = config.RAW / "bathymetry"
    for name, fn in GRIDS.items():
        p = bathy / fn
        if not p.exists():
            print(f"missing {p}; skipped", file=sys.stderr)
            continue
        res["grids"][name] = analyse_grid(p, l_now, l_max, secs, geo, north_lat)
        print(name, json.dumps(res["grids"][name]["sectors"], indent=None)[:600])
    etopo1 = bathy / "etopo1_ncaspian.tif"
    if etopo1.exists():
        import rasterio

        with rasterio.open(etopo1) as r:
            res_deg = abs(r.transform.a)
        res["etopo1_skipped"] = (f"local subset has {res_deg:g}-degree pixels ({r.width} x {r.height}); "
                                 "a 1-arc-minute grid is needed" if abs(res_deg - 1 / 60) > 1e-6
                                 else "1-arc-minute grid; not analysed here (coarser than the 15-arc-second grids)")

    # cross-grid summary of the offshore-edge medians and of the band share
    coastal = [k for k in SECTOR_ORDER if k != "open_shelf"]
    med = {k: [g["sectors"][k]["offshore_edge"]["p50_km"] for g in res["grids"].values()] for k in coastal}
    res["summary"] = {
        "offshore_edge_p50_km_min_over_grids": {k: min(v) for k, v in med.items()},
        "offshore_edge_p50_km_max_over_grids": {k: max(v) for k, v in med.items()},
        "band_share_coastal_min": min(g["band_share_coastal_sectors"] for g in res["grids"].values()),
        "band_share_coastal_max": max(g["band_share_coastal_sectors"] for g in res["grids"].values()),
        "n_grids": len(res["grids"]),
        "edge_readings_band_changed_km2_max": max(g["edge_readings"]["band_changed_km2"]
                                                  for g in res["grids"].values()),
        "edge_readings_north_edge_sea_at_max_km2_max": max(g["edge_readings"]["north_edge_sea_at_max_km2"]
                                                           for g in res["grids"].values()),
    }
    out = config.RESULTS / "study_area" / "sector_edges.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
