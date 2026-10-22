"""Generality test, local static layers (PREREG Amendment 16 items 4 and 6).

Great Salt Lake:
1. Arms. The Union Pacific Lakeside Subdivision (OpenStreetMap) splits the box into a
   northern and a southern part (its lake-crossing vertices, ordered west to east and
   extended along the parallels of its end points to the box edges). North of it, the
   medial line of the Promontory peninsula (per-row midpoint of the never-water cells of
   GSW max_extent, 250 m; caspian.generality.medial_line) separates the north arm
   (west) from Bear River Bay (east, excluded). Farmington Bay (excluded) is the part of
   the south side east of Antelope Island's medial line and south of the West Davis
   County Causeway, closed in the south by the parallel through the island's
   southernmost never-water cell.
2. Salt ponds: OpenStreetMap landuse=salt_pond polygons, buffered by 120 m.
3. Extremes: the near-maximum extent of each arm from the year of its record maximum
   (north 1987, south 1986, from scripts/g1_ee.py extents), the near-minimum from the
   year of its lowest April-October level.
4. Open water: near-minimum water whose 5 km surroundings are >= 80% water, per arm.
5. Zone v6: the band connected to open water, dilated 2 km, inside the arms;
   cover: 5 km tiles containing a zone cell plus one ring.

Outputs: results/generality/<basin>/{arms_laea,cover_laea,ponds_laea}.geojson,
{zone,cover}_lonlat.geojson, static.json; data/interim/generality/<basin>/static_250m.npz
Usage: python scripts/g1_static.py <basin>
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pyproj
import shapely
from affine import Affine
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, box, mapping, shape
from shapely.ops import linemerge, polygonize, transform as stransform, unary_union

from caspian import generality as G
from caspian.config import INTERIM, RAW, RESULTS

RES = 250.0
TILE_M = 5000.0
POND_BUFFER_M = 120.0


def to_xy(rc, tr):
    """(row, col) at cell centres -> (x, y)."""
    r, c = rc
    return (tr[2] + tr[0] * (c + 0.5), tr[5] + tr[4] * (r + 0.5))


def to_rc(xy, tr):
    x, y = xy
    return (int(np.floor((y - tr[5]) / tr[4])), int(np.floor((x - tr[2]) / tr[0])))


def gsl(basin: str) -> int:
    b = G.BASINS[basin]
    crs = b["crs"]
    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    z = np.load(INTERIM / "generality" / basin / "extents_250m.npz")
    tr = [float(v) for v in z["transform"]]
    shape_ = z["gsw_max"].shape
    x0, y1 = tr[2], tr[5]
    x1, y0 = x0 + tr[0] * shape_[1], y1 + tr[4] * shape_[0]
    bx = box(x0, y0, x1, y1)
    land = np.asarray(z["gsw_max"]) == 0                     # never water in GSW (1984-2021)
    osm = json.loads((RAW / "generality" / basin / "osm_gsl.json").read_text(encoding="utf-8"))

    def line_xy(el):
        return [fwd(p["lon"], p["lat"]) for p in el["geometry"]]

    rail = [line_xy(e) for e in osm["elements"] if e["type"] == "way" and e.get("tags", {}).get("railway")]
    caus = [line_xy(e) for e in osm["elements"] if e["type"] == "way" and e.get("tags", {}).get("highway")]
    # 1a. the railway as one west-east polyline: lake-crossing vertices (west of Ogden, 112.0 W)
    xmax = fwd(-112.0, 41.2)[0]
    pts = sorted({p for l in rail for p in l if p[0] < xmax})
    pts = [pts[0]] + [p for p in pts[1:]]
    rl = LineString(pts).simplify(30.0)
    rcoords = list(rl.coords)
    rcut = LineString([(x0 - 1000, rcoords[0][1])] + rcoords + [(x1 + 1000, rcoords[-1][1])])
    parts = list(shapely.ops.split(bx, rcut).geoms)
    north_pt = Point(fwd(-112.75, 41.45))
    north_side = [p for p in parts if p.contains(north_pt)][0]
    south_side = unary_union([p for p in parts if not p.contains(north_pt)])
    # 1b. Promontory medial line: start on the peninsula at 41.30 N, 112.50 W, trace north and south
    r0, c0 = to_rc(fwd(-112.50, 41.30), tr)
    lab, _ = ndimage.label(land)
    # stop where the land run gets wider than 20 km (the peninsula's base joins the mainland)
    wmax = int(20_000 / RES)
    up = G.medial_line(land, (r0, c0), direction=-1, max_shift=4, max_width=wmax)
    down = G.medial_line(land, (r0 + 1, c0), direction=+1, max_shift=4, max_width=wmax)
    keep = sorted(down + up)                                    # north (small row) first
    med = [to_xy(p, tr) for p in keep]
    north_end = med[0]
    south_end = med[-1]
    mline = LineString([(north_end[0], y1 + 1000)] + med + [(south_end[0], south_end[1] - 20_000)])
    # QC: the straight northern extension should cross no near-maximum water
    hi_any = np.zeros(shape_, bool)
    for k in z.files:
        if k.startswith("high_"):
            hi_any |= np.asarray(z[k]) > 0
    ext_rows = range(0, max(to_rc(north_end, tr)[0], 0))
    ext_cross = int(sum(hi_any[r, to_rc(north_end, tr)[1]] for r in ext_rows))
    npieces = list(shapely.ops.split(north_side, mline).geoms)
    north_arm = [p for p in npieces if p.contains(north_pt)][0]
    bear = unary_union([p for p in npieces if not p.contains(north_pt)])
    # 1c. Farmington Bay
    cl = linemerge(unary_union([LineString(c) for c in caus]))
    cl = max(cl.geoms, key=lambda g: g.length) if cl.geom_type == "MultiLineString" else cl
    cc_ = list(cl.coords)
    cw, ce = (cc_[0], cc_[-1]) if cc_[0][0] < cc_[-1][0] else (cc_[-1], cc_[0])
    ri, ci = to_rc(fwd(-112.22, 41.0), tr)                     # Antelope Island
    isl = lab == lab[ri, ci] if land[ri, ci] else None
    if isl is None:
        raise SystemExit("Antelope Island seed is not never-water land")
    rows = np.flatnonzero(isl.any(axis=1))
    tip_r = rows.max()
    tip_c = float(np.flatnonzero(isl[tip_r]).mean())
    imed = G.medial_line(isl, (tip_r, tip_c), direction=-1, max_shift=4)
    imed_xy = [to_xy(p, tr) for p in imed if to_xy(p, tr)[1] < cw[1]]
    tip = to_xy((tip_r, tip_c), tr)
    ring = [ce, (x1 + 1000, ce[1]), (x1 + 1000, tip[1]), tip] + imed_xy + [cw] + cc_[::(1 if cc_[0] == cw else -1)]
    farm = shapely.make_valid(Polygon(ring)).buffer(0)
    farm = farm.intersection(south_side)
    south_arm = south_side.difference(farm)
    # 2. salt ponds
    ponds = []
    for e in osm["elements"]:
        if e.get("tags", {}).get("landuse") != "salt_pond":
            continue
        if e["type"] == "way" and len(e["geometry"]) >= 4:
            ponds.append(shapely.make_valid(Polygon(line_xy(e))))
        elif e["type"] == "relation":
            ls = [LineString([fwd(p["lon"], p["lat"]) for p in m["geometry"]]) for m in e.get("members", [])
                  if m.get("role") == "outer" and m.get("geometry")]
            ponds += list(polygonize(unary_union(ls)))
    ponds = unary_union(ponds).buffer(POND_BUFFER_M) if ponds else Polygon()
    # 3. extremes per arm
    arms = {"north": north_arm, "south": south_arm}
    win = json.loads(str(z["windows"]))
    T = Affine(*tr[:6])
    armlab = rasterize([(arms["north"], 1), (arms["south"], 2)], out_shape=shape_, transform=T, fill=0, dtype="uint8")
    high = np.zeros(shape_, bool)
    low = np.zeros(shape_, bool)
    for k, arm in ((1, "north"), (2, "south")):
        high |= (armlab == k) & (np.asarray(z[f"high_{win[arm]['high_year']}"]) > 0)
        low |= (armlab == k) & (np.asarray(z[f"low_{win[arm]['low_year']}"]) > 0)
    other = armlab == 0
    high |= other & (np.asarray(z[f"high_{win['south']['high_year']}"]) > 0)
    low |= other & (np.asarray(z[f"low_{win['south']['low_year']}"]) > 0)
    # 4. open water per arm (5 km disk, >= 80% water), 5. zone and cover
    rad = int(round(G.OPEN_WATER_RADIUS_M / RES))
    seed = np.zeros(shape_, bool)
    for k in (1, 2):
        # the arm's main water body: the largest component of its open-water mask, so that a
        # walled evaporation pond large enough to pass the 5 km test is not a seed
        ow = G.open_water(low & (armlab == k), rad) & (armlab == k)
        seed |= G.largest_component(ow)
    zone = G.zone_v6(high, low, seed, int(round(2000 / RES))) & (armlab > 0)
    tiles = G.tile_cover(zone, tr, TILE_M, 1)
    tile_polys = [box(i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M) for i, j in tiles]
    cover = unary_union(tile_polys).intersection(unary_union(list(arms.values())))
    # outputs
    od = RESULTS / "generality" / basin
    od.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "properties": {"arm": "north", "role": "sector", "gauge": b["gauges"]["north"]},
              "geometry": mapping(north_arm)},
             {"type": "Feature", "properties": {"arm": "south", "role": "sector", "gauge": b["gauges"]["south"]},
              "geometry": mapping(south_arm)},
             {"type": "Feature", "properties": {"arm": "bear_river_bay", "role": "excluded"}, "geometry": mapping(bear)},
             {"type": "Feature", "properties": {"arm": "farmington_bay", "role": "excluded"}, "geometry": mapping(farm)}]
    gj = lambda fs: {"type": "FeatureCollection", "crs_proj4": crs, "features": fs}
    (od / "arms_laea.geojson").write_text(json.dumps(gj(feats)), encoding="utf-8")
    (od / "cover_laea.geojson").write_text(json.dumps(gj([{"type": "Feature", "properties": {"i": i, "j": j},
                                                             "geometry": mapping(p)} for (i, j), p in zip(tiles, tile_polys)])),
                                           encoding="utf-8")
    (od / "ponds_laea.geojson").write_text(json.dumps(gj([{"type": "Feature", "properties": {"buffer_m": POND_BUFFER_M},
                                                             "geometry": mapping(ponds)}])), encoding="utf-8")
    zpolys = [shape(g).simplify(RES / 2) for g, v in shapes(zone.astype(np.uint8), mask=zone, transform=T) if v == 1]
    zll = [stransform(inv, p.segmentize(1000)) for p in zpolys if p.area > 0]
    (od / "zone_lonlat.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(p)} for p in zll]}), encoding="utf-8")
    cll = stransform(inv, cover.segmentize(1000))
    (od / "cover_lonlat.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(cll)}]}), encoding="utf-8")
    np.savez_compressed(INTERIM / "generality" / basin / "static_250m.npz", transform=np.array(tr), high=high, low=low,
                        seed=seed, zone=zone, armlab=armlab, land=land)
    km2 = RES * RES / 1e6
    rec = {"grid_250m": {"transform": tr, "shape": list(shape_)}, "windows": win,
           "areas_km2": {"north_arm": round(north_arm.area / 1e6, 1), "south_arm": round(south_arm.area / 1e6, 1),
                         "bear_river_bay_excluded": round(bear.area / 1e6, 1),
                         "farmington_bay_excluded": round(farm.area / 1e6, 1), "salt_ponds_buffered": round(ponds.area / 1e6, 1),
                         "high": round(high.sum() * km2, 1), "low": round(low.sum() * km2, 1),
                         "open_water_seed": {a: round(float((seed & (armlab == k)).sum() * km2), 1) for k, a in ((1, "north"), (2, "south"))},
                         "zone": round(zone.sum() * km2, 1), "cover": round(cover.area / 1e6, 1),
                         "cover_by_arm": {a: round(cover.intersection(p).area / 1e6, 1) for a, p in arms.items()}},
           "n_cover_tiles": len(tiles), "n_zone_polygons": len(zll),
           "railway_vertices": len(rcoords), "promontory_medial_rows": len(keep),
           "promontory_north_extension_high_water_cells": ext_cross,
           "antelope_island_tip_lonlat": [round(v, 4) for v in inv(*tip)],
           "causeway_ends_lonlat": [[round(v, 4) for v in inv(*cw)], [round(v, 4) for v in inv(*ce)]],
           "rules": "Amendment 16 items 4 and 6"}
    (od / "static.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def main(argv) -> int:
    basin = argv[0]
    if basin == "great_salt_lake":
        return gsl(basin)
    raise SystemExit("only great_salt_lake has a static-layer builder")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
