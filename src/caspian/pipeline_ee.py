"""Cost-efficient Earth Engine pipeline for water classification and water frequency.

Stage 0  active zone (v6): the sea-connected band between the near-maximum
         1995-96 and the near-minimum 2025-26 water extents, dilated by 2 km at 250 m
         on the equal-area grid. Its 5 km tile cover clips stage-1 and
         stage-2 inputs, so that computation outside the zone is skipped.
Stage 1  per-scene table: fixed-bin MNDWI histogram of clear in-zone pixels
         (240 m, native projection), Otsu threshold, the histogram itself, and in-zone
         clear / snow-ice counts. One reduceRegion per scene, run as profiled
         interactive batches (scripts/a3_stage1_interactive.py).
Stage 2  per-pixel aggregation: classification is band arithmetic with the
         acquisition's pre-computed threshold; overlapping scenes of one acquisition
         are mosaicked so each overpass counts once; water and valid counts are
         summed in one pass per (epoch, level-bin) group, as uint16.
"""

from __future__ import annotations

import ee
import pyproj
from shapely.geometry import mapping

from . import masks, water
from .config import CRS_EQUAL_AREA
from .geography import DOMAIN

LOWSTAND = ("2025-05-01", "2026-09-30")
HIGHSTAND = ("1995-05-01", "1996-09-30")   # around the record maximum
ZONE_DILATE_M = 2000
OCC_LO, OCC_HI = 2, 98   # GSW occurrence (%) band that marks a changing water status
ZONE_SCALE = 250   # zone mask grid (m)
ZONE_TILE_M = 5000  # tile cover of the zone used to clip stage-2 inputs (m)
HIST_SCALE = 240   # per-scene MNDWI histogram sampling for Otsu (m); 240 m matches 60 m
                   # thresholds to <= 2 bins on 19 test scenes at 1/16 of the cost
HIGH_PCT, LOW_PCT = 90, 10   # extreme-extent composites
MIN_OBS_HIGH = 3   # fewer clear 1995-96 observations than this: use GSW max_extent instead
HIGH_MIN_FRAC, LOW_MIN_FRAC = 0.10, 0.90   # zone v6 extents: share of clear observations that are water
DARK_WATER = 0.03  # zone composites: NIR and SWIR1 both below this = clear deep water
CONN_SCALE = 1000  # grid of the sea-connectivity pass (m)
SEA_RADIUS_M, SEA_MIN_FRAC = 20_000, 0.80   # open sea: 2025-26 water whose 20 km surroundings are >= 80% water
CONN_MAX_M = 200_000   # cumulativeCost reach from the open sea through 1995-96 water
GSW = "JRC/GSW1_4/GlobalSurfaceWater"

# Earth Engine parses WKT1 (GDAL flavour), not the WKT2 that pyproj writes by default
LAEA_WKT = pyproj.CRS(CRS_EQUAL_AREA).to_wkt(version="WKT1_GDAL")


def laea(scale: float):
    """The equal-area projection at ``scale`` metres."""
    return ee.Projection(LAEA_WKT).atScale(scale)


def laea_transform(res: float, bounds_lonlat):
    """crsTransform and pixel dimensions of an equal-area grid aligned to multiples
    of ``res`` that covers a lon/lat bounding box (w, s, e, n)."""
    import math

    t = pyproj.Transformer.from_crs("EPSG:4326", CRS_EQUAL_AREA, always_xy=True)
    w, s, e, n = bounds_lonlat
    xs, ys = t.transform([w, e, w, e], [s, s, n, n])
    x0, x1 = math.floor(min(xs) / res) * res, math.ceil(max(xs) / res) * res
    y0, y1 = math.floor(min(ys) / res) * res, math.ceil(max(ys) / res) * res
    return [res, 0, x0, 0, -res, y1], (int(round((x1 - x0) / res)), int(round((y1 - y0) / res)))


def domain_geometry():
    """The study domain with planar (lon/lat straight-line) edges, as in shapely.
    ee.Geometry reads GeoJSON with geodesic edges by default; the domain's 8.2-degree
    northern edge then bulges up to about 8 km north of the local polygon. Zone v5,
    the extents_v6 export and zone v6 were built with geodesic edges, a superset in
    the north (southern edges move by under 1 km), and everything is clipped to the
    planar domain locally."""
    return ee.Geometry(mapping(DOMAIN), None, False)


def _mndwi_col(col):
    return col.map(lambda im: water.ee_nd(im, "green", "swir1").rename("mndwi"))


def lowstand_water(region=None):
    """Median-MNDWI water mask from Landsat 8/9 May-September scenes of 2025-2026
    (zone v3/v4; kept for reproducing those assets)."""
    region = region or domain_geometry()
    cols = [masks.landsat(s, region, *LOWSTAND) for s in ("OLI_L8", "OLI_L9")]
    col = cols[0].merge(cols[1]).filter(ee.Filter.calendarRange(5, 9, "month"))
    return _mndwi_col(col).median().gt(0).rename("low_water")


def highstand_water(region=None):
    """Median-MNDWI water mask from Landsat 5 TM May-September scenes of 1995-1996
    (zone v4; kept for reproducing that asset)."""
    region = region or domain_geometry()
    col = masks.landsat("TM_L5", region, *HIGHSTAND).filter(ee.Filter.calendarRange(5, 9, "month"))
    return _mndwi_col(col).median().gt(0).rename("high_water")


def active_zone(region=None, scale: float = 250):
    """Zone v3, on an EPSG:4326 grid; kept for reproducing that asset."""
    region = region or domain_geometry()
    proj = ee.Projection("EPSG:4326").atScale(scale)
    occ = ee.Image(GSW).select("occurrence").unmask(0).reproject(proj)
    low = lowstand_water(region).reproject(proj)
    changed = occ.gt(OCC_LO).And(occ.lt(OCC_HI))
    lost_since = occ.gte(OCC_HI).And(low.Not())
    radius_px = max(int(round(ZONE_DILATE_M / scale)), 1)
    zone = changed.Or(lost_since).focalMax(radius_px, "circle", "pixels").reproject(proj)
    return zone.selfMask().clip(region).rename("zone")


def active_zone_v4(region=None, scale: float = 250):
    """Zone v4, on an EPSG:4326 grid; superseded by v5."""
    region = region or domain_geometry()
    proj = ee.Projection("EPSG:4326").atScale(scale)
    hi = highstand_water(region).unmask(0).reproject(proj)
    lo = lowstand_water(region).unmask(0).reproject(proj)
    band = hi.And(lo.Not())
    edge_hi = hi.focalMax(1, "square", "pixels").neq(hi.focalMin(1, "square", "pixels"))
    edge_lo = lo.focalMax(1, "square", "pixels").neq(lo.focalMin(1, "square", "pixels"))
    radius_px = max(int(round(ZONE_DILATE_M / scale)), 1)
    zone = band.Or(edge_hi).Or(edge_lo).focalMax(radius_px, "circle", "pixels").reproject(proj)
    return zone.selfMask().clip(region).rename("zone")


def highstand_extent(region=None, scale: float = ZONE_SCALE):
    """Near-maximum water extent of the 1995-96 high stand: water in at least
    100 - HIGH_PCT % of clear May-September Landsat-5 observations (MNDWI
    HIGH_PCT-th percentile > 0). Where fewer than MIN_OBS_HIGH clear observations
    exist, the GSW maximum extent (1984-2021) is used instead, and the band
    ``filled`` records where."""
    region = region or domain_geometry()
    proj = laea(scale)
    col = (masks.landsat("TM_L5", region, *HIGHSTAND, bands=["green", "swir1"])
           .filter(ee.Filter.calendarRange(5, 9, "month")))
    m = _mndwi_col(col)
    wet = m.reduce(ee.Reducer.percentile([HIGH_PCT])).gt(0).unmask(0)
    n = m.count().unmask(0)
    filled = n.lt(MIN_OBS_HIGH)
    gsw_max = ee.Image(GSW).select("max_extent").unmask(0)
    hi = wet.where(filled, gsw_max)
    return ee.Image.cat([hi.rename("high_extent"), filled.rename("filled")]).reproject(proj)


def lowstand_extent(region=None, scale: float = ZONE_SCALE):
    """Near-minimum water extent of the 2025-26 low stand: water in at least
    100 - LOW_PCT % of clear May-September Landsat-8/9 observations (MNDWI
    LOW_PCT-th percentile > 0)."""
    region = region or domain_geometry()
    proj = laea(scale)
    cols = [masks.landsat(s, region, *LOWSTAND, bands=["green", "swir1"]) for s in ("OLI_L8", "OLI_L9")]
    col = cols[0].merge(cols[1]).filter(ee.Filter.calendarRange(5, 9, "month"))
    lo = _mndwi_col(col).reduce(ee.Reducer.percentile([LOW_PCT])).gt(0).unmask(0)
    return lo.rename("low_extent").reproject(proj)


def active_zone_v5(region=None, scale: float = ZONE_SCALE):
    """Zone v5: pixels inside the near-maximum 1995-96 extent but outside the
    near-minimum 2025-26 extent, plus both shorelines, dilated by ZONE_DILATE_M on
    the equal-area grid (an isotropic kernel, unlike v3/v4 on EPSG:4326).
    Bands: zone (1 inside, masked outside) and filled (GSW gap fill used)."""
    region = region or domain_geometry()
    proj = laea(scale)
    hi_img = highstand_extent(region, scale)
    hi = hi_img.select("high_extent")
    lo = lowstand_extent(region, scale)
    band = hi.And(lo.Not())
    edge_hi = hi.focalMax(1, "square", "pixels").neq(hi.focalMin(1, "square", "pixels"))
    edge_lo = lo.focalMax(1, "square", "pixels").neq(lo.focalMin(1, "square", "pixels"))
    radius_px = max(int(round(ZONE_DILATE_M / scale)), 1)
    zone = band.Or(edge_hi).Or(edge_lo).focalMax(radius_px, "circle", "pixels").reproject(proj)
    return (ee.Image.cat([zone.rename("zone"), hi_img.select("filled").uint8()])
            .updateMask(zone).clip(region))


def zone_water(img):
    """Water indicator for the zone composites only (not the classification):
    MNDWI > 0, or NIR and SWIR1 both below DARK_WATER. Clear deep water in the south
    of the domain has near-zero NIR and SWIR1 and an MNDWI that scatters around 0."""
    m = water.ee_nd(img, "green", "swir1")
    dark = img.select("nir").lt(DARK_WATER).And(img.select("swir1").lt(DARK_WATER))
    return m.gt(0).Or(dark).rename("w")


def _water_fraction(col):
    return col.map(zone_water).mean()


def extents_v6(region=None, scale: float = ZONE_SCALE):
    """Near-maximum 1995-96 and near-minimum 2025-26 water extents (zone v6):
    water in at least HIGH_MIN_FRAC (resp. LOW_MIN_FRAC) of clear May-September
    observations, by zone_water(). GSW max_extent fills the 1995-96 extent where
    there are fewer than MIN_OBS_HIGH clear observations. Bands: high, low, filled."""
    region = region or domain_geometry()
    bands = ["green", "nir", "swir1"]
    hi_col = masks.landsat("TM_L5", region, *HIGHSTAND, bands=bands).filter(ee.Filter.calendarRange(5, 9, "month"))
    lo_cols = [masks.landsat(s, region, *LOWSTAND, bands=bands) for s in ("OLI_L8", "OLI_L9")]
    lo_col = lo_cols[0].merge(lo_cols[1]).filter(ee.Filter.calendarRange(5, 9, "month"))
    n_hi = hi_col.map(zone_water).count().unmask(0)
    filled = n_hi.lt(MIN_OBS_HIGH)
    hi = _water_fraction(hi_col).gte(HIGH_MIN_FRAC).unmask(0).where(filled, ee.Image(GSW).select("max_extent").unmask(0))
    lo = _water_fraction(lo_col).gte(LOW_MIN_FRAC).unmask(0)
    return ee.Image.cat([hi.rename("high"), lo.rename("low"), filled.rename("filled")]).reproject(laea(scale))


def sea_connected(high, low, scale: float = CONN_SCALE):
    """Cells of the 1995-96 extent connected to the open sea through 1995-96 water:
    least-cost distance (at most CONN_MAX_M) from the open sea, on a ``scale`` grid
    with max pooling so narrow channels stay connected. The open sea is 2025-26
    water whose SEA_RADIUS_M surroundings are at least SEA_MIN_FRAC water, which
    excludes inland lakes. Seeding from the whole sea front keeps the propagation
    distance short (Earth Engine's cumulativeCost does not carry across many tiles).
    1 = connected."""
    proj = laea(scale)
    hi_c = high.reduceResolution(ee.Reducer.max(), maxPixels=1024).reproject(proj)
    lo_c = low.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(proj)
    radius_px = int(round(SEA_RADIUS_M / scale))
    sea = lo_c.gte(0.5).And(lo_c.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(radius_px, "pixels"))
                            .reproject(proj).gte(SEA_MIN_FRAC))
    cost = ee.Image(1).updateMask(hi_c.gt(0).Or(sea)).reproject(proj)
    dist = cost.cumulativeCost(sea.selfMask(), CONN_MAX_M, False)
    return dist.mask().gt(0).unmask(0).reproject(proj).rename("connected")


def active_zone_v6(extents, region=None, scale: float = ZONE_SCALE):
    """Zone v6 from the exported extents_v6 image: the sea-connected band
    between the 1995-96 near-maximum and the 2025-26 near-minimum water extents,
    dilated by ZONE_DILATE_M on the equal-area grid. Bands: zone (1 inside, masked
    outside), filled (GSW gap fill)."""
    region = region or domain_geometry()
    proj = laea(scale)
    hi, lo = extents.select("high"), extents.select("low")
    conn = sea_connected(hi, lo).reproject(proj).focalMax(1, "square", "pixels").reproject(proj)
    band = hi.And(lo.Not()).And(conn)
    radius_px = max(int(round(ZONE_DILATE_M / scale)), 1)
    zone = band.focalMax(radius_px, "circle", "pixels").reproject(proj)
    return (ee.Image.cat([zone.rename("zone"), extents.select("filled").uint8()])
            .updateMask(zone).clip(region))


def zone_cover(path):
    """The zone's tile cover (a GeoJSON FeatureCollection written by scripts/a2_zone.py)
    as one ee.Geometry."""
    import json
    from pathlib import Path

    feats = json.loads(Path(path).read_text(encoding="utf-8"))["features"]
    return ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"]), {}) for f in feats]).geometry()


def zone_tiles(zone, region=None, cell_m: float = ZONE_TILE_M):
    """Polygons of the ZONE_TILE_M equal-area cells that contain any zone pixel
    (a coarse cover used to clip stage-2 inputs)."""
    region = region or domain_geometry()
    proj = laea(cell_m)
    coarse = (zone.select("zone").unmask(0).gt(0)
              .reduceResolution(ee.Reducer.max(), maxPixels=1024).reproject(proj))
    return coarse.selfMask().reduceToVectors(geometry=region, crs=proj, geometryType="polygon",
                                             eightConnected=False, maxPixels=1e9, bestEffort=False)


def scene_table(col, zone, scale: float = HIST_SCALE, min_samples: int = 300, clip=None):
    """Stage-1 table: one feature per scene with the Otsu threshold (fixed-bin MNDWI
    histogram of clear in-zone pixels, the binning of water.otsu_threshold) and the
    in-zone pixel counts used for screening. One reduceRegion per scene, in the
    scene's native projection (reprojecting to the equal-area grid cost a fifth of
    the total in the profiler). Counts are of ``scale``-metre pixels:
    zone_px (zone), zone_clear (clear, i.e. counted in the histogram), zone_snow
    (QA or NDSI snow/ice), zone_snow_ndsi (NDSI test alone)."""
    z = zone.select(0).unmask(0).gt(0)
    if clip is not None:
        # skip the footprint outside the zone cover: 35% cheaper, same thresholds
        col = col.map(lambda im: im.clip(clip))
    reducer = (water.ee_histogram_reducer().combine(ee.Reducer.count(), "", True)
               .combine(ee.Reducer.count().setOutputs(["zone_snow"]), "", False)
               .combine(ee.Reducer.count().setOutputs(["zone_snow_ndsi"]), "", False)
               .combine(ee.Reducer.count().setOutputs(["zone_snow_qa"]), "", False)
               .combine(ee.Reducer.count().setOutputs(["zone_px"]), "", False))

    def feat(img):
        m = water.ee_nd(img, "green", "swir1").rename("mndwi").updateMask(z)
        layers = ee.Image.cat([m, img.select("qa_snow").And(z).selfMask(),
                               img.select("qa_snow_ndsi").And(z).selfMask(),
                               img.select("qa_snow_qa").And(z).selfMask(), z.selfMask()])
        geom = img.geometry()
        r = layers.reduceRegion(reducer, geom, scale=scale, maxPixels=1e9, bestEffort=True, tileScale=2)
        hist_key = ee.List(r.keys().filter(ee.Filter.stringContains("item", "histogram"))).get(0)
        n = ee.Number(ee.Algorithms.If(r.get("count"), r.get("count"), 0))
        ok = n.gte(min_samples)
        thr = ee.Algorithms.If(ok, water.ee_otsu_fixed(r.get(hist_key)), None)
        # the 256 bucket counts, so threshold rules can be evaluated locally
        hist = ee.Algorithms.If(n.gt(0), ee.Array(r.get(hist_key)).slice(1, 1, 2).project([0]).toList(), None)
        return ee.Feature(geom.centroid(100), {
            "scene": img.get("system:index"), "sensor": img.get("sensor"), "acq": img.get("acq"),
            "time": img.get("system:time_start"), "otsu_threshold": thr, "otsu_n": n,
            "zone_px": r.get("zone_px"), "zone_clear": n,
            "zone_snow": r.get("zone_snow"), "zone_snow_ndsi": r.get("zone_snow_ndsi"),
            "zone_snow_qa": r.get("zone_snow_qa"),
            "cloud_scene": img.get("cloud_scene"), "hist": hist})

    return ee.FeatureCollection(col.map(feat))


def classify_with_threshold(img, threshold, ndvi_veg: float = water.NDVI_VEG):
    """Per-pixel classes (0 dry, 1 water, 2 emergent vegetation) using a
    pre-computed scene threshold."""
    m = water.ee_nd(img, "green", "swir1")
    n = water.ee_nd(img, "nir", "red")
    w = m.gt(ee.Number(threshold))
    veg = w.Not().And(n.gt(ndvi_veg).unmask(0))
    known = w.Or(n.mask())          # water needs only MNDWI; dry or vegetated needs NDVI
    return (ee.Image(water.DRY).where(veg, water.VEG).where(w, water.WATER)
            .updateMask(m.mask()).updateMask(known).uint8().rename("class"))


def water_valid(img, threshold, ndvi_veg: float = water.NDVI_VEG, observed: bool = False):
    """Stage-2 per-scene [w, v] pair: w = water, v = water or dry (emergent
    vegetation is not a valid observation, PREREG section 3); masked = unobserved.
    Same classes as classify_with_threshold, in the form the counts need. All bands
    carry one mask, so w <= v in every image and every per-acquisition mosaic (a
    separate NDVI mask on v gave w > v). With ``observed``, a third band o = 1
    wherever the pixel was classified, so o - v counts emergent vegetation."""
    m = water.ee_nd(img, "green", "swir1")
    n = water.ee_nd(img, "nir", "red")
    w = m.gt(ee.Number(threshold))
    v = w.Or(n.lte(ndvi_veg).unmask(0))
    known = w.Or(n.mask())
    bands = [w, v, w.Or(w.Not())] if observed else [w, v]
    names = ["w", "v", "o"] if observed else ["w", "v"]
    return ee.Image.cat(bands).rename(names).uint16().updateMask(m.mask()).updateMask(known)


def per_acquisition(col):
    """Mosaic the scenes of each acquisition (property ``acq``) so that an overpass
    covering a pixel twice (adjacent rows, overlapping tiles) counts once."""
    keys = ee.List(col.aggregate_array("acq")).distinct()
    return ee.ImageCollection(keys.map(
        lambda k: ee.ImageCollection(col.filter(ee.Filter.eq("acq", k))).mosaic()))


def group_counts(wv, name: str, kinds=("w", "v")):
    """Water and valid (and, with kinds w, v, o, observed) counts of a collection of
    water_valid() images, one pass, each acquisition once; zero where no acquisition
    observed the pixel. uint16 bands <kind>_<name>."""
    kinds = list(kinds)
    zero = ee.ImageCollection([ee.Image.constant([0] * len(kinds)).rename(kinds).uint16()])
    total = zero.merge(per_acquisition(wv)).sum()
    return total.rename([f"{k}_{name}" for k in kinds]).uint16()
