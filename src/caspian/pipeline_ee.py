"""Cost-efficient Earth Engine pipeline for A3/A6 (D-015).

Stage 0  active zone: the only area where waterlines can occur between 1984 and
         2026 = JRC GSW pixels whose water status changed (0 < occurrence < 100)
         plus pixels permanently wet through 2021 but dry in a 2025-2026 low-stand
         composite; dilated by 2 km and clipped to the domain.
Stage 1  per-scene table: MNDWI histogram inside the zone (coarse scale), Otsu
         threshold, and in-zone clear / snow-ice fractions for screening. One
         reduceRegion per scene, exported as a table.
Stage 2  per-pixel aggregation: classification is band arithmetic with the
         scene's pre-computed threshold (no per-tile reductions), summed into
         (epoch, level-bin) groups over the zone only.
"""

from __future__ import annotations

import ee
from shapely.geometry import mapping

from . import masks, water
from .geography import DOMAIN

LOWSTAND = ("2025-05-01", "2026-09-30")
ZONE_DILATE_M = 2000
OCC_LO, OCC_HI = 2, 98   # GSW occurrence (%) band that marks a changing water status (D-023)
ZONE_SCALE = 250   # zone mask grid (m)
HIST_SCALE = 60    # per-scene MNDWI histogram sampling for Otsu (m)


def domain_geometry():
    return ee.Geometry(mapping(DOMAIN))


def lowstand_water(region=None):
    """Median-MNDWI water mask from Landsat 8/9 May-September scenes of 2025-2026."""
    region = region or domain_geometry()
    cols = [masks.landsat(s, region, *LOWSTAND) for s in ("OLI_L8", "OLI_L9")]
    col = cols[0].merge(cols[1]).filter(ee.Filter.calendarRange(5, 9, "month"))
    m = col.map(lambda im: im.normalizedDifference(["green", "swir1"]).rename("mndwi")).median()
    return m.gt(0).rename("low_water")


def active_zone(region=None, scale: float = 250):
    """Zone mask computed on a coarse grid (D-019): GSW occurrence and the low-stand
    composite are evaluated at ``scale`` metres (Earth Engine reads pyramid levels),
    then dilated by ZONE_DILATE_M with a pixel kernel. A 2 km dilation at 60 m cost
    17.7 EECU-h for the domain (D-019); at 250 m the kernel is 8 px."""
    region = region or domain_geometry()
    proj = ee.Projection("EPSG:4326").atScale(scale)
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    occ = gsw.select("occurrence").unmask(0).reproject(proj)
    low = lowstand_water(region).reproject(proj)
    changed = occ.gt(OCC_LO).And(occ.lt(OCC_HI))
    lost_since = occ.gte(OCC_HI).And(low.Not())
    radius_px = max(int(round(ZONE_DILATE_M / scale)), 1)
    zone = changed.Or(lost_since).focalMax(radius_px, "circle", "pixels").reproject(proj)
    return zone.selfMask().clip(region).rename("zone")


def scene_table(col, zone, scale: float = HIST_SCALE, min_samples: int = 300):
    """Stage-1 table: one feature per scene with the Otsu threshold and in-zone
    observation fractions."""
    z = zone.unmask(0).gt(0)

    def feat(img):
        m = img.normalizedDifference(["green", "swir1"]).rename("mndwi")
        clear = img.select("green").mask().gt(0)
        snow = img.select("qa_snow").unmask(0).gt(0)
        layers = ee.Image.cat([m.updateMask(z), z.rename("zone_px"), z.And(clear).rename("zone_clear"),
                               z.And(snow).rename("zone_snow")])
        geom = img.geometry()
        hist = layers.select("mndwi").reduceRegion(
            ee.Reducer.histogram(256, 0.01).combine(ee.Reducer.count(), "", True),
            geom, scale, maxPixels=1e9, bestEffort=True, tileScale=2)
        sums = layers.select(["zone_px", "zone_clear", "zone_snow"]).reduceRegion(
            ee.Reducer.sum(), geom, scale * 4, maxPixels=1e9, bestEffort=True, tileScale=2)
        n = ee.Number(ee.Algorithms.If(hist.get("mndwi_count"), hist.get("mndwi_count"), 0))
        ok = n.gte(min_samples)
        thr = ee.Algorithms.If(ok, water.ee_otsu(hist.get("mndwi_histogram")), None)
        return ee.Feature(geom.centroid(100), {
            "scene": img.get("system:index"), "sensor": img.get("sensor"),
            "time": img.get("system:time_start"), "otsu_threshold": thr, "otsu_n": n,
            "zone_px": sums.get("zone_px"), "zone_clear": sums.get("zone_clear"),
            "zone_snow": sums.get("zone_snow"),
            "cloud_scene": img.get("CLOUD_COVER")})

    return ee.FeatureCollection(col.map(feat))


def classify_with_threshold(img, threshold, ndvi_veg: float = water.NDVI_VEG):
    """Stage-2 per-pixel classes using a pre-computed scene threshold."""
    m = img.normalizedDifference(["green", "swir1"])
    n = img.normalizedDifference(["nir", "red"])
    w = m.gt(ee.Number(threshold))
    veg = w.Not().And(n.gt(ndvi_veg))
    return (ee.Image(water.DRY).where(veg, water.VEG).where(w, water.WATER)
            .updateMask(m.mask()).uint8().rename("class"))


def group_counts(classified, zone, name: str):
    """Water and valid (water or dry) counts of a classified collection; zero where
    no scene observed the pixel. Returns two uint8 bands (capped at 255)."""
    zero = ee.ImageCollection([ee.Image.constant(0).uint8()])
    w = zero.merge(classified.map(lambda im: im.eq(water.WATER).uint8())).sum()
    v = zero.merge(classified.map(lambda im: im.lte(water.WATER).uint8())).sum()
    return (ee.Image.cat([w.min(255).uint8().rename(f"w_{name}"), v.min(255).uint8().rename(f"v_{name}")])
            .updateMask(zone.unmask(0).gt(0)))
