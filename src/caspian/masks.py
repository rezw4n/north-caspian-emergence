"""Earth Engine collections with harmonised band names and quality masks.

Landsat Collection 2 Level-2 surface reflectance (scale 2.75e-5, offset -0.2)
and Sentinel-2 L2A harmonised (scale 1e-4) with Cloud Score+.

Common band names: blue, green, red, nir, swir1, swir2 (reflectance, unitless).

Landsat QA_PIXEL bits (USGS Collection 2 Level-2 Science Product Guide):
0 fill, 1 dilated cloud, 2 cirrus (OLI only), 3 cloud, 4 cloud shadow, 5 snow,
6 clear, 7 water. QA_RADSAT bit i set = band i saturated.

Snow and ice (PREREG section 3): a pixel is excluded if the product QA flags it
(Landsat QA_PIXEL bit 5; Sentinel-2 Scene Classification class 11) or if it passes
the conservative pass-1 snow test of the Theia LIS algorithm, NDSI > 0.400 and red
reflectance > 0.200 (Gascoin et al. 2019, ESSD 11, Eq. 2 and Table 1). The red
criterion is the one LIS uses to avoid false snow on turbid water. Two diagnostic
bands are kept for the stage-1 accounting, both masked to cloud-free pixels (D-030+;
run 1 counted cloud, shadow and cirrus pixels too): ``qa_snow`` (1 = excluded as
snow or ice: QA flag, or the NDSI test in NDSI_MONTHS) and ``qa_snow_ndsi`` (the
NDSI test alone, in every month) and ``qa_snow_qa`` (the product QA flag alone, in
every month), so that false positives in summer can be
checked.

Every image carries the properties ``sensor``, ``cloud_scene`` (scene cloud
percentage from the product metadata) and ``acq`` (acquisition key: sensor, orbit
and date for Landsat, sensor and datatake for Sentinel-2). Overlapping scenes of
one acquisition share ``acq`` and are mosaicked before counting (D-027).
"""

from __future__ import annotations

LANDSAT = {
    # sensor key: (collection id, SR band names in order blue, green, red, nir, swir1, swir2,
    #              RADSAT bit indices of those bands)
    "TM_L5": ("LANDSAT/LT05/C02/T1_L2", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
              [0, 1, 2, 3, 4, 6]),
    "ETM_L7": ("LANDSAT/LE07/C02/T1_L2", ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
               [0, 1, 2, 3, 4, 6]),
    "OLI_L8": ("LANDSAT/LC08/C02/T1_L2", ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
               [1, 2, 3, 4, 5, 6]),
    "OLI_L9": ("LANDSAT/LC09/C02/T1_L2", ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
               [1, 2, 3, 4, 5, 6]),
}
S2 = ("COPERNICUS/S2_SR_HARMONIZED", ["B2", "B3", "B4", "B8", "B11", "B12"])
S2_CSPLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S2_SCL_SNOW = 11
COMMON = ["blue", "green", "red", "nir", "swir1", "swir2"]
# bands the water classification needs (MNDWI, NDVI, snow test)
CLASSIFY_BANDS = ["green", "red", "nir", "swir1"]

# Theia LIS pass-1 snow test (Gascoin et al. 2019, Table 1: n1, r1)
NDSI_SNOW = 0.400
RED_SNOW = 0.200
# Months in which snow/ice exclusion applies to the NDSI test (both sensors) and to the
# Sentinel-2 SCL flag; the Landsat CFMask snow bit applies in every month (D-063). PREREG
# Amendment 3 rule for the NDSI test, extended to the QA flags by Amendment 9:
# cloud-masked test batches flag 1.2% (Landsat NDSI), 3.1% (Sentinel-2 NDSI) and 1.9%
# (Sentinel-2 SCL class 11) of clear in-zone pixels in July; the rule is evaluated on
# the full stage-1 run 2 (D-030+).
NDSI_MONTHS = (11, 12, 1, 2, 3, 4)

# Landsat 7 Scan Line Corrector failure (USGS): ETM+ scenes after this date are SLC-off.
L7_SLC_OFF = "2003-05-31"


def _select(names, bands):
    """Product band names and common names for the requested common bands."""
    idx = [COMMON.index(b) for b in bands]
    return [names[i] for i in idx], list(bands)


def _cold(time_start):
    """1 in NDSI_MONTHS (when snow and ice are possible on the North Caspian), else 0,
    as a constant image."""
    import ee

    month = ee.Number(ee.Date(time_start).get("month"))
    return ee.Image(ee.Number(ee.Algorithms.If(ee.List(list(NDSI_MONTHS)).contains(month), 1, 0)))


def ndsi_snow(sr, time_start=None):
    """LIS pass-1 snow test on a harmonised reflectance image (needs green, red, swir1).
    With ``time_start``, the test applies only in NDSI_MONTHS (0 elsewhere)."""
    import ee

    ndsi = sr.normalizedDifference(["green", "swir1"])
    flag = ndsi.gt(NDSI_SNOW).And(sr.select("red").gt(RED_SNOW))
    if time_start is None:
        return flag
    month = ee.Number(ee.Date(time_start).get("month"))
    active = ee.List(list(NDSI_MONTHS)).contains(month)
    return flag.And(ee.Image(ee.Number(ee.Algorithms.If(active, 1, 0))))


def landsat(sensor: str, region, start: str, end: str, bands=COMMON, ids=None):
    """Harmonised, masked Landsat collection for one sensor (optionally only the
    scenes whose system:index is in ``ids``)."""
    import ee

    cid, names, radsat_bits = LANDSAT[sensor]
    need = list(dict.fromkeys(list(bands) + ["green", "red", "swir1"]))  # snow test bands
    src, common = _select(names, need)
    bits = [radsat_bits[COMMON.index(b)] for b in need]

    def prep(img):
        qa = img.select("QA_PIXEL")
        bad = (qa.bitwiseAnd(1 << 0).neq(0)            # fill
               .Or(qa.bitwiseAnd(1 << 1).neq(0))       # dilated cloud
               .Or(qa.bitwiseAnd(1 << 3).neq(0))       # cloud
               .Or(qa.bitwiseAnd(1 << 4).neq(0)))      # cloud shadow
        if sensor.startswith("OLI"):
            bad = bad.Or(qa.bitwiseAnd(1 << 2).neq(0))  # cirrus
        bad = bad.Or(img.select("QA_RADSAT").bitwiseAnd(sum(1 << b for b in bits)).neq(0))
        sr = img.select(src, common).multiply(2.75e-05).add(-0.2)
        snow_ndsi = ndsi_snow(sr)                                   # every month (diagnostic)
        snow_qa = qa.bitwiseAnd(1 << 5).neq(0)                       # every month (diagnostic)
        cold = _cold(img.get("system:time_start"))
        # PREREG Amendments 3 and 9, evaluated on stage-1 run 2 (results/qc/snow_rules_run2.json,
        # D-063): the CFMask snow bit flags at most 0.22% of clear in-zone pixels in May-September,
        # so it applies in every month; the NDSI test (3.69%) only in NDSI_MONTHS.
        snow = snow_qa.Or(snow_ndsi.And(cold))
        cloud_free = bad.Not()
        acq = (ee.String(sensor).cat("_").cat(ee.Number(img.get("WRS_PATH")).format("%03d")).cat("_")
               .cat(ee.Date(img.get("system:time_start")).format("YYYYMMdd")))
        return (sr.select(list(bands)).updateMask(cloud_free.And(snow.Not()))
                .addBands([snow.updateMask(cloud_free).rename("qa_snow"),
                           snow_ndsi.updateMask(cloud_free).rename("qa_snow_ndsi"),
                           snow_qa.updateMask(cloud_free).rename("qa_snow_qa")])
                .copyProperties(img, ["system:time_start", "CLOUD_COVER", "WRS_PATH", "WRS_ROW",
                                      "SPACECRAFT_ID", "SUN_ELEVATION"])
                .set({"sensor": sensor, "cloud_scene": img.get("CLOUD_COVER"), "acq": acq}))

    col = ee.ImageCollection(cid).filterBounds(region).filterDate(start, end)
    if ids is not None:
        col = col.filter(ee.Filter.inList("system:index", ids if isinstance(ids, ee.List) else list(ids)))
    return col.map(prep)


def sentinel2(region, start: str, end: str, cs_threshold: float = 0.60, bands=COMMON, ids=None):
    """Harmonised Sentinel-2 L2A masked with Cloud Score+ (cs_cdf >= threshold kept)
    and for snow/ice (Scene Classification class 11, or the LIS pass-1 test)."""
    import ee

    cid, names = S2
    need = list(dict.fromkeys(list(bands) + ["green", "red", "swir1"]))
    src, common = _select(names, need)
    cs = ee.ImageCollection(S2_CSPLUS)
    col = ee.ImageCollection(cid).filterBounds(region).filterDate(start, end)
    if ids is not None:
        col = col.filter(ee.Filter.inList("system:index", ids if isinstance(ids, ee.List) else list(ids)))
    col = col.linkCollection(cs, ["cs_cdf"])

    def prep(img):
        clear = img.select("cs_cdf").gte(cs_threshold)
        sr = img.select(src, common).multiply(1e-4)
        snow_ndsi = ndsi_snow(sr)                                   # every month (diagnostic)
        snow_qa = img.select("SCL").eq(S2_SCL_SNOW)                   # every month (diagnostic)
        snow = snow_qa.Or(snow_ndsi).And(_cold(img.get("system:time_start")))
        # one datatake = one overpass, shared by all its tiles. Fallback: the first token
        # of system:index (<datatake sensing time>_<granule time>_<tile>)
        datatake = ee.Algorithms.If(img.propertyNames().contains("DATATAKE_IDENTIFIER"),
                                    img.get("DATATAKE_IDENTIFIER"),
                                    ee.String(img.get("system:index")).split("_").get(0))
        acq = ee.String("MSI_").cat(ee.String(datatake))
        return (sr.select(list(bands)).updateMask(clear.And(snow.Not()))
                .addBands([snow.updateMask(clear).rename("qa_snow"),
                           snow_ndsi.updateMask(clear).rename("qa_snow_ndsi"),
                           snow_qa.updateMask(clear).rename("qa_snow_qa")])
                .copyProperties(img, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE",
                                      "SPACECRAFT_NAME"])
                .set({"sensor": "MSI_S2", "cloud_scene": img.get("CLOUDY_PIXEL_PERCENTAGE"), "acq": acq}))

    return col.map(prep)
