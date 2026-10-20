"""Earth Engine collections with harmonised band names and quality masks.

Landsat Collection 2 Level-2 surface reflectance (scale 2.75e-5, offset -0.2)
and Sentinel-2 L2A harmonised (scale 1e-4) with Cloud Score+.

Common band names: blue, green, red, nir, swir1, swir2 (reflectance, unitless).

Landsat QA_PIXEL bits (USGS Collection 2 Level-2 Science Product Guide):
0 fill, 1 dilated cloud, 2 cirrus (OLI only), 3 cloud, 4 cloud shadow, 5 snow,
6 clear, 7 water. QA_RADSAT bit i set = band i saturated.
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
COMMON = ["blue", "green", "red", "nir", "swir1", "swir2"]

# Landsat 7 Scan Line Corrector failure (USGS): ETM+ scenes after this date are SLC-off.
L7_SLC_OFF = "2003-05-31"


def landsat(sensor: str, region, start: str, end: str):
    """Harmonised, masked Landsat collection for one sensor."""
    import ee

    cid, bands, radsat_bits = LANDSAT[sensor]

    def prep(img):
        qa = img.select("QA_PIXEL")
        bad = (qa.bitwiseAnd(1 << 0).neq(0)            # fill
               .Or(qa.bitwiseAnd(1 << 1).neq(0))       # dilated cloud
               .Or(qa.bitwiseAnd(1 << 3).neq(0))       # cloud
               .Or(qa.bitwiseAnd(1 << 4).neq(0))       # cloud shadow
               .Or(qa.bitwiseAnd(1 << 5).neq(0)))      # snow / ice
        if sensor.startswith("OLI"):
            bad = bad.Or(qa.bitwiseAnd(1 << 2).neq(0))  # cirrus
        sat_mask = sum(1 << b for b in radsat_bits)
        bad = bad.Or(img.select("QA_RADSAT").bitwiseAnd(sat_mask).neq(0))
        sr = img.select(bands, COMMON).multiply(2.75e-05).add(-0.2)
        snowice = qa.bitwiseAnd(1 << 5).neq(0).rename("qa_snow")
        return (sr.updateMask(bad.Not()).addBands(snowice)
                .copyProperties(img, ["system:time_start", "CLOUD_COVER", "WRS_PATH", "WRS_ROW",
                                      "SPACECRAFT_ID", "SUN_ELEVATION"])
                .set("sensor", sensor))

    return ee.ImageCollection(cid).filterBounds(region).filterDate(start, end).map(prep)


def sentinel2(region, start: str, end: str, cs_threshold: float = 0.60):
    """Harmonised Sentinel-2 L2A masked with Cloud Score+ (cs_cdf >= threshold kept)."""
    import ee

    cid, bands = S2
    cs = ee.ImageCollection(S2_CSPLUS)
    col = ee.ImageCollection(cid).filterBounds(region).filterDate(start, end).linkCollection(
        cs, ["cs_cdf"])

    def prep(img):
        clear = img.select("cs_cdf").gte(cs_threshold)
        sr = img.select(bands, COMMON).multiply(1e-4)
        return (sr.updateMask(clear)
                .copyProperties(img, ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE", "MGRS_TILE",
                                      "SPACECRAFT_NAME"])
                .set("sensor", "MSI_S2"))

    return col.map(prep)
