"""Project-wide constants: paths, projections, study-area bounds.

Anything that defines the analysis (projection, study area, Earth Engine
project) lives here so that it is set once.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
STAGE1 = INTERIM / "stage1_v3"   # stage-1 scene tables in use (run 3: corrected normalized differences)
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics.json"
DOCS = ROOT / "docs"
OUTPUTS = ROOT / "outputs"
FIG_OUT = ROOT / "figures" / "out"   # manuscript figures (figures/README.md)

EE_PROJECT = "ee-rezwanahmed"

# Exchange CRS (geographic) and the equal-area analysis CRS.
CRS_EXCHANGE = "EPSG:4326"
CRS_EQUAL_AREA = (
    "+proj=laea +lat_0=45.5 +lon_0=50 +x_0=0 +y_0=0 "
    "+datum=WGS84 +units=m +no_defs +type=crs"
)

# Search box for the North Caspian (west, south, east, north), degrees WGS84.
# Generous on purpose: it is used only to find granules and scenes. The study-area
# polygon itself is defined and frozen in caspian.geography.
NC_BBOX = (46.0, 43.8, 54.5, 47.6)
