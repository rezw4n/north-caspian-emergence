"""Pre-registered analysis constants (docs/PREREG.md v1.0 and amendments).

One place for every number the pre-registration fixes, so code and Methods text
(via tools/export_params.py and \\metric{params.*}) quote the same values. Changing
any of these is a PREREG amendment, logged in docs/DECISIONS.md.
"""

from __future__ import annotations

# section 4: waterline DEM
BIN_WIDTH_M = 0.10
BIN_WIDTH_SENS_M = (0.05, 0.20)
BIN_ORIGIN_M = -30.00
CONTOUR_LEVEL = 0.5            # water-frequency contour
MIN_VALID_OBS = 5              # valid observations per pixel, bin and epoch
DEM_RES_M = 30
E2_END = "2010-12-31"          # E2 ends; E3 starts 2011-01-01 (E1 ends at the level maximum)

# section 2: independent heights
EMERGENCE_LAG_DAYS = 30        # acquisition later than the pixel's emergence date plus this

# section 6: spatial blocking
BLOCK_SIZE_M = 25_000
BLOCK_SIZE_SENS_M = (12_500, 25_000, 50_000)
SPLIT = {"train": 0.6, "calib": 0.2, "test": 0.2}
SEED = 20261020

# sections 7-8: accuracy and conformal intervals
BOOTSTRAP_N = 2000
CONFORMAL_LEVEL = 0.90
MONDRIAN_MIN_N = 50

# section 9: hindcast
PERMUTATIONS_N = 10_000
ALPHA = 0.05
VEGETATED_SECTOR_FRACTION = 0.10

# section 10: scenarios (m below the 2026 level)
SCENARIOS_M = (0.5, 1.0, 2.0, 3.0, 5.0)
NEAR_TERM_RANGE_M = (0.5, 2.0)

# Amendment 3: edge-guided Otsu robustness sample
EDGE_OTSU_SAMPLE_N = 500

# Amendments 3 and 9: a snow/ice test (NDSI, or a product QA flag) that flags more than
# this share (%) of clear in-zone pixels in any May-September month is applied only in
# the snow season (caspian.masks.NDSI_MONTHS); otherwise it applies in every month
SNOW_RULE_MAX_PCT = 1.0
