# North Caspian pipeline. Run inside the environment, e.g.
#   micromamba run -n caspian make all
#
# Two stages:
#   1. Raw inputs. Acquired once by the named targets below; never run by `make all`.
#        make fetch       network, with credentials: DAHITI_API_KEY, HYDROWEB_API_KEY
#        make ee-inputs   Earth Engine; spends EECU, so it refuses to run without
#                         EE_OK=1 (pilot first and check the budget: CLAUDE.md)
#      They write git-ignored files under data/ and committed acquisition records
#      (results/provenance/*.json with each raw file's SHA-256; for the scene
#      inventory results/inventory/inventory_meta.json, the summary_*.csv tables and
#      results/inventory/scenes_csv.json, the checksum of scenes.csv).
#   2. `make all` rebuilds every result, QC figure, macro table and the manuscript from
#      those raw inputs and acquisition records. It first checks every raw input
#      against its record (tools/check_inputs.py: checksums, and the inventory numbers
#      recomputed from scenes.csv), so it never builds on a raw file other than the one
#      recorded. File targets carry their real prerequisites, so the order also holds
#      under `make -j`, and only out-of-date steps rerun; `make -B all` forces a full
#      rebuild. A missing raw input stops the build with the name of the target that
#      acquires it.
#      `make all` includes the Phase A pipeline (stage 2b: H1-H3, the Great Salt Lake test,
#      the robustness checks), about 20-23 h on one machine; `make -n -B all` lists every
#      step in order, and `make inputs` every input that an acquisition step writes.

PYTHON   ?= python
MAMBA    ?= micromamba
ENVNAME  ?= caspian
TECTONIC ?= tectonic

# Raw inputs read by `make all` (git-ignored; see stage 1 above).
DAHITI   ?= data/raw/dahiti/dahiti_39_caspian_sea_water_level.csv
HYDROWEB ?= data/raw/hydroweb/extracted/HYDROWEB_LAKES_OPE/hydroprd_L_caspian.txt
# Directory holding the Hydroweb zip named in results/provenance/hydroweb_l_caspian.json.
HYDROWEB_DIR ?= data/raw/hydroweb
SCENES   ?= data/interim/inventory/scenes.csv
# CASPCOM General Catalogue of the Caspian Sea level (monthly post levels, Baltic system);
# its SHA-256 is in results/provenance/caspcom_catalogue.json, checked by the offset script.
CASPCOM  ?= data/raw/gauges/caspcom/Sea-Level-EN.zip
BATHY    ?= data/raw/bathymetry
# 15-arc-second relief subsets (bathymetry-download) and ETOPO1 (ee-etopo1).
GRIDS    := $(BATHY)/gebco_2022_ncaspian.nc $(BATHY)/gebco_2026_ncaspian.nc $(BATHY)/etopo_2022_ncaspian.nc \
            $(BATHY)/srtm15p_v2.7_ncaspian.nc
ETOPO1   := $(BATHY)/etopo1_ncaspian.tif
# Acquisition records written by ee-scene-inventory together with SCENES (committed).
INV_META ?= results/inventory/inventory_meta.json
INV_REC  := results/inventory/scenes_csv.json results/inventory/summary_sensor_year.csv \
            results/inventory/summary_sensor_month.csv
# Acquisition records with the SHA-256 of each raw file (committed).
PROV_REC := $(addprefix results/provenance/,dahiti_39.json hydroweb_l_caspian.json gebco_2022.json \
            gebco_2026.json etopo_2022.json srtm15p_v2.7.json etopo1.json)
# Git-ignored marker: every raw input matched its record (tools/check_inputs.py).
VERIFIED ?= data/interim/.raw_inputs_verified

PKG := $(wildcard src/caspian/*.py)

# $(call need,<file>,<target>,<what it needs>): stop if a raw input is missing.
need = $(if $(wildcard $(1)),,$(error missing raw input $(1): run `make $(2)` first, which needs $(3)))
# Earth Engine targets run only with EE_OK=1.
ee_ok = $(if $(EE_OK),,$(error `make $@` uses Earth Engine and spends EECU: pilot first, check results/inventory/ee_budgets.json, then rerun with EE_OK=1))
# Batch stages take STEP explicitly.
need_step = $(if $(STEP),,$(error `make $@` needs STEP=$(1)))

.PHONY: help env lock test verify-0 verify-A verify-B \
        fetch dahiti-download hydroweb-download swot-download bathymetry-download osm-download refs caspcom-download \
        wdpa-download \
        ee-inputs ee-scene-inventory ee-etopo1 ee-zone ee-stage1 ee-era5-winds ee-erl-landsat \
        check-inputs dahiti-qc study-area scene-levels qc-figures params metrics macros check-refs \
        check-quotes manuscript figures all phase-a inputs

help:
	@echo Targets
	@echo   all            rebuild results, QC figures, numbers.tex and manuscript/main.pdf from the raw inputs
	@echo   phase-a        the Phase A results only: H1-H3, Great Salt Lake, robustness, about 20-23 h, also built by all
	@echo   inputs         list every input that make all reads and an acquisition step writes
	@echo   fetch          acquire network inputs: dahiti-download hydroweb-download swot-download bathymetry-download osm-download refs caspcom-download wdpa-download
	@echo   ee-inputs      acquire Earth Engine inputs: ee-scene-inventory ee-etopo1, with EE_OK=1
	@echo   ee-zone ee-stage1 ee-era5-winds   Earth Engine batch stages, with EE_OK=1 and STEP=...
	@echo   check-inputs   check raw inputs against their acquisition records: checksums, inventory numbers
	@echo   qc-figures     QC figures in outputs/qc, also built by all
	@echo   figures        display items that need no results, in figures/out, also built by all
	@echo   check-quotes   verbatim quotes in refs/notes against the texts in data/raw/lit, not run by all
	@echo   env lock test verify-0 verify-A verify-B check-refs

## environment -------------------------------------------------------------
env:
	$(MAMBA) create -y -n $(ENVNAME) -f requirements.lock
	$(MAMBA) run -n $(ENVNAME) python -m pip install --no-deps -r requirements-pip.txt
	$(MAMBA) run -n $(ENVNAME) python -m pip install --no-deps --no-build-isolation -e .

lock:
	$(PYTHON) tools/pin_env.py
	$(MAMBA) env export -n $(ENVNAME) --explicit --md5 > requirements.lock

## checks ------------------------------------------------------------------
test:
	$(PYTHON) -m pytest -q -p no:cacheprovider

verify-0:
	$(PYTHON) tools/verify.py 0

verify-A:
	$(PYTHON) tools/verify.py A

verify-B:
	$(PYTHON) tools/verify.py B

## stage 1a: raw inputs over the network (credentials; never run by `make all`) ----
fetch: dahiti-download hydroweb-download swot-download bathymetry-download osm-download refs caspcom-download \
       wdpa-download

# CASPCOM post levels (public HTTP). Writes $(CASPCOM), results/provenance/caspcom_catalogue.json.
caspcom-download:
	$(PYTHON) scripts/a7_dahiti_bhs_offset.py fetch

# DAHITI target 39; needs DAHITI_API_KEY. Writes $(DAHITI), results/provenance/dahiti_39.json.
dahiti-download:
	$(PYTHON) scripts/p0_dahiti_download.py

# Hydroweb L_caspian; needs HYDROWEB_API_KEY and py_hydroweb (requirements-pip.txt).
# Writes $(HYDROWEB), results/provenance/hydroweb_l_caspian.json.
hydroweb-download:
	$(PYTHON) scripts/a4_hydroweb_download.py

# SWOT L2 HR Raster 250 m (hydroweb.next); needs HYDROWEB_API_KEY.
swot-download:
	$(PYTHON) scripts/a5_swot_download.py

# Public HTTP subsets: GEBCO_2026 and GEBCO_2022 (CEDA), ETOPO 2022 (NCEI), SRTM15+ (Scripps).
bathymetry-download:
	$(PYTHON) scripts/a7_get_gebco.py
	$(PYTHON) scripts/a7_get_gebco.py 2022
	$(PYTHON) scripts/a7_get_etopo.py
	$(PYTHON) scripts/a7_get_srtm15.py

# OpenStreetMap layers via the Overpass API.
osm-download:
	$(PYTHON) scripts/a8_osm_exposure.py

# WDPA country downloads (Protected Planet public file store; non-commercial terms, D-101, D-103).
# Writes data/raw/wdpa/ (never redistributed) and results/provenance/wdpa.json.
wdpa-download:
	$(PYTHON) -I scripts/a8_wdpa_exposure.py fetch

# doi.org and Crossref/DataCite: refs/references.bib and refs/verification.csv, then the
# documented metadata corrections of refs/bib_corrections.csv (D-131, D-134).
refs:
	$(PYTHON) tools/fetch_bibtex.py
	$(PYTHON) tools/apply_bib_corrections.py .

## stage 1b: raw inputs from Earth Engine (spends EECU; never run by `make all`) ----
ee-inputs: ee-scene-inventory ee-etopo1

# Scene metadata only. Writes $(SCENES), $(INV_META), results/inventory/summary_sensor_*.csv;
# then records the checksum of $(SCENES) in results/inventory/scenes_csv.json (only if the
# numbers recomputed from it equal the committed ones).
ee-scene-inventory:
	$(ee_ok)
	$(PYTHON) scripts/a2_scene_inventory.py
	$(PYTHON) tools/check_inputs.py --record-scenes --scenes $(SCENES)

# ETOPO1 subset (Earth Engine catalogue copy). Writes data/raw/bathymetry/etopo1_ncaspian.tif.
ee-etopo1:
	$(ee_ok)
	$(PYTHON) scripts/a7_get_etopo1.py

# Batch stages (D-015); each is submitted, monitored and fetched by its own script.
ee-zone:
	$(ee_ok)
	$(call need_step,submit or area)
	$(PYTHON) scripts/a2_zone.py $(STEP)

ee-stage1:
	$(ee_ok)
	$(call need_step,pilot or submit or status or fetch)
	$(PYTHON) scripts/a3_stage1_thresholds.py $(STEP)

ee-era5-winds:
	$(ee_ok)
	$(call need_step,submit or status or fetch)
	$(PYTHON) scripts/a5_era5_winds.py $(STEP)

## raw-input guards: a missing file stops the build and names its target ---------
$(DAHITI):
	$(call need,$@,dahiti-download,DAHITI_API_KEY)

$(HYDROWEB):
	$(call need,$@,hydroweb-download,HYDROWEB_API_KEY)

$(SCENES) $(INV_META) $(INV_REC):
	$(call need,$@,ee-scene-inventory EE_OK=1,Earth Engine)

$(GRIDS):
	$(call need,$@,bathymetry-download,network access)

$(CASPCOM):
	$(call need,$@,caspcom-download,network access)

$(ETOPO1):
	$(call need,$@,ee-etopo1 EE_OK=1,Earth Engine)

## stage 2: results from raw inputs ----------------------------------------
# Every raw input matches its acquisition record (a mismatch or missing file stops here).
# Results that read raw inputs are ordered after it (order-only: `| $(VERIFIED)`).
$(VERIFIED): tools/check_inputs.py $(DAHITI) $(HYDROWEB) $(SCENES) $(GRIDS) $(ETOPO1) $(INV_META) \
        $(INV_REC) $(PROV_REC)
	$(PYTHON) tools/check_inputs.py --dahiti $(DAHITI) --hydroweb $(HYDROWEB) --hydroweb-dir $(HYDROWEB_DIR) \
	    --scenes $(SCENES) --bathy $(BATHY) --stamp $@

# Also writes outputs/qc/dahiti_level_series.{pdf,png}.
results/qc/dahiti_summary.json: scripts/p0_dahiti_qc.py $(DAHITI) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/p0_dahiti_qc.py

# Also writes results/study_area/sectors.geojson.
results/study_area/sector_areas.json: scripts/a0_study_area.py $(PKG)
	$(PYTHON) scripts/a0_study_area.py

# Sector edges against the present shoreline in each 15-arc-second grid. The script
# imports scripts/a0_study_area.py for the geodesic reading of the same polygons. The
# ETOPO1 subset is read only to record why it is skipped (its pixel size), but that
# record is part of the output, so ETOPO1 stays a prerequisite.
results/study_area/sector_edges.json: scripts/a0_sector_edges.py scripts/a0_study_area.py $(DAHITI) $(GRIDS) \
        $(ETOPO1) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a0_sector_edges.py

# DAHITI - Hydroweb cross-checks (pairs within +/-5 and +/-20 days). Each run writes a
# summary and its pair table (grouped targets, so the table is never older than the
# summary); also outputs/qc/level_crosscheck{,_w20d}.{pdf,png}.
results/qc/level_crosscheck.json results/qc/level_crosscheck_pairs.csv &: scripts/a4_level_crosscheck.py \
        $(DAHITI) $(HYDROWEB) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a4_level_crosscheck.py

results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv &: scripts/a4_level_crosscheck.py \
        $(DAHITI) $(HYDROWEB) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a4_level_crosscheck.py 20

# Reads the +/-20-day summary and pair table (it stops if their pair counts differ).
# Also writes data/interim/levels/scene_levels.csv.
results/levels/scene_levels_summary.json: scripts/a4_scene_levels.py $(DAHITI) $(HYDROWEB) $(SCENES) \
        results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a4_scene_levels.py

# QC figure cited in docs/DATUMS.md (GEBCO_2026 Type Identifier); also writes the .png.
QC_FIGS := outputs/qc/gebco_2026_tid.pdf
outputs/qc/gebco_2026_tid.pdf: figures/qc_gebco_tid.py $(BATHY)/gebco_2026_ncaspian.nc $(PKG) | $(VERIFIED)
	$(PYTHON) figures/qc_gebco_tid.py $(BATHY)/gebco_2026_ncaspian.nc

## stage 2b: the Phase A pipeline (D-110) ------------------------------------------
# Every committed result from the stage-2 counts on: the DEM and H2 (validation), H1 (the
# hindcast and its red-team runs), H3 (near-term emergence, WDPA exposure, the ATL24 check),
# the Great Salt Lake test, the robustness checks and their verdicts. Commands and arguments
# are the ones that produced the committed outputs (their run.json records; DECISIONS D-069
# to D-104). Steps that write many git-ignored files (DEM tiles, emergence dates, the
# robustness splits) have a stamp file as their target, data/interim/.stamp_<step>
# (tools/make_stamp.py), written only after the step succeeded.
#
# Runtime: about 20-23 h in all, one step at a time. HEAVY marks the steps with a recorded
# wall time of 1 h or more (run.json "seconds"); their flood-level maps are cached under
# data/interim/hindcast/ and data/interim/emergence/, so a rebuild that finds the caches is
# faster. The heavy steps need several GB of memory each: do not run them under -j.
#
# Inputs written by acquisition steps (Earth Engine or network; never run by `make all`). A
# missing one stops the build and names the command that acquires it; `make inputs` lists
# them all. Their content is fixed by the committed acquisition records named here (the
# validation and GLO-30 scripts check the ICESat-2 row counts and GLO-30 files against
# theirs when they read them; tools/check_inputs.py does not cover these inputs).
need_acq = $(if $(wildcard $(1)),,$(error missing input $(1): it is acquired by $(2), never by `make all`))
STAMP := data/interim/.stamp_
# Stage-2 counts (scripts/a6_stage2_export.py, Earth Engine): the main Landsat export mainL
# (D-079; results/inventory/stage2_mainL.json, which names the plan it ran with), the D-064
# screening sample sensB, and the R-block alternatives (scripts/r18_rblock_export.py;
# results/inventory/stage2_<tag>.json).
STAGE2_DIR  ?= data/interim/stage2
RBLOCK_TAGS := rR_fixed0 rR_awei rR_ndvi020 rR_ndvi040 rR_snow rR_bins005 rR_tau015 rR_subtm rR_subetm rR_slcoff \
               rR_suboli rR_s2
STAGE2_EE   := $(addprefix $(STAGE2_DIR)/,mainL sensB $(RBLOCK_TAGS))
STAGE2_PLAN := $(STAGE2_DIR)/plan_main_tau9900cm_landsat.csv
# ICESat-2 ATL06 and ATL08 (scripts/a6_icesat2.py; results/provenance/icesat2_atl06.json, _atl08.json)
ICESAT2     ?= data/interim/icesat2
# Copernicus GLO-30 tiles (scripts/a6_glo30.py download; results/provenance/copernicus_glo30.json)
GLO30_DIR   ?= data/raw/glo30
# GEDI L2A shots (scripts/a6_gedi.py export, Earth Engine; results/inventory/gedi_export.json)
GEDI_DIR    ?= data/interim/gedi
# H3 exposure layers: OpenStreetMap (osm-download; results/provenance/osm_exposure.json); the
# habitat rasters, retrieved without a script (results/provenance/habitat_rasters.json); the
# WDPA (wdpa-download; results/provenance/wdpa.json)
OSM_DIR     ?= data/raw/osm
HABITAT_DIR ?= data/raw/habitat
WDPA_DIR    ?= data/raw/wdpa
# Great Salt Lake (D-094): downloads (scripts/g1_data.py levels, osm, vdatum, products;
# results/provenance/generality_great_salt_lake_*.json) and Earth Engine outputs
# (scripts/g1_ee.py extents, scenes, stage1, dep3, stage2; results/generality/great_salt_lake/
# extents.json, scene_inventory.json; results/inventory/generality_great_salt_lake_stage2_main.json)
GSL_RAW     ?= data/raw/generality/great_salt_lake
GSL_INT     ?= data/interim/generality/great_salt_lake
GSL_NET     := $(GSL_RAW)/levels/gsl_daily_levels.csv $(GSL_RAW)/osm_gsl.json \
               $(GSL_RAW)/vdatum_navd88_minus_egm2008.npz $(GSL_RAW)/bathymetry
GSL_EE      := $(addprefix $(GSL_INT)/,extents_250m.npz scenes.csv stage1.csv dep3_60m.npy stage2_main)
ACQ_INPUTS  := $(STAGE2_EE) $(STAGE2_PLAN) $(ICESAT2)/ATL06 $(ICESAT2)/ATL08 $(GLO30_DIR) $(GEDI_DIR) $(OSM_DIR) \
               $(HABITAT_DIR) $(WDPA_DIR) $(GSL_NET) $(GSL_EE)

$(STAGE2_EE):
	$(call need_acq,$@,scripts/a6_stage2_export.py or scripts/r18_rblock_export.py (Earth Engine))
$(STAGE2_PLAN):
	$(call need_acq,$@,the stage-2 plan that the main export ran with (results/inventory/stage2_mainL.json))
$(ICESAT2)/ATL06 $(ICESAT2)/ATL08:
	$(call need_acq,$@,scripts/a6_icesat2.py ATL06|ATL08 (NSIDC; Earthdata login))
$(GLO30_DIR):
	$(call need_acq,$@,scripts/a6_glo30.py download (AWS Open Data))
$(GEDI_DIR):
	$(call need_acq,$@,scripts/a6_gedi.py export (Earth Engine))
$(OSM_DIR):
	$(call need_acq,$@,`make osm-download` (Overpass API))
$(HABITAT_DIR):
	$(call need_acq,$@,the retrieval recorded in results/provenance/habitat_rasters.json (no script))
$(WDPA_DIR):
	$(call need_acq,$@,`make wdpa-download` (Protected Planet))
$(GSL_NET):
	$(call need_acq,$@,scripts/g1_data.py levels|osm|vdatum|products great_salt_lake (network))
$(GSL_EE):
	$(call need_acq,$@,scripts/g1_ee.py extents|scenes|stage1|dep3|stage2 great_salt_lake (Earth Engine))

# --- stage-2 counts, vegetation labels, waterlines, DEM, emergence dates ------------------
# Main stage-2 counts (D-076, D-079): the Landsat export as the tag `main`, one source tag
# (5 min). Writes data/interim/stage2/main/tile_*.npz.
S2MAIN := results/provenance/stage2_main_combine.json
$(S2MAIN): scripts/a6_stage2_combine.py $(STAGE2_DIR)/mainL $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_stage2_combine.py main mainL

# Vegetation fraction of the emerged band and the vegetated-sector labels of H1 (D-071).
results/levels/vegetation_fraction.json results/levels/sector_labels.csv &: scripts/a6_vegetation.py $(S2MAIN) \
        results/study_area/extents_v6_250m.npz $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_vegetation.py main

# Waterlines of every (epoch, level bin) (PREREG section 4), then the static-contour rule
# (D-075), which rewrites the point table with its `static` column: one recipe, so the
# table is never used without that column (8 min). Writes
# data/interim/waterlines/main/points.csv.gz.
results/levels/static_contour_rule.json: scripts/a6_waterlines.py scripts/a6_static_rule.py $(S2MAIN) $(STAGE2_PLAN) \
        results/study_area/blocks_25km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_waterlines.py main $(STAGE2_PLAN)
	$(PYTHON) scripts/a6_static_rule.py main

# Waterline DEM, TIN at 30 m in 60 km tiles (PREREG section 4; D-069; 8 min):
# data/processed/dem/main/dem_*.tif.
DEM_MAIN := $(STAMP)dem_main
$(DEM_MAIN): scripts/a6_dem.py results/levels/static_contour_rule.json \
        results/study_area/zone_v6_cover_buffered.geojson $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_dem.py main
	$(PYTHON) tools/make_stamp.py $@ data/processed/dem/main --pattern "dem_*.tif"

# Emergence date of every DEM pixel from the daily level record (D-069), through the
# scene-level code path: data/processed/dem/main/emergence/.
EMERGENCE_DIR := data/processed/dem/main/emergence
EMERGENCE_MAIN := $(STAMP)emergence_main
$(EMERGENCE_MAIN): scripts/a6_emergence.py scripts/a4_scene_levels.py $(DEM_MAIN) $(DAHITI) \
        $(HYDROWEB) results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_emergence.py main
	$(PYTHON) tools/make_stamp.py $@ $(EMERGENCE_DIR) --pattern "emergence_*.tif"

# --- H2: the DEM against ICESat-2, GLO-30 and GEDI (PREREG sections 5-8; D-084 to D-091) ---
VAL  := results/validation/main
VALV := results/validation/main_vegetated
VALG := results/validation/main_glo30
VALE := results/validation/main_gedi
# $(call val_fits,<sources>,<folder>): the files of one `evaluate` run.
val_fits = $(foreach s,$(1),$(addprefix $(2)/$(s)_,datum_constant.json product_offsets.json scale_model.json \
           calibration.json counts.json coverage.csv accuracy.csv accuracy_blocks.csv paired_bootstrap.csv))
val_vario = $(foreach s,$(1),$(2)/$(s)_semivariogram.csv $(2)/$(s)_semivariogram_detrended.csv)
PRODUCT_GRIDS := $(GRIDS) $(ETOPO1)
CORRECTION := results/study_area/geoid_correction_primary.npz
H2_DEM := $(DEM_MAIN) $(EMERGENCE_MAIN) results/levels/static_contour_rule.json
# prepare: the ICESat-2 quality steps, geoid (EIGEN-6C4, D-078), dryness test, and the DEM
# and the five products at each point (20 min). Writes data/interim/validation/main/.
$(VAL)/prepare.json $(VAL)/ATL06_filter_cascade.csv $(VAL)/ATL08_filter_cascade.csv &: scripts/a6_validate_dem.py \
        $(H2_DEM) $(ICESAT2)/ATL06 $(ICESAT2)/ATL08 results/provenance/icesat2_atl06.json \
        results/provenance/icesat2_atl08.json results/study_area/geoid_egm2008_minus_eigen6c4.npz $(PRODUCT_GRIDS) \
        $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main prepare --emergence $(EMERGENCE_DIR)
# blocksize: semivariograms on the training blocks; the final size, 50 km (D-084, D-085).
# Also the point annotations (emerged band, vegetation share; D-084), about 30 min.
$(VAL)/block_size_check.json $(call val_vario,ATL06 ATL08,$(VAL)) &: scripts/a6_validate_dem.py $(VAL)/prepare.json \
        $(S2MAIN) results/study_area/extents_v6_250m.npz results/study_area/blocks_25km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main blocksize
# evaluate on the test blocks (blocks_50km.csv), and the 25 and 12.5 km block sensitivities (D-088).
$(call val_fits,ATL06 ATL08,$(VAL)) &: scripts/a6_validate_dem.py $(VAL)/block_size_check.json \
        results/study_area/blocks_50km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate
$(call val_fits,ATL06 ATL08,$(VAL)/sensitivity_25000m) &: scripts/a6_validate_dem.py $(VAL)/block_size_check.json \
        results/study_area/blocks_25km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate --block-size 25000
$(call val_fits,ATL06 ATL08,$(VAL)/sensitivity_12500m) &: scripts/a6_validate_dem.py $(VAL)/block_size_check.json \
        results/study_area/blocks_12km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate --block-size 12500
# The vegetated population (brief section 10, check 9; D-084, D-088).
$(VALV)/block_size_check.json $(call val_vario,ATL06 ATL08,$(VALV)) &: scripts/a6_validate_dem.py $(VAL)/prepare.json \
        $(VAL)/block_size_check.json results/study_area/blocks_25km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main blocksize --population vegetated
$(call val_fits,ATL06 ATL08,$(VALV)) &: scripts/a6_validate_dem.py $(VALV)/block_size_check.json \
        results/study_area/blocks_50km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate --population vegetated
# The H2 decision (PREREG section 7; D-086).
$(VAL)/h2_decision.json: scripts/a6_h2_decision.py $(VAL)/ATL06_paired_bootstrap.csv $(VAL)/ATL08_paired_bootstrap.csv $(PKG)
	$(PYTHON) scripts/a6_h2_decision.py main
# GLO-30 and GEDI, each alone on the H2 blocks, the block size inherited (D-091); the H2 rule
# applied descriptively, and ETOPO 2022's source datasets at their test points.
$(VALG)/prepare.json $(VALG)/GLO30_filter_cascade.csv &: scripts/a6_glo30.py scripts/a6_validate_dem.py $(H2_DEM) \
        $(GLO30_DIR) results/provenance/copernicus_glo30.json $(CORRECTION) $(PRODUCT_GRIDS) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_glo30.py points main --emergence $(EMERGENCE_DIR)
$(VALE)/prepare.json $(VALE)/GEDI_filter_cascade.csv &: scripts/a6_gedi.py scripts/a6_validate_dem.py $(H2_DEM) \
        $(GEDI_DIR) results/inventory/gedi_export.json results/study_area/geoid_egm2008_minus_eigen6c4.npz \
        $(PRODUCT_GRIDS) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_gedi.py points main --emergence $(EMERGENCE_DIR)
$(VALG)/block_size_check.json: scripts/a6_validate_dem.py $(VALG)/prepare.json $(VAL)/block_size_check.json $(PKG)
	$(PYTHON) scripts/a6_validate_dem.py main blocksize --sources GLO30
$(VALE)/block_size_check.json: scripts/a6_validate_dem.py $(VALE)/prepare.json $(VAL)/block_size_check.json $(PKG)
	$(PYTHON) scripts/a6_validate_dem.py main blocksize --sources GEDI
$(call val_fits,GLO30,$(VALG)) &: scripts/a6_validate_dem.py $(VALG)/block_size_check.json $(S2MAIN) \
        results/study_area/extents_v6_250m.npz results/study_area/blocks_50km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate --sources GLO30
$(call val_fits,GEDI,$(VALE)) &: scripts/a6_validate_dem.py $(VALE)/block_size_check.json $(S2MAIN) \
        results/study_area/extents_v6_250m.npz results/study_area/blocks_50km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate --sources GEDI
$(call val_fits,GEDI,$(VALE)/subset_night) &: scripts/a6_validate_dem.py $(VALE)/block_size_check.json $(S2MAIN) \
        results/study_area/extents_v6_250m.npz results/study_area/blocks_50km.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_validate_dem.py main evaluate --sources GEDI --subset night
$(VALG)/GLO30_rule_check.json: scripts/a6_h2_decision.py $(VALG)/GLO30_paired_bootstrap.csv $(PKG)
	$(PYTHON) scripts/a6_h2_decision.py main --additional GLO30
$(VALE)/GEDI_rule_check.json: scripts/a6_h2_decision.py $(VALE)/GEDI_paired_bootstrap.csv $(PKG)
	$(PYTHON) scripts/a6_h2_decision.py main --additional GEDI
$(VALE)/subset_night/GEDI_rule_check.json: scripts/a6_h2_decision.py $(VALE)/subset_night/GEDI_paired_bootstrap.csv $(PKG)
	$(PYTHON) scripts/a6_h2_decision.py main --additional GEDI --subset night
$(VALG)/GLO30_etopo2022_sources.json: scripts/a6_etopo_sources.py scripts/a6_validate_dem.py \
        $(call val_fits,GLO30,$(VALG)) $(BATHY)/etopo_2022_ncaspian.nc $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_etopo_sources.py main GLO30
$(VALE)/GEDI_etopo2022_sources.json: scripts/a6_etopo_sources.py scripts/a6_validate_dem.py \
        $(call val_fits,GEDI,$(VALE)) $(BATHY)/etopo_2022_ncaspian.nc $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a6_etopo_sources.py main GEDI
H2_RESULTS := $(VAL)/prepare.json $(VAL)/block_size_check.json $(call val_fits,ATL06 ATL08,$(VAL)) $(VAL)/h2_decision.json \
              $(call val_fits,ATL06 ATL08,$(VAL)/sensitivity_25000m) $(call val_fits,ATL06 ATL08,$(VAL)/sensitivity_12500m) \
              $(VALV)/block_size_check.json $(call val_fits,ATL06 ATL08,$(VALV)) \
              $(VALG)/prepare.json $(VALG)/block_size_check.json $(call val_fits,GLO30,$(VALG)) \
              $(VALG)/GLO30_rule_check.json $(VALG)/GLO30_etopo2022_sources.json \
              $(VALE)/prepare.json $(VALE)/block_size_check.json $(call val_fits,GEDI,$(VALE)) \
              $(call val_fits,GEDI,$(VALE)/subset_night) $(VALE)/GEDI_rule_check.json \
              $(VALE)/subset_night/GEDI_rule_check.json $(VALE)/GEDI_etopo2022_sources.json

# --- H1: the hindcast audit (PREREG section 9; D-070, D-072, D-082, D-083, D-089) ---------
HC := results/hindcast
hc_tables = $(addprefix $(1)/,extent.csv displacement.csv noise_floor.csv observed.csv block_h1.csv blocks.csv.gz run.json)
h1_tables = $(addprefix $(1)/,h1_mispredicts.csv h1_mispredicts_pooled_floor.csv h1_mispredicts_detail.csv \
            h1_block_test.csv h1_decision.json)
HC_INPUTS := scripts/a7_hindcast.py $(S2MAIN) $(PRODUCT_GRIDS) $(CORRECTION) results/study_area/sea_mask_1km.npz \
             results/study_area/beyond_state_250m.npz results/study_area/blocks_25km.csv \
             results/study_area/zone_v6_cover_buffered.geojson results/study_area/extents_v6_250m.npz $(PKG)
# HEAVY (6,540 s): the primary hindcast, total and shape variants.
$(call hc_tables,$(HC)/main) $(HC)/main/shape_offsets.json &: $(HC_INPUTS) | $(VERIFIED)
	$(PYTHON) scripts/a7_hindcast.py main --vertical-correction $(CORRECTION)
# The H1 decisions (PREREG section 9; D-072), here and for both red-team runs.
$(call h1_tables,$(HC)/main) &: scripts/a7_h1_decision.py $(call hc_tables,$(HC)/main) results/levels/sector_labels.csv \
        $(PKG)
	$(PYTHON) scripts/a7_h1_decision.py main
# Displacement by level band and ETOPO 2022's sources over the emerged band (D-087, D-108).
$(HC)/main/displacement_by_level_band.csv $(HC)/main/etopo2022_sources_emerged_band.json \
        $(HC)/main/etopo2022_sources_by_level_band.csv &: scripts/a7_level_bands.py $(HC)/main/displacement.csv \
        $(BATHY)/etopo_2022_ncaspian.nc results/study_area/extents_v6_250m.npz $(DEM_MAIN) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a7_level_bands.py main
# Red team, HEAVY (7,485 s): without the record-low water envelope (D-082 sensitivity).
$(call hc_tables,$(HC)/main_noenv) $(HC)/main_noenv/shape_offsets.json &: $(HC_INPUTS) | $(VERIFIED)
	$(PYTHON) scripts/a7_hindcast.py main --vertical-correction $(CORRECTION) --no-envelope --out-suffix _noenv
$(call h1_tables,$(HC)/main_noenv) &: scripts/a7_h1_decision.py $(call hc_tables,$(HC)/main_noenv) \
        results/levels/sector_labels.csv $(PKG)
	$(PYTHON) scripts/a7_h1_decision.py main_noenv
# Red team, HEAVY (4,694 s): the shape constant fitted per sector (scripts/a7_sector_shape.py).
$(call hc_tables,$(HC)/main_sectorshape) $(HC)/main_sectorshape/sector_offsets.json &: scripts/a7_sector_shape.py \
        $(HC_INPUTS) $(HC)/main/run.json $(HC)/main/shape_offsets.json | $(VERIFIED)
	$(PYTHON) scripts/a7_sector_shape.py main --vertical-correction $(CORRECTION)
$(call h1_tables,$(HC)/main_sectorshape) &: scripts/a7_h1_decision.py $(call hc_tables,$(HC)/main_sectorshape) \
        results/levels/sector_labels.csv $(PKG)
	$(PYTHON) scripts/a7_h1_decision.py main_sectorshape
H1_RESULTS := $(call hc_tables,$(HC)/main) $(HC)/main/shape_offsets.json $(call h1_tables,$(HC)/main) \
              $(HC)/main/displacement_by_level_band.csv $(HC)/main/etopo2022_sources_emerged_band.json \
              $(HC)/main/etopo2022_sources_by_level_band.csv \
              $(call hc_tables,$(HC)/main_noenv) $(HC)/main_noenv/shape_offsets.json $(call h1_tables,$(HC)/main_noenv) \
              $(call hc_tables,$(HC)/main_sectorshape) $(HC)/main_sectorshape/sector_offsets.json \
              $(call h1_tables,$(HC)/main_sectorshape)

# --- H3: near-term emergence (PREREG section 10; D-081, D-090, D-095, D-096, D-103) --------
EM := results/emergence
h3_tables = $(addprefix $(1)/,shifts.csv error_models.json loeo_errors.csv transfer.csv emergence.csv fused_summary.csv \
            exposure.csv scenario_dates.csv run.json)
h3_dec = $(addprefix $(1)/,h3_decision.csv h3_summary.csv h3_decision.json)
H3_INPUTS := scripts/a8_emergence.py scripts/a7_hindcast.py $(call hc_tables,$(HC)/main) $(DEM_MAIN) $(DAHITI) \
             results/levels/projection_pathways.csv $(OSM_DIR) $(HABITAT_DIR) $(PRODUCT_GRIDS) $(CORRECTION) \
             results/study_area/blocks_25km.csv $(PKG)
# HEAVY (8,292 s): the registered run, error-model form linear_hold. Caches its surfaces in
# data/interim/emergence/main/, which the constant-form run shares, so that run waits for
# this one (order-only).
$(call h3_tables,$(EM)/main) &: $(H3_INPUTS) | $(VERIFIED)
	$(PYTHON) scripts/a8_emergence.py main --dem-tag main --vertical-correction $(CORRECTION)
$(call h3_dec,$(EM)/main) &: scripts/a8_h3_decision.py $(EM)/main/run.json $(EM)/main/emergence.csv $(PKG)
	$(PYTHON) scripts/a8_h3_decision.py main
# HEAVY (8,131 s): the D-081 constant-form sensitivity (D-096).
$(call h3_tables,$(EM)/main_constant) &: $(H3_INPUTS) | $(VERIFIED) $(EM)/main/run.json
	$(PYTHON) scripts/a8_emergence.py main --dem-tag main --vertical-correction $(CORRECTION) --sensitivity-form constant
$(call h3_dec,$(EM)/main_constant) &: scripts/a8_h3_decision.py $(EM)/main_constant/run.json \
        $(EM)/main_constant/emergence.csv $(PKG)
	$(PYTHON) scripts/a8_h3_decision.py main_constant
# WDPA protected areas and Ramsar sites as exposure layers (D-101, D-103): `layers` writes
# data/interim/wdpa/; `exposure` (2,756 s) reads both runs' cached surfaces and checks
# them against emergence.csv and exposure.csv before any WDPA number.
$(EM)/wdpa/layers.json: scripts/a8_wdpa_exposure.py $(WDPA_DIR) results/provenance/wdpa.json $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a8_wdpa_exposure.py layers
$(EM)/wdpa/exposure_main.csv $(EM)/wdpa/exposure_main_constant.csv $(EM)/wdpa/run.json &: scripts/a8_wdpa_exposure.py \
        scripts/a8_emergence.py scripts/f1_erl_display.py $(EM)/wdpa/layers.json $(call h3_tables,$(EM)/main) \
        $(call h3_tables,$(EM)/main_constant) $(HC)/main/run.json $(OSM_DIR) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a8_wdpa_exposure.py exposure
# Sensitivity (D-103 addendum, 649 s): the protected areas without the Kazakh northern-Caspian
# reserved zone; after the registered tables (order-only: one heavy read at a time).
$(EM)/wdpa/exposure_main_noksrz.csv $(EM)/wdpa/exposure_main_constant_noksrz.csv $(EM)/wdpa/run_noksrz.json &: \
        scripts/a8_wdpa_exposure.py scripts/a8_emergence.py scripts/f1_erl_display.py $(EM)/wdpa/layers.json \
        $(call h3_tables,$(EM)/main) $(call h3_tables,$(EM)/main_constant) $(HC)/main/run.json $(OSM_DIR) $(PKG) \
        | $(VERIFIED) $(EM)/wdpa/run.json
	$(PYTHON) scripts/a8_wdpa_exposure.py exposure --layers wdpa_protected_areas_noksrz --suffix _noksrz
# ATL24 check of the near-term surfaces (Amendment 17; D-095). ATL24 v002 has no photon over
# the domain (results/provenance/icesat2_atl24.json; nothing under data/raw/icesat2/atl24), so
# the tables are written empty and the surface stage is verified on 300 random cells (153 s).
ATL24_OUT := $(addprefix results/validation/atl24/,photon_cascade.csv granules.json celltracks_summary.csv \
             fused_check.csv verdicts.json depth_classes.csv correction_at_cells.csv water_surface_constant.json \
             error_model_range.csv surfaces_check_cells.csv run.json)
$(ATL24_OUT) &: scripts/a8_atl24_check.py scripts/a8_emergence.py scripts/a7_hindcast.py $(call hc_tables,$(HC)/main) \
        $(HC)/main/shape_offsets.json $(call h3_tables,$(EM)/main) $(VAL)/ATL06_datum_constant.json \
        $(VAL)/ATL08_datum_constant.json results/qc/dahiti_summary.json results/provenance/icesat2_atl24.json \
        $(DEM_MAIN) $(CORRECTION) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a8_atl24_check.py --check-surfaces 300
H3_RESULTS := $(call h3_tables,$(EM)/main) $(call h3_dec,$(EM)/main) $(call h3_tables,$(EM)/main_constant) \
              $(call h3_dec,$(EM)/main_constant) $(EM)/wdpa/layers.json $(EM)/wdpa/exposure_main.csv \
              $(EM)/wdpa/exposure_main_constant.csv $(EM)/wdpa/run.json $(EM)/wdpa/exposure_main_noksrz.csv \
              $(EM)/wdpa/exposure_main_constant_noksrz.csv $(EM)/wdpa/run_noksrz.json $(ATL24_OUT)

# --- A9: Great Salt Lake (PREREG Amendment 16; D-094) --------------------------------------
# The local steps, in the order they ran. The static layers, scene levels, thresholds and
# plans were inputs of the basin's Earth Engine stage 1 and stage 2 as well; rebuilt here,
# they must come out as committed (VB.15).
GSL := results/generality/great_salt_lake
GSL_STATIC := $(addprefix $(GSL)/,arms_laea.geojson cover_laea.geojson ponds_laea.geojson zone_lonlat.geojson \
              cover_lonlat.geojson static.json)
$(GSL_STATIC) &: scripts/g1_static.py $(GSL_INT)/extents_250m.npz $(GSL_RAW)/osm_gsl.json $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/g1_static.py great_salt_lake
$(GSL)/scene_levels.json: scripts/g1_plan.py $(GSL_INT)/scenes.csv $(GSL_RAW)/levels/gsl_daily_levels.csv $(PKG) \
        | $(VERIFIED)
	$(PYTHON) scripts/g1_plan.py levels great_salt_lake
$(GSL)/threshold_rule_summary.json: scripts/g1_plan.py $(GSL_INT)/stage1.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/g1_plan.py thresholds great_salt_lake
$(GSL)/plan_main.json: scripts/g1_plan.py $(GSL)/scene_levels.json $(GSL)/threshold_rule_summary.json \
        $(GSL_INT)/stage1.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/g1_plan.py plan great_salt_lake main
$(GSL)/plan_pilot.json: scripts/g1_plan.py $(GSL)/scene_levels.json $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/g1_plan.py plan great_salt_lake pilot --placeholder 0.2
# Product lineage over each arm's emerged band (ETOPO 2022's FABDEM share; D-094, D-107).
$(GSL)/lineage.json: scripts/g1_lineage.py $(GSL)/static.json $(GSL_RAW)/bathymetry $(GSL_INT)/dep3_60m.npy \
        $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/g1_lineage.py great_salt_lake
# Hindcast per arm, five products and the 3DEP lidar control (1,865 s). It writes each masked
# stage-2 tile once and reuses it, so the masked folders are removed first.
GSL_HC := $(foreach a,north south,$(foreach r,hindcast hindcast_dep3,$(addprefix $(GSL)/$(a)/$(r)/,block_h1.csv \
          blocks.csv.gz displacement.csv extent.csv noise_floor.csv observed.csv run.json shape_offsets.json)))
$(GSL)/blocks_25km.csv $(GSL_HC) &: scripts/g1_hindcast.py scripts/a7_hindcast.py $(GSL_STATIC) \
        $(GSL_INT)/stage2_main $(GSL_INT)/dep3_60m.npy $(GSL_RAW)/vdatum_navd88_minus_egm2008.npz \
        $(GSL_RAW)/bathymetry $(PKG) | $(VERIFIED)
	$(PYTHON) tools/make_stamp.py --clean data/interim/generality/great_salt_lake/stage2_main_north_masked \
	    data/interim/generality/great_salt_lake/stage2_main_south_masked
	$(PYTHON) scripts/g1_hindcast.py great_salt_lake
$(GSL)/h1_mispredicts.csv $(GSL)/h1_mispredicts_detail.csv $(GSL)/h1_decision.json &: scripts/g1_decide.py $(GSL_HC) \
        $(GSL_RAW)/levels/gsl_daily_levels.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/g1_decide.py great_salt_lake
GSL_RESULTS := $(GSL_STATIC) $(GSL)/scene_levels.json $(GSL)/threshold_rule_summary.json $(GSL)/plan_main.json \
               $(GSL)/plan_pilot.json $(GSL)/lineage.json $(GSL)/blocks_25km.csv $(GSL_HC) $(GSL)/h1_mispredicts.csv \
               $(GSL)/h1_mispredicts_detail.csv $(GSL)/h1_decision.json

# --- robustness checks (brief section 10; docs/ROBUSTNESS_PLAN.md; D-092, D-097, D-104) -----
# scripts/r19_local_runs.py runs each check locally (no Earth Engine: the R-block and sensB
# counts are inputs). Several of its commands merge into the check's existing summary.json,
# so each check's commands run in one recipe, in the order they were run, with the
# bootstrap counts they used (D-097: 2,000 for ATL08 and the no-static ATL06 run; 1,000 for
# the rest). Variant counts and DEMs that r19 reuses when present are removed first. ATL06
# validations take 15-20 min each; whole-variant runs 10-30 min.
ROB := results/robustness
R19 := $(PYTHON) scripts/r19_local_runs.py
ROB_CODE := scripts/r19_local_runs.py scripts/r00_pipeline.py scripts/r00_counts.py \
            $(wildcard scripts/r0[4-9]_*.py scripts/r1[0-4]_*.py) scripts/a6_waterlines.py scripts/a6_dem.py \
            scripts/a6_emergence.py scripts/a6_validate_dem.py scripts/a4_scene_levels.py $(PKG)
ROB_MAIN := $(H2_DEM) $(VAL)/prepare.json $(VAL)/block_size_check.json $(call val_fits,ATL06 ATL08,$(VAL)) \
            results/study_area/blocks_50km.csv
# check 4: the setup-corrected levels (DEM rebuilt; ATL08) and the Komsomolets scatter (sensB counts).
$(ROB)/r04_screening_setupcorr/summary.json $(ROB)/r04_screening/setup_correction_shifts.csv &: $(ROB_CODE) \
        $(ROB_MAIN) $(STAGE2_PLAN) | $(VERIFIED)
	$(R19) setupcorr --dem --sources ATL08
$(ROB)/r04_screening_komsomolets/summary.json $(addprefix $(ROB)/r04_screening/komsomolets_sensB,.csv _summary.csv \
        _all_taus.csv _summary_all_taus.csv) &: $(ROB_CODE) $(STAGE2_DIR)/sensB results/inventory/stage2_sensB.json \
        | $(VERIFIED)
	$(R19) komsomolets
# check 5: the level-series alternatives (the Hydroweb-constant one with its DEM).
$(ROB)/r05_levels/summary.json: $(ROB_CODE) $(ROB_MAIN) $(STAGE2_PLAN) $(DAHITI) $(HYDROWEB) \
        results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv | $(VERIFIED)
	$(R19) levels --level-names hydroweb_const --dem --sources ATL08
	$(R19) levels --level-names dahiti_pchip,window1y
# check 6, validation side: the datum constant, product offsets and tide systems (about 1.5 h).
$(ROB)/r06_datums_validation/summary.json: $(ROB_CODE) $(ROB_MAIN) | $(VERIFIED)
	$(R19) datums --sources ATL08 --n-boot 1000
	$(R19) datums --sources ATL06 --n-boot 1000
# check 7: kriging against TIN on a block sample, and 0.20 m bins.
$(ROB)/r07_interpolation/summary.json $(ROB)/r07_interpolation/gp_vs_tin.csv &: $(ROB_CODE) $(ROB_MAIN) | $(VERIFIED)
	$(R19) gp --n-boot 1000
$(ROB)/r07_interpolation_bw020/summary.json: $(ROB_CODE) $(ROB_MAIN) $(S2MAIN) | $(VERIFIED)
	$(PYTHON) tools/make_stamp.py --clean data/interim/stage2/main__bw020
	$(R19) bw020 --sources ATL08 --n-boot 1000
	$(R19) validate --check r07_interpolation_bw020 --variant main__bw020 --summary-check r07_interpolation_bw020 \
	    --summary-key h2 --sources ATL06 --n-boot 1000
# check 8: 20 random reassignments of the 50 km blocks and the calibration/test swap
# (scripts/r08_blocking.py, 22 min), folded with the block-size sensitivities and the
# sector-shape run (r19 fold).
ROB_SPLITS := $(STAMP)r08_splits
$(ROB_SPLITS): scripts/r08_blocking.py $(ROB_CODE) $(ROB_MAIN) | $(VERIFIED)
	$(PYTHON) scripts/r08_blocking.py splits --main main --sources ATL08 --n-boot 1000
	$(PYTHON) tools/make_stamp.py $@ $(ROB)/r08_blocking --pattern "split_*/validation/*"
$(ROB)/r08_blocking/summary.json $(ROB)/r06_datums_sectorshape/summary.json &: $(ROB_CODE) $(ROB_SPLITS) $(ROB_MAIN) \
        $(call val_fits,ATL06 ATL08,$(VAL)/sensitivity_25000m) $(call val_fits,ATL06 ATL08,$(VAL)/sensitivity_12500m) \
        $(call h1_tables,$(HC)/main) $(call h1_tables,$(HC)/main_sectorshape)
	$(R19) fold
# check 9: emergent vegetation counted as land or as water (whole DEM; ATL08, then ATL06).
$(ROB)/r09_vegetation/summary.json: $(ROB_CODE) $(ROB_MAIN) $(S2MAIN) | $(VERIFIED)
	$(PYTHON) tools/make_stamp.py --clean data/interim/stage2/main__veg_as_dry data/interim/stage2/main__veg_as_water
	$(R19) vegetation --mode as_dry --sources ATL08 --n-boot 1000
	$(R19) vegetation --mode as_water --sources ATL08 --n-boot 1000
	$(R19) validate --check r09_vegetation --variant main__veg_as_dry --summary-check r09_vegetation \
	    --summary-key alternatives.as_dry.h2 --sources ATL06 --n-boot 1000
	$(R19) validate --check r09_vegetation --variant main__veg_as_water --summary-check r09_vegetation \
	    --summary-key alternatives.as_water.h2 --sources ATL06 --n-boot 1000
# check 10: morphology between epochs, with the limb-difference map; check 11: prior work.
$(ROB)/r10_morphology/summary.json: $(ROB_CODE) $(ROB_MAIN) $(call hc_tables,$(HC)/main) | $(VERIFIED)
	$(R19) morphology --map
$(ROB)/r11_prior_work/summary.json: $(ROB_CODE) $(DEM_MAIN) $(HC)/main/observed.csv refs/literature_values.csv \
        results/study_area/extents_v6_250m.npz | $(VERIFIED)
	$(R19) prior
# The DEM without the static-contour rule (D-034), emergence + 365 days (D-069), nearest-
# neighbour products (D-070).
$(ROB)/r12_static_rule/summary.json: $(ROB_CODE) $(ROB_MAIN) | $(VERIFIED)
	$(R19) nostatic --sources ATL08
	$(R19) validate --check r12_static_rule --variant main__nostatic --summary-check r12_static_rule --summary-key h2 \
	    --sources ATL06
$(ROB)/r13_emergence_lag/summary.json: $(ROB_CODE) $(ROB_MAIN) | $(VERIFIED)
	$(R19) lag365 --sources ATL08
	$(R19) lag365 --sources ATL06 --n-boot 1000
$(ROB)/r14_product_resampling/summary.json: $(ROB_CODE) $(ROB_MAIN) $(PRODUCT_GRIDS) | $(VERIFIED)
	$(R19) nearest --sources ATL08
# H1 with E2 + E3 pooled (HEAVY, 5,378 s; a full hindcast on the cached flood-level maps).
$(ROB)/h1_e2e3/summary.json: $(ROB_CODE) $(HC_INPUTS) $(call hc_tables,$(HC)/main) $(HC)/main/shape_offsets.json \
        $(HC)/main/d089_rerun_comparison.json results/levels/sector_labels.csv | $(VERIFIED)
	$(R19) h1-e2e3
# The R-block alternatives (scripts/r18_rblock_export.py; --alt:tag:kind), 6 min each. The
# first builds the reference TIN around R (main__Rnear) that the others reuse, so it runs
# first and removes the old one.
RBLOCKS := fixed_mndwi0:rR_fixed0:full awei_sh:rR_awei:full ndvi020:rR_ndvi020:full ndvi040:rR_ndvi040:full \
           snow_all_months:rR_snow:full bins005:rR_bins005:full tau015:rR_tau015:sub sub_tm:rR_subtm:sub \
           sub_etm:rR_subetm:sub sub_slcoff:rR_slcoff:sub sub_oli:rR_suboli:sub s2_add:rR_s2:add
RBLOCK_FIRST := $(ROB)/rblock_fixed_mndwi0/summary.json
RBLOCK_DEPS = $(ROB_CODE) $(ROB_MAIN) $(S2MAIN) $(STAGE2_DIR)/$(1) results/inventory/stage2_$(1).json \
              $(HC)/main/noise_floor.csv
$(RBLOCK_FIRST): $(call RBLOCK_DEPS,rR_fixed0) | $(VERIFIED)
	$(PYTHON) tools/make_stamp.py --clean data/processed/dem/main__Rnear data/interim/waterlines/main__Rnear
	$(R19) rblock --alt fixed_mndwi0 --tag rR_fixed0 --kind full --n-boot 1000
define rblock_rule
$(ROB)/rblock_$(1)/summary.json: $(call RBLOCK_DEPS,$(2)) | $(VERIFIED) $(RBLOCK_FIRST)
	$(R19) rblock --alt $(1) --tag $(2) --kind $(3) --n-boot 1000
endef
rb_field = $(word $(2),$(subst :, ,$(1)))
rb_rule = $(call rblock_rule,$(call rb_field,$(1),1),$(call rb_field,$(1),2),$(call rb_field,$(1),3))
$(foreach r,$(filter-out fixed_mndwi0:%,$(RBLOCKS)),$(eval $(call rb_rule,$(r))))
# Stage-2 seam gaps (D-097 QC finding).
$(ROB)/r21_seam_gaps/summary.json: scripts/r21_seam_gaps.py $(S2MAIN) results/study_area/zone_v6_cover_buffered.geojson \
        $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/r21_seam_gaps.py
# The summaries scripts/r20_verdicts.py reads (its rule is with the metrics inputs below).
ROB_SUMMARIES := $(addprefix $(ROB)/,r04_screening_setupcorr/summary.json r04_screening_komsomolets/summary.json \
                 r05_levels/summary.json r06_datums_validation/summary.json r06_datums_sectorshape/summary.json \
                 r07_interpolation/summary.json r07_interpolation_bw020/summary.json r08_blocking/summary.json \
                 r09_vegetation/summary.json r10_morphology/summary.json r11_prior_work/summary.json \
                 r12_static_rule/summary.json r13_emergence_lag/summary.json r14_product_resampling/summary.json \
                 h1_e2e3/summary.json) $(foreach r,$(RBLOCKS),$(ROB)/rblock_$(call rb_field,$(r),1)/summary.json)
ROB_RESULTS := $(ROB_SUMMARIES) $(ROB)/r21_seam_gaps/summary.json $(ROB)/verdicts.json $(ROB)/headline_table.csv

# Everything above, in `all` and `phase-a` (ERL display summaries: below, with the figures).
PHASE_A_RESULTS := $(S2MAIN) results/levels/vegetation_fraction.json results/levels/sector_labels.csv \
                   results/levels/static_contour_rule.json $(DEM_MAIN) $(EMERGENCE_MAIN) $(H2_RESULTS) $(H1_RESULTS) \
                   $(H3_RESULTS) $(GSL_RESULTS) $(ROB_RESULTS)

# export_params.py imports or parses the scripts whose constants the Methods quote.
PARAM_SCRIPTS := $(addprefix scripts/,a4_scene_levels.py a3_stage1_interactive.py a3_thresholds.py a5_setup_model.py a5_swot_sector_levels.py \
                 a5_sea_mask.py a6_screening.py a6_stage2_plan.py a6_stage2_export.py a6_static_rule.py \
                 a6_waterlines.py a6_vegetation.py a6_dem.py a7_beyond_state.py a7_hindcast.py)
results/params.json: tools/export_params.py $(PARAM_SCRIPTS) figures/fig_study_area.py figures/_erl.py \
        figures/tab_report_card.py figures/erl_fig_then_now.py $(PKG)
	$(PYTHON) tools/export_params.py

# Display items that need no results (figures/; vector PDF, 600 dpi TIFF and a PNG
# preview in figures/out/). The level figure reads the raw level series through the
# scene-level code path (scripts/a4_scene_levels.py) and data/interim/levels/scene_levels.csv,
# which the scene-levels target writes with its summary.
FIG_OUT := figures/out
FIGS := $(FIG_OUT)/fig1_levels.pdf $(FIG_OUT)/fig_study_area.pdf $(FIG_OUT)/supp_setup_model.pdf
$(FIG_OUT)/fig1_levels.pdf: figures/fig1_levels.py figures/_common.py scripts/a4_scene_levels.py \
        scripts/a3_thresholds.py scripts/a6_emergence.py $(DAHITI) $(HYDROWEB) \
        results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv \
        results/levels/scene_levels_summary.json results/metrics.json $(PKG) | $(VERIFIED)
	$(PYTHON) figures/fig1_levels.py

$(FIG_OUT)/fig_study_area.pdf: figures/fig_study_area.py figures/_common.py results/study_area/sector_areas.json \
        results/study_area/zone_v6_cover_buffered.geojson results/study_area/blocks_25km.csv \
        results/study_area/sea_mask_1km.npz $(PKG)
	$(PYTHON) figures/fig_study_area.py

$(FIG_OUT)/supp_setup_model.pdf: figures/supp_setup_model.py figures/_common.py results/levels/setup_model.json $(PKG)
	$(PYTHON) figures/supp_setup_model.py

# Results display items (figures/fig1_emergence.py ... fig4_audit.py, figures/ed_*.py).
# They read the frozen result tables in results/ and the git-ignored products of the
# pipeline (DEM and emergence tiles, waterline and ICESat-2 point tables, the stage-2
# plan, OSM layers); figures/_data.py caches the decimated and selected reads in
# data/interim/figures/ and rebuilds a cache when an input changes. Nothing is recomputed.
FIG_DEPS := figures/_common.py figures/_data.py results/metrics.json $(PKG)
STUDY_MAP := results/study_area/sectors.geojson results/study_area/sea_mask_1km.npz \
             results/study_area/extents_v6_250m.npz
RESULT_FIGS := $(FIG_OUT)/fig1_emergence.pdf $(FIG_OUT)/fig2_method.pdf $(FIG_OUT)/fig3_dem_accuracy.pdf \
               $(FIG_OUT)/fig4_audit.pdf $(FIG_OUT)/ed_block_size.pdf $(FIG_OUT)/ed_vegetation.pdf \
               $(FIG_OUT)/ed_screening.pdf $(FIG_OUT)/ed_h1_blocks_shape.pdf $(FIG_OUT)/ed_h2_hexbins.pdf
$(FIG_OUT)/fig1_emergence.pdf: figures/fig1_emergence.py $(FIG_DEPS) $(STUDY_MAP)
	$(PYTHON) figures/fig1_emergence.py
$(FIG_OUT)/fig2_method.pdf: figures/fig2_method.py $(FIG_DEPS) results/levels/screening_sensitivity.json \
        results/hindcast/main/noise_floor.csv results/study_area/blocks_25km.csv results/study_area/blocks_50km.csv \
        results/levels/sector_labels.csv
	$(PYTHON) figures/fig2_method.py
$(FIG_OUT)/fig3_dem_accuracy.pdf: figures/fig3_dem_accuracy.py $(FIG_DEPS) $(STUDY_MAP) \
        results/study_area/blocks_50km.csv $(wildcard results/validation/main/*.csv results/validation/main/*.json)
	$(PYTHON) figures/fig3_dem_accuracy.py
$(FIG_OUT)/fig4_audit.pdf: figures/fig4_audit.py $(FIG_DEPS) $(STUDY_MAP) $(wildcard results/hindcast/main/*.csv)
	$(PYTHON) figures/fig4_audit.py
$(FIG_OUT)/ed_block_size.pdf: figures/ed_block_size.py $(FIG_DEPS) \
        $(wildcard results/validation/main/*.csv results/validation/main/sensitivity_*/*.csv)
	$(PYTHON) figures/ed_block_size.py
$(FIG_OUT)/ed_vegetation.pdf: figures/ed_vegetation.py $(FIG_DEPS) \
        $(wildcard results/validation/main/*_accuracy.csv results/validation/main_vegetated/*_accuracy.csv)
	$(PYTHON) figures/ed_vegetation.py
$(FIG_OUT)/ed_screening.pdf: figures/ed_screening.py $(FIG_DEPS) results/levels/screening_sensitivity.json
	$(PYTHON) figures/ed_screening.py
$(FIG_OUT)/ed_h1_blocks_shape.pdf: figures/ed_h1_blocks.py $(FIG_DEPS) $(STUDY_MAP) results/hindcast/main/block_h1.csv
	$(PYTHON) figures/ed_h1_blocks.py
$(FIG_OUT)/ed_h2_hexbins.pdf: figures/ed_h2_hexbins.py $(FIG_DEPS) $(wildcard results/validation/main/*.csv \
        results/validation/main/*.json)
	$(PYTHON) figures/ed_h2_hexbins.py
manuscript/sections/ed_tables_draft.tex: figures/ed_sector_tables.py figures/_common.py results/metrics.json \
        results/hindcast/main/h1_mispredicts.csv
	$(PYTHON) figures/ed_sector_tables.py
# H3 (results/emergence/main/): Fig. 5; Table 3 (written into tables_draft.tex between its
# T3 markers) and the Supplementary Tables of scenario dates and further exposure.
RESULT_FIGS += $(FIG_OUT)/fig5_next_metres.pdf
H3_TABLES := $(wildcard results/emergence/main/*.csv results/emergence/main/*.json)
$(FIG_OUT)/fig5_next_metres.pdf: figures/fig5_next_metres.py $(FIG_DEPS) $(H3_TABLES)
	$(PYTHON) figures/fig5_next_metres.py
manuscript/sections/ed_tables_h3_draft.tex: figures/tab_h3.py figures/_common.py results/metrics.json \
        results/emergence/main/scenario_dates.csv refs/verification.csv
	$(PYTHON) figures/tab_h3.py

# ERL main display items (D-100, D-102). scripts/f1_erl_display.py rebuilds, with the
# hindcast's own functions, the shorelines and agreement counts the maps draw, checks every
# recomputed median and area against the frozen tables, and writes the small summaries that
# build_metrics reads (results/display/); it also reads git-ignored products (stage-2 tiles,
# flood-level caches, the Great Salt Lake grids). The Landsat composites are an Earth Engine
# acquisition (ee-erl-landsat, never run by `make all`); their record is committed.
ERL_DISPLAY := results/display/erl_hero.json results/display/erl_agreement.json results/display/erl_gsl.json
results/display/erl_hero.json: scripts/f1_erl_display.py scripts/a7_hindcast.py results/hindcast/main/run.json \
        results/hindcast/main/shape_offsets.json results/hindcast/main/displacement.csv \
        results/emergence/main/emergence.csv $(STUDY_MAP) results/study_area/zone_v6_cover_buffered.geojson \
        $(S2MAIN) $(STAGE2_PLAN) $(PRODUCT_GRIDS) $(CORRECTION) $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/f1_erl_display.py caspian
results/display/erl_agreement.json: results/display/erl_hero.json ;
results/display/erl_gsl.json: scripts/f1_erl_display.py scripts/g1_hindcast.py scripts/a7_hindcast.py \
        results/generality/great_salt_lake/h1_decision.json results/generality/great_salt_lake/arms_laea.geojson \
        $(wildcard results/generality/great_salt_lake/*/hindcast*/displacement.csv \
        results/generality/great_salt_lake/*/hindcast*/run.json) $(GSL_STATIC) $(GSL_INT)/stage2_main \
        $(GSL_INT)/dep3_60m.npy $(GSL_RAW)/vdatum_navd88_minus_egm2008.npz $(GSL_RAW)/bathymetry $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/f1_erl_display.py gsl
ERL_DEPS := figures/_erl.py $(FIG_DEPS)
RESULT_FIGS += $(FIG_OUT)/erl_fig_then_now.pdf $(FIG_OUT)/erl_fig_shorelines.pdf $(FIG_OUT)/erl_fig_gsl.pdf \
               $(FIG_OUT)/erl_fig_agreement.pdf $(FIG_OUT)/erl_fig_timeline.pdf $(FIG_OUT)/erl_tab_report_card.pdf \
               $(FIG_OUT)/erl_fig_audit.pdf
# The Letter's audit figure: the ERL version of fig4_audit, from the same hindcast tables (D-102 addendum).
$(FIG_OUT)/erl_fig_audit.pdf: figures/erl_fig_audit.py $(ERL_DEPS) $(wildcard results/hindcast/main/*.csv)
	$(PYTHON) figures/erl_fig_audit.py
$(FIG_OUT)/erl_fig_then_now.pdf: figures/erl_fig_then_now.py $(ERL_DEPS) results/provenance/erl_landsat.json $(STUDY_MAP)
	$(PYTHON) figures/erl_fig_then_now.py
$(FIG_OUT)/erl_fig_shorelines.pdf: figures/erl_fig_shorelines.py $(ERL_DEPS) results/display/erl_hero.json $(STUDY_MAP)
	$(PYTHON) figures/erl_fig_shorelines.py
$(FIG_OUT)/erl_fig_gsl.pdf: figures/erl_fig_gsl.py $(ERL_DEPS) results/display/erl_gsl.json \
        results/generality/great_salt_lake/arms_laea.geojson
	$(PYTHON) figures/erl_fig_gsl.py
$(FIG_OUT)/erl_fig_agreement.pdf: figures/erl_fig_agreement.py $(ERL_DEPS) results/display/erl_agreement.json $(STUDY_MAP)
	$(PYTHON) figures/erl_fig_agreement.py
$(FIG_OUT)/erl_fig_timeline.pdf: figures/erl_fig_timeline.py $(ERL_DEPS) scripts/a8_emergence.py \
        results/levels/projection_pathways.csv $(H3_TABLES)
	$(PYTHON) figures/erl_fig_timeline.py
# The report card: Table 1 in tables_erl.tex (between its ERL-T1 markers) and a rendered preview.
manuscript/sections/tables_erl.tex: figures/tab_report_card.py $(ERL_DEPS)
	$(PYTHON) figures/tab_report_card.py
$(FIG_OUT)/erl_tab_report_card.pdf: manuscript/sections/tables_erl.tex ;
# Landsat then-and-now composites (Earth Engine, interactive, 3 EECU-h cap; pilot first).
ee-erl-landsat:
	$(ee_ok)
	$(PYTHON) scripts/f2_erl_landsat.py pilot
	$(PYTHON) scripts/f2_erl_landsat.py fetch

# Every file tools/build_metrics.py reads (tests/test_makefile.py checks this list).
METRIC_INPUTS := $(INV_META) results/study_area/sector_areas.json results/study_area/sector_edges.json \
                 results/qc/dahiti_summary.json \
                 results/qc/level_crosscheck_w20d.json results/levels/scene_levels_summary.json \
                 results/params.json refs/literature_values.csv results/levels/setup_model.json \
                 results/qc/threshold_rule_summary.json results/provenance/geoid_difference.json \
                 results/study_area/blocks_25km.csv results/study_area/zone_v6_cover_buffered.geojson \
                 results/levels/screening_sensitivity.json results/levels/dahiti_bhs_offset.json \
                 results/generality/great_salt_lake/h1_decision.json results/generality/great_salt_lake/status.json \
                 results/generality/lake_urmia/status.json
METRIC_INPUTS += results/robustness/verdicts.json

# Robustness verdicts (brief section 10; rule D-092) from the per-check summaries, which
# stage 2b rebuilds (ROB_SUMMARIES), and their headline table.
results/robustness/verdicts.json: scripts/r20_verdicts.py $(ROB_SUMMARIES) | $(VERIFIED)
	$(PYTHON) scripts/r20_verdicts.py
results/robustness/headline_table.csv: scripts/r22_report_table.py results/robustness/verdicts.json
	$(PYTHON) scripts/r22_report_table.py

# tools/build_metrics.py reads the H1-H3 tables of stage 2b directly.
METRIC_INPUTS += $(PHASE_A_RESULTS)

# DAHITI -> BHS-77 offset from the CASPCOM post levels (PREREG section 5; D-093). The script
# stops if $(CASPCOM) differs from its acquisition record. Also writes the annual differences.
results/levels/dahiti_bhs_offset.json: scripts/a7_dahiti_bhs_offset.py $(DAHITI) $(CASPCOM) \
        results/provenance/caspcom_catalogue.json refs/caspcom_bulletin_levels.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a7_dahiti_bhs_offset.py

# ATL24 check of the near-term surfaces (PREREG Amendment 17, D-095): inputs read by build_metrics
METRIC_INPUTS += results/provenance/icesat2_atl24.json results/validation/atl24/run.json \
                 results/validation/atl24/verdicts.json results/validation/atl24/water_surface_constant.json

# Stage-2 seam-gap QC (robustness agent, scripts/r21_seam_gaps.py), read by build_metrics
METRIC_INPUTS += results/robustness/r21_seam_gaps/summary.json

# WDPA protected-area and Ramsar exposure (D-103): counts and areas from scripts/a8_wdpa_exposure.py
# layers | exposure on the H3 runs' cached surfaces (stage 2b).
METRIC_INPUTS += results/emergence/wdpa/layers.json results/emergence/wdpa/run.json results/emergence/wdpa/run_noksrz.json

# ERL display summaries and the Landsat acquisition record (display_erl_metrics)
METRIC_INPUTS += $(ERL_DISPLAY) results/provenance/erl_landsat.json

# Values the ERL Letter quotes from committed result files (erl_text_metrics, D-107)
METRIC_INPUTS += results/generality/great_salt_lake/lineage.json results/hindcast/main/d089_rerun_comparison.json \
                 results/robustness/r08_blocking/summary.json

## Revision after the internal review (PREREG Amendment 18; D-112) ------------------------
# Each workstream adds its metrics through tools/metrics_rev_<ws>.py (build_metrics.revision_metrics).
METRIC_INPUTS += $(wildcard tools/metrics_rev_*.py)

# Workstream S, H1 statistics (D-113 to D-115). HEAVY (about 4,600 s): the raster part reads
# the cached flood maps and stage-2 tiles and reproduces the registered tables first.
REV_S_RASTERS := $(addprefix results/revision/s/rasters/,band_overlap.csv oos_area.csv test_displacement.csv \
  floor_e1_bins.csv checks.json run.json)
$(REV_S_RASTERS) &: scripts/rev_s_rasters.py scripts/a7_sector_shape.py $(HC_INPUTS) $(call hc_tables,$(HC)/main) \
        $(HC)/main/shape_offsets.json | $(VERIFIED)
	$(PYTHON) scripts/rev_s_rasters.py
REV_S := $(addprefix results/revision/s/,item1_counts.csv item2_p_mispredict.csv item2_count_distribution.csv \
  item2_dq.csv item4_area_bias.csv item4_oos.csv item4_overlap.csv item5_sector_values.csv item5_tests.csv \
  item5_null.csv item5_block_groups.csv item8_binomial.csv item8_hysteresis.csv r3m4_test_blocks.csv \
  r3m4_floor_halves.csv r3m4_floor_halves_levels.csv r3minor1_floor.csv r1minor11_medians.csv \
  r2minor8_record_low.csv summary.json)
$(REV_S) &: scripts/rev_s_tables.py $(call h1_tables,$(HC)/main) $(call hc_tables,$(HC)/main) \
        $(call h1_tables,$(HC)/main_noenv) $(call hc_tables,$(HC)/main_noenv) $(call h1_tables,$(HC)/main_sectorshape) \
        results/levels/sector_labels.csv results/robustness/r10_morphology/summary.json \
        results/robustness/r10_morphology/morphology_detail.csv $(S2MAIN) $(REV_S_RASTERS) $(PKG)
	$(PYTHON) scripts/rev_s_tables.py
METRIC_INPUTS += $(REV_S)
REVISION_RESULTS += $(REV_S_RASTERS) $(REV_S)

results/metrics.json: tools/build_metrics.py $(METRIC_INPUTS) | $(VERIFIED)
	$(PYTHON) tools/build_metrics.py

manuscript/numbers.tex: tools/make_macros.py results/metrics.json
	$(PYTHON) tools/make_macros.py

# Offline: references.bib and verification.csv agree with refs/cite_list.csv and every
# entry is well formed (`make refs` regenerates them over the network).
check-refs:
	$(PYTHON) tools/fetch_bibtex.py --check

# Offline: every verbatim quote in refs/notes/*.md (sections "checked programmatically")
# and refs/literature_values.csv is found in the retrieved text kept under data/raw/lit
# (checksums in refs/notes/SOURCES.csv). Not part of `all`: data/raw/lit is git-ignored
# and no target acquires it, so on a fresh clone the sources are first retrieved again
# from the URLs in SOURCES.csv and the notes.
check-quotes:
	$(PYTHON) scripts/check_quotes.py

manuscript/main.pdf: manuscript/main.tex manuscript/numbers.tex $(wildcard manuscript/sections/*.tex) \
        refs/references.bib $(FIGS) $(RESULT_FIGS) | check-refs
	$(TECTONIC) -X compile manuscript/main.tex

# Environmental Research Letters Letter and its Supplementary Information (D-099, D-107).
# Both use IOP's template class manuscript/iopjournal.cls (unmodified; D-107 addendum). The
# Letter keeps its .aux (git-ignored), which the Supplementary reads through xr-hyper for the
# numbers of the Letter's figures, so the Letter is built first.
manuscript/main_erl.pdf: manuscript/main_erl.tex manuscript/iopjournal.cls manuscript/numbers.tex $(wildcard manuscript/sections/*.tex) \
        refs/references.bib $(FIGS) $(RESULT_FIGS) | check-refs
	$(TECTONIC) -X compile --keep-intermediates manuscript/main_erl.tex
manuscript/supplementary_erl.pdf: manuscript/supplementary_erl.tex manuscript/iopjournal.cls manuscript/main_erl.pdf manuscript/numbers.tex \
        $(wildcard manuscript/sections/*.tex) refs/references.bib $(FIGS) $(RESULT_FIGS) | check-refs
	$(TECTONIC) -X compile manuscript/supplementary_erl.tex
# Main-text word count of the Letter as ERL defines it (excluding title, abstract, captions,
# tables and references); also the abstract's length. Not part of `all`.
.PHONY: erl-wordcount
erl-wordcount:
	$(PYTHON) tools/word_count_erl.py

## named steps -------------------------------------------------------------
check-inputs: $(VERIFIED)
dahiti-qc: results/qc/dahiti_summary.json
study-area: results/study_area/sector_areas.json results/study_area/sector_edges.json
scene-levels: results/qc/level_crosscheck.json results/qc/level_crosscheck_pairs.csv \
              results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv \
              results/levels/scene_levels_summary.json
qc-figures: $(QC_FIGS)
figures: $(FIGS) $(RESULT_FIGS) manuscript/sections/ed_tables_draft.tex manuscript/sections/ed_tables_h3_draft.tex
params: results/params.json
metrics: results/metrics.json manuscript/numbers.tex
macros: manuscript/numbers.tex
manuscript: manuscript/main.pdf manuscript/main_erl.pdf manuscript/supplementary_erl.pdf
phase-a: $(PHASE_A_RESULTS)
# Every input `make all` reads that an acquisition step writes, one per line (stage 1 and 2b).
inputs:
	$(foreach f,$(DAHITI) $(HYDROWEB) $(SCENES) $(CASPCOM) $(GRIDS) $(ETOPO1) $(ACQ_INPUTS),$(info $(f)))

RESULTS_ALL := results/qc/dahiti_summary.json results/study_area/sector_areas.json \
               results/study_area/sector_edges.json \
               results/qc/level_crosscheck.json results/qc/level_crosscheck_pairs.csv \
               results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv \
               results/levels/scene_levels_summary.json results/levels/dahiti_bhs_offset.json \
               results/params.json results/metrics.json

all: $(VERIFIED) $(PHASE_A_RESULTS) $(RESULTS_ALL) $(QC_FIGS) $(FIGS) $(RESULT_FIGS) manuscript/sections/ed_tables_draft.tex \
     manuscript/sections/ed_tables_h3_draft.tex manuscript/numbers.tex manuscript/main.pdf \
     manuscript/main_erl.pdf manuscript/supplementary_erl.pdf
