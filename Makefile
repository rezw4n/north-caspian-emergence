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
        ee-inputs ee-scene-inventory ee-etopo1 ee-zone ee-stage1 ee-era5-winds \
        check-inputs dahiti-qc study-area scene-levels qc-figures params metrics macros check-refs \
        check-quotes manuscript figures all

help:
	@echo Targets
	@echo   all            rebuild results, QC figures, numbers.tex and manuscript/main.pdf from the raw inputs
	@echo   fetch          acquire network inputs: dahiti-download hydroweb-download swot-download bathymetry-download osm-download refs caspcom-download
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
fetch: dahiti-download hydroweb-download swot-download bathymetry-download osm-download refs caspcom-download

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

# doi.org and Crossref/DataCite: refs/references.bib and refs/verification.csv.
refs:
	$(PYTHON) tools/fetch_bibtex.py

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

# export_params.py imports or parses the scripts whose constants the Methods quote.
PARAM_SCRIPTS := $(addprefix scripts/,a4_scene_levels.py a3_stage1_interactive.py a3_thresholds.py a5_setup_model.py a5_swot_sector_levels.py \
                 a5_sea_mask.py a6_screening.py a6_stage2_plan.py a6_stage2_export.py a6_static_rule.py \
                 a6_waterlines.py a6_vegetation.py a6_dem.py a7_beyond_state.py a7_hindcast.py)
results/params.json: tools/export_params.py $(PARAM_SCRIPTS) figures/fig_study_area.py $(PKG)
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

# DAHITI -> BHS-77 offset from the CASPCOM post levels (PREREG section 5; D-093). The script
# stops if $(CASPCOM) differs from its acquisition record. Also writes the annual differences.
results/levels/dahiti_bhs_offset.json: scripts/a7_dahiti_bhs_offset.py $(DAHITI) $(CASPCOM) \
        results/provenance/caspcom_catalogue.json refs/caspcom_bulletin_levels.csv $(PKG) | $(VERIFIED)
	$(PYTHON) scripts/a7_dahiti_bhs_offset.py

# ATL24 check of the near-term surfaces (PREREG Amendment 17, D-095): inputs read by build_metrics
METRIC_INPUTS += results/provenance/icesat2_atl24.json results/validation/atl24/run.json \
                 results/validation/atl24/verdicts.json results/validation/atl24/water_surface_constant.json

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
manuscript: manuscript/main.pdf

RESULTS_ALL := results/qc/dahiti_summary.json results/study_area/sector_areas.json \
               results/study_area/sector_edges.json \
               results/qc/level_crosscheck.json results/qc/level_crosscheck_pairs.csv \
               results/qc/level_crosscheck_w20d.json results/qc/level_crosscheck_pairs_w20d.csv \
               results/levels/scene_levels_summary.json results/levels/dahiti_bhs_offset.json \
               results/params.json results/metrics.json

all: $(VERIFIED) $(RESULTS_ALL) $(QC_FIGS) $(FIGS) $(RESULT_FIGS) manuscript/sections/ed_tables_draft.tex \
     manuscript/sections/ed_tables_h3_draft.tex manuscript/numbers.tex manuscript/main.pdf
