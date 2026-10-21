# Figures

One script per figure, writing vector PDF, a 600 dpi TIFF and a PNG preview to `figures/out/` at the exact journal width (`_common.py`: 88 and 180 mm, Nature Communications guidelines read 2026-10-21, `docs/journal_specs.md`). `make figures` builds them, and `make all` runs it.

| Script | Display item | Inputs |
|---|---|---|
| `fig1_levels.py` | Fig. 1 inset / Supplementary Fig.: basin level, levelled scenes per month, epochs E1–E3 | raw DAHITI and Hydroweb via `scripts/a4_scene_levels.py`, `data/interim/levels/scene_levels.csv`, `results/metrics.json` |
| `fig_study_area.py` | Supplementary Fig.: domain, sectors, stage-2 cover, 25 km blocks by split | `results/study_area/` (sectors, cover, blocks, sea mask) |
| `supp_setup_model.py` | Supplementary Fig.: setup-model calibration per sector | `results/levels/setup_model.json` |
| `fig1_emergence.py` | Fig. 1: emergence year of the emerged band, sectors, deltas, OSM ports and platforms; level series; Landsat acquisitions per month | `data/processed/dem/main/emergence/` (tiles, `level_series_daily.csv`), `results/study_area/` (sectors, sea mask, `extents_v6_250m.npz`), `data/raw/osm/`, `data/interim/stage2/plan_main_tau9900cm_landsat.csv` |
| `fig2_method.py` | Fig. 2: waterline–level schematic; waterlines of one training block; surge-screening sensitivity; E1 against E2 + E3 | `data/interim/waterlines/main/points.csv.gz`, `results/levels/screening_sensitivity.json`, `results/hindcast/main/noise_floor.csv`, block tables |
| `fig3_dem_accuracy.py` | Fig. 3: DEM map and test blocks; DEM against ATL06 (hexbin); error against interpolation distance and by sector; conformal coverage | `data/processed/dem/main/`, `data/interim/validation/main/` (points, annotations), `results/validation/main/` |
| `fig4_audit.py` | Fig. 4: area bias by level; displacement by level band; per-block displacement maps; sector summary against the noise floor | `results/hindcast/main/` |
| `ed_block_size.py` | Supplementary Fig.: H2 block-size sensitivity (12.5, 25, 50 km) | `results/validation/main/`, `sensitivity_*m/` |
| `ed_vegetation.py` | Supplementary Fig.: bare against vegetated emerged seabed (check 9) | `results/validation/main/`, `main_vegetated/` |
| `ed_screening.py` | Supplementary Fig.: surge-screening sensitivity and the acquisitions each threshold would remove | `results/levels/screening_sensitivity.json`, stage-2 plan |
| `ed_h1_blocks.py` | Supplementary Figs.: per-block displacement maps of every product, shape and total variants | `results/hindcast/main/block_h1.csv` |
| `ed_h2_hexbins.py` | Supplementary Fig.: every model against ICESat-2 (ATL06, ATL08) | as `fig3_dem_accuracy.py` |
| `fig5_next_metres.py` | Fig. 5: emerged area against decline (as used against fused, 90% intervals); per-sector emergence at 1 and 2 m; exposure by category; scenario dates of the primary pathways | `results/emergence/main/` (emergence, exposure, scenario dates, H3 decision) |
| `tab_h3.py` | Table 3 (1 m, written into `tables_draft.tex` between its T3 markers) and Supplementary Tables (2 m, further habitat layers, scenario dates by pathway) in `manuscript/sections/ed_tables_h3_draft.tex`, `\metric{}` macros only | `results/metrics.json`, `results/emergence/main/scenario_dates.csv`, `refs/verification.csv` |
| `ed_sector_tables.py` | Supplementary Tables: full H2 and H1 sector tables, written to `manuscript/sections/ed_tables_draft.tex` as `\metric{}` macros only | `results/metrics.json`, `results/hindcast/main/h1_mispredicts.csv` |

The results figures show frozen results; they recompute nothing. `_data.py` reads the large products at reduced size (rasters decimated tile by tile, point tables in chunks of selected columns) and caches them in `data/interim/figures/`; each cache stores the size and modification time of its inputs and is rebuilt when one changes. The ICESat-2 selection is checked against the committed accuracy tables (count and DEM NMAD/RMSE) before it is drawn. Peak memory of any script is a few hundred MB.

Known data defect (2026-10-21): `results/hindcast/main/block_h1.csv` has rows whose `median_abs_m` is negative (−2,780,457 m, the decoded int32 minimum, not a distance); the maps hatch them as undefined and `results/metrics.json` counts them (`h1.block_h1.n_undefined*`). Fixing it needs the hindcast driver and a rerun, not a figure change.

Draft legends are in `manuscript/sections/figure_captions_draft.tex`; numbers in them come from `results/metrics.json`, and display constants quoted there (scale bar, graticule, map grid, minimum bin size) are exported by `tools/export_params.py`. Draft tables are in `manuscript/sections/tables_draft.tex` (T1–T3) and `ed_tables_draft.tex` (generated).

QC figures (`qc_*.py`) write vector PDF and a PNG preview to `outputs/qc/`; `make all` runs them (`make qc-figures`).
