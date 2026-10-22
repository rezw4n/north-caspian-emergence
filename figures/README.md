# Figures

One script per figure, writing vector PDF, a 600 dpi TIFF and a PNG preview to `figures/out/` at the exact journal width (`_common.py`: 88 and 180 mm, Nature Communications guidelines read 2026-10-21). `make figures` builds them, and `make all` runs it.

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

## ERL main display items

Drawn to the Environmental Research Letters rules: 150 mm wide, text 8 pt or more, panels "(a)", scale bar and graticule on every map, product identity by colour and line style or marker. Typeface TeX Gyre Heros, a Helvetica version (IOP restricts figure fonts to Times, Helvetica, Courier or Symbol), embedded; its files come from tectonic's cache (`_erl.font_files`). The Landsat composites are JPEG inside their PDF (`_erl.jpeg_images`); the TIFF stays lossless at 600 dpi. Shared settings and display constants in `_erl.py` (exported as `params.erl_fig_*`); every number drawn comes from `results/metrics.json` (`_erl.num`, formatted as the text's macros). The shorelines, counts and arrows are rebuilt by `scripts/f1_erl_display.py` with the hindcast's and the H3 driver's own functions, checked against the frozen tables, and cached in `data/interim/figures/erl/`; its small summaries (`results/display/`) feed `erl.*` metrics. The existing figures above move to the Supplementary.

| Script | Display item | Inputs |
|---|---|---|
| `erl_fig_then_now.py` | Then and now: Landsat true-colour composites of summer 1996 (TM) and 2025 (OLI) with the observed shorelines of four E3 level bins (earlier versions drew emergence-date contours of the waterline DEM); the basin level with each bin's acquisition period | `data/raw/erl_landsat/` (`scripts/f2_erl_landsat.py`, Earth Engine, `make ee-erl-landsat EE_OK=1`; record `results/provenance/erl_landsat.json`), `results/display/erl_waterlines.json` and `data/interim/figures/erl/waterlines_main.npz` (`scripts/f3_erl_waterlines.py`), `level_series_daily.csv`, metrics `rev.fig.f1.*` |
| `erl_fig_shorelines.py` | Where the maps put the shoreline: observed against GEBCO_2026, ETOPO 2022 and SRTM15+ shorelines at −28.05 m, six zooms at −28.05 and −27.05 m, arrows | `results/display/erl_hero.json`, `data/interim/figures/erl/hero_main.npz` |
| `erl_fig_gsl.py` | Great Salt Lake: both arms at a representative level with the products and the 3DEP control (mostly 2016 lidar); medians against the noise floors; the grids' heights over each arm's emerged band | `results/display/erl_gsl.json`, `data/interim/figures/erl/gsl.npz`, `results/revision/g/gsl_band_height_histograms.csv`, metrics `generality.*`, `rev.g.gsl.*` |
| `erl_fig_agreement.py` | Where the first metre emerges: how many of the five grids place each cell in the first metre on the composite observed baseline (a) and on the shifted grids (b), majority hatched, the observed 2025–26 shoreline; (c) the share of each grid's 1 m band already dry | `results/display/erl_nearterm_map.json`, `data/interim/figures/erl/nearterm_map.npz` (`scripts/f4_erl_nearterm_map.py`, the set-up of the near-term revision analysis), metrics `rev.fig.f5.*`, `rev.n.already_dry.*` |
| `erl_fig_timeline.py` | When and how much: the level record joined to the seven primary pathways, the range pathways shaded; E(Δ) at 0.1 m on the composite baseline and the shifted grids with the decline-normalised envelopes to 2.1 m, the grids as distributed in grey | `level_series_daily.csv`, `results/levels/projection_pathways.csv` (via `scripts/a8_emergence.py`), `results/revision/n/curves.csv`, `curves_composite.csv`, `envelopes.csv`, metrics `h3.*`, `rev.n.env.*` |
| `erl_fig_audit.py` | The Letter's audit figure: area bias by level (after the shift, and on the datum chain, one y-range), displacement by level band in the validated and the not-validated sectors, per-level D/Q by sector; the ERL version of `fig4_audit.py` without its per-block maps | `results/hindcast/main/` (as `fig4_audit.py`), `results/revision/s/item2_dq.csv` |
| `tab_report_card.py` | Table 1, the product report card, in `manuscript/sections/tables_erl.tex` (ERL-T1 markers) and a preview `erl_tab_report_card.png` | metrics `h1.*`, `h2.atl08.*`, `generality.*`, `h3.emergence.*`, `erl.card.*` |

Draft legends: `manuscript/sections/figure_captions_erl.tex`.

QC figures (`qc_*.py`) write vector PDF and a PNG preview to `outputs/qc/`; `make all` runs them (`make qc-figures`).
