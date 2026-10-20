# Figures

One script per figure, writing vector PDF, a 600 dpi TIFF and a PNG preview to `figures/out/` at the exact journal width (`_common.py`: 88 and 180 mm, Nature Communications guidelines read 2026-10-21, `docs/journal_specs.md`). `make figures` builds them, and `make all` runs it.

| Script | Display item | Inputs |
|---|---|---|
| `fig1_levels.py` | Fig. 1 inset / Supplementary Fig.: basin level, levelled scenes per month, epochs E1–E3 | raw DAHITI and Hydroweb via `scripts/a4_scene_levels.py`, `data/interim/levels/scene_levels.csv`, `results/metrics.json` |
| `fig_study_area.py` | Supplementary Fig.: domain, sectors, stage-2 cover, 25 km blocks by split | `results/study_area/` (sectors, cover, blocks, sea mask) |
| `supp_setup_model.py` | Supplementary Fig.: setup-model calibration per sector | `results/levels/setup_model.json` |

Draft legends are in `manuscript/sections/figure_captions_draft.tex`; numbers in them come from `results/metrics.json`, and display constants quoted there (scale bar, graticule) are exported by `tools/export_params.py`.

QC figures (`qc_*.py`) write vector PDF and a PNG preview to `outputs/qc/`; `make all` runs them (`make qc-figures`).
