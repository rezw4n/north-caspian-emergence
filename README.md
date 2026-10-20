# North Caspian seabed emergence

Code and documentation for *Four decades of retreating shorelines audit projected seabed emergence in the North Caspian Sea* (working title).

**Author:** Rezwan Ahmed, Department of Geography and Environment, University of Dhaka.

The project turns four decades of satellite waterlines, each paired with the altimetry water level on its acquisition date, into a seabed topography of the emerged North Caspian shelf. It then uses that record to hindcast-test the global bathymetry products behind seabed-emergence projections. The plan is `docs/RESEARCH_BRIEF.md`; the working rules are `CLAUDE.md`.

## Setup (Windows, PowerShell or Git Bash)

```bash
# 1. micromamba (user-level binary); then build the locked environment
#    (same three steps as `make env`): conda lock, pip-only packages, the package itself
micromamba create -y -n caspian -f requirements.lock
micromamba run -n caspian python -m pip install --no-deps -r requirements-pip.txt
micromamba run -n caspian python -m pip install --no-deps --no-build-isolation -e .

# 2. credentials (never committed)
earthengine authenticate            # Earth Engine, project ee-rezwanahmed
setx EARTHDATA_TOKEN "<token>"      # NASA Earthdata
setx DAHITI_API_KEY "<key>"         # DAHITI water levels
setx HYDROWEB_API_KEY "<key>"       # hydroweb.next (Hydroweb levels, SWOT)

# 3. raw inputs (git-ignored, under data/); never fetched by `make all`
micromamba run -n caspian make fetch                 # network downloads with the keys above
micromamba run -n caspian make ee-inputs EE_OK=1     # Earth Engine: spends EECU; pilot first

# 4. checks and rebuild
micromamba run -n caspian make verify-0
micromamba run -n caspian make all       # results, QC figures, numbers.tex, manuscript PDF
micromamba run -n caspian make -B all    # force a full rebuild
```

`make all` stops with the name of the acquiring target if a raw input is missing. Before it rebuilds any result it checks every raw input against its acquisition record (`make check-inputs`, `tools/check_inputs.py`): the SHA-256 in `results/provenance/*.json` and `results/inventory/scenes_csv.json`, and the scene-inventory numbers in `results/inventory/inventory_meta.json` recomputed from `scenes.csv`. A raw file that differs from the recorded one stops the build. Re-acquiring an input (`make fetch`, `make ee-inputs`) rewrites its record, so a changed input shows up in `git diff` of `results/`. `make help` lists the targets; the Makefile header describes the two stages.

## Layout

| Path | Contents |
|---|---|
| `src/caspian/` | package (io, masks, water, levels, surge, waterlines, dem, conformal, hindcast, exposure) |
| `scripts/` | entry points per phase |
| `figures/` | one script per figure, writing to `outputs/figures/` (QC figures to `outputs/qc/`) |
| `tests/` | pytest suite |
| `tools/verify.py` | phase checks (`make verify-0`, `verify-A`, `verify-B`) |
| `results/metrics.json` | the single source of every number in the manuscript |
| `refs/` | bibliography and citation audit trail |
| `docs/` | brief, status, decisions, pre-registration, datums, gate memos |
| `manuscript/` | journal template; numbers enter through generated macros |
| `data/` | git-ignored raw, interim and processed data |

Code: MIT licence. Data products: CC BY 4.0 (on release).
