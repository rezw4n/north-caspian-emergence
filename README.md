# North Caspian seabed emergence

Code and documentation for *Four decades of retreating shorelines audit projected seabed emergence in the North Caspian Sea* (working title).

**Author:** Rezwan Ahmed, Department of Geography and Environment, University of Dhaka.

The project turns four decades of satellite waterlines, each paired with the altimetry water level on its acquisition date, into a seabed topography of the emerged North Caspian shelf. It then uses that record to hindcast-test the global bathymetry products behind seabed-emergence projections. The plan is `docs/RESEARCH_BRIEF.md`; the working rules are `CLAUDE.md`.

## Setup (Windows, PowerShell or Git Bash)

```bash
# 1. micromamba (user-level binary); then build the locked environment
micromamba create -y -n caspian -f requirements.lock
micromamba run -n caspian python -m pip install --no-deps --no-build-isolation -e .

# 2. credentials (never committed)
earthengine authenticate            # Earth Engine, project ee-rezwanahmed
setx EARTHDATA_TOKEN "<token>"      # NASA Earthdata

# 3. inputs the PI supplies (git-ignored)
#    data/raw/dahiti/<DAHITI Caspian Sea water-level series>.csv

# 4. checks and rebuild
micromamba run -n caspian make verify-0
micromamba run -n caspian make all
```

## Layout

| Path | Contents |
|---|---|
| `src/caspian/` | package (io, masks, water, levels, surge, waterlines, dem, conformal, hindcast, exposure) |
| `scripts/` | entry points per phase |
| `figures/` | one script per figure, writing to `outputs/figures/` |
| `tests/` | pytest suite |
| `tools/verify.py` | phase checks (`make verify-0`, `verify-A`, `verify-B`) |
| `results/metrics.json` | the single source of every number in the manuscript |
| `refs/` | bibliography and citation audit trail |
| `docs/` | brief, status, decisions, pre-registration, datums, gate memos |
| `manuscript/` | journal template; numbers enter through generated macros |
| `data/` | git-ignored raw, interim and processed data |

Code: MIT licence. Data products: CC BY 4.0 (on release).
