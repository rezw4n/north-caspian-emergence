# Data release rules

Rules for any public release of this repository or its data (Zenodo, ESSD, a public GitHub repository). The repository is private until the PI decides otherwise; making it public is an irreversible step for the PI (CLAUDE.md, Stop conditions).

## Never released

| Item | Why | Instead |
|---|---|---|
| `data/` (raw and interim data) | Large, and some licences forbid redistribution | Provenance records in `results/provenance/`; download scripts |
| Secrets (`.env`, credential files, tokens) | Security (CLAUDE.md rule 6) | — |
| `results/levels/dahiti_bhs_annual_diffs.csv` | Per-gauge annual means of CASPCOM levels; the site states "All Rights Reserved" and gives no licence (D-093, D-101) | Release δ and σ (`results/levels/dahiti_bhs_offset.json`, after checking that it holds only derived summaries) and `scripts/a7_dahiti_bhs_offset.py`, which recomputes them from the catalogue |
| WDPA polygons, raw or clipped | WDPA terms forbid redistributing the data (D-101) | Derived exposure counts and areas only, with the UNEP-WCMC and IUCN citation |

## Released with conditions

| Item | Condition |
|---|---|
| Anything derived from Copernicus DEM GLO-30 (H2 tables) | The licence Art. 6 notice: "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved", plus the licence's no-liability sentence (D-091, D-101) |
| `refs/caspcom_bulletin_levels.csv` | Short quotations with bulletin and page locations, cited |

## Before any release

1. Check the files against this list. `results/levels/dahiti_bhs_offset.json` must hold only δ, σ and method fields, not per-gauge values.
2. Re-read each source's terms, which change without notice.
3. **Rewrite the git history** to drop `results/levels/dahiti_bhs_annual_diffs.csv`, which was committed in 4e95e9a. Approved by the PI on 2026-10-21 (D-109).
   - Do it only when no agent is working and the working tree is clean; a rewrite on a dirty tree destroys uncommitted work.
   - Keep a backup branch.
   - Verify that the PREREG commit c2b94d8 and the first metric commit 812662e keep their hashes; they predate 4e95e9a.
   - Then force-push, and rerun the history audit (no secrets, no restricted files).
4. The PI approves the release and makes the repository public.
