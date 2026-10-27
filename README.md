# Global bathymetry grids misplace the retreating North Caspian shoreline by kilometres

Rezwan Ahmed (ORCID 0009-0002-8003-099X; corresponding author, rezwan490@gmail.com) and Al Sakib (ORCID 0009-0007-6064-4827), Department of Geography and Environment, University of Dhaka; Tanveer Hossain Munim (ORCID 0009-0000-2295-9325), Regional Integrated Multi-Hazard Early Warning System (RIMES), Dhaka; and Kazi Farha Farzana Suhi, Department of Geography, University of Florida.

This repository holds the code, derived data and manuscript sources for the paper submitted to Remote Sensing Applications: Society and Environment. It contains the analysis package, the pipeline scripts, the figure and table scripts, the derived tables behind every figure and number in the paper, the pre-registration with its dated amendments, and the LaTeX sources of the article and its Supplementary material. The numbers in the manuscript are not typed by hand: they are read from `results/metrics.json` and inserted as LaTeX macros (`manuscript/numbers.tex`).

## Requirements

The code was developed and run on Windows with Python 3.11. The conda environment is defined in `environment.yml` (top-level packages, conda-forge) and locked in `requirements.lock` (win-64, every package with its URL and checksum). A few pip-only packages are listed in `requirements-pip.txt`. With micromamba:

```bash
micromamba create -y -n caspian -f requirements.lock
micromamba run -n caspian python -m pip install --no-deps -r requirements-pip.txt
micromamba run -n caspian python -m pip install --no-deps --no-build-isolation -e .
```

`make env` runs the same three commands. On other platforms, create the environment from `environment.yml` instead of the lock file. The manuscript is compiled with tectonic.

## Rebuilding the results

The raw inputs are not in the repository. They are downloaded once into `data/` (git-ignored) by separate targets, which are never run by `make all`:

```bash
make fetch                  # DAHITI, Hydroweb, SWOT, GEBCO, ETOPO 2022, SRTM15+, OpenStreetMap, WDPA, CASPCOM, references
make ee-inputs EE_OK=1      # Landsat and Sentinel-2 scene inventory and ETOPO1, via Google Earth Engine
```

Some downloads need credentials, set as environment variables: `DAHITI_API_KEY`, `HYDROWEB_API_KEY` and, for ICESat-2, `EARTHDATA_TOKEN`. Earth Engine needs `earthengine authenticate` and a registered Cloud project. The Earth Engine batch stages (`ee-zone`, `ee-stage1`, `ee-era5-winds`, `ee-erl-landsat`) are also run by hand, with `EE_OK=1` and, where needed, `STEP=...`; they use Earth Engine compute quota. The remaining inputs (the stage-2 Earth Engine exports, ICESat-2, GLO-30, GEDI, habitat layers and the Great Salt Lake inputs) are acquired by their own scripts. `make inputs` lists every such input, and `make all` names the acquiring script when one is missing. Each acquisition writes a record with the SHA-256 checksum of every raw file to `results/provenance/` or `results/inventory/`.

Then:

```bash
micromamba run -n caspian make all
```

`make all` first checks every raw input against its recorded checksum. It then rebuilds the results, figures, tables, `results/metrics.json`, `manuscript/numbers.tex` and the PDFs. A full rebuild takes about 20 to 23 hours on one desktop machine. `make help` lists the targets, `make -n -B all` lists every step in order, and `make test` runs the test suite.

## Repository layout

- `src/caspian/`: the analysis package (water classification, waterlines, level assignment, datums, hindcast, conformal intervals, area integration).
- `scripts/`: pipeline entry points, one per step; `scripts/README.md` lists them.
- `figures/`: one script per figure and table.
- `tools/`: metrics, LaTeX macros, input checks, reference retrieval and the data package builder.
- `results/`: derived tables, run records, provenance records and `metrics.json`.
- `outputs/`: quality-control figures.
- `manuscript/`: LaTeX sources of the article (`rse_main.tex`), its Supplementary material (`rse_supplementary.tex`), the highlights and the cover letter. `main_erl.tex` and `supplementary_erl.tex` are the earlier, shorter version of the paper; the Supplementary material reuses their section files.
- `docs/`: the pre-registration (`PREREG.md`), the vertical datum chain (`DATUMS.md`) and the data release rules (`DATA_RELEASE.md`).
- `refs/`: the bibliography and the list of verified references.
- `tests/`: the pytest suite.

## Data sources and licences

The input data come from their providers under their own terms; the Supplementary Information cites each of them.

- Landsat and Sentinel-2 surface reflectance (USGS, ESA/Copernicus), accessed through Google Earth Engine.
- Water levels: DAHITI (DGFI-TUM), Hydroweb and SWOT (hydroweb.next), USGS gauges for Great Salt Lake.
- Bathymetry and relief: GEBCO_2026, GEBCO_2022, ETOPO 2022, ETOPO1 (NOAA NCEI), SRTM15+ (Scripps).
- Heights for validation: ICESat-2 (NSIDC), GEDI (LP DAAC), Copernicus DEM GLO-30, USGS 3DEP.
- ERA5 winds (Copernicus, ECMWF); ESA WorldCover; OpenStreetMap (Open Database License).
- Gauge levels of the Coordinating Committee on Hydrometeorology of the Caspian Sea (CASPCOM). The catalogue states no licence, so the gauge values are not redistributed; the repository holds only derived offset statistics and the script that recomputes them from the catalogue.
- World Database on Protected Areas (UNEP-WCMC and IUCN). Its terms forbid redistribution, so only derived counts and areas are included, for non-commercial use.

Results derived from the Copernicus DEM were produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. `docs/DATA_RELEASE.md` lists what may and may not be redistributed.

## Citation

Please cite the paper and this repository; `CITATION.cff` has the details.

## Licence

The code is released under the MIT licence (`LICENSE`). The derived data package (`make data-package`, which writes `release/zenodo/`) is released under CC BY 4.0, with the exceptions listed above.
