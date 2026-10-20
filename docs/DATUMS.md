# Vertical and horizontal datums

Every height in this project is converted to one reference before it is compared with another. This file records the full chain. Each entry gives what the product's own documentation says, the source we read, and anything still open. Offsets that the documentation cannot resolve are estimated empirically in Phase A, with uncertainty, and logged here and in `results/metrics.json`.

**Target reference for analysis:** heights in metres in the Baltic Height System 1977 (BHS-77, normal heights; the system in which Caspian levels are conventionally quoted), horizontal coordinates in WGS84. Areas and grids use the equal-area projection in D-002 of `docs/DECISIONS.md`.

## Chain (to be completed in Phase A)

```
h_ell (ellipsoidal, satellite) ──(− N or ζ of a geoid/quasi-geoid model)──► H (physical height above that model)
H_model ──(+ δ_model→BHS, estimated empirically with uncertainty)──► H_BHS-77
```

| Quantity | Reference surface as documented | Source read | Status |
|---|---|---|---|
| DAHITI water level | Normal height: ellipsoidal altimetry height minus a (quasi-)geoid model (EIGEN-6C3stat in the 2015 description; EIGEN-6C4 in v6.0) | Schwatke et al. 2015, HESS 19, §2, Table 2; DAHITI product page (accessed 2026-10-20) | Verified for the method; the series file's own header is checked when it arrives |
| Official Caspian levels (Roshydromet, Kazhydromet) | Baltic Height System (Kronstadt tide-gauge zero) | — | To verify in Phase A |
| G-REALM level (cross-check) | — | — | To verify in Phase A |
| ICESat-2 ATL06 / ATL08 / ATL24 | — | — | To verify in Phase A from the product ATBDs and data dictionaries |
| Copernicus DEM GLO-30 | — | — | To verify in Phase A from the product handbook |
| GEBCO_2026 | GEBCO states that source data are assimilated "assuming all of them to be referred to mean sea level", and that some shallow-water sources use other datums. Nothing specific to inland seas is stated. File attribute `long_name`: "Elevation relative to sea level" | gebco.net gridded-bathymetry page and CEDA file metadata (read 2026-10-20); `results/provenance/gebco_2026.json` | Reference **unresolved for the Caspian**: estimated empirically (PREREG §5, total vs shape error) |
| ETOPO 2022 v1 (15") | File attributes: `vert_crs_name = EGM2008`, `vert_crs_epsg = EPSG:3855`, for the surface elevation, source-ID and geoid-height layers | NCEI THREDDS file attributes (read 2026-10-20); `results/provenance/etopo_2022.json` | Documented as EGM2008. Inside the Caspian, the base layer is GEBCO 2022 (MacFerrin et al. 2025), so the effective reference of the bathymetry is checked empirically |
| SRTM15+ V2.7 | Not stated in `README_V2.7.txt` or the file attributes | Scripps distribution (read 2026-10-20); `results/provenance/srtm15p_v2.7.json` | Unresolved: check Tozer et al. 2019 (doi:10.1029/2019ea000658) and estimate empirically |
| SWOT L2 HR | — | — | To verify in Phase A from the product description document |

## DAHITI

- **Source:** Database for Hydrological Time Series of Inland Waters (DAHITI), DGFI-TUM, https://dahiti.dgfi.tum.de; method in Schwatke, Dettmering, Bosch & Seitz (2015), *Hydrology and Earth System Sciences* 19, 4345–4364, doi:10.5194/hess-19-4345-2015. Product page "Water level from altimetry" read on 2026-10-20.
- **Units:** metres (ASCII header: "column 2: water level from altimetry [m]", "column 3: error [m]"; NetCDF attribute `unit = "m"`).
- **Vertical reference:** normal heights. Schwatke et al. (2015, §2): "We compute normal heights by subtracting a (quasi-)geoid model (N) from the ellipsoidal heights. For this purpose, the EIGEN-6c3stat (Förste et al., 2012) model is used, which supplements the EGM2008 geoid model with additional GOCE gravity data." The current product page (data version v6.0) lists the geoid model as EIGEN-6C4 (Förste et al., 2014). Because the geoid model is subtracted from the ellipsoidal height on the same ellipsoid, the resulting normal heights do not depend on which ellipsoid the altimetry was processed on.
- **Relation to BHS-77:** DAHITI heights are physical heights, but they are not tied to the Kronstadt zero. The offset δ_DAHITI→BHS is unknown a priori and is estimated in Phase A from overlapping published BHS levels and the co-located ICESat-2/waterline comparison, with its uncertainty.
- **Caveat stated by the producers:** for large lakes a uniform surface is assumed and wind-driven level differences are neglected (Schwatke et al. 2015, §4.3.4). On the North Caspian this matters (wind setup); see brief §8 A5.
- **File in use:** *none yet: the PI is to place the DAHITI Caspian Sea series in `data/raw/dahiti/`.* When it arrives, this line records the file name, DAHITI target ID, software version and download date from its header, and the parsed time span (from `results/qc/dahiti_summary.json`).

## GEBCO (2026 and 2022 grids)

- **Source:** GEBCO_2026 Grid documentation (CEDA, `GEBCO_Grid_docmentation.pdf`, read 2026-10-20), §4 "GEBCO Grid, Vertical and horizontal datum":
  - "The GEBCO grid can be assumed to be relative to WGS84."
  - "GEBCO's global elevation models are generated by the assimilation of heterogeneous data types, assuming all of them to be referred to Mean Sea Level. However, in some shallow water areas, the grid includes data from sources having a vertical datum other than mean sea level."
- **Units:** metres; pixel-centre registered, on a 15 arc-second grid.
- **Type Identifier (§3, Table 2):** "70 Pre-generated grid - depth value is taken from a pre-generated grid that is based on mixed source data types, e.g. single beam, multibeam, interpolation etc."
  - In the domain, the North Caspian sea area is coded 70.
  - Sub-sea-level land is coded 40, which suggests a labelling convention for land below 0 m (see `outputs/qc/gebco_2026_tid.png`).
- **Implication:** the vertical reference of the Caspian pre-generated grid (chart datum, the Baltic Height System, or a level-referenced depth) is not documented. It is estimated empirically, and hindcast errors are reported both as total error and as shape error (PREREG §5).
