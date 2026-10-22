# Citation requests from workstream G (Great Salt Lake, H2, ETOPO 2022)

For workstream C, who resolves each request in `refs/` (Crossref or DataCite where a DOI
exists, BibTeX by content negotiation, `verification.csv`, `claims.csv`). Every document
below was retrieved and read by workstream G on 2026-10-22. The local copies are in
`data/raw/revision_g/` (git-ignored); each copy has a `.meta.json` file with its URL,
retrieval time and SHA-256. The quotes were checked against those copies. Locations are
1-based PDF page positions; the printed page number is in brackets.

Tier notes follow CLAUDE.md: official data or product documentation from USGS, NOAA and the
data providers is Tier A. Nature Communications is Tier A.

## A. The 3DEP surface used as the Great Salt Lake control (citation audit major; R1.M5; the SI `\todo{check}`)

The current citation, `usgs_3dep10m` (the Earth Engine catalogue page), supports neither
"lidar" nor "2016". The sources below establish that the 3DEP 1/3 arc-second seamless DEM is
built from several sources. Over the lake-bed band it is mostly, but not only, the 2016 Great
Salt Lake lidar. The provenance table is in `results/revision/g/dep3_provenance.json`.

1. **Quantum Spatial, Inc. (2017). *Utah 2016 – Great Salt Lake AOIs LiDAR Project Report*. Submitted 14 August 2017 to the USGS (contract AV2408).**
   - **URL:** https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/UT_CentralUT_Southern_ID_2016_A16/UT_GreatSaltLakeQL1_2016/reports/29083_Utah2016_GreatSaltLake_ProjectReport.pdf
   - **No DOI.** Local copy: `3dep/workunits/UT_GreatSaltLakeQL1_2016/project_report.pdf`.
   - **Supports:** the dates of the 2016 lidar, and that its DEMs are hydro-flattened and on NAVD 88.
     - PDF p. 5 [2]: "LiDAR data was acquired on September 3, 2016 through November 30, 2016 in 48 total lifts."
     - PDF p. 26 [23]: "Class 2 (Ground) lidar points in conjunction with the hydro breaklines were used to create hydro-flattened raster DEMs with a 0.5-meter cell size for the QL1 AOIs and a 1-meter cell size for the QL2 AOI."
     - PDF p. 5 [2]: "All geospatial deliverables were produced in NAD83 UTM Zone 12, meters; NAVD88 (GEOID12B), meters."
   - **Claims:** Methods (Generality), SI generality "Positive control", the Great Salt Lake figure caption.

2. **Quantum Spatial, Inc. (2017). *Utah 2016 LiDAR – Great Salt Lake AOIs*, project-level FGDC metadata (pubdate 2017-10-24), distributed by the USGS.**
   - **URL:** https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/metadata/UT_CentralUT_Southern_ID_2016_A16/UT_GreatSaltLakeQL1_2016/best_use_xml/29083_Utah2016_GreatSaltLakeAOIs_ProjectLevel_Metadata.xml
   - **No DOI.** Local copy: `3dep/workunits/UT_GreatSaltLakeQL1_2016/project_metadata.xml`.
   - **Supports:** hydro-flattening and fall-2016 collection.
     - Purpose: "The purpose of these lidar data was to produce high accuracy 3D hydro-flattened digital elevation models (DEMs) with 0.5 and 1 meter cell sizes."
     - Lineage, process step 3: "Elevation values were assigned to all Inland Ponds and Lakes, Inland Pond and Lake Islands, Inland Stream and River Islands, using TerraModeler functionality."
     - Abstract: "Ground Conditions: Lidar was partially collected in fall 2016, while no snow was on the ground and rivers were at or below normal levels."
   - **Claims:** as item 1. One of items 1 and 2 is enough if C prefers a single source.

3. **U.S. Geological Survey, 3D Elevation Program: Work Unit Extent Spatial Metadata (WESM), 3DEP Elevation Index map service, work units UT_GreatSaltLakeQL1_2016 and UT_GreatSaltLakeQL2_2016.**
   - **URL:** https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer (layers 24 and 27). Responses are in `3dep/`, and the query log is in `dep3_provenance.json`.
   - **No DOI.**
   - **Supports:** quality levels QL1 and QL2; collection 2016-09-03 to 2016-11-18 as WESM records it. The project report gives 30 November 2016 as the last lift, so cite the report for the dates. Lidar Base Specification 1.2.
   - **Claims:** "QL1/QL2" in the SI. If C wants to limit the number of documentation references, drop this item and keep items 1 and 4.

4. **U.S. Geological Survey (2024–2026). 1/3 arc-second DEM tiles n41w112, n41w113, n41w114, n42w112, n42w113 and n42w114, with their FGDC metadata and source-footprint GeoPackages.**
   - **URL pattern:** https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current/n41w113/USGS_13_n41w113.xml (and `.gpkg`).
   - **No DOI.** Local copies: `3dep/tiles/`.
   - **Supports:** that the seamless DEM is a mosaic of sources, and which source feeds each part of the band.
     - Abstract of the n41w113 XML: "Seamless 3DEP data are derived from diverse source data that are processed to a common coordinate system and unit of vertical measure."
     - The GeoPackages list, per footprint, the source project, method and dates. From them: the 2016 lidar is 91–92% of the north arm's band and 64% of the south arm's. The rest is lidar of 2013–2023 and, in the south arm, legacy non-lidar DEMs of 1966–1973: 10% in the current tiles, 17% in the tiles as of 2022-05-04.
   - **Claims:** the SI generality "Positive control" (new wording), and the response to R1.M5.

5. **Archuleta, C.M., Constance, E.W., Arundel, S.T., Lowe, A.J., Mantey, K.S. and Phillips, L.A. (2017). *The National Map seamless digital elevation model specifications*. USGS Techniques and Methods 11-B9.**
   - **DOI:** 10.3133/tm11B9. The authors, title and year match the Crossref record (checked by G on 2026-10-22).
   - **Local copy:** `3dep/docs/tm11B9.pdf`.
   - **Supports:** that the seamless DEMs are hydro-flattened.
     - PDF p. 9 [1]: "The seamless DEMs distributed by the USGS are hydroflattened, topographic raster DEMs that represent the bare ground (bare earth) surface excluding trees, buildings, and any other surface objects."
   - **Claims:** "evaluated only above its flattened lake surfaces" (Methods; SI).
   - **Tier:** A (USGS official product documentation).

6. **Optional context, not a citation for the Letter: the Earth Engine STAC record of `USGS/3DEP/10m_collection`** (`3dep/docs/ee_stac_USGS_3DEP_10m_collection.json`).
   - It gives `gee:user_uploaded: true`, "Dataset uploaded by Farmers Business Network", and a temporal interval ending 2022-05-04.
   - So which tile versions Earth Engine served is not documented. The SI should say that the band shares are given for the current tiles and for the tiles as of 2022-05-04.

## B. ETOPO 2022's lake processing (R1.M7; R3.M6)

7. **NOAA National Centers for Environmental Information (2022). *ETOPO 2022 User Guide*, Rev. 1.2, 13 October 2022.**
   - **URL:** https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/docs/1.2%20ETOPO%202022%20User%20Guide.pdf. The ISO record of doi:10.25921/fd45-gt74 links it.
   - **Local copy:** `etopo/ETOPO_2022_User_Guide_v1.2.pdf`.
   - **Supports:**
     - the vertical reference, PDF p. 2: "referenced in meters relative to the Earth Gravitational Model of 2008 (EGM2008) geoid surface (EPSG:3855)", and "For most purposes, EGM2008 is an adequate approximation of mean sea level.";
     - the lake layer, PDF p. 6 (source table): "GEBCO Lake Depths | MSL | 12 | GEBCO Hydrolakes outlines and GEBCO elevations | GEBCO 2022 | Global surveyed lake depths (very large lakes)".
   - **Warning:**
     - The guide numbers GEBCO Lake Depths **12** and BedMachine **11**. MacFerrin et al. (2025, Table 1; already in `refs/`, key `macferrin2025`) give **11** for GEBCO Lake Depths.
     - The North Caspian source-ID subset holds codes 1, 9, 10 and 11 only, consistent with the ESSD numbering.
     - **Cite `macferrin2025` for the meaning of source ID 11**, and the user guide only for the datum and the lake layer's description.
   - **Not in either document:** a lake-surface elevation, its epoch, or any capping of lake depths. See item 8 and the D-130 addendum.

8. **Lehner, B. and Messager, M.L. (2016). *HydroLAKES – Technical Documentation*, Version 1.0.**
   - **URL:** https://data.hydrosheds.org/file/technical-documentation/HydroLAKES_TechDoc_v10.pdf
   - **No DOI.** Local copy: `etopo/HydroLAKES_TechDoc_v10.pdf`. The data record read is `HydroLAKES_points_v10_shp.zip`.
   - **Supports:** that the lake outline ETOPO 2022 uses for the Caspian is the SRTM Water Body Data shoreline of February 2000, and its lake-surface elevation.
     - PDF p. 2: compiled from "foremost the SRTM Water Body Data (SWBD; Slater et al., 2006) for regions from 56˚S to 60˚N".
     - PDF p. 5: "All lake shorelines were delineated as they appeared at the time of the data collection in February 2000."
     - Caspian Sea record (Hylak_id 1): `Poly_src` = "SWBD", `Elevation` = −29. `Elevation` is defined on PDF p. 14 as the lake-surface elevation in metres above sea level, the majority pixel elevation of EarthEnv-DEM90.
   - **Claims:** the description of ETOPO 2022's lake processing (SI; response to R1.M7).
   - **Tier:** A (provider documentation).

9. **Messager, M.L., Lehner, B., Grill, G., Nedeva, I. and Schmitt, O. (2016). Estimating the volume and age of water stored in global lakes using a geo-statistical approach. *Nature Communications* 7, 13603.**
   - **DOI:** 10.1038/ncomms13603. The title matches Crossref (`etopo/crossref_10.1038_ncomms13603.json`).
   - **Read:** the open-access full text from Europe PMC (PMC5171767), local copy `etopo/Messager2016_ncomms13603_EuropePMC_PMC5171767.xml`.
   - **Supports:** HydroLAKES as a dataset, and its SWBD source between 56° S and 60° N (the same sentence as the technical documentation).
   - **Tier:** A.
   - **Use:** cite it beside item 8 where the Letter names HydroLAKES.

## C. Great Salt Lake grid sources (R1.M5; R2.M11; R1 minor 22)

These are already in `refs/`. Only claim rows need adding.

10. **`gebco2026` (GEBCO grid documentation), Table 2:** TID 0 = "Land" (PDF p. 17). It supports "every band cell of both arms is coded Land in GEBCO_2026 and GEBCO_2022". The GEBCO_2022 documentation (`gebco2022`) has the same table; C should confirm its page.

11. **`tozer2019`, §5.1:** "Each grid cell that is constrained by a sounding contains a unique integer (SID)". It supports "SRTM15+ holds no sounding in either arm's band (source ID 0 throughout)".
    - Tozer et al. 2019 §2.3.2 also says that SRTM CGIAR-CSI V4.1 is the onshore source between 60° N and 56° S. That supports "the grids' Great Salt Lake heights are SRTM land heights".

12. **`macferrin2025`, Table 3:** Great Salt Lake is not among the lakes whose GEBCO bathymetry ETOPO 2022 includes. The paper uses "a 'flat surface' for remaining lakes worldwide" (PDF p. 5). This supports "ETOPO 2022 holds no lake depths at Great Salt Lake; its band is 100% FABDEM".

13. **`copdem` (Copernicus DEM product handbook, issue 5), PDF p. 8:** "This DEM is derived from an edited DSM named WorldDEMTM, i.e. flattening of water bodies and consistent flow of rivers has been included." It supports "FABDEM, built from the Copernicus DEM, carries a flattened water surface where the lake stood in 2011–2015".

14. **SRTM acquisition dates (11–22 February 2000).** Needed only if the text gives these exact dates.
    - We did not read a source for them. The HydroLAKES documentation (item 8) supports "February 2000".
    - Candidate: Farr, T.G. et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics* 45, RG2004, doi:10.1029/2005RG000183. C must retrieve and read it before it is cited; the reference is not typed into `refs/` by G.
    - Our gauge comparison gives the same answer for the whole of February 2000 (D-128 addendum). The text can therefore say "February 2000" with item 8.

## D. Context for workstream V (not a G claim)

15. **NOAA NCEI, *ETOPO1 Data Sources*** (https://www.ncei.noaa.gov/sites/default/files/2023-01/Data%20Sources.pdf; local copy `etopo/NCEI_ETOPO_Data_Sources.pdf`).
    - Page 2 says the Caspian grid "was shifted 28-m vertically to be consistent with mean sea level before use in building ETOPO1".
    - This bears on ETOPO1's vertical reference in the Caspian (`docs/DATUMS.md`, which records it as "not recorded"). Passed to V and C; G does not use it.
