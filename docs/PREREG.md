# Pre-registration — North Caspian seabed emergence

**Version 1.0, 2026-10-20.** Registered by commit to this repository before any hindcast, DEM-accuracy or projection metric is computed. At the time of registration:

- the only Earth Engine computations run are a scene-metadata inventory and an EECU cost pilot whose grouping is by calendar month, not by water level;
- no water level has been attached to any scene, because the DAHITI series is not yet available;
- no bathymetry product has been downloaded.

Amendments go at the end of this file with a date and reason, and are logged in `docs/DECISIONS.md`. Every result is reported whichever way it comes out (brief §5).

Numeric thresholds appear only where they can be justified independently of the results. Elsewhere the decision rule compares a quantity with its own measured noise floor.

---

## 1. Questions and hypotheses

**H1 (audit).** Global bathymetry products mispredict the extent of North Caspian seabed that emerged between the mid-1990s and 2026, and the error is largest in deltaic and vegetated sectors.

Prior evidence that cuts against H1 is recorded here, before any analysis: Bocharov et al. (2026, doi:10.21046/2070-7401-2026-23-3-291-303) found that whole-sea areas from the chart-based DEM Caspy-30 differ from satellite areas by at most 5.75%. H1 is therefore tested **spatially and per sector**. The whole-domain area bias is reported alongside.

**H2 (topography).** A surge-screened, level-binned waterline DEM of the emerged band has lower vertical error than the bathymetry products, judged against independent ICESat-2 and Copernicus DEM GLO-30 heights.

**H3 (consequence).** Correcting the products for their hindcast error changes near-term emergence (0.5–2 m below the 2026 level) and exposure by a decision-relevant amount (§9). Results are reported with calibrated intervals.

## 2. Data

- **Study domain and sectors:** frozen in D-014 (`results/study_area/sectors.geojson`). The sectors are: west (Kizlyar–Dagestan–Kalmyk), Volga delta, Volga–Ural interfluve, Ural delta, northeast (Tengiz–Kashagan), Komsomolets–Kaidak, and open shelf.
- **Deltaic** sectors: Volga delta, Ural delta. **Vegetated** sectors: those where emergent vegetation covers at least 10% of the emerged band. The vegetation fraction comes from the A3 vegetation class and is computed before any hindcast metric. The 10% bar is a descriptive cut used to label sectors; the H1 test below does not depend on it.
- **Water levels:** DAHITI (primary), cross-checked against G-REALM and Hydroweb where available.
  - Altimetry starts in 1992.
  - Landsat scenes from 1984 to August 1992 need gauge levels (monthly or better, Baltic Height System).
  - **Fallback if no pre-1992 gauge series is obtained:** the rising limb is restricted to 1992–1995, and the two-limb band is reported as the levels actually crossed twice in the altimetry era.
- **Bathymetry products audited:** GEBCO_2026 (doi:10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa), ETOPO 2022 and SRTM15+ (latest), plus any regional chart-based DEM that becomes accessible.
  - The products share soundings. ETOPO 2022 uses GEBCO 2022 as its base layer for seas and large lakes, so the products are reported as **partly dependent**, never as independent replications.
- **Independent heights:**
  - ICESat-2 ATL06 and ATL08 photons over seabed that was dry at acquisition (land-surface flags; acquisition date later than the pixel's emergence date from A6 plus 30 days).
  - GLO-30 over pixels dry throughout its 2011–2015 acquisition window.
  - ATL24 bathymetry where available.

## 3. Water classification and screening (A3–A5)

- **Per-scene classification:** MNDWI with an edge-guided Otsu threshold (`caspian.water.ee_classify`). An emergent-vegetation class (NDVI > 0.30 while not water) counts as not observed for water frequency.
- **Ice:** observations flagged as snow or ice by QA, or with an NDSI test, are excluded. So are months that the empirical ice-season analysis (A2) identifies as ice-affected.
- **Surge screening:** an empirical setup model is fitted to ERA5 wind stress (lags 0–48 h) using **training blocks only**.
  - The screening threshold is chosen by the sensitivity curve of within-bin contour scatter in training blocks, as the smallest threshold beyond which further screening does not reduce scatter by more than its bootstrap standard error.
  - The choice never uses ICESat-2 or GLO-30 test data.
  - The number of scenes removed by each threshold is reported.

## 4. Waterline DEM (A6)

- **Level bins:** 0.10 m wide (sensitivity 0.05 and 0.20 m), origin −30.00 m on the analysis datum.
- **Epochs:**
  - E1: rising limb, from the start of the level record to the date of its maximum (expected about 1995);
  - E2: from that maximum to 2010-12-31;
  - E3: 2011-01-01 to the last scene.
- **Contours:** the 50% water-frequency contour per bin and epoch, extracted at sub-pixel precision (marching squares). A bin is used only where the pixel has at least 5 valid observations.
- **Contour attributes:** each contour gets the bin centre level. Its 1σ combines:
  - the level-interpolation σ (Brownian bridge, `caspian.levels`);
  - the bin half-width as a uniform error (width/√12);
  - the residual setup uncertainty.
- **Gridding:** TIN interpolation to 30 m in the equal-area CRS (D-002), with the interpolation distance stored per pixel. GP/kriging is the comparison method (§10, check 7). The DEM is built from E2 and E3 together; E1 is used for the two-limb consistency test.

## 5. Datum chain

Every product and reference height is converted to the analysis datum (BHS-77) through `docs/DATUMS.md`. Offsets that the documentation does not fix are estimated as follows.

- **DAHITI → BHS:** estimated empirically, with a bootstrap σ, from overlapping published BHS levels.
- **ICESat-2 → waterline DEM:** a constant, estimated from co-located heights in **training blocks only**, with a bootstrap σ.
- **Products:** each product's vertical reference in the Caspian is estimated empirically. Hindcast errors are reported twice:
  - **(a) total error**, with the product on its documented or estimated reference;
  - **(b) shape error**, after removing the single constant offset that minimises the product's area error across all levels.

  This separates a datum offset from errors in the shape of the seabed.

## 6. Spatial blocking (validation and calibration)

- **Blocks:** square, aligned to the equal-area grid, initial size 25 km. Each block takes the sector that contains its centroid.
- **Assignment:** blocks are randomly assigned to training, calibration and test in proportions 0.6 / 0.2 / 0.2, stratified by sector, with seed 20261020.
- **Block size check, before any test-block evaluation:** fit the empirical semivariogram of the DEM-minus-ICESat-2 residuals in training blocks.
  - If the practical range exceeds 25 km, enlarge the block to the next multiple of 25 km at or above the range and reassign with the same seed.
  - The final size is logged.
- **Block-size sensitivity:** sizes of 12.5, 25 and 50 km are reported in the robustness checks.

## 7. Accuracy metrics (H2)

- **Metrics** for the waterline DEM and each product: bias, RMSE, NMAD and the 5th/95th percentiles of the error. Each is computed within blocks and summarised over test blocks, overall and per sector.
- **Decision rule for H2:** H2 is supported if the waterline DEM's test-block NMAD and RMSE are both lower than every product's. The comparison is paired and made on the same ICESat-2 points, with a block-bootstrap 95% interval (2,000 resamples of test blocks) on the difference that excludes zero.
  - Partial support (some sectors, or some products) is reported as such.
  - **If H2 fails, the brief §16 pivot applies.**

## 8. Conformal intervals for DEM heights

- **Method:** split conformal with locally normalised scores |y − ŷ| / s(x), where s(x) = exp(β·[1, log(1 + interpolation distance), contour density, local slope]).
  - β is fitted by least squares on log |residual| in **training blocks**.
  - q̂ is computed on **calibration blocks**, at nominal level 0.90.
- **Mondrian:** q̂ is computed per sector when that sector has at least 50 calibration points. Otherwise the sector uses the pooled q̂, and this is reported.
- **Evaluation:** empirical coverage on **test blocks** with its Clopper–Pearson 95% interval, overall and per sector.
  - Coverage counts as consistent with 90% if 0.90 lies inside that interval.
  - Any inconsistency is flagged in the G3 memo and in the paper.
  - Because points within a block are correlated, a block-bootstrap interval is also reported; the Clopper–Pearson interval is then optimistic.
- **Assumption, stated in the paper:** calibration and test points are exchangeable at the block level, and spatial blocking is how that is made plausible.

## 9. Hindcast audit (H1)

- **Predicted extent:** for each epoch and level bin with an observed extent (pixels with water frequency ≥ 0.5 within the valid mask), each product's predicted extent is the flood fill from the open-sea seed, using cells below the level (`caspian.hindcast`).
- **Metrics** per product × sector × level: area bias (absolute and relative), IoU, F1, and shoreline displacement (median, 95th percentile, signed median).
- **Observational noise floor:** displacement between the two independent half-samples of scenes in the same bin and epoch (odd and even scene index), and between E1 and E2/E3 at levels crossed twice.
- **Decision rule for H1:**
  - **Mispredicts** in a sector: the product's median absolute shoreline displacement exceeds the 95th percentile of the noise-floor displacement in that sector, at the majority of the sector's observed levels.
  - **Largest in deltaic and vegetated sectors:** a block-level permutation test (10,000 permutations) of the difference in median displacement between deltaic or vegetated sectors and the rest, one-sided at α = 0.05.
- **Alternative explanation tested before attributing error to bathymetry (§10, check 10):** seabed change between epochs, measured as the E1-versus-E2/E3 contour difference at the same level.

## 10. Near-term emergence and exposure (H3)

- **Fused surface:**
  - the waterline DEM inside the observed range;
  - below it, each product corrected by the error model learned in §9. The model is a per-sector function of depth below the lowest observed level, fitted on E2 and E3.

  The correction is marked as extrapolation. Its transfer is tested by leave-one-epoch-out: fit on two epochs and predict the third epoch's extents.
- **Scenarios:** 0.5, 1, 2, 3 and 5 m below the 2026 level, mapped to the dates implied by published projection pathways, which are verified before they are used.
- **Area intervals:** conformalised directly. The leave-one-epoch-out hindcast area errors at held-out observed levels are the calibration scores. Exposure intervals follow from the area intervals. Per-pixel intervals are never summed as if independent.
- **Decision rule for H3:** the correction is decision-relevant for a scenario if the corrected emerged area lies outside the 90% interval of the uncorrected estimate, or the reverse. Exposure changes are reported as counts of assets, and areas of habitat and protected land, that change status, with intervals.

## 11. Robustness checks (brief §10)

Every robustness check listed in brief §10 is run as specified there. The headline results are recomputed under each alternative, and the change is reported in Extended Data.

## 12. Deviations

- Any change to these rules after registration is an amendment (date, reason, and whether results had been seen).
- Choices the plan leaves open are made on training data only and logged in `docs/DECISIONS.md`.
