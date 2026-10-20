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

---

## Amendment 1 — 2026-10-20 (before any hindcast, DEM-accuracy or projection metric was computed)

**Change.** Two products are added to the audit (§2), because they are the grids that published Caspian emergence and exposure estimates actually used:
- **GEBCO_2022**, used by Court et al. (2025, doi:10.1038/s43247-025-02212-5) and shown by Samant & Prange (2023);
- **ETOPO1**, used by Akbari et al. (2020, doi:10.1088/1748-9326/abaad8).

GEBCO_2026, ETOPO 2022 and SRTM15+ V2.7 remain the current-product audit.

**Provenance finding recorded before analysis.** Products were compared only with each other; nothing observed was used.
- GEBCO_2026 and SRTM15+ V2.7 are identical in 60% of domain cells (mean difference −0.35 m).
- ETOPO 2022 differs from GEBCO_2026 by a median of −0.36 m (5th–95th percentile −3.9 to +1.0 m).
- In the GEBCO_2026 Type Identifier grid, the North Caspian sea area is coded **70 (pre-generated grid)**. Code 40 ("predicted, satellite gravity") occurs on the sub-sea-level land of the Caspian Depression, not on the seabed, which suggests a labelling convention for land below 0 m.

The products are therefore treated as one partly shared lineage (as §2 already required). Results are presented per product, with no claim of independent replication.

**Reason.** Brief §8 A7 asks for an audit of "the bathymetry products behind emergence projections". Testing the exact versions those projections used is the direct test. Adding products cannot be tuned toward a result, because no hindcast metric exists yet.

**Correction (2026-10-20, D-041).** Amendment 1 justified adding ETOPO1 as one of "the grids that published Caspian emergence and exposure estimates actually used". That overstates its role in Akbari et al. (2020).
- They used ETOPO1 only as a depth map (their §2.2).
- Their area, vulnerability and desiccation estimates come from Landsat NDWI water masks regressed on observed levels (§2.6, §3.1–3.2).
- Their only use of ETOPO1 in the results is a qualitative check that the band flooded in 1977–1995 matched depths of less than 3 m (§3.1).

ETOPO1 stays in the audit as the depth map of a published Caspian vulnerability study. Nothing in the analysis changes, because no hindcast metric existed when ETOPO1 was added (review finding 12).

**Correction (2026-10-20, integration).** The label "predicted, satellite gravity" for Type Identifier code 40 follows the GEBCO_2026 grid documentation. Its TID table defines code 40 as "Predicted based on satellite-derived gravity data - depth value is an interpolated value guided by satellite-derived gravity data" (found by script in `data/raw/lit/10_5285_4f68d5c7_45eb_f999_e063_7086abc036fa.pdf`; to be added to `refs/notes/gebco2026.md` under its checked quotes). The grid file's own `flag_meanings` name the same code `predicted_altimetry` (`results/provenance/gebco_2026.json`), and the regenerated QC figure uses that name (D-045). Both names refer to one code. The finding (code 40 on sub-sea-level land, code 70 on the North Caspian seabed) and the analysis are unchanged.

---

## Amendment 2 — 2026-10-20 (before any hindcast, DEM-accuracy or projection metric was computed; no waterline, contour or DEM exists yet)

**Change.** GEDI L2A footprint ground elevations are added as a third set of independent heights (§2), and as the pre-specified substitute for ICESat-2 if ICESat-2 cannot be retrieved.

**Data.** GEDI L2A Elevation and Height Metrics, Version 2 (Dubayah et al. 2021, doi:10.5067/GEDI/GEDI02_A.002). Earth Engine collection `LARSE/GEDI/GEDI02_A_002_MONTHLY` (25 m rasterised footprints).
- 189 monthly images intersect the domain, from 2019-04 to 2025-07.
- No acquisitions from 2023-03-17 to 2024-04-24 (instrument in storage).
- Ground height: `elev_lowestmode`, "relative to reference ellipsoid"; the datum is WGS84 (GEDI02 User Guide v2.1, Table 1).

**Shot selection, fixed now.** Every rule follows the product documentation or mirrors the ICESat-2 rules in §2.
- `quality_flag` = 1. Per the User Guide, this already includes the energy, amplitude, surface-tracking and DEM-difference criteria and a 0.9 land sensitivity threshold.
- `degrade_flag` = 0.
- BEAM0000 and BEAM0001 shots before 2020-02-06 are excluded (known ranging issue, User Guide §8).
- Gross-error screen, as the User Guide recommends: |`elev_lowestmode` − `digital_elevation_model`| ≤ 50 m. The 50 m bound removes only cloud and noise returns; on this terrain, real ground lies within a few metres of the reference.
- **Dry at acquisition:** acquisition date later than the emergence date (A6) plus 30 days, for every 30 m DEM pixel within one pixel of the shot. The neighbourhood covers the 25 m footprint and its geolocation error.
- **Sensitivity check, not a filter:** night shots only (`solar_elevation` < 0), reported in the robustness checks.

**Role.**
1. **In every case** GEDI is reported as an additional independent check, with the same blocks, metrics and sectors as GLO-30 (§7). It is never pooled with ICESat-2.
2. **Substitution rule.** Suppose that when the waterline DEM is frozen (A6 complete, before any test-block evaluation) ICESat-2 ATL06/ATL08 granules for the domain still cannot be retrieved through NASA Earthdata. GEDI then takes ICESat-2's place in four places:
   - §5: the constant from co-located heights in training blocks;
   - §6: the semivariogram block-size check;
   - §7: the H2 decision rule, paired on the same GEDI shots;
   - §8: conformal coverage on test blocks.

   The switch and its date are logged in `docs/DECISIONS.md`, and the paper states that ICESat-2 was unavailable.
3. **If ICESat-2 becomes retrievable** after GEDI-based test results have been computed, the ICESat-2 results are reported as well, labelled as a check made after the decision. The GEDI-based decision stands.

**Reason.** Since 2026-10-20, NASA's Common Metadata Repository has returned HTTP 500 for every ICESat-2 granule search. Every ICESat-2 access route depends on it (earthaccess, NSIDC, SlideRule, OpenAltimetry). The AWS copy of the archive can be read only from inside AWS us-west-2.
- Fixing the substitute now, before any DEM exists, keeps the validation source from being chosen after the results are seen.
- GEDI is lidar, independent of the optical imagery and of the bathymetry products, and covers 2019–2025, when most of the emergence occurred.
- Its known weaknesses are a larger footprint and lower ground precision than ICESat-2. They are reported, not hidden.

---

## Amendment 3 — 2026-10-20 (before any stage-1 production run, and before any hindcast, DEM-accuracy or projection metric was computed)

**Changes to §3.**
1. **Otsu sample.**
   - The production per-scene threshold is Otsu's threshold of the MNDWI histogram of clear pixels inside the active zone (D-027), the band within which the altimetry-era shoreline moves. It replaces "edge-guided Otsu" (Canny edges on MNDWI).
   - Histogram: 256 equal bins on [−1, 1], the binning of the tested NumPy reference, sampled at 240 m.
   - Scale check, made before this amendment on 20 scenes from all five sensors (April–October; `results/qc/stage1_scale_check.json`). For the 19 with a threshold at both scales, the 240 m and 60 m thresholds were identical for 8 scenes, one bin apart (0.0078) for 8, and two bins apart (0.0156) for 3. 240 m costs about 1/16 as much (D-028).
   - **Robustness check:** the edge-guided threshold is computed for a random sample of 500 scenes, stratified by sensor (seed 20261020). The headline DEM accuracy and hindcast results are recomputed with water frequencies in which those scenes use the edge-guided threshold, and the threshold differences are reported.
2. **Snow and ice test, specified.** "Flagged as snow or ice by QA, or with an NDSI test" is implemented as follows:
   - **QA:** Landsat QA_PIXEL bit 5; Sentinel-2 Scene Classification class 11.
   - **NDSI test:** the conservative pass-1 snow test of the Theia LIS algorithm, NDSI > 0.400 and red reflectance > 0.200 (Gascoin et al. 2019, ESSD 11, 493–514, Table 1).

   The NDSI test alone is recorded per scene. If it flags more than 1% of clear in-zone pixels in any May–September month (when snow and ice are physically implausible), the test is restricted to November–April. This rule is fixed now, and is applied from stage-1 data before any stage-2 run.
3. **Counting unit.** Water frequencies count each acquisition (overpass) at most once per pixel; overlapping scenes of one acquisition are mosaicked. The odd/even half-sample split for the noise floor (§9) is defined on acquisitions.

**Reason.**
- The edge-guided sample needs a Canny edge pass for every one of about 75,600 scenes. Measured in pilot 1, this was the dominant cost: about 550 EECU-s per scene with the threshold computed inside the image graph. That is beyond the Community-tier quota (D-009, D-015).
- Items 2 and 3 fix defects found in the 2026-10-20 technical review: Sentinel-2 had no snow mask, and overlapping scenes were counted twice. No water frequency, waterline or metric existed when these changes were made.

**Correction (2026-10-20, integration).** Item 1's "active zone (D-027)" referred to zone v5. Stage 1 (runs 1 and 2) and stage 2 use zone v6 (D-029), the sea-connected band with the dark-water rule, so the histogram sample is the clear pixels inside zone v6, clipped to its 5 km tile cover. The rule is otherwise unchanged.

**What had been seen when Amendment 3 was made (disclosure added 2026-10-20).** The amendment was not committed before stage-1 run 1 started. The file was last written at 20:15:23 local time, and run 1's first batch is logged at 20:16:47 (`results/inventory/stage1_interactive_log.jsonl`). Its heading "before any stage-1 production run" therefore rests on file times, not on a commit. When it was written, the following had been seen:
- the stage-1 pilot and scale-check thresholds (20 scenes; `results/qc/stage1_scale_check.json`, 19:11);
- the D-028 profiling;
- the stage-2 cost micro-pilot on one 0.3° box (`results/inventory/stage2_micropilot.json`, 19:17), and a calendar-month (not level-binned) water-fraction map from it (19:41);
- the zone v5 and v6 composites at 1 km (D-029).

No level-binned water frequency, waterline, contour, DEM or metric existed. Stage-1 run 1's monthly snow shares (`results/qc/stage1_run1_monthly.csv`) were computed after Amendment 3 and are used in Amendment 9.

---

**What had been seen when Amendments 4–8 were made (2026-10-20).** No hindcast, DEM-accuracy or projection metric had been computed. No level-binned water frequency, observed extent, waterline, contour, DEM, conformal score or coverage existed, and no product had been compared with any observed extent. Input-level results had been seen:
- the scene levels and epoch counts (DECISIONS D-021, D-035 to D-037);
- the near-maximum 1995–96 and near-minimum 2025–26 zone composites, inspected at 1 km for zone QC (D-029);
- a calendar-month (not level-binned) water-fraction map of one 0.3° box on the northeast coast, from the stage-2 cost micro-pilot (D-027);
- stage-1 thresholds of the pilot and scale-check scenes (D-028);
- product shorelines at two DAHITI levels, computed from the products alone for the sector layout check (D-040), and the product-to-product comparison of Amendment 1.

---

## Amendment 4 — 2026-10-20 (before any hindcast, DEM-accuracy or projection metric was computed, and before any waterline, contour or DEM existed; made after scene levels had been attached and the epoch counts seen, with the other input-level results listed above)

**Changes to §4.**
1. **E1 end, specified.** "The date of its maximum" is read as the date of the maximum of the 3-record running median of the DAHITI series (`caspian.levels.smoothed_extremes`, k = 3): 1995-07-16, at −25.816 m. E2 starts after it.
   - **Reason:** the single highest record (1995-07-06, −25.672 m) lies 0.137 m above every other record and 2.8σ above its neighbours (D-035, D-037), so it cannot stand for the sea's maximum.
   - **Disclosed limitation:** the top of the running median is flat; its two highest values differ by 0.002 m. The single-record maximum and the 5- and 7-record medians give 1995-07-06 (860 E1 scenes), k = 3 gives 873, and a 9-record median gives 1995-05-28 (812). The E1 end is therefore uncertain by several weeks across the 1995 plateau, and up to 61 Landsat-5 scenes near the high stand depend on it.
   - **Commitment:** the k = 3 end is primary. Every result that uses E1 (the two-limb consistency test, and the §9 comparisons of E1 with E2/E3) is also reported with E1 ending on the single-record maximum (k = 1). The E1-end sensitivity over k = 1, 3, 5, 7 and 9 is reported.
   - If the PI prefers the literal single-record reading, a further amendment withdraws this item before the A6 two-limb test is run.
2. **Level assignment, stated for the record.** §4 refers to `caspian.levels` for the level-interpolation σ but does not state the rules.
   - A scene takes its level from DAHITI by Brownian-bridge interpolation when its bracketing DAHITI records are at most 60 days apart. The 60-day value is the default of `caspian.levels`, committed in aabacee before v1.0 of this plan and before any level data existed; it was not chosen on the data (no change).
   - Otherwise, within the Hydroweb record, the scene takes its level from Hydroweb, shifted by the moving median of the ±20-day DAHITI − Hydroweb pair differences within ±2 years of the scene (D-036).
   - The record noise σ and the bridge rate q are estimated jointly from cross-validated interpolation residuals (D-035).
   - **Contour 1σ.** The level σ of a gap-filled scene has a random part and an offset part (`sigma_random_m`, `sigma_offset_m`). The offset part is common to the gap-filled scenes of a period, and is not averaged down within a bin when the contour 1σ is formed.

**Reason.** Review findings 7, 8, 22, 27 and 28 of the 2026-10-20 technical review.
- Item 1 reads an ambiguous phrase of §4. It was chosen after the scene levels and epoch counts had been seen; it moved E1 from 860 to 873 scenes.
- Item 2 records rules that the plan referred to only through `caspian.levels`. The gap-fill offset and the joint σ/q estimator replaced earlier methods (D-020, D-021) after the levels had been seen. They remove biases shown by simulation and by the pair data, and were not chosen to move any result.

**Signed off by the lead, 2026-10-20** (the consolidating agent's choice in item 1: k = 1 reporting extended to every E1-based result).

---

## Amendment 5 — 2026-10-20 (before any hindcast, DEM-accuracy or projection metric was computed; no DEM, conformal score or coverage existed; input-level results seen as listed above, none of them heights or scores)

**Changes to §8.**
1. A sector's calibration count is its number of **usable** calibration points: reference height, prediction and scale covariate all present. A sector with fewer than 50 usable points, including a sector with none, uses the pooled q̂, computed over all usable calibration points of all sectors. The rule is unchanged; this makes it exact.
2. The record (q̂, n, pooled or not) is reported for every sector, including sectors with no calibration point, in the G3 memo and the paper.
3. Points without a DEM prediction (outside the TIN hull) or without a scale value are excluded from calibration and from coverage alike. The number excluded is reported per sector, together with the fraction of test points that have no prediction. Infinite scores and interval bounds are kept, which is conservative.

**Reason.** Review findings 11 and 31 (D-031). The code computed a sector q̂ whatever the sector's size (infinite below 9 points, which reports 100% coverage) and recorded no fallback. Calibration dropped missing points, while coverage counted them as misses. The §8 rule (at least 50 points, otherwise the pooled q̂, reported) is unchanged; the code now implements it as written.

---

## Amendment 6 — 2026-10-20 (before any hindcast, DEM-accuracy or projection metric was computed; no level-binned observed extent or waterline existed; input-level results seen as listed above)

**Changes to §9.**
1. **Waterline.** A waterline cell is a cell of known water with at least one 4-neighbour of known land. Unknown cells are those outside the valid mask (zone edge, cloud or low-count holes), outside the domain, or beyond the array border. An edge against an unknown cell is not a waterline.
2. **Known states outside the active zone.** Inside the domain, a cell outside the zone is:
   - water, if it is connected, outside the zone, to the open-sea seed;
   - land, if it is connected, outside the zone, to a designated hinterland land seed;
   - unknown otherwise.

   Cells outside the domain, and a one-cell rim inside the domain edge, are unknown. If the open sea and a land seed are connected outside the zone, the analysis stops with an error. These states hold for sea-connected water at levels between the 2025–26 low stand and the 1995–96 high stand. The land seeds are fixed before any hindcast metric is computed. A product's state is known wherever its DEM is finite.
3. **Shoreline displacement** is symmetric. It pools two sets of distances, keeping only cells whose state on the other extent is known:
   - from each predicted waterline cell to the nearest observed waterline cell;
   - from each observed waterline cell to the nearest predicted waterline cell.

   Reported: the median and 95th percentile of the absolute distances, and the signed median (positive: the product predicts too much water). Cells that cannot be measured are counted and reported. Per sector, the waterline cells lying in the sector are measured against the nearest waterline of the other extent anywhere in the domain.
4. **No-waterline rules.** A product with no waterline in the domain has infinite displacement, and that level counts as exceeding the noise floor. With no observed waterline anywhere, or no measurable cell in the sector, the displacement is undefined.
5. **Observed levels of a sector** are the levels at which the observed extent has a waterline in that sector, whatever the product.
   - The majority rule is evaluated over observed levels at which at least one waterline cell could be measured. The others are reported as not evaluable.
   - A product waterline in a sector, at a level that is not observed there, is reported but does not count toward the majority rule.
6. **Noise floor.** The same definitions apply, with half-sample A (or E1) in the place of the product, and with its own valid mask.
7. **Area bias, IoU and F1** include errors in cells outside the valid mask whose state is known under item 2. Correctly predicted cells outside the valid mask are not counted. The values within the mask are also reported.

**Reason.** Review findings 2, 9 and 10 (D-030):
- edges against unobserved cells were counted as shorelines;
- prediction errors beyond the zone edge were censored;
- the measure was directed, so a product that missed a bay scored near zero.

§9 named the displacement metric but did not define it. The definitions are fixed without sight of any hindcast result.

---

## Amendment 7 — 2026-10-20 (clarification of §9; before any hindcast, DEM-accuracy or projection metric, and before any noise-floor number, was computed; input-level results seen as listed above)

**Clarifications of §9.**
1. **Noise floor compared per level** (D-032). In the "Mispredicts" rule, the 95th percentile of the noise-floor displacement is taken per level within the sector.
   - At each evaluable observed level of the sector, the product's median absolute displacement in the sector is compared with the 95th percentile of the absolute noise-floor displacements in the sector at the same level.
   - At a level observed in more than one epoch, the distances of those epochs are pooled, for the product and the noise floor alike.
   - Each half-sample's valid mask applies the §4 rule (at least 5 valid observations) to that half's own counts.
   - A level is evaluable when at least one product distance and at least one noise-floor distance could be measured in the sector. Other observed levels are reported as not evaluable.
   - "Majority" means more than half of the sector's evaluable observed levels.
   - **Sensitivity, reported but not deciding:** a single 95th percentile pooled over all the sector's levels and epochs.
2. **Observed extent connected to the open sea** (D-033). The observed extent at each epoch and level is restricted to sea-connected water, as the predicted extent already is.
   - Observed water is sea if it is 8-connected, through cells of known water, to the same open-sea seed as the products' flood fill. Known water means observed water inside the valid mask, plus known open sea outside the zone (Amendment 6, item 2).
   - Observed water that is not so connected takes the land state if it is bordered only by known land. It is unknown if it touches a cell of unknown state.
   - The same applies to each half-sample and epoch extent in the noise floor.
   - The area of each class is reported per sector and level.

**Reason.**
- **Item 1.** §9 counts levels ("at the majority of the sector's observed levels"), so the threshold is taken at the same level. The number of acquisitions per bin and the seabed slope vary with level, so a pooled threshold would be lenient at some levels and strict at others.
- **Item 2.** The predicted extent is a flood fill from the open sea. Without the restriction, inland water inside the zone (delta channels, ilmens, lakes, flooded fields) would count as product error, chiefly in the deltaic sectors that H1 tests. In a review probe, a river channel gave a perfect product a p95 of 10 px. Item 2 applies to §9 the analysis mask already fixed in D-014: only water hydrologically connected to the open sea counts.

**Signed off by the lead, 2026-10-20** (the consolidating agent's choices: unconnected observed water takes the land state or unknown; the ≥ 5-observation rule applies to each half-sample; no pooled substitute for a level that is not evaluable; "majority" means strictly more than half).

---

## Amendment 8 — 2026-10-20 (change to §4; before any hindcast, DEM-accuracy or projection metric was computed, and before any contour was extracted; input-level results seen as listed above)

**Change to §4 (contours and gridding).** Contour points whose location does not move with level are excluded from the DEM. Channel banks in the Volga delta and other river-fed water, for example, give a 50% contour at the same place in every level bin.
- The exact rule and its thresholds are fixed on **training blocks only**, using no calibration- or test-block data. They are fixed before any calibration- or test-block evaluation, and logged in `docs/DECISIONS.md` (D-034) with the date.
- The rule must keep contours on real steep slopes.
- Reported: the fraction and length of contour excluded per sector and level, and the DEM accuracy without the exclusion, as a robustness check.
- Whether the same exclusion applies to the §9 observed waterlines is decided with the rule, before any hindcast metric is computed.

**Reason.** The DEM gives each contour its bin level. A contour set by a bank rather than by the sea would put many heights at one location, and would inflate the contour density used in §8. The sea connection of Amendment 7 does not remove these contours, because the channels join the sea.

**Clarification (2026-10-20, integration).** The exclusion rule is fixed on the training blocks of the initial assignment (25 km blocks, seed 20261020, §6), before the §6 block-size check. If §6 enlarges and reassigns the blocks, the rule is not re-tuned. The cells used to fix it that fall in calibration or test blocks of the final assignment are counted and reported. As a robustness check, the headline DEM accuracy and coverage are recomputed with those cells excluded.

**Signed off by the lead, 2026-10-20** (the consolidating agent's choice: only the constraint is fixed by this amendment; the rule itself is fixed later, on training blocks, as above).

---

## Amendment 9 — 2026-10-20 (before any stage-2 run, water frequency, waterline, DEM, hindcast or projection metric)

**What had been seen.** Cloud-masked stage-1 test batches (20 July scenes per sensor, made with the code that applies this amendment), in which the Landsat QA snow bit flagged 0.0% of clear in-zone pixels, Sentinel-2 SCL class 11 flagged 1.9%, and the LIS NDSI test flagged 1.2% (Landsat) and 3.1% (Sentinel-2). The extension was chosen after seeing these fractions. As registered, section 3 excluded QA-flagged snow and ice in every month.

**Change.** The Amendment 3 rule (a test that flags more than 1% of clear in-zone pixels in any May-September month is applied only in November-April) is extended from the NDSI test to the product QA snow flags. Each test is evaluated separately on the full stage-1 run 2. In May-September no test replaces the QA snow screen. The cloud screens remain in force in all months (Landsat QA cloud, shadow, cirrus and saturation; Sentinel-2 Cloud Score+), and pixels a QA flag would mark as snow are classified by the scene threshold like any other pixel. The share each test flags is recorded per scene.

**Robustness check.** Headline results recomputed with both tests applied in every month.

**Reason.** Snow and ice cannot occur on the North Caspian in May-September; flags there are false positives, most likely on bright turbid water, which is where waterlines lie.

---

## Amendment 10 — 2026-10-20 (change to §9; before any hindcast, DEM-accuracy or projection metric was computed; no level-binned observed extent or waterline existed)

**What had been seen.** The input-level results listed before Amendment 4, and since then: the stage-1 run-1 thresholds and monthly snow shares (D-048, D-056), the cloud-masked test batches of Amendment 9, and the zone-v6 seaward-ring check (D-049), which compares scene thresholds with the land share just outside the zone and uses no product. The product-only shorelines of the sector layout check (D-039, D-040) were known: at the record-maximum level, GEBCO_2022, GEBCO_2026 and SRTM15+ flood land north of the domain edge (47.35 N). They motivated this change. No product had been compared with any observed extent.

**Changes to §9 (Amendment 6, item 2).**
1. **Known states beyond the domain.** Cells outside the domain, and in the one-cell rim inside its edge, stay unknown unless an independent beyond-domain state is supplied. Where it is supplied, such a cell outside the zone takes that state, water or land, and enters the area metrics and the displacement like any other known cell outside the valid mask (Amendment 6, items 3 and 7).
   - The beyond-domain state comes from observations independent of the products, and holds at every evaluated level: for example, land where a cell lies farther than the zone dilation from the observed near-maximum 1995–96 water extent.
   - Its construction is fixed and logged in `docs/DECISIONS.md` (D-050) before any hindcast metric is computed.
2. **Sectors beyond the domain.** In the per-sector metrics, a cell outside the domain belongs to the sector of the nearest cell inside the domain.
3. **Zone used for the outside states.** The zone that separates labelled from observed cells is the footprint over which water frequencies were computed (the stage-2 tile cover, buffered by one 5 km cell; D-049, D-060), not zone v6 itself, so that waterlines inside that footprint are observed rather than labelled.

**Reason.** Verifier finding (D-050). With every cell beyond the domain unknown, a product that floods land beyond the domain edge was not scored there: its predicted water counted only as unmeasured. The censoring that Amendment 6 removed at the zone edge therefore remained at the domain edge, and it affects three of the audited products at the record-maximum level. Without a beyond-domain state the Amendment 6 rule is unchanged; the amendment only adds known states, and fixes how they may be obtained, without sight of any hindcast result.

## Amendment 11 — 2026-10-21 (changes and clarifications to §2, §5, §6, §7, §9 and §10; before any stage-2 main output, water frequency, waterline, DEM, residual, hindcast or projection metric existed)

**What had been seen:**
- input-level results only: stage-1 run 3 thresholds; the setup-model fit on SWOT; the counts of one sensitivity block, checked for water ≤ valid and found to be open water at every level;
- geoid, datum and product-documentation facts;
- no screening statistic S(τ), no waterline, no DEM, no ICESat-2 residual and no hindcast quantity.

**Changes:**
1. **§6 block-size check needs evidence of spatial structure (D-069).** The exponential practical range sets the block size only if a permutation test detects spatial autocorrelation of the training-block residuals. The test statistic is the mean semivariance below 7.5 km divided by the mean over all pairs up to 150 km; there are 999 permutations, and structure is detected at p ≤ 0.05. Otherwise the 25 km blocks are kept. Reason: on synthetic white noise the exponential fit returned a 63 km range, which would have enlarged the blocks for no reason.
2. **§7 H2 (D-069):**
   - metrics are pooled over test-block points, with the median of per-block metrics reported as a sensitivity;
   - products enter with one training-block constant offset each ("matched", as the DEM's ICESat-2 constant), and the "total" variant is reported alongside;
   - ATL06 and ATL08 are never pooled: H2 is supported only if the rule holds on both, and is partially supported if it holds on one.
3. **§2 dryness test (D-069):**
   - **emergence date of a DEM pixel:** the first date of the daily DAHITI level series after the last date on which the level was at or above the pixel's height;
   - **neighbourhood:** 1 pixel for ATL06 and 2 pixels for ATL08;
   - the ATL08 `segment_watermask` is not used (a static mask);
   - **robustness check:** emergence + 365 days.
4. **§2 vegetated sectors (D-071):**
   - the emerged band is water in the 1995–96 extent and not water in the 2025–26 extent;
   - a cell is vegetated when its E3 emergent-vegetation share is at least 0.5;
   - a sector is vegetated when at least 10% of its band cells are vegetated.
5. **§5 datum chain (D-070, D-073, D-074):**
   - the working datum is the DAHITI reference; a constant offset to BHS cancels in every hindcast metric and is used only to report levels;
   - **total error** converts products as EGM2008 heights with N(EGM2008, NGA) − N(EIGEN-6C4); the variant without NGA's −0.41 m constant is a sensitivity;
   - **shape error** fits its constant on training-block cells only (on a 0.01 m grid over ±5 m), and is refitted without the held-out epoch in the leave-one-epoch-out check.
6. **§9 H1 decision variant (D-072):** the shape variant decides "mispredicts", and the total variant is reported alongside. **H1 block statistic (D-070):** per block, the median absolute displacement pooled over all its units; blocks with at least 20 distances; 10,000 label permutations.
7. **§4 static contours (D-034, D-075):** points with rho_level below rho* are excluded from the DEM. rho* is Otsu's split of training-block rho_level if valid (η ≥ 0.70, valley ratio ≤ 0.50, within [0.2, 0.8]); otherwise rho* = 0.5.
8. **§3 water classification (D-068):** normalized differences take negative reflectance as 0, so Earth Engine's masking of negative inputs no longer drops clear-water observations. Stage 1 was rerun under this definition, and run-2 thresholds are not used.

**Reason.** Each item either fixes a rule that the text left open, or corrects a defect found by tests or diagnostics before any outcome existed (items 1 and 8). None was chosen with sight of an outcome; the decision log holds the details and alternatives.

## Amendment 12 — 2026-10-21 02:40 (data, §3–§4; before any main stage-2 output, water frequency, waterline, DEM or metric existed)

**Change.** The main stage-2 water frequencies, and therefore the waterlines, the DEM, the hindcast observed extents and the emergence analysis, use **Landsat 5/7/8/9 only**. Sentinel-2 MSI is used in stage 1 (thresholds and snow diagnostics) and in a robustness check, run on test blocks only if the quota allows.

**Reason (D-079).** Measured costs: Sentinel-2 costs about 4.5 times Landsat per km² (about 700 EECU-hours interactive), which exceeds the 70% monthly Earth Engine cap set by the project rules. Batch execution runs at 0.5–2.5 EECU-s per second per task with two tasks at a time, so it would take several days. Landsat alone provides 4,103 acquisitions in 43 level bins over 1984–2026. **Acquisition halves:** parities were assigned over all sensors in time order. Within Landsat they remain disjoint and independent, with a median parity-0 share per bin of 0.502.

## Amendment 13 — 2026-10-21 04:10 (change to §9 observed extents; before any hindcast or emergence metric; the DEM was being built, and no residual or hindcast quantity had been computed)

**Change.** A cell that is water in the observed extent at the record low (the three lowest E3 level bins, counts pooled) is water in every observed extent at higher levels: an observed dry state there is set to water. This applies to the observed extents of §9 (and §10), to the noise-floor halves, and to E1 against E2/E3. The number of cells changed is reported per unit, and the hindcast without this rule is a sensitivity analysis.

**Reason (D-082).** QC of the stage-2 observations, with no product involved, showed that dark, clear deep water is often classified dry: up to 4% of record-low water cells in some bins, and whole blocks of open water with frequencies near 0.5. Under sea-connected flooding, a cell under water at the lowest level of the record is under water at every higher level, so these dry states are observation errors. Left in, they would create spurious observed shorelines far offshore and inflate every product's displacement.
