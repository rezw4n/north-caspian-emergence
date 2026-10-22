"""Writer metrics of the revision: ranges and shares the Letter and the Supplementary
quote that no other revision module states directly.

Loaded by tools/build_metrics.revision_metrics, which calls add(m) after every other
metrics_rev_<name>.py module (name order), so every key here is built only from keys those
modules have already put in m, or from keys of the registered pipeline. Nothing is recomputed
from result files; each record names the keys it was taken from. Every key starts "rev.w.".
"""

from __future__ import annotations

PRODUCTS = ["gebco2026", "gebco2022", "etopo2022", "etopo1", "srtm15p"]
MAJORITY = ["gebco2026", "gebco2022", "etopo1", "srtm15p"]   # the four grids that mispredict
COASTS3 = ["west", "interfluve", "northeast"]                  # validated coasts besides the open shelf
UNVALIDATED = ["volga", "ural", "komsomolets"]
SECTORS = ["west", "volga", "ural", "interfluve", "northeast", "komsomolets", "open_shelf"]


def _v(m: dict, key: str):
    """The value of a metrics record, or None when the key is absent."""
    rec = m.get(key)
    if rec is None:
        return None
    return rec["value"] if isinstance(rec, dict) else rec


def _put(m: dict, key: str, value, unit: str, source: str, note: str | None = None, decimals: int | None = None):
    if value is None:
        return
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    if decimals is not None:
        rec["decimals"] = decimals
    m[f"rev.w.{key}"] = rec


def _range(m: dict, key: str, keys: list[str], unit: str, note: str, decimals: int | None = None):
    vals = [_v(m, k) for k in keys]
    if not keys or any(v is None for v in vals):
        return
    src = "metrics.json keys " + ", ".join(sorted(set(k.rsplit(".", 2)[0] + ".*" for k in keys)))
    _put(m, f"{key}_min", min(vals), unit, src, note, decimals)
    _put(m, f"{key}_max", max(vals), unit, src, note, decimals)


def h1(m: dict) -> None:
    note = "over the four mispredicting grids (shape variant), three validated coasts: west, interfluve, northeast"
    _range(m, "h1.coasts3.d_km", [f"rev.s.headline.{s}.d_km_{e}" for s in COASTS3 for e in ("min", "max")], "km", note)
    _range(m, "h1.coasts3.q_km", [f"rev.s.headline.{s}.q_km" for s in COASTS3], "km", "sector floors, " + note)
    _range(m, "h1.coasts3.dq", [f"rev.s.headline.{s}.dq_median_{e}" for s in COASTS3 for e in ("min", "max")],
           "ratio", "per-level D/Q medians, " + note, 1)
    _range(m, "h1.coasts3.p", [f"rev.s.headline.{s}.p_mispredict_{e}" for s in COASTS3 for e in ("min", "max")],
           "probability", "moving-block bootstrap P(mispredict), " + note, 2)
    unote = "over the four mispredicting grids (shape variant), sectors whose observations are not validated"
    _range(m, "h1.unvalidated.d_km",
           [f"rev.s.headline.{s}.d_km_{e}" for s in ("volga", "ural") for e in ("min", "max")]
           + [f"rev.s.item2.dq.shape.{p}.komsomolets.d_km" for p in MAJORITY], "km", unote)
    _range(m, "h1.unvalidated.q_km", [f"rev.s.headline.{s}.q_km" for s in ("volga", "ural")]
           + [f"rev.s.item2.dq.shape.{MAJORITY[0]}.komsomolets.q_km"], "km", "sector floors, " + unote)
    _range(m, "h1.komsomolets.p", [f"rev.s.item2.p_mispredict.shape.{p}.komsomolets" for p in MAJORITY],
           "probability", "P(mispredict) in Komsomolets Bay, four grids, shape", 2)
    _range(m, "h1.komsomolets.d_km", [f"rev.s.item2.dq.shape.{p}.komsomolets.d_km" for p in MAJORITY], "km",
           "sector median displacement in Komsomolets Bay, four grids, shape")
    _range(m, "h1.komsomolets.dq", [f"rev.s.item2.dq.shape.{p}.komsomolets.median" for p in MAJORITY], "ratio",
           "median per-level D/Q in Komsomolets Bay, four grids, shape", 1)
    _range(m, "h1.komsomolets.q_km", [f"rev.s.item2.dq.shape.{p}.komsomolets.q_km" for p in MAJORITY], "km",
           "Komsomolets Bay floor")
    pe = _v(m, "rev.s.item2.count.shape.etopo2022.validated4.p_equal_registered")
    if pe is not None:
        _put(m, "h1.etopo2022.validated4.p_at_least_one", round(1 - pe, 2), "probability",
             "1 - rev.s.item2.count.shape.etopo2022.validated4.p_equal_registered",
             "ETOPO 2022: bootstrap probability of at least one mispredicted validated sector (registered count 0)", 2)
    # test blocks only: validated sectors mispredicted, four grids
    _range(m, "h1.test.validated4.n", [f"rev.s.r3m4.test.shape.{p}.n_validated" for p in MAJORITY], "sectors",
           "of 4 validated sectors, test blocks only, four grids")
    # halves-only floor: sectors mispredicted, by grid
    _range(m, "h1.halves.gebco_etopo1.n", [f"rev.s.r3m4.halves_floor.shape.{p}.n_sectors"
                                            for p in ("gebco2026", "gebco2022", "etopo1")], "sectors",
           "of 7, halves-only floor, GEBCO grids and ETOPO1")
    # record-low rule: open-shelf grids mispredicting with and without it
    _range(m, "h1.noenv.validated4.n", [f"rev.s.item1.no_record_low_rule.shape.{p}.validated4.n_mispredict"
                                         for p in MAJORITY], "sectors", "of 4 validated sectors without the record-low rule")
    _range(m, "h1.sectorshift.validated4.n", [f"rev.s.item1.sector_constants.sector_shape.{p}.validated4.n_mispredict"
                                               for p in MAJORITY], "sectors", "of 4 validated sectors with per-sector shifts")


def s_area(m: dict) -> None:
    for scope in ("domain", "validated"):
        _range(m, f"s.slope.{scope}", [f"rev.s.item4.area_bias.shape.{p}.{scope}.theil_sen_pct_per_m" for p in MAJORITY],
               "% per m", f"Theil-Sen slope of per-level area bias against level, four grids, {scope}", 0)
        _range(m, f"s.slope_lo.{scope}", [f"rev.s.item4.area_bias.shape.{p}.{scope}.theil_sen_lo" for p in MAJORITY],
               "% per m", f"lower 95% bound of that slope, four grids, {scope}", 1)
        _range(m, f"s.iqr.{scope}", [f"rev.s.item4.area_bias.shape.{p}.{scope}.iqr_pct" for p in PRODUCTS],
               "%", f"interquartile range of per-level area bias, five grids, {scope}", 0)
    _range(m, "s.oos.c_e12_m", [f"rev.s.item4.oos.{p}.c_e12_m" for p in PRODUCTS], "m",
           "shape constant fitted on E1 and E2 training blocks, five grids")
    _range(m, "s.oos.c_all_m", [f"rev.s.item4.oos.{p}.c_all_m" for p in PRODUCTS], "m",
           "shape constant fitted on all epochs (registered), five grids")
    _range(m, "s.oos.validated_pct", [f"rev.s.item4.oos.{p}.calibtest_validated.c_e12.median_pct" for p in PRODUCTS],
           "%", "median E3 area bias in validated sectors, calibration and test blocks, constant from E1 and E2", 0)
    _range(m, "s.oos.c_all_pct", [f"rev.s.item4.oos.{p}.calibtest.c_all.median_pct" for p in PRODUCTS],
           "%", "median E3 area bias, calibration and test blocks, registered all-epoch constant", 0)


def s_deltaic(m: dict) -> None:
    three = ["gebco2026", "gebco2022", "etopo1"]     # the block tests significant in the shape variant
    _range(m, "s.item5.sector_p", [f"rev.s.item5.shape.{p}.sector.p" for p in three], "p",
           "exact sector-level one-sided p, the three grids whose block test was significant (shape)", 2)
    _range(m, "s.item5.block_p_holm", [f"rev.s.item5.shape.{p}.block.p_holm" for p in three], "p",
           "Holm-adjusted block p over 10 tests, the same three grids (shape)", 3)
    _range(m, "s.item5.sector_rank_p", [f"rev.s.item5.shape.{p}.ranksum.p" for p in three], "p",
           "rank-sum sector-level p, the same three grids (shape)", 2)


def g_gsl(m: dict) -> None:
    four = ["gebco2026", "gebco2022", "etopo1", "srtm15p"]
    arms = ("north", "south")
    _range(m, "g.gsl.flat_share_pct", [f"rev.g.gsl.{a}.{p}.modal_share_pct" for a in arms for p in four], "%",
           "share of the emerged band at the grid's one flat height, four SRTM-derived grids, both arms", 0)
    fl = [_v(m, f"rev.g.gsl.{a}.{p}.modal_height_navd88_m") for a in arms for p in four]
    if fl and None not in fl and len(set(fl)) == 1:
        _put(m, "g.gsl.flat_height_navd88_m", fl[0], "m NAVD 88", "rev.g.gsl.<arm>.<grid>.modal_height_navd88_m",
             "the one flat height of the four SRTM-derived grids, both arms", 2)
        above = [round(fl[0] - _v(m, f"rev.g.gsl.{a}.gauge_feb2000_navd88_m"), 2) for a in arms]
        if None not in above:
            _put(m, "g.gsl.flat_above_feb2000_min_m", min(above), "m", "flat height minus rev.g.gsl.<arm>.gauge_feb2000_navd88_m", None, 2)
            _put(m, "g.gsl.flat_above_feb2000_max_m", max(above), "m", "flat height minus rev.g.gsl.<arm>.gauge_feb2000_navd88_m", None, 2)
    _range(m, "g.gsl.ctrl.north.four_exceeding", [f"rev.g.gsl.control_levels.{p}.shape.north.n_exceeding" for p in four],
           "levels", "north arm, at the control's levels, four SRTM-derived grids")
    _range(m, "g.gsl.ctrl.south.exceeding", [f"rev.g.gsl.control_levels.{p}.shape.south.n_exceeding" for p in PRODUCTS],
           "levels", "south arm, at the control's levels, five grids")
    _range(m, "g.gsl.ctrl.ratio", [f"rev.g.gsl.control_levels.ratio.{p}.{a}.median" for p in PRODUCTS for a in arms],
           "ratio", "median per-level grid displacement over the control's, five grids, both arms", 0)
    for a in arms:
        _range(m, f"g.gsl.{a}.lidar2016_pct", [f"rev.g.gsl.dep3.current.{a}.lidar2016_pct",
                                              f"rev.g.gsl.dep3.asof2022.{a}.lidar2016_pct"], "%",
               f"share of the {a} arm's 3DEP band from the autumn-2016 lidar, current and 2022 tile metadata", 0)
    _range(m, "g.gsl.south_nonlidar_pct", ["rev.g.gsl.dep3.current.south.non_lidar_pct",
                                           "rev.g.gsl.dep3.asof2022.south.non_lidar_pct"], "%",
           "share of the south arm's 3DEP band from legacy non-lidar DEMs (1966-1973), current and 2022 tile metadata", 0)


def g_h2_etopo(m: dict) -> None:
    pts = {s: _v(m, f"h2.atl08.n_points.{s}") for s in SECTORS}
    known = {s: v for s, v in pts.items() if v is not None}
    if "northeast" in known and len(known) >= 2:
        _put(m, "g.h2.ne_points_share_pct", round(100 * known["northeast"] / sum(known.values())), "%",
             "h2.atl08.n_points.<sector>", "share of the ATL08 test points in the northeast sector's test blocks", 0)
    _range(m, "g.etopo.below.n_sectors", [f"rev.g.etopo.bands.below_m27p7.{p}.n_sectors_majority" for p in PRODUCTS],
           "sectors", "of 7: sectors where a grid exceeds the floor at most levels below -27.7 m")
    _range(m, "g.etopo.above.dq", [f"rev.g.etopo.bands.above_m27p7.etopo2022.{s}.median_DQ" for s in SECTORS],
           "ratio", "ETOPO 2022 per-sector median D/Q above -27.7 m", 2)
    _range(m, "g.etopo.below.dq", [f"rev.g.etopo.bands.below_m27p7.etopo2022.{s}.median_DQ" for s in SECTORS],
           "ratio", "ETOPO 2022 per-sector median D/Q below -27.7 m", 2)


def g_etopo_cap(m: dict) -> None:
    """The level at which ETOPO 2022 caps GEBCO_2022's Caspian lake depths, as the rule that
    revision analysis G tested and recorded in the note of its metric (matched at every cell)."""
    import re

    rec = m.get("rev.g.etopo.lake_clamp.share_matching_min_rule_pct")
    if not isinstance(rec, dict) or rec.get("value") != 100.0:
        return
    hit = re.search(r"min\(GEBCO_2022,\s*(-?\d+(?:\.\d+)?)\s*m\)", rec.get("note", ""))
    if hit:
        _put(m, "g.etopo.cap_m", float(hit.group(1)), "m",
             "note of rev.g.etopo.lake_clamp.share_matching_min_rule_pct (results/revision/g/etopo/lake_clamp.json)",
             "ETOPO 2022 = min(GEBCO_2022, cap) at every North Caspian lake-depth cell", 0)


def s_bootstrap(m: dict) -> None:
    """Design of revision analysis S's moving-block bootstrap of the verdicts (A18 item 2),
    read from its run summary."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "results" / "revision" / "s" / "summary.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    src = "results/revision/s/summary.json"
    _put(m, "s.boot.n", d.get("n_boot"), "resamples", src, "moving-block bootstrap over levels")
    _put(m, "s.boot.block_levels", d.get("block_levels"), "levels", src, "levels per bootstrap block")
    bw = _v(m, "params.bin_width_m")
    if bw is not None and d.get("block_levels") is not None:
        _put(m, "s.boot.block_m", round(d["block_levels"] * bw, 2), "m", f"{src} x params.bin_width_m",
             "length of a bootstrap block in level", 1)
    _put(m, "s.boot.seed", str(d["seed"]) if d.get("seed") is not None else None, "text", src,
         "seed, as text so that it prints without separators")


def _pct(m: dict, key: str, keys: list[str], note: str, decimals: int = 0, scale: float = 100.0):
    """Range of shares (0-1) in keys, as percentages."""
    vals = [_v(m, k) for k in keys]
    if not keys or any(not isinstance(v, (int, float)) for v in vals):
        return
    src = "metrics.json keys " + ", ".join(sorted(set(k.rsplit(".", 1)[0] + ".*" for k in keys)))
    _put(m, f"{key}_min", round(min(vals) * scale, decimals + 2), "%", src, note, decimals)
    _put(m, f"{key}_max", round(max(vals) * scale, decimals + 2), "%", src, note, decimals)


def n_near(m: dict) -> None:
    """Near-term (revision analysis N, A18 item 10): percentages, ranges over grids and text years."""
    n = "rev.n."
    _pct(m, "n.dry.unc1_pct", [n + "already_dry.uncorrected.1.share_zv6_min", n + "already_dry.uncorrected.1.share_zv6_max"],
         "share of each uncorrected grid's 1 m band not water in the 2025-26 extent")
    _pct(m, "n.dry.unc2_pct", [n + "already_dry.uncorrected.2.share_zv6_min", n + "already_dry.uncorrected.2.share_zv6_max"],
         "the same at 2 m")
    _pct(m, "n.dry.shifted1_pct", [n + "already_dry.shifted.1.share_zv6_min", n + "already_dry.shifted.1.share_zv6_max"],
         "the same for the shifted grids at 1 m", 1)
    _pct(m, "n.isolated.unc1_pct", [n + "summary.uncorrected.1.isolated_share_min", n + "summary.uncorrected.1.isolated_share_max"],
         "isolated water in the uncorrected 1 m bands")
    _pct(m, "n.isolated.shifted1_pct", [n + "summary.shifted.1.isolated_share_min", n + "summary.shifted.1.isolated_share_max"],
         "isolated water in the shifted 1 m bands", 1)
    _pct(m, "n.isolated.composite1_pct", [n + "composite.summary.common_location.1.isolated_share_min",
                                          n + "composite.summary.common_location.1.isolated_share_max"],
         "isolated water on the composite baseline at 1 m", 1)
    # variance shares, pathway by grid (area scale)
    pw = [n + f"var.{s}.domain.{y}.area.pathway" for s in ("shifted", "uncorrected") for y in ("2035", "2050")]
    pw.append(n + "var.common_location_composite.domain.2050.area.pathway")
    _pct(m, "n.var.pathway_pct", pw, "pathway share of the sum of squares of emerged area at 2035 and 2050 "
         "(shifted, uncorrected; composite at 2050)")
    for scale in ("area", "log"):
        k = n + f"var.common_location_composite.domain.2035.{scale}.product"
        if _v(m, k) is not None:
            _put(m, f"n.var.composite2035.grid_{scale}_pct", round(100 * _v(m, k), 1), "%", k,
                 "grid share of the sum of squares, composite baseline, 2035", 0)
    # decline-normalised envelopes at 1 m, all pairs
    grids = PRODUCTS
    _range(m, "n.env.shifted1.lo", [n + f"env.shifted.{p}.all_pairs.decline.1.lo_km2" for p in grids], "km2",
           "lower bound of the decline-normalised envelope, shifted grids, 1 m")
    _range(m, "n.env.shifted1.hi", [n + f"env.shifted.{p}.all_pairs.decline.1.hi_km2" for p in grids], "km2",
           "upper bound of the decline-normalised envelope, shifted grids, 1 m")
    _range(m, "n.env.composite1.lo", [n + f"env.common_location_composite.{p}.all_pairs.decline.1.lo_km2" for p in grids],
           "km2", "lower bound, composite baseline, 1 m")
    _range(m, "n.env.composite1.hi", [n + f"env.common_location_composite.{p}.all_pairs.decline.1.hi_km2" for p in grids],
           "km2", "upper bound, composite baseline, 1 m")
    qs = [_v(m, n + f"env.q.shifted.{p}.domain.E3_pairs.scale") for p in grids]
    if qs and all(isinstance(q, (int, float)) for q in qs):
        _put(m, "n.env.q_e3_shifted_min", min(qs), "ratio", "rev.n.env.q.shifted.<grid>.domain.E3_pairs.scale", None, 1)
        _put(m, "n.env.q_e3_shifted_max", max(qs), "ratio", "rev.n.env.q.shifted.<grid>.domain.E3_pairs.scale", None, 1)
    # vertical-error simulation
    _pct(m, "n.sim.sd_rel_shifted1_pct", [n + f"sim.150km.{p}.E_shifted_1m_domain.sd_rel" for p in grids],
         "relative standard deviation of E(1 m) on the shifted grids over the simulated fields")
    _range(m, "n.sim.sd_shifted1_km2", [n + f"sim.150km.{p}.E_shifted_1m_domain.sd_km2" for p in grids], "km2",
           "standard deviation of E(1 m) on the shifted grids over the simulated fields")
    obs = _v(m, n + "sim.observed_agreement_240m.uncorrected.1")
    if isinstance(obs, (int, float)):
        _put(m, "n.sim.agree_unc1_observed_pct", round(100 * obs, 2), "%",
             n + "sim.observed_agreement_240m.uncorrected.1", "five-grid agreement, grids as distributed, 1 m, 480 m cells", 1)
    _pct(m, "n.sim.agree_unc1_ref_p05_pct", [n + f"sim.150km.{p}.agreement_uncorrected_1m.p05" for p in grids],
         "5th percentile of the agreement of five perturbed copies of one grid", 2)
    _pct(m, "n.sim.agree_unc1_ref_p95_pct", [n + f"sim.150km.{p}.agreement_uncorrected_1m.p95" for p in grids],
         "95th percentile of the agreement of five perturbed copies of one grid", 1)
    _pct(m, "n.sim.agree_unc1_groups_below_pct",
         [n + f"sim.150km.{p}.agreement_uncorrected_1m.share_groups_at_or_below_observed" for p in grids],
         "share of reference groups whose agreement is at or below the observed")
    # protected areas
    for surf, lab in (("common_location_composite", "composite"), ("shifted", "shifted")):
        for var, tag in (("wdpa_protected_areas", "with"), ("wdpa_protected_areas_noksrz", "without")):
            _pct(m, f"n.wdpa.{lab}.{tag}_pct", [n + f"exp.{var}.{surf}.1.min_share", n + f"exp.{var}.{surf}.1.max_share"],
                 f"protected-area share of the 1 m band, {lab}, {tag} WDPA record 342651")
    # pathway trend windows: years as text
    for w in ("1996_2021", "2006_2025", "2016_2025"):
        for d in ("1m", "2m"):
            y = _v(m, n + f"trend.{w}.year_reaches_{d}")
            if y is not None:
                _put(m, f"n.trend.{w}.year_{d}", str(int(y)), "text", n + f"trend.{w}.year_reaches_{d}")
        r = _v(m, n + f"trend.{w}.rate_cm_per_yr")
        if r is not None:
            _put(m, f"n.trend.{w}.fall_cm_per_yr", round(abs(r), 1), "cm per year", n + f"trend.{w}.rate_cm_per_yr", None, 1)
    import re

    rng = sorted({int(h.group(1)) for k in m for h in [re.match(r"rev\.n\.sim\.(\d+)km\.", k)] if h})
    if rng:
        _put(m, "n.sim.range_primary_km", max(rng), "km", "the rev.n.sim.<range>km.* keys of workstream N",
             "the primary correlation range of the simulated vertical-error fields (D-125)")
    if len(rng) > 1:
        _put(m, "n.sim.range_secondary_km", min(rng), "km", "the rev.n.sim.<range>km.* keys of workstream N",
             "the sensitivity correlation range of the simulated vertical-error fields (D-125)")
    for r in rng:
        t = f"n.sim.r{r}km"
        _pct(m, f"{t}.sd_rel_shifted1_pct", [n + f"sim.{r}km.{p}.E_shifted_1m_domain.sd_rel" for p in PRODUCTS],
             f"relative SD of E(1 m), shifted grids, {r} km range")
        _range(m, f"{t}.sd_shifted1_km2", [n + f"sim.{r}km.{p}.E_shifted_1m_domain.sd_km2" for p in PRODUCTS], "km2",
               f"SD of E(1 m), shifted grids, {r} km range")
        _pct(m, f"{t}.agree_unc1_ref_p05_pct", [n + f"sim.{r}km.{p}.agreement_uncorrected_1m.p05" for p in PRODUCTS],
             f"5th percentile of five-copy agreement, {r} km range", 2)
        _pct(m, f"{t}.agree_unc1_ref_p95_pct", [n + f"sim.{r}km.{p}.agreement_uncorrected_1m.p95" for p in PRODUCTS],
             f"95th percentile of five-copy agreement, {r} km range", 1)
        _pct(m, f"{t}.agree_unc1_groups_below_pct",
             [n + f"sim.{r}km.{p}.agreement_uncorrected_1m.share_groups_at_or_below_observed" for p in PRODUCTS],
             f"share of reference groups at or below the observed agreement, {r} km range")
    jit = sorted({float(h.group(1)) for k in m for h in [re.match(r"rev\.n\.sens\.L26\+(\d+(?:\.\d+)?)\.", k)] if h})
    if len(jit) == 1:
        _put(m, "n.sens.jitter_m", jit[0], "m", "the rev.n.sens.L26+<value>.* keys of workstream N",
             "change of the 2026 reference level in N's sensitivity", 1)
    a = _v(m, n + "anchor.observed_fall_2021_2026_m")
    if a is not None:
        _put(m, "n.anchor.observed_fall_m", round(a, 2), "m", n + "anchor.observed_fall_2021_2026_m", None, 2)


def r_rasters(m: dict) -> None:
    """Revision analysis R (A18 items 3, 6, 7): ranges over the four mispredicting grids and sectors."""
    r = "rev.r."
    for name, secs in (("coasts3", ("west", "interfluve", "northeast")), ("westinter", ("west", "interfluve")),
                       ("openshelf", ("openshelf",)), ("validated4", ("west", "interfluve", "northeast", "openshelf"))):
        _range(m, f"r.control.{name}_km", [r + f"control.rounded.{p}.{s}.D_control_median_km" for p in MAJORITY
                                             for s in secs], "km",
               f"median displacement of the whole-metre, native-cell control, four grids' classes, {name}")
    for k in ("sigma_median", "horizontal_median"):
        u = "m" if k == "sigma_median" else "km"
        _range(m, f"r.budget.{k}", [r + f"budget.{k}.{s}_{u}" for s in SECTORS_R], u,
               f"vertical error budget (1 sigma) per sector, {k}", 2 if u == "m" else 2)
    g = [_v(m, r + f"budget.g_median.{s}") for s in SECTORS_R]
    if g and None not in g:
        _put(m, "r.km_per_m_min", round(1 / max(g), 2), "km per m", "1 / max rev.r.budget.g_median.<sector>",
             "horizontal shift of the shoreline per metre of level, from the median shelf gradients", 0)
        _put(m, "r.km_per_m_max", round(1 / min(g), 2), "km per m", "1 / min rev.r.budget.g_median.<sector>", None, 0)
    _range(m, "r.gmedian", [r + f"budget.g_median.{s}" for s in SECTORS_R], "m per km",
           "median shelf gradient per sector", 2)
    d = [_v(m, r + f"constants.{p}.c_D_minus_c_area_m") for p in MAJORITY]
    if d and None not in d:
        _put(m, "r.cD_below_min_m", round(min(abs(x) for x in d), 2), "m", r + "constants.<grid>.c_D_minus_c_area_m",
             "how far the displacement-fitted constant lies below the area constant", 1)
        _put(m, "r.cD_below_max_m", round(max(abs(x) for x in d), 2), "m", r + "constants.<grid>.c_D_minus_c_area_m", None, 1)
    for fam, lab in (("budget.mispredicts_augmented", "budget"), ("res.mispredicts", "res"),
                     ("constants.bestcase", "bestcase")):
        suffix = "n_no_single_constant_validated" if lab == "bestcase" else "n_validated4"
        _range(m, f"r.{lab}.validated4", [r + f"{fam}.{p}.{suffix}" for p in MAJORITY], "sectors",
               f"validated sectors still mispredicted, four grids, {lab}")


SECTORS_R = ["west", "volga", "ural", "interfluve", "northeast", "komsomolets", "openshelf"]
POSTS_DELTA = ["baku", "fort", "kbg", "krasn", "kuuli", "makh", "nkamni"]   # the seven posts behind delta
POSTS_NORTH = ["lag", "tul", "pesh", "kul"]                               # North Caspian posts, not used for delta


def v_datums(m: dict) -> None:
    """Revision analysis V (A18 item 9): sums and ranges over grids, posts and terms."""
    import json
    from pathlib import Path

    so = Path(__file__).resolve().parents[1] / "results" / "hindcast" / "main_sectorshape" / "sector_offsets.json"
    if so.exists():
        sm = json.loads(so.read_text(encoding="utf-8")).get("smooth_sigma_m")
        if sm is not None:
            _put(m, "redteam.sector_smooth_sigma_km", round(sm / 1000, 2), "km",
                 "results/hindcast/main_sectorshape/sector_offsets.json (smooth_sigma_m)",
                 "Gaussian standard deviation of the red-team R2 per-sector offset map (R3 minor 3)", 0)
    v = "rev.v."
    ch = [_v(m, v + f"depaudit.shape.{p}.n_changed") for p in PRODUCTS]
    if ch and None not in ch:
        _put(m, "v.depaudit.shape_n_changed", int(sum(ch)), "verdicts", v + "depaudit.shape.<grid>.n_changed",
             "shape-variant verdicts changed by the sector-shifted levels, five grids x seven sectors")
        _put(m, "v.depaudit.n_cases", 5 * 7, "verdicts", "five grids x seven sectors")
    _range(m, "v.planar.reduction_pct", [v + f"planar.{p}.objective_reduction_pct" for p in PRODUCTS], "%",
           "reduction of the Eq. S11 area objective by the planar field", 0)
    for p in ("gebco2026", "etopo1"):
        a = _v(m, v + f"planar.verdicts.domain_constant.{p}.median_sector_median_m")
        b = _v(m, v + f"planar.verdicts.planar.{p}.median_sector_median_m")
        if a and b:
            _put(m, f"v.planar.{p}.before_km", round(a / 1000, 2), "km", v + f"planar.verdicts.domain_constant.{p}.median_sector_median_m", None, 1)
            _put(m, f"v.planar.{p}.after_km", round(b / 1000, 2), "km", v + f"planar.verdicts.planar.{p}.median_sector_median_m", None, 1)
    _range(m, "v.posts.delta_dev_m", [v + f"posts.{p}.minus_delta_m" for p in POSTS_DELTA], "m",
           "per-post offset minus delta, the seven posts behind delta", 2)
    _range(m, "v.posts.north_dev_m", [v + f"posts.{p}.minus_delta_m" for p in POSTS_NORTH], "m",
           "per-post offset minus delta, the four North Caspian posts", 2)
    _range(m, "v.bhs.k_m", [v + f"bhs.{p}.k_m" for p in PRODUCTS], "m",
           "constant to add to the distributed heights to express them in BHS-77", 2)
    spread = []
    for p in PRODUCTS:
        hi, lo = _v(m, v + f"bhs.{p}.k_sector_max_m"), _v(m, v + f"bhs.{p}.k_sector_min_m")
        if hi is not None and lo is not None:
            spread.append(hi - lo)
    if spread:
        _put(m, "v.bhs.sector_spread_min_m", round(min(spread), 2), "m", v + "bhs.<grid>.k_sector_max_m - min", None, 1)
        _put(m, "v.bhs.sector_spread_max_m", round(max(spread), 2), "m", v + "bhs.<grid>.k_sector_max_m - min", None, 1)
    _range(m, "v.cref.c_m", [v + "cref.atl06.c_m", v + "cref.atl08.c_m"], "m", "ICESat-2 constant c_ref", 2)
    _range(m, "v.cref.resid_documented_m", [v + "cref.atl06.resid_documented_min_m", v + "cref.atl08.resid_documented_min_m"],
           "m", "part of c_ref left after the documented terms", 2)
    _range(m, "v.cref.resid_tp_m", [v + "cref.atl06.resid_tp_m", v + "cref.atl08.resid_tp_m"], "m",
           "part of c_ref left after a T/P-ellipsoid reduction", 2)
    _range(m, "v.depth2026_m", [v + "depth2026.shape_min_m", v + "depth2026.shape_max_m"], "m",
           "mean depth of the domain's sea at the 2026 level, shifted grids", 1)
    for k, src, dp in (("v.dep.common_m", "dep.all.m", 2), ("v.levels.annual_1995_bhs_m", "levels.annual_1995_bhs_m", 1),
                       ("v.levels.annual_2025_bhs_m", "levels.annual_2025_bhs_m", 1),
                       ("v.levels.annual_fall_m", "levels.annual_fall_1995_2025_m", 1),
                       ("v.levels.janSep_above_annual_m", "levels.janSep_minus_annual_median_m", 2),
                       ("v.levels.e3_hydroweb_pct", "levels.e3_below_28_share_hydroweb_pct", 0)):
        x = _v(m, v + src)
        if x is not None:
            _put(m, k[2:], round(x, 3), "m" if k.endswith("_m") else "%", v + src, None, dp)


def add(m: dict) -> None:
    v_datums(m)
    r_rasters(m)
    n_near(m)
    s_bootstrap(m)
    g_etopo_cap(m)
    h1(m)
    s_area(m)
    s_deltaic(m)
    g_gsl(m)
    g_h2_etopo(m)
