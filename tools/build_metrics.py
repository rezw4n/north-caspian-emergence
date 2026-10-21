"""Assemble results/metrics.json, the single source of every number in the manuscript
(CLAUDE.md integrity rule 2), from the committed result files.

Each metric is {"value", "unit", "source", "note"?}. Keys are dotted paths
(e.g. "levels.dahiti.n_records"). Rebuilt by `make metrics`; never edited by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(rel: str) -> dict:
    p = RES / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def put(metrics: dict, key: str, value, unit: str, source: str, note: str | None = None) -> None:
    if value is None:
        return
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    metrics[key] = rec


def _half_up(x: float) -> int:
    import math

    return int(math.floor(x + 0.5))


def study_area_metrics(m: dict) -> None:
    """Domain and sector areas (scripts/a0_study_area.py) and the sector-layout check
    against the shorelines traced in the 15-arc-second grids (scripts/a0_sector_edges.py)."""
    sa = load("study_area/sector_areas.json")
    src = "results/study_area/sector_areas.json"
    put(m, "study_area.domain_km2", sa.get("domain_km2"), "km2", src,
        "edges straight in lon/lat, densified before the equal-area projection")
    for k, v in sa.get("sectors_km2", {}).items():
        put(m, f"study_area.sector_km2.{k}", v, "km2", src)
    geo = sa.get("geodesic_reading") or {}
    if geo:
        import math

        gnote = "the same polygons read with geodesic edges, as Earth Engine reads lon/lat GeoJSON"
        put(m, "study_area.geodesic.domain_km2", geo["domain_km2"], "km2", src, gnote)
        put(m, "study_area.geodesic.north_edge_offset_km", _half_up(geo["north_edge_offset_km"]), "km", src,
            "largest distance between the geodesic and the lon/lat reading of the domain's northern edge, "
            "rounded; " + gnote)
        put(m, "study_area.geodesic.other_edge_offset_max_km", math.ceil(geo["other_edges_max_offset_km"]), "km",
            src, "upper bound (rounded up) over every other edge of the domain and the sectors; " + gnote)

    se = load("study_area/sector_edges.json")
    grids = se.get("grids") or {}
    if not grids:
        return
    src = "results/study_area/sector_edges.json"
    note = (f"over the grids {', '.join(grids)}; shorelines at the DAHITI levels "
            f"{se['present_level_m']} m ({se['present_level_time'][:10]}) and {se['record_max_level_m']} m, no offset")
    put(m, "study_area.layout.n_grids", len(grids), "grids", src, note)
    shares = [100 * g["band_share_coastal_sectors"] for g in grids.values()]
    put(m, "study_area.layout.band_share_coastal_pct_min", _half_up(min(shares)), "%", src,
        "share of the seabed between the record-maximum and present shorelines inside the coastal sectors; " + note)
    put(m, "study_area.layout.band_share_coastal_pct_max", _half_up(max(shares)), "%", src, note)
    coastal = [k for k in next(iter(grids.values()))["sectors"] if k != "open_shelf"]
    reach = {k: [g["sectors"][k]["max_offshore_km"] for g in grids.values()] for k in coastal}
    edge = {k: [g["sectors"][k]["offshore_edge"]["p50_km"] for g in grids.values()] for k in coastal}
    put(m, "study_area.layout.max_offshore_km_min", _half_up(min(min(v) for v in reach.values())), "km", src,
        "farthest distance of a sector's sea cells from the present shoreline, smallest over sectors and grids; "
        + note)
    put(m, "study_area.layout.max_offshore_km_max", _half_up(max(max(v) for v in reach.values())), "km", src, note)
    for k in coastal:
        put(m, f"study_area.layout.max_offshore_km.{k}.min", _half_up(min(reach[k])), "km", src, note)
        put(m, f"study_area.layout.max_offshore_km.{k}.max", _half_up(max(reach[k])), "km", src, note)
        put(m, f"study_area.layout.edge_p50_km.{k}.min", _half_up(min(edge[k])), "km", src,
            "median signed distance of the boundary shared with the open shelf from the present shoreline "
            "(negative: on land); " + note)
        put(m, f"study_area.layout.edge_p50_km.{k}.max", _half_up(max(edge[k])), "km", src, note)
    readings = [g["edge_readings"] for g in grids.values() if "edge_readings" in g]
    if readings:
        import math

        pct = [100 * g["edge_readings"]["band_changed_km2"] / g["band_km2_total"]
               for g in grids.values() if "edge_readings" in g and g["band_km2_total"]]
        put(m, "study_area.geodesic.band_changed_pct_max", math.ceil(max(pct)), "%", src,
            "upper bound (rounded up), over the grids, of the share of the seabed between the record-maximum and "
            "present shorelines inside the domain that changes domain or sector between the lon/lat and the "
            "geodesic edge readings; " + note)
        put(m, "study_area.geodesic.band_changed_km2_max", max(r["band_changed_km2"] for r in readings), "km2",
            src, "largest over the grids; " + note)
        put(m, "study_area.geodesic.north_edge_sea_at_max_km2_max",
            max(r["north_edge_sea_at_max_km2"] for r in readings), "km2", src,
            "sea at the record-maximum level between the two readings of the northern edge, largest over the "
            "grids; " + note)


def input_stage_metrics(m: dict) -> None:
    """Input-level facts quoted in Methods, all fixed before any waterline, DEM or
    hindcast quantity existed: the setup-model fit on SWOT (scripts/a5_setup_model.py;
    D-065, D-066), the per-acquisition threshold rule's outcome (scripts/a3_thresholds.py;
    D-061), the EGM2008 - EIGEN-6C4 geoid difference (scripts/a7_geoid_difference.py;
    D-074) and the 25 km block assignment (scripts/a6_blocks.py; PREREG section 6)."""
    def put_dp(key, value, unit, source, note=None, dp=2):
        put(m, key, value, unit, source, note)
        if key in m:
            m[key]["decimals"] = dp

    sm = load("levels/setup_model.json")
    src = "results/levels/setup_model.json"
    if sm:
        inp = sm.get("inputs") or {}
        put(m, "surge.n_passes", inp.get("n_passes"), "passes", src, "SWOT passes after block-pass screening")
        put(m, "surge.n_block_passes", inp.get("n_block_passes"), "block-passes", src, "after screening")
        fitted = {k: v for k, v in (sm.get("sectors") or {}).items() if v.get("fitted")}
        put(m, "surge.n_sectors_fitted", len(fitted), "sectors", src)
        put(m, "surge.n_sectors_proxy", sum(1 for v in sm["sectors"].values() if v.get("proxy")), "sectors", src,
            "sectors without SWOT open-sea passes that use a neighbour's model (D-066)")
        r2 = [v["r2_calib"] for v in fitted.values() if v.get("r2_calib") is not None]
        sd = [v["resid_sd_calib_m"] for v in fitted.values() if v.get("resid_sd_calib_m") is not None]
        cnote = "calibration blocks (never used in the fit); fitted sectors only"
        if r2:
            put_dp("surge.r2_calib_min", min(r2), "", src, cnote)
            put_dp("surge.r2_calib_max", max(r2), "", src, cnote)
        if sd:
            put_dp("surge.resid_sd_calib_min_m", min(sd), "m", src, cnote)
            put_dp("surge.resid_sd_calib_max_m", max(sd), "m", src, cnote)
        for s, v in fitted.items():
            put(m, f"surge.n_train_passes.{s}", v.get("n_train_passes"), "passes", src)
            put(m, f"surge.n_calib_passes.{s}", v.get("n_calib_passes"), "passes", src)
            if v.get("r2_calib") is not None:
                put_dp(f"surge.r2_calib.{s}", v["r2_calib"], "", src, cnote)

    tr = load("qc/threshold_rule_summary.json")
    src = "results/qc/threshold_rule_summary.json"
    if tr:
        put(m, "thresholds.n_acquisitions", tr.get("n_acquisitions"), "acquisitions", src,
            "open-water-month acquisitions of stage-1 run 3")
        if tr.get("share_valid_open_water") is not None:
            put_dp("thresholds.valid_pct", round(100 * tr["share_valid_open_water"], 1), "%", src,
                   "acquisitions whose pooled Otsu threshold passes the D-061 rule", dp=0)

    gd = (load("provenance/geoid_difference.json").get("summary") or {})
    src = "results/provenance/geoid_difference.json"
    dn = gd.get("dN_m_domain_bounds") or {}
    if dn:
        gnote = "N(EGM2008) - N(EIGEN-6C4), ICGEM, tide-free, degree 2190, over the domain bounds"
        put_dp("datum.dN_min_m", dn.get("min_m"), "m", src, gnote)
        put_dp("datum.dN_max_m", dn.get("max_m"), "m", src, gnote)
        put_dp("datum.dN_sd_m", dn.get("sd_m"), "m", src, gnote, dp=3)
    off = gd.get("N_EGM2008_NGA_minus_ICGEM_region") or {}
    if off:
        put_dp("datum.nga_offset_m", off.get("mean_m"), "m", src,
               "NGA/PROJ EGM2008 minus ICGEM EGM2008 (the NGA height offset), mean over the product box")

    cov = load("study_area/zone_v6_cover_buffered.geojson")
    if cov.get("features"):
        put(m, "study_area.stage2_cover_km2", cov["features"][0]["properties"].get("area_km2"), "km2",
            "results/study_area/zone_v6_cover_buffered.geojson",
            "zone-v6 5 km tile cover extended seaward by one cell, clipped to the domain (D-060)")

    import csv
    from collections import Counter
    bp = RES / "study_area" / "blocks_25km.csv"
    if bp.exists():
        rows = list(csv.DictReader(bp.open(encoding="utf-8")))
        src = "results/study_area/blocks_25km.csv"
        put(m, "blocks.n_total", len(rows), "blocks", src, "25 km blocks whose centre lies in a sector")
        for k, n in Counter(r["split"] for r in rows).items():
            put(m, f"blocks.n.{k}", n, "blocks", src)


def datum_metrics(m: dict) -> None:
    """DAHITI -> BHS-77 offset from the CASPCOM post levels (scripts/a7_dahiti_bhs_offset.py;
    PREREG section 5; D-093). Used only to report levels in BHS-77; it cancels in every
    hindcast and validation metric (D-070, D-074)."""
    r = load("levels/dahiti_bhs_offset.json")
    src = "results/levels/dahiti_bhs_offset.json"
    if not r:
        return

    def put_dp(key, value, unit, note=None, dp=2):
        put(m, key, value, unit, src, note)
        if key in m:
            m[key]["decimals"] = dp

    p, tw = r.get("primary") or {}, r.get("posts_and_years_bootstrap") or {}
    note = "delta = H_BHS-77 - H_DAHITI; mean over the CASPCOM century posts of the per-post mean annual difference"
    put_dp("datum.bhs.delta_m", p.get("delta_m"), "m", note)
    put_dp("datum.bhs.sigma_years_m", p.get("sigma_m"), "m",
           f"moving-block bootstrap over years ({p.get('block_years')}-year blocks)", dp=3)
    put_dp("datum.bhs.sigma_posts_years_m", tw.get("sigma_m"), "m", "bootstrap over posts and year blocks", dp=2)
    put_dp("datum.bhs.between_post_sd_m", r.get("between_post_sd_m"), "m", "SD of the per-post offsets", dp=2)
    rng = r.get("between_post_range_m") or [None, None]
    put_dp("datum.bhs.post_min_m", rng[0], "m", "most negative per-post offset")
    put_dp("datum.bhs.post_max_m", rng[1], "m", "least negative per-post offset")
    put(m, "datum.bhs.n_posts", p.get("n_posts"), "posts", src)
    put(m, "datum.bhs.n_years", p.get("n_years"), "years", src, "calendar years with a difference at any post")
    put(m, "datum.bhs.n_post_years", p.get("n_post_years"), "post-years", src)
    put(m, "datum.bhs.first_year", str(p["first_year"]) if p.get("first_year") else None, "year", src)
    put(m, "datum.bhs.last_year", str(p["last_year"]) if p.get("last_year") else None, "year", src)
    dr = r.get("common_drift") or {}
    put_dp("datum.bhs.drift_m_per_decade", dr.get("trend_m_per_decade"), "m per decade",
           "OLS trend of the year-mean difference after removing post effects", dp=3)
    sens = r.get("sensitivities") or {}
    vals = [s["delta_m"] for s in sens.values() if s.get("delta_m") is not None]
    if vals:
        put_dp("datum.bhs.sens_min_m", min(vals), "m", "range over the D-093 sensitivities")
        put_dp("datum.bhs.sens_max_m", max(vals), "m", "range over the D-093 sensitivities")
    nc = sens.get("north_caspian_posts") or {}
    put_dp("datum.bhs.north_delta_m", nc.get("delta_m"), "m", "Peshnoy, Tyuleniy, Lagan and Kulaly posts")
    bs = r.get("bulletin_sea_mean_offset") or {}
    put_dp("datum.bhs.bulletin_delta_m", bs.get("mean_m"), "m",
           "CASPCOM bulletin 'mean level of the Caspian Sea' minus DAHITI, mean over the years with both")
    put(m, "datum.bhs.bulletin_n_years", bs.get("n_years"), "years", src)


PRODUCT_KEY = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
               "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}


def results_metrics(m: dict, tag: str = "main") -> None:
    """Headline results from the frozen result tables: H2 (results/validation/<tag>),
    H1 (results/hindcast/<tag>) and, when present, H3 (results/emergence/<tag>)."""
    import pandas as pd

    val = RES / "validation" / tag
    if (val / "h2_decision.json").exists():
        dec = load(f"validation/{tag}/h2_decision.json")
        for v, d in dec["decision"].items():
            put(m, f"h2.decision.{v}", d, "text", f"validation/{tag}/h2_decision.json")
        bs = load(f"validation/{tag}/block_size_check.json")
        put(m, "h2.block_size_km", bs["final_block_size_m"] / 1000, "km", f"validation/{tag}/block_size_check.json")
        if "h2.block_size_km" in m:
            m["h2.block_size_km"]["decimals"] = 0 if float(m["h2.block_size_km"]["value"]).is_integer() else 1
        for src in ("ATL06", "ATL08"):
            s = src.lower()
            acc = pd.read_csv(val / f"{src}_accuracy.csv")
            o = acc[acc.scope == "overall"].set_index("model")
            f = f"validation/{tag}/{src}_accuracy.csv"
            put(m, f"h2.{s}.n_test_points", int(o.loc["dem", "n"]), "points", f)
            put(m, f"h2.{s}.n_test_blocks", int(o.loc["dem", "n_blocks"]), "blocks", f)
            for metric in ("nmad", "rmse", "bias"):
                put(m, f"h2.{s}.dem.{metric}_m", round(float(o.loc["dem", metric]), 3), "m", f)
                for p, k in PRODUCT_KEY.items():
                    put(m, f"h2.{s}.{k}.{metric}_m", round(float(o.loc[f"{p}|matched", metric]), 3), "m", f)
            for sec, r in acc[(acc.model == "dem") & (acc.scope != "overall") & (acc.n > 0)].set_index("scope").iterrows():
                put(m, f"h2.{s}.dem.{sec}.nmad_m", round(float(r["nmad"]), 3), "m", f)
                put(m, f"h2.{s}.dem.{sec}.bias_m", round(float(r["bias"]), 3), "m", f)
            pb = pd.read_csv(val / f"{src}_paired_bootstrap.csv")
            pb = pb[(pb.scope == "overall") & (pb.variant == "matched")]
            fb = f"validation/{tag}/{src}_paired_bootstrap.csv"
            for r in pb.itertuples():
                k = f"h2.{s}.diff.{r.metric}.{PRODUCT_KEY[r.product]}"
                put(m, k + "_m", round(float(r.diff), 3), "m", fb, "DEM minus product")
                put(m, k + "_ci_low_m", round(float(r.ci_low), 3), "m", fb)
                put(m, k + "_ci_high_m", round(float(r.ci_high), 3), "m", fb)
            cov = pd.read_csv(val / f"{src}_coverage.csv").set_index("scope")
            fc = f"validation/{tag}/{src}_coverage.csv"
            for sec in cov.index:
                if pd.notna(cov.loc[sec, "coverage"]):
                    put(m, f"h2.{s}.coverage.{sec}", round(float(cov.loc[sec, "coverage"]), 3), "fraction", fc)
            for size in (25000, 12500):
                sd = val / f"sensitivity_{size}m"
                # key label as committed: 25000 -> "sens25km", 12500 -> "sens12km" (12.5 km blocks; the
                # source path in each record names the size exactly)
                lab = f"sens{size // 1000}km"
                if (sd / f"{src}_accuracy.csv").exists():
                    a2 = pd.read_csv(sd / f"{src}_accuracy.csv")
                    o2 = a2[a2.scope == "overall"].set_index("model")
                    fs = f"validation/{tag}/sensitivity_{size}m/{src}_accuracy.csv"
                    put(m, f"h2.{lab}.{s}.n_test_blocks", int(o2.loc["dem", "n_blocks"]), "blocks", fs)
                    put(m, f"h2.{lab}.{s}.dem.nmad_m", round(float(o2.loc["dem", "nmad"]), 3), "m", fs)
                    put(m, f"h2.{lab}.{s}.dem.rmse_m", round(float(o2.loc["dem", "rmse"]), 3), "m", fs)
                    c2 = pd.read_csv(sd / f"{src}_coverage.csv").set_index("scope")
                    put(m, f"h2.{lab}.{s}.coverage.overall", round(float(c2.loc["overall", "coverage"]), 3),
                        "fraction", fs.replace("accuracy", "coverage"))
                    # display items (tables_draft.tex T2, figures/ed_block_size.py): every model at this size
                    put(m, f"h2.{lab}.{s}.n_test_points", int(o2.loc["dem", "n"]), "points", fs)
                    put(m, f"h2.{lab}.{s}.dem.bias_m", round(float(o2.loc["dem", "bias"]), 3), "m", fs)
                    for p, k in PRODUCT_KEY.items():
                        for metric in ("nmad", "rmse", "bias"):
                            put(m, f"h2.{lab}.{s}.{k}.{metric}_m", round(float(o2.loc[f"{p}|matched", metric]), 3), "m", fs)
                    pb2 = pd.read_csv(sd / f"{src}_paired_bootstrap.csv")
                    pb2 = pb2[(pb2.scope == "overall") & (pb2.variant == "matched")]
                    fb2 = fs.replace("accuracy", "paired_bootstrap")
                    for r in pb2.itertuples():
                        k = f"h2.{lab}.{s}.diff.{r.metric}.{PRODUCT_KEY[r.product]}"
                        put(m, k + "_m", round(float(r.diff), 3), "m", fb2, "DEM minus product")
                        put(m, k + "_ci_low_m", round(float(r.ci_low), 3), "m", fb2)
                        put(m, k + "_ci_high_m", round(float(r.ci_high), 3), "m", fb2)
            display_h2_metrics(m, tag, src)
    hc = RES / "hindcast" / tag
    if (hc / "h1_decision.json").exists():
        so = load(f"hindcast/{tag}/shape_offsets.json")
        for p, k in PRODUCT_KEY.items():
            put(m, f"h1.shape_offset.{k}_m", so["products"][p]["offset_m"], "m", f"hindcast/{tag}/shape_offsets.json")
        mp = pd.read_csv(hc / "h1_mispredicts.csv")
        for (p, v), g in mp.groupby(["product", "variant"]):
            put(m, f"h1.mispredicts.{v}.{PRODUCT_KEY[p]}.n_sectors", int((g["mispredicts"] == True).sum()), "sectors",  # noqa: E712
                f"hindcast/{tag}/h1_mispredicts.csv")
        bt = pd.read_csv(hc / "h1_block_test.csv")
        for r in bt.itertuples():
            put(m, f"h1.block_test.{r.variant}.{PRODUCT_KEY[r.product]}.p", round(float(r.p_value), 4), "p",
                f"hindcast/{tag}/h1_block_test.csv")
            put(m, f"h1.block_test.{r.variant}.{PRODUCT_KEY[r.product]}.diff_km", round(float(r.statistic) / 1000, 2), "km",
                f"hindcast/{tag}/h1_block_test.csv", "median block displacement, deltaic or vegetated minus the rest")
        lb = pd.read_csv(hc / "displacement_by_level_band.csv")
        lb = lb[lb.variant == "shape"]
        names = {"(-29.0, -28.3]": "below_m28p3", "(-28.3, -27.7]": "m28p3_m27p7", "(-27.7, -27.0]": "m27p7_m27p0",
                 "(-27.0, -26.3]": "m27p0_m26p3", "(-26.3, -25.5]": "above_m26p3"}
        # records, not itertuples: itertuples renames the column "srtm15p_v2.7" (not an
        # identifier), which silently dropped SRTM15+ from these keys
        for r in lb.to_dict("records"):
            for p, k in PRODUCT_KEY.items():
                put(m, f"h1.disp_band.{names.get(r['band'], r['band'])}.{k}_km",
                    round(float(r[p]) / 1000, 2) if p in r and pd.notna(r[p]) else None,
                    "km", f"hindcast/{tag}/displacement_by_level_band.csv")
        # PREREG section 1: the whole-domain area bias is reported beside the per-sector test
        # (Bocharov et al. 2026 found whole-sea chart areas within 5.75% of satellite areas)
        ex = pd.read_csv(hc / "extent.csv")
        ex = ex[(ex["sector"] == "domain") & (ex["epoch"] != "pooled")]
        fe = f"hindcast/{tag}/extent.csv"
        for (p, v), g in ex.groupby(["product", "variant"]):
            rel = 100 * g["area_bias_rel_total"].astype(float)
            b = f"h1.domain_area_bias.{v}.{PRODUCT_KEY[p]}"
            put(m, f"{b}.median_pct", round(float(rel.median()), 1), "%", fe,
                "median over (epoch, level) units of the domain total relative area bias")
            put(m, f"{b}.max_abs_pct", round(float(rel.abs().max()), 1), "%", fe, "largest |relative area bias|")
            put(m, f"{b}.median_iou", round(float(g["iou_total"].astype(float).median()), 3), "IoU", fe)
            put(m, f"{b}.n_units", int(len(g)), "units", fe)
        es = load(f"hindcast/{tag}/etopo2022_sources_emerged_band.json")
        for name, share in es.get("share", {}).items():
            key = "fabdem" if "FABDEM" in name else "gebco_lake" if "GEBCO" in name else "other"
            put(m, f"h1.etopo2022.source_share.{key}", round(100 * share, 1), "%", f"hindcast/{tag}/etopo2022_sources_emerged_band.json")
        nz = pd.read_csv(hc / "noise_floor.csv")
        nz = nz[(nz.kind == "pooled_level") & (nz.sector != "all") & (nz.n > 0)]
        for sec, g in nz.groupby("sector"):
            put(m, f"h1.noise_p95.{sec}_km", round(float(g["p95_m"].median()) / 1000, 2), "km", f"hindcast/{tag}/noise_floor.csv",
                "median over levels of the noise-floor 95th percentile")
        display_h1_metrics(m, tag)
    h2_additional_metrics(m, tag)
    display_input_metrics(m)
    if (RES / "emergence" / tag / "h3_decision.json").exists():
        h3_metrics(m, tag)


H3_LAYERS = {"osm_platforms": ("platforms", "platforms and wells newly onshore"),
             "osm_pipelines": ("pipelines", "pipeline length that emerges (km)"),
             "osm_channels": ("channels", "channel and canal length that emerges (km)"),
             "osm_ports": ("ports", "ports, piers and ferry terminals losing the sea within the threshold"),
             "osm_settlements": ("settlements", "settlements losing the sea within the threshold"),
             "osm_wetlands": ("osm_wetlands", "OSM wetland area that emerges (km2)"),
             "worldcover2021_herbaceous_wetland": ("worldcover", "ESA WorldCover 2021 herbaceous wetland that emerges (km2)"),
             "gwl_fcs30_2020_wetland": ("gwl", "GWL_FCS30 2020 wetland that emerges (km2)"),
             "glwd_v2_wetland": ("glwd", "GLWD v2 wetland that emerges (km2)")}


def _scen(s: float) -> str:
    """Scenario label in keys: 0.5 -> "0p5", 1.0 -> "1"."""
    return f"{float(s):g}".replace(".", "p")


def _and(items) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def h3_metrics(m: dict, tag: str) -> None:
    """H3 values from the frozen tables of scripts/a8_emergence.py and caspian.h3
    (results/emergence/<tag>/; D-081, D-090), read for Fig. 5 and Table 3: the 2026
    level, the emerged area of each surface with its conformal interval (domain,
    "emergence" functional), the verdicts, exposure at every scenario and the scenario
    dates of every pathway. Direct reads; nothing is recomputed."""
    import math

    import pandas as pd

    em = RES / "emergence" / tag
    run = load(f"emergence/{tag}/run.json")
    lv = run.get("level_2026") or {}
    put(m, "h3.level_2026_m", lv.get("level_m"), "m (DAHITI reference)", f"emergence/{tag}/run.json",
        f"{lv.get('method')} of the {lv.get('n_records')} DAHITI records of 2026")
    if "h3.level_2026_m" in m:
        m["h3.level_2026_m"]["decimals"] = 2
    put(m, "h3.level_2026.n_records", lv.get("n_records"), "records", f"emergence/{tag}/run.json")
    cst = run.get("constants") or {}
    for key, name in (("PORT_ACCESS_M", "h3.port_access_km"), ("COAST_M", "h3.coast_km")):
        if cst.get(key) is not None:
            put(m, name, int(cst[key] / 1000) if float(cst[key] / 1000).is_integer() else cst[key] / 1000, "km",
                f"emergence/{tag}/run.json", "access threshold of the exposure rule (scripts/a8_emergence.py)")
    sc = run.get("scenarios_m") or []
    if sc:
        put(m, "h3.scenarios_text", _and(f"{s:g}" for s in sc), "text", f"emergence/{tag}/run.json",
            "scenario depths below the 2026 level (m)")
    e = pd.read_csv(em / "emergence.csv")
    fe = f"emergence/{tag}/emergence.csv"
    d = e[(e["region"] == "domain") & (e["functional"] == "emergence")]
    for r in d.itertuples():
        b = f"h3.emergence.{PRODUCT_KEY[r.product]}.{r.surface}.{_scen(r.scenario_m)}"
        for col, name in (("estimate_km2", "estimate"), ("lo_km2", "lo"), ("hi_km2", "hi")):
            v = float(getattr(r, col))
            put(m, f"{b}.{name}_km2", round(v, 1) if math.isfinite(v) else None, "km2", fe,
                "domain, emerged area below the surface's own 2026 shoreline; lo/hi: split-conformal 90% interval")
    for (p, s_), g in d.groupby(["product", "surface"]):
        put(m, f"h3.n_scores.{PRODUCT_KEY[p]}.{s_}", int(g["n_scores"].iloc[0]), "scores", fe,
            "held-out unit errors calibrating the interval (domain, emergence)")
    dec = load(f"emergence/{tag}/h3_decision.json")
    for p, v in (dec.get("domain") or {}).items():
        k = PRODUCT_KEY[p]
        fd = f"emergence/{tag}/h3_decision.json"
        put(m, f"h3.verdict.{k}", v["verdict"], "text", fd, dec.get("primary"))
        rel = [s for s in str(v.get("relevant_scenarios", "")).split(",") if s]
        put(m, f"h3.relevant_scenarios.{k}", _and(rel) if rel else "none", "text", fd,
            "near-term scenarios (m) at which the correction is decision-relevant")
        put(m, f"h3.n_relevant.{k}", v.get("n_relevant"), "scenarios", fd)
        put(m, "h3.n_near_term", v.get("n_near_term"), "scenarios", fd)
    sm = pd.read_csv(em / "h3_summary.csv")
    sm = sm[sm["functional"] == "emergence"]
    for r in sm.itertuples():
        put(m, f"h3.verdict.{PRODUCT_KEY[r.product]}.{r.region}", r.verdict, "text", f"emergence/{tag}/h3_summary.csv",
            "per region, emergence functional")
    x = pd.read_csv(em / "exposure.csv")
    fx = f"emergence/{tag}/exposure.csv"
    x = x[(x["region"] == "domain") & x["layer"].isin(list(H3_LAYERS))]
    for lay, g in x.groupby("layer"):
        short, what = H3_LAYERS[lay]
        put(m, f"h3.exposure.{short}.n_features", int(g["n_features"].iloc[0]), "features", fx,
            "features of the layer inside the domain")
    for r in x.itertuples():     # every scenario (brief section 15: exposure per scenario)
        short, what = H3_LAYERS[r.layer]
        count = r.rule in ("onshore", "access")
        b = f"h3.exposure.{short}.{PRODUCT_KEY[r.product]}.{r.surface}.{_scen(r.scenario_m)}"
        unit = "features" if count else ("km" if r.rule == "length" else "km2")
        for col, name in (("amount", "amount"), ("amount_lo", "lo"), ("amount_hi", "hi")):
            v = float(getattr(r, col))
            if not math.isfinite(v):
                continue
            put(m, f"{b}.{name}", int(round(v)) if count else round(v, 1), unit, fx,
                what + "; lo/hi at the equivalent levels of the area-interval bounds")
            if r.rule == "length":
                m[f"{b}.{name}"]["decimals"] = 1   # km to 0.1; areas follow the km2 convention (0 dp)
    tr = pd.read_csv(em / "transfer.csv")       # PREREG section 10 leave-one-epoch-out transfer test
    ft = f"emergence/{tag}/transfer.csv"
    for r in tr[(tr["region"] == "domain") & (tr["form"] == "linear_hold")].itertuples():
        b = f"h3.transfer.{PRODUCT_KEY[r.product]}.{r.fold}"
        put(m, f"{b}.n_heldout_units", int(r.n_heldout_usable), "units", ft, "held-out usable units of the fold")
        put(m, f"{b}.shift_error_median_m", round(float(r.shift_transfer_median_m), 2), "m", ft,
            "median predicted minus held-out matching shift")
        put(m, f"{b}.share_extrapolated", round(float(r.share_extrapolated), 2), "fraction", ft,
            "held-out units deeper than the fold's training range")
        put(m, f"{b}.emergence_err_corrected_km2", round(float(r.emergence_median_abs_corrected_km2), 1), "km2", ft,
            "median absolute emergence error of the corrected product at the held-out epoch's scored units")
        put(m, f"{b}.emergence_err_uncorrected_km2", round(float(r.emergence_median_abs_uncorrected_km2), 1), "km2",
            ft, "the same for the uncorrected product")
        put(m, f"{b}.emergence_corrected_better", "yes" if r.emergence_median_abs_corrected_km2
            < r.emergence_median_abs_uncorrected_km2 else "no", "text", ft)
    sd = pd.read_csv(em / "scenario_dates.csv")
    fs = f"emergence/{tag}/scenario_dates.csv"
    for r in sd.itertuples():
        pid = "".join(ch if ch.isalnum() else "_" for ch in str(r.pathway_id).lower())
        pid = "_".join(t for t in pid.split("_") if t)
        if bool(r.reached) and pd.notna(r.date):
            val = str(int(math.floor(float(r.date))))
        else:
            val = str(r.note) if pd.notna(r.note) else "not reached"
        put(m, f"h3.date.{pid}.{_scen(r.delta_m)}", val, "year", fs,
            f"{r.role} pathway ({r.source}, {r.scenario}, {r.statistic}); calendar year in which its fall after 2026 "
            f"reaches the depth (anchor {r.anchor})")


def display_h2_metrics(m: dict, tag: str, src: str) -> None:
    """H2 values read for the display items (figures/fig3_dem_accuracy.py, figures/ed_*.py,
    manuscript/sections/tables_draft.tex): per-sector accuracy of every model, coverage
    with its intervals, the training-block constants, and the vegetated population (brief
    section 10 check 9). Direct reads of the committed tables; nothing is recomputed."""
    import pandas as pd

    val = RES / "validation" / tag
    s = src.lower()
    f = f"validation/{tag}/{src}_accuracy.csv"
    acc = pd.read_csv(val / f"{src}_accuracy.csv")
    for r in acc[(acc.scope != "overall") & (acc.n > 0)].itertuples():
        k = "dem" if r.model == "dem" else (PRODUCT_KEY[r.model.split("|")[0]] if r.model.endswith("|matched") else None)
        if k is None:
            continue
        if k == "dem":
            put(m, f"h2.{s}.n_points.{r.scope}", int(r.n), "points", f, "test-block points in the sector")
            put(m, f"h2.{s}.n_blocks.{r.scope}", int(r.n_blocks), "blocks", f, "test blocks with points in the sector")
            put(m, f"h2.{s}.dem.{r.scope}.rmse_m", round(float(r.rmse), 3), "m", f)
        else:
            for metric in ("nmad", "rmse", "bias"):
                put(m, f"h2.{s}.{k}.{r.scope}.{metric}_m", round(float(getattr(r, metric)), 3), "m", f, "matched variant")
    cov = pd.read_csv(val / f"{src}_coverage.csv")
    fc = f"validation/{tag}/{src}_coverage.csv"
    for r in cov[cov.n > 0].itertuples():
        base = f"h2.{s}.coverage.{r.scope}"
        put(m, base + "_ci_low", round(float(r.ci_low), 3), "fraction", fc, "binomial 95% interval (points as independent)")
        put(m, base + "_ci_high", round(float(r.ci_high), 3), "fraction", fc, "binomial 95% interval (points as independent)")
        put(m, base + "_boot_low", round(float(r.boot_low), 3), "fraction", fc, "test-block bootstrap 95% interval")
        put(m, base + "_boot_high", round(float(r.boot_high), 3), "fraction", fc, "test-block bootstrap 95% interval")
        put(m, base + "_n_blocks", int(r.n_blocks), "blocks", fc)
        put(m, base + "_consistent", "consistent" if bool(r.consistent_with_nominal) else "not consistent", "text",
            fc, "0.90 inside the binomial 95% interval (PREREG section 8)")
    sectors = sorted(k.split(".")[-1] for k in m if k.startswith("study_area.sector_km2."))
    seen = set(acc.loc[(acc.model == "dem") & (acc.n > 0), "scope"])
    missing = [x for x in sectors if x not in seen]
    put(m, f"h2.{s}.sectors_without_test_points", ", ".join(missing) if missing else "none", "text", f,
        "sectors with no test-block points (no per-sector accuracy or coverage)")
    dc = load(f"validation/{tag}/{src}_datum_constant.json").get("constant") or {}
    put(m, f"h2.{s}.datum_constant_m", round(float(dc["constant_m"]), 3) if dc else None, "m",
        f"validation/{tag}/{src}_datum_constant.json", "training-block median of DEM - H (D-069, D-078)")
    put(m, f"h2.{s}.datum_constant_sigma_m", round(float(dc["sigma_boot_m"]), 3) if dc else None, "m",
        f"validation/{tag}/{src}_datum_constant.json", "training-block bootstrap SD")
    po = load(f"validation/{tag}/{src}_product_offsets.json").get("product_offsets") or {}
    for p, k in PRODUCT_KEY.items():
        if p in po:
            put(m, f"h2.{s}.offset.{k}_m", round(float(po[p]["offset_m"]), 3), "m",
                f"validation/{tag}/{src}_product_offsets.json", "training-block median of y_ref - product (matched variant)")
    vg = RES / "validation" / f"{tag}_vegetated"
    if (vg / f"{src}_accuracy.csv").exists():
        a3 = pd.read_csv(vg / f"{src}_accuracy.csv")
        o3 = a3[a3.scope == "overall"].set_index("model")
        fv = f"validation/{tag}_vegetated/{src}_accuracy.csv"
        put(m, f"h2.veg.{s}.n_test_points", int(o3.loc["dem", "n"]), "points", fv)
        put(m, f"h2.veg.{s}.n_test_blocks", int(o3.loc["dem", "n_blocks"]), "blocks", fv)
        for metric in ("nmad", "rmse", "bias"):
            put(m, f"h2.veg.{s}.dem.{metric}_m", round(float(o3.loc["dem", metric]), 3), "m", fv)
            for p, k in PRODUCT_KEY.items():
                put(m, f"h2.veg.{s}.{k}.{metric}_m", round(float(o3.loc[f"{p}|matched", metric]), 3), "m", fv)
        c3 = pd.read_csv(vg / f"{src}_coverage.csv").set_index("scope")
        put(m, f"h2.veg.{s}.coverage.overall", round(float(c3.loc["overall", "coverage"]), 3), "fraction",
            fv.replace("accuracy", "coverage"))


ADDITIONAL_SOURCES = (("GLO30", "glo30"), ("GEDI", "gedi"))


def h2_additional_metrics(m: dict, tag: str) -> None:
    """The additional independent checks of the waterline DEM (D-091): Copernicus GLO-30
    (PREREG section 2) and GEDI L2A (Amendment 2), evaluated on the H2 blocks with the H2
    metrics (results/validation/<tag>_glo30/, <tag>_gedi/, scripts/a6_validate_dem.py).
    Keys h2.glo30.* and h2.gedi.* mirror the h2.atl06/atl08 families, plus per-sector
    paired differences and the split counts; h2.gedi_night.* is the Amendment 2
    night-only sensitivity. The H2 decision itself is ICESat-2's (D-086). Direct reads of
    the committed tables; nothing is recomputed; NaN values are not written."""
    import math

    import pandas as pd

    def putf(key, value, unit, source, note=None):
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return
        put(m, key, value, unit, source, note)

    def r3(v):
        return round(float(v), 3)

    def block(s, rel, src, per_sector=True):
        d = RES / rel
        acc = pd.read_csv(d / f"{src}_accuracy.csv")
        o = acc[acc.scope == "overall"].set_index("model")
        f = f"{rel}/{src}_accuracy.csv"
        putf(f"h2.{s}.n_test_points", int(o.loc["dem", "n"]), "points", f)
        putf(f"h2.{s}.n_test_blocks", int(o.loc["dem", "n_blocks"]), "blocks", f)
        for metric in ("nmad", "rmse", "bias"):
            putf(f"h2.{s}.dem.{metric}_m", r3(o.loc["dem", metric]), "m", f)
            for p, k in PRODUCT_KEY.items():
                putf(f"h2.{s}.{k}.{metric}_m", r3(o.loc[f"{p}|matched", metric]), "m", f, "matched variant")
        if per_sector:
            for sec, r in acc[(acc.model == "dem") & (acc.scope != "overall") & (acc.n > 0)].set_index("scope").iterrows():
                putf(f"h2.{s}.dem.{sec}.nmad_m", r3(r["nmad"]), "m", f)
                putf(f"h2.{s}.dem.{sec}.bias_m", r3(r["bias"]), "m", f)
        pb = pd.read_csv(d / f"{src}_paired_bootstrap.csv")
        pb = pb[(pb.variant == "matched") & (pb.n_points > 0)]
        fb = f"{rel}/{src}_paired_bootstrap.csv"
        for r in pb.itertuples():
            if r.scope != "overall" and not per_sector:
                continue
            k = (f"h2.{s}.diff.{r.metric}.{PRODUCT_KEY[r.product]}" if r.scope == "overall"
                 else f"h2.{s}.diff.{r.scope}.{r.metric}.{PRODUCT_KEY[r.product]}")
            putf(k + "_m", r3(r.diff), "m", fb, "DEM minus product, matched variant")
            putf(k + "_ci_low_m", r3(r.ci_low), "m", fb, "test-block bootstrap 95% interval")
            putf(k + "_ci_high_m", r3(r.ci_high), "m", fb, "test-block bootstrap 95% interval")
        cov = pd.read_csv(d / f"{src}_coverage.csv")
        fc = f"{rel}/{src}_coverage.csv"
        for r in cov[cov.n > 0].itertuples():
            if r.scope != "overall" and not per_sector:
                continue
            base = f"h2.{s}.coverage.{r.scope}"
            putf(base, r3(r.coverage), "fraction", fc)
            putf(base + "_ci_low", r3(r.ci_low), "fraction", fc, "binomial 95% interval (points as independent)")
            putf(base + "_ci_high", r3(r.ci_high), "fraction", fc, "binomial 95% interval (points as independent)")
            putf(base + "_boot_low", r3(r.boot_low), "fraction", fc, "test-block bootstrap 95% interval")
            putf(base + "_boot_high", r3(r.boot_high), "fraction", fc, "test-block bootstrap 95% interval")
            putf(base + "_n_blocks", int(r.n_blocks), "blocks", fc)
        cnt = load(f"{rel}/{src}_counts.json")
        fn = f"{rel}/{src}_counts.json"
        by = {}
        for r in cnt["by_split_sector"]:
            by[r["split"]] = by.get(r["split"], 0) + int(r["n"])
        for sp in ("train", "calib", "test"):
            putf(f"h2.{s}.n_points_split.{sp}", by.get(sp), "points", fn)
            putf(f"h2.{s}.n_blocks_split.{sp}", cnt["n_blocks_with_points"].get(sp), "blocks", fn,
                 "blocks of the split holding points of this source")
        es = d / f"{src}_etopo2022_sources.json"          # scripts/a6_etopo_sources.py (D-091)
        if es.exists():
            fe = f"{rel}/{src}_etopo2022_sources.json"
            for name, r in json.loads(es.read_text(encoding="utf-8"))["by_etopo_source"].items():
                k = {"FABDEM": "fabdem", "GEBCO Lake Depths": "gebco_lake"}.get(name, "other")
                b = f"h2.{s}.etopo_src.{k}"
                putf(b + ".share", round(100 * float(r["share"]), 1), "%", fe, "test-block points by ETOPO 2022 source")
                putf(b + ".n_points", int(r["n_points"]), "points", fe)
                putf(b + ".n_blocks", int(r["n_blocks"]), "blocks", fe)
                for metric in ("nmad", "rmse", "bias"):
                    putf(f"{b}.dem.{metric}_m", r3(r["dem"][metric]), "m", fe)
                    putf(f"{b}.etopo2022.{metric}_m", r3(r["etopo_2022_matched"][metric]), "m", fe, "matched variant")
        rc = d / f"{src}_rule_check.json"
        if rc.exists():
            for v, st in json.loads(rc.read_text(encoding="utf-8"))["outcome"].items():
                putf(f"h2.{s}.rule_check.{v}", st, "text", f"{rel}/{src}_rule_check.json",
                     "PREREG section 7 rule applied descriptively; the H2 decision is ICESat-2's (D-086)")

    for src, s in ADDITIONAL_SOURCES:
        rel = f"validation/{tag}_{s}"
        if not (RES / rel / f"{src}_accuracy.csv").exists():
            continue
        block(s, rel, src)
        display_h2_metrics(m, f"{tag}_{s}", src)
        pj = load(f"{rel}/prepare.json")
        putf(f"h2.{s}.n_points_prepared", pj.get("n_points"), "points", f"{rel}/prepare.json",
             "dry points with every model column, before the bare-seabed population (D-084)")
        if s == "gedi":
            putf("h2.gedi.n_night_prepared", pj.get("n_night"), "points", f"{rel}/prepare.json")
        sub = RES / rel / "subset_night"
        if (sub / f"{src}_accuracy.csv").exists():
            block(f"{s}_night", f"{rel}/subset_night", src, per_sector=False)


def display_h1_metrics(m: dict, tag: str) -> None:
    """H1 values read for the display items (figures/fig4_audit.py, figures/ed_*.py):
    per-sector level counts behind each "mispredicts" call, the sector summary of the
    per-level median displacement, the block-test counts and the unit counts."""
    import numpy as np
    import pandas as pd

    hc = RES / "hindcast" / tag
    f = f"hindcast/{tag}/h1_mispredicts.csv"
    for r in pd.read_csv(hc / "h1_mispredicts.csv").itertuples():
        k = PRODUCT_KEY[r.product]
        put(m, f"h1.n_exceeding.{r.variant}.{k}.{r.sector}", int(r.n_exceeding), "levels", f,
            "evaluable levels whose median displacement exceeds the sector noise-floor 95th percentile")
        put(m, f"h1.n_evaluable.{r.variant}.{k}.{r.sector}", int(r.n_evaluable), "levels", f)
    det = pd.read_csv(hc / "h1_mispredicts_detail.csv")
    det = det[det["evaluable"] == True]  # noqa: E712
    fd = f"hindcast/{tag}/h1_mispredicts_detail.csv"
    for (p, v, sec), g in det.groupby(["product", "variant", "sector"]):
        put(m, f"h1.disp_median.{v}.{PRODUCT_KEY[p]}.{sec}_km", round(float(g["median"].median()) / 1000, 2), "km", fd,
            "median over the evaluable levels of the per-level median displacement (pooled epochs), as quoted in D-087")
        put(m, f"h1.noise_p95_evaluable.{v}.{PRODUCT_KEY[p]}.{sec}_km", round(float(g["p95_noise"].median()) / 1000, 2),
            "km", fd, "median over the same evaluable levels of the sector noise-floor 95th percentile")
    for r in pd.read_csv(hc / "h1_block_test.csv").itertuples():
        b = f"h1.block_test.{r.variant}.{PRODUCT_KEY[r.product]}"
        put(m, b + ".n_blocks", int(r.n_blocks), "blocks", f"hindcast/{tag}/h1_block_test.csv", "25 km blocks with values")
        put(m, b + ".n_group", int(r.n_group), "blocks", f"hindcast/{tag}/h1_block_test.csv", "deltaic or vegetated sectors")
        put(m, b + ".n_rest", int(r.n_rest), "blocks", f"hindcast/{tag}/h1_block_test.csv")
    # block_h1.csv rows whose median is not a distance (negative: the decoded int32 minimum,
    # 2026-10-21 figure review); counted so the display items can state them
    bh = pd.read_csv(hc / "block_h1.csv")
    min_n = int(load("params.json").get("params.h1_min_block_n", {}).get("value", 20))
    for (p, v), g in bh.groupby(["product", "variant"]):
        bad = g["median_abs_m"] < 0
        put(m, f"h1.block_h1.n_undefined.{v}.{PRODUCT_KEY[p]}", int(bad.sum()), "blocks", f"hindcast/{tag}/block_h1.csv",
            "rows with a negative median_abs_m (not a distance)")
        put(m, f"h1.block_h1.n_undefined_used.{v}.{PRODUCT_KEY[p]}", int((bad & (g["n"] >= min_n)).sum()), "blocks",
            f"hindcast/{tag}/block_h1.csv", "the same, among blocks with at least params.h1_min_block_n distances")
    ex = pd.read_csv(hc / "extent.csv", usecols=["product", "variant", "epoch", "bin", "sector", "area_obs_km2"])
    dom = ex[(ex["sector"] == "domain") & (ex["area_obs_km2"] > 0)]
    put(m, "h1.n_units_domain", int(dom[["epoch", "bin"]].drop_duplicates().shape[0]), "units",
        f"hindcast/{tag}/extent.csv", "(epoch, level) units with observed sea inside the domain")
    for e, g in dom[["epoch", "bin"]].drop_duplicates().groupby("epoch"):
        put(m, f"h1.n_units_domain.{e}", int(len(g)), "units", f"hindcast/{tag}/extent.csv")
    nz = pd.read_csv(hc / "noise_floor.csv")
    ev = nz[(nz["kind"] == "e1_vs_e23") & (nz["sector"] != "all") & (nz["n"] > 0)]
    put(m, "h1.e1_vs_e23.n_rows", int(len(ev)), "sector-levels", f"hindcast/{tag}/noise_floor.csv",
        "E1 against E2 + E3 shoreline comparisons with distances (kind e1_vs_e23)")
    put(m, "h1.e1_vs_e23.n_infinite", int((~np.isfinite(ev["median_m"].astype(float))).sum()), "sector-levels",
        f"hindcast/{tag}/noise_floor.csv", "of those, comparisons with an infinite median (no counterpart shoreline)")
    so = load(f"hindcast/{tag}/shape_offsets.json")
    put(m, "h1.shape_fit.n_units", len(so.get("units", [])) or None, "units", f"hindcast/{tag}/shape_offsets.json",
        "(epoch, level) units in the shape-offset fit")
    lb = pd.read_csv(hc / "displacement_by_level_band.csv")
    edges = sorted({float(x) for b in lb["band"] for x in b.strip("(]").split(",")})
    for i, v in enumerate(edges):
        put(m, f"h1.disp_band.edge_{i}_m", v, "m (DAHITI reference)", f"hindcast/{tag}/displacement_by_level_band.csv",
            "level-band edges (scripts/a7_level_bands.BANDS), lowest first")
        m[f"h1.disp_band.edge_{i}_m"]["decimals"] = 1


def display_input_metrics(m: dict) -> None:
    """The surge-screening sensitivity of D-077 (results/levels/screening_sensitivity.json),
    read for Fig. 2c and its Supplementary Figure."""
    ss = load("levels/screening_sensitivity.json")
    if not ss:
        return
    src = "results/levels/screening_sensitivity.json"
    put(m, "screening.n_blocks", ss.get("n_blocks"), "blocks", src, "sampled training blocks (tag sensB)")
    put(m, "screening.chosen_tau", "no screening" if ss.get("chosen") == "inf" else ss.get("chosen"), "text", src)
    for tau, r in (ss.get("results") or {}).items():
        lab = "inf" if tau == "inf" else f"{float(tau):.2f}".replace(".", "p")
        put(m, f"screening.S_m.{lab}", r["S_m"], "m", src, "median over block-bins of the median odd/even displacement")
        put(m, f"screening.se_m.{lab}", round(float(r["se_m"]), 1), "m", src, "bootstrap SE")
        put(m, f"screening.n_block_bins.{lab}", int(r["n_block_bins"]), "block-bins", src)
        for k in (f"screening.S_m.{lab}", f"screening.se_m.{lab}"):
            m[k]["decimals"] = 0  # metres on a 60 m grid


def generality_metrics(m: dict) -> None:
    """A9 generality test (brief section 11; PREREG Amendment 16; D-094). From
    results/generality/<basin>/h1_decision.json when the basin was run (scripts/g1_decide.py),
    and from its status.json (why it was not run, or the run's cost and inputs).
    The inputs are literal paths, so that tests/test_makefile.py checks that they are
    prerequisites of results/metrics.json. An infinite median (no predicted waterline at
    most levels) is stored as "inf" in the decision file and kept as a category here."""
    inputs = {"great_salt_lake": (load("generality/great_salt_lake/h1_decision.json"),
                                  load("generality/great_salt_lake/status.json")),
              "lake_urmia": ({}, load("generality/lake_urmia/status.json"))}

    def num(key, x, unit, src, note=None):
        if isinstance(x, str) and x in ("inf", "-inf"):
            put(m, key, "infinite" if x == "inf" else "-infinite", "category", src, note)
        else:
            put(m, key, x, unit, src, note)

    def flag(key, x, src, note=None):
        put(m, key, None if x is None else ("true" if x else "false"), "category", src, note)

    for basin, (dec, st) in inputs.items():
        key = f"generality.{basin}"
        if dec:
            src = f"results/generality/{basin}/h1_decision.json"
            v = dec["verdict"]
            put(m, f"{key}.verdict", v["verdict"], "category", src, "PREREG Amendment 16 item 9; shape variant decides")
            put(m, f"{key}.n_products_mispredict", v["n_mispredict"], "products", src)
            put(m, f"{key}.n_products_evaluable", v["n_evaluable"], "products", src)
            put(m, f"{key}.positive_control", v["positive_control"], "category", src)
            put(m, f"{key}.verdict_total_variant", dec["verdict_total_variant"]["verdict"], "category", src)
            for prod, vv in dec["products"].items():
                for var, e in vv.items():
                    k = f"{key}.{prod}.{var}"
                    put(m, f"{k}.n_sectors_mispredict", e["n_sectors_mispredict"], "sectors", src)
                    put(m, f"{k}.n_sectors_evaluable", e["n_sectors_evaluable"], "sectors", src)
                    flag(f"{k}.mispredicts_in_basin", e.get("in_basin"), src)
                    for arm, x in e.get("arms", {}).items():
                        num(f"{k}.{arm}.median_displacement_m", x.get("median_displacement_m"), "m", src,
                            "median over evaluable levels of the per-level median absolute displacement")
                        num(f"{k}.{arm}.median_noise_p95_m", x.get("median_noise_p95_m"), "m", src)
                        put(m, f"{k}.{arm}.n_evaluable_levels", x.get("n_evaluable"), "levels", src)
                        put(m, f"{k}.{arm}.n_observed_levels", x.get("n_observed_levels"), "levels", src)
                        put(m, f"{k}.{arm}.n_exceeding_levels", x.get("n_exceeding"), "levels", src)
                        flag(f"{k}.{arm}.mispredicts", x.get("mispredicts"), src)
                        flag(f"{k}.{arm}.mispredicts_pooled_floor", x.get("mispredicts_pooled_floor"), src,
                             "sensitivity: one noise floor per arm, pooled over all levels")
                        for side in ("above", "below"):
                            num(f"{k}.{arm}.median_displacement_{side}_2013_m",
                                x.get(f"median_displacement_{side}_2013_m"), "m", src,
                                "descriptive (Amendment 16 item 9): levels above/below the arm's 2013-07-01 level")
            for arm, x in dec.get("arms", {}).items():
                num(f"{key}.{arm}.noise_floor_p95_median_m", x.get("noise_floor_p95_median_m"), "m", src)
                put(m, f"{key}.{arm}.n_noise_levels", x.get("n_noise_levels"), "levels", src)
            for arm, x in dec.get("control_arms", {}).items():
                num(f"{key}.control.{arm}.median_displacement_m", x.get("median_displacement_m"), "m", src)
                num(f"{key}.control.{arm}.median_noise_p95_m", x.get("median_noise_p95_m"), "m", src)
                put(m, f"{key}.control.{arm}.n_evaluable_levels", x.get("n_evaluable"), "levels", src)
                put(m, f"{key}.control.{arm}.n_exceeding_levels", x.get("n_exceeding"), "levels", src)
                flag(f"{key}.control.{arm}.mispredicts", x.get("mispredicts"), src)
            for arm, offs in (dec.get("shape_offsets") or {}).items():
                for prod, r in offs.items():
                    num(f"{key}.shape_offset.{arm}.{prod}_m", r.get("offset_m"), "m", src,
                        "D-070 constant (positive: product heights too high)")
            for arm, lv in (dec.get("level_2013_07_01_m") or {}).items():
                num(f"{key}.{arm}.level_2013_07_01_m", lv, "m (NAVD 88)", src)
        src = f"results/generality/{basin}/status.json"
        if not dec:
            put(m, f"{key}.verdict", st.get("verdict", "not run: no record"), "category", src,
                "PREREG Amendment 16 item 9; docs/gates/STOP-2026-10-21-generality.md")
            put(m, f"{key}.positive_control", st.get("positive_control"), "category", src)
        for k in ("eecu_h_spent", "eecu_h_cap", "eecu_h_stage2", "eecu_h_projected_stage2_area_scaled",
                  "eecu_h_projected_stage2_request_scaled", "eecu_h_projected_total_low", "eecu_h_projected_total_high"):
            put(m, f"{key}.{k}", st.get(k), "EECU-h", src)
        put(m, f"{key}.n_scenes", st.get("n_scenes"), "scenes", src)
        put(m, f"{key}.n_acquisitions", st.get("n_acquisitions"), "acquisitions", src)
        for arm, x in st.get("arms", {}).items():
            put(m, f"{key}.{arm}.ngvd29_to_navd88_m", x.get("ngvd29_to_navd88_m"), "m", src, "VERTCON 3.0 at the gauge")
            if f"{key}.{arm}.ngvd29_to_navd88_m" in m:
                m[f"{key}.{arm}.ngvd29_to_navd88_m"]["decimals"] = 3
            put(m, f"{key}.{arm}.e1_end", x.get("e1_end"), "date", src)
            put(m, f"{key}.{arm}.n_scenes_with_level", x.get("n_scenes_with_level"), "scenes", src)
            put(m, f"{key}.{arm}.n_groups_planned", x.get("n_groups_planned"), "groups", src)
            put(m, f"{key}.{arm}.n_bins_planned", x.get("n_bins_planned"), "bins", src)


def main() -> int:
    m: dict = {}
    inv = load("inventory/inventory_meta.json")
    src = "results/inventory/inventory_meta.json"
    put(m, "inventory.n_scenes_total", inv.get("n_scenes_total"), "scenes", src)
    for k, v in inv.get("n_by_sensor", {}).items():
        put(m, f"inventory.n_scenes.{k}", v, "scenes", src)
    for k, v in inv.get("n_cloud_below_60_by_sensor", {}).items():
        put(m, f"inventory.n_scenes_cc_lt60.{k}", v, "scenes", src)
    if inv.get("landsat_path_rows"):
        put(m, "inventory.n_landsat_path_rows", len(inv["landsat_path_rows"]), "path/rows", src)
    put(m, "inventory.n_s2_tiles", inv.get("n_s2_tiles"), "tiles", src)
    for k, v in inv.get("first_by_sensor", {}).items():
        put(m, f"inventory.first.{k}", v[:10], "date", src)
    for k, v in inv.get("last_by_sensor", {}).items():
        put(m, f"inventory.last.{k}", v[:10], "date", src)

    study_area_metrics(m)
    input_stage_metrics(m)
    datum_metrics(m)

    dq = load("qc/dahiti_summary.json")
    src = "results/qc/dahiti_summary.json"
    put(m, "levels.dahiti.n_records", dq.get("n_records"), "records", src)
    put(m, "levels.dahiti.first", (dq.get("first") or "")[:10] or None, "date", src)
    put(m, "levels.dahiti.last", (dq.get("last") or "")[:10] or None, "date", src)
    put(m, "levels.dahiti.min_m", dq.get("level_min_m"), "m (DAHITI reference)", src)
    put(m, "levels.dahiti.max_m", dq.get("level_max_m"), "m (DAHITI reference)", src)
    put(m, "levels.dahiti.date_of_min", (dq.get("time_of_min") or "")[:10] or None, "date", src)
    put(m, "levels.dahiti.date_of_max", (dq.get("time_of_max") or "")[:10] or None, "date", src)
    if dq.get("level_max_m") is not None and dq.get("level_min_m") is not None:
        put(m, "levels.dahiti.range_m", round(dq["level_max_m"] - dq["level_min_m"], 3), "m", src,
            "highest single record minus lowest single record (each one noisy record; use "
            "levels.dahiti.smoothed.range_m for the fall of the sea)")
    sm = dq.get("smoothed") or {}
    if sm:
        note = f"{sm['rule']} (D-021 correction)"
        put(m, "levels.dahiti.smoothed.window_records", sm.get("k"), "records", src, note)
        put(m, "levels.dahiti.smoothed.max_m", sm.get("level_max_m"), "m (DAHITI reference)", src, note)
        put(m, "levels.dahiti.smoothed.min_m", sm.get("level_min_m"), "m (DAHITI reference)", src, note)
        put(m, "levels.dahiti.smoothed.date_of_max", (sm.get("time_of_max") or "")[:10] or None, "date", src, note)
        put(m, "levels.dahiti.smoothed.date_of_min", (sm.get("time_of_min") or "")[:10] or None, "date", src, note)
        put(m, "levels.dahiti.smoothed.range_m", sm.get("range_m"), "m", src,
            "maximum minus minimum of the running median; " + note)

    put(m, "levels.dahiti.sigma_formal_max_m", dq.get("sigma_max_m"), "m", src, "formal wse_u")
    rpy = {int(k): v for k, v in (dq.get("records_per_year") or {}).items()}
    # the two sampling periods compared in Methods (Water levels, gap filling); the years
    # are stored as text so that the macros print 1993, not 1\,993
    dense_years, sparse_years = (1993, 2021), (2022, 2025)
    sparse = [rpy[y] for y in range(sparse_years[0], sparse_years[1] + 1) if y in rpy]
    if sparse:
        put(m, "levels.dahiti.records_per_year_2022_2025_min", min(sparse), "records per year", src)
        put(m, "levels.dahiti.records_per_year_2022_2025_max", max(sparse), "records per year", src)
        put(m, "levels.dahiti.sparse_period", f"{sparse_years[0]}--{sparse_years[1]}", "years", src,
            "years of the records_per_year_2022_2025 metrics")
    dense = [rpy[y] for y in range(dense_years[0], dense_years[1] + 1) if y in rpy]
    if dense:
        put(m, "levels.dahiti.records_per_year_1993_2021_median", int(sorted(dense)[len(dense) // 2]),
            "records per year", src)
        put(m, "levels.dahiti.dense_period", f"{dense_years[0]}--{dense_years[1]}", "years", src,
            "years of the records_per_year_1993_2021 metric")
    put(m, "levels.dahiti.max_gap_days", dq.get("max_gap_days"), "days", src, "fractional days")
    put(m, "levels.dahiti.median_gap_days", dq.get("median_gap_days"), "days", src, "fractional days")
    put(m, "levels.dahiti.n_gaps_over_60_days", dq.get("n_gaps_over_60_days"), "gaps", src)

    xc = load("qc/level_crosscheck_w20d.json")
    src = "results/qc/level_crosscheck_w20d.json"
    put(m, "levels.hydroweb.n_records", xc.get("hydroweb_records"), "records", src)
    put(m, "levels.crosscheck.n_pairs", xc.get("n_pairs"), "pairs", src)
    put(m, "levels.crosscheck.window_days", xc.get("max_half_gap_days"), "days", src)
    put(m, "levels.hydroweb.last", (xc.get("hydroweb_last") or "")[:10] or None, "date", src)
    put(m, "levels.crosscheck.offset_median_m", xc.get("offset_median_m"), "m", src, "DAHITI minus Hydroweb")
    put(m, "levels.crosscheck.nmad_m", xc.get("diff_nmad_m"), "m", src)
    put(m, "levels.crosscheck.drift_m_per_decade", xc.get("drift_m_per_decade"), "m per decade", src)

    sl = load("levels/scene_levels_summary.json")
    src = "results/levels/scene_levels_summary.json"
    def put_dp(key, value, unit, source, note=None, dp=3):
        put(m, key, value, unit, source, note)
        if key in m:
            m[key]["decimals"] = dp

    joint = "joint estimate with q from cross-validated interpolation residuals (D-018 correction)"
    put(m, "levels.dahiti_noise_floor_m", sl.get("dahiti_noise_floor_m"), "m", src, joint)
    put(m, "levels.se.dahiti_noise_floor_m", sl.get("dahiti_noise_floor_se_m"), "m", src,
        "moving-block bootstrap SE of levels.dahiti_noise_floor_m")
    put(m, "levels.hydroweb_noise_floor_m", sl.get("hydroweb_noise_floor_m"), "m", src,
        "Hydroweb record noise with q fixed to the DAHITI estimate")
    put(m, "levels.q_m2_per_day", sl.get("q_dahiti_m2_per_day"), "m2 per day", src,
        "effective diffusion of the linear-interpolation error (absorbs sub-sampling variability); " + joint)
    put(m, "levels.se.q_m2_per_day", sl.get("q_dahiti_se_m2_per_day"), "m2 per day", src,
        "moving-block bootstrap SE of levels.q_m2_per_day")
    put(m, "levels.epoch_E1_end", (sl.get("epoch_E1_end") or "")[:10] or None, "date", src,
        sl.get("epoch_E1_end_rule"))
    put(m, "levels.max_gap_days", sl.get("max_gap_days"), "days", src,
        "gap rule: caspian.levels default fixed in commit aabacee before PREREG v1.0 and before the DAHITI "
        "data arrived; not in PREREG v1.0; stated for the record in PREREG Amendment 4 item 2 (D-038)")
    off = sl.get("hydroweb_offset") or {}
    if off:
        note = off.get("method")
        hw_ = off.get("half_window_years")
        put(m, "levels.gapfill.half_window_years", int(hw_) if hw_ is not None and float(hw_).is_integer() else hw_,
            "years", src, note)
        put(m, "levels.gapfill.offset_median_m", off.get("filled_scenes_offset_median_m"), "m", src,
            "median over gap-filled scenes of the moving-median offset applied; " + note)
        put_dp("levels.gapfill.offset_min_m", off.get("filled_scenes_offset_min_m"), "m", src, note)
        put_dp("levels.gapfill.offset_max_m", off.get("filled_scenes_offset_max_m"), "m", src, note)
        put(m, "levels.gapfill.offset_sigma_p50_m", off.get("filled_scenes_offset_sigma_p50_m"), "m", src,
            "median over gap-filled scenes of the common offset 1-sigma (SE of the moving median and the "
            "window-local tau); " + (off.get("tau_rule") or ""))
        put_dp("levels.gapfill.offset_sigma_p50_record_tau_m", off.get("filled_scenes_offset_sigma_p50_with_record_tau_m"),
               "m", src, "the same with the record-wide tau instead (superseded; first D-020 correction)")
        put(m, "levels.gapfill.resid_nmad_m", off.get("pair_residual_nmad_m"), "m", src,
            "NMAD of the pair differences about the moving median (per-record random term)")
        put_dp("levels.gapfill.tau_m", off.get("excess_sd_tau_m"), "m", src,
            "record-wide (all years pooled) year-to-year offset variability beyond sampling noise (method of "
            "moments); the floor of the applied tau, not the applied value (see levels.gapfill.tau_p50_m)")
        put_dp("levels.gapfill.tau_p50_m", off.get("filled_scenes_tau_p50_m"), "m", src,
               "median over gap-filled scenes of the applied window-local tau; " + (off.get("tau_rule") or ""))
        put_dp("levels.gapfill.tau_min_m", off.get("filled_scenes_tau_min_m"), "m", src, "applied tau, minimum")
        put_dp("levels.gapfill.tau_max_m", off.get("filled_scenes_tau_max_m"), "m", src, "applied tau, maximum")
        put(m, "levels.gapfill.tau_min_years", off.get("tau_min_years"), "years", src,
            "minimum number of years in the local tau window")
        put(m, "levels.gapfill.n_scenes_tau_at_floor", off.get("n_filled_scenes_tau_at_record_floor"), "scenes",
            src, "gap-filled scenes whose local tau is at or below the record-wide floor")
    dg = sl.get("diagnostics") or {}
    if dg:
        put(m, "levels.gapfill.n_scenes_other_bin_pooled_offset", dg.get("n_hydroweb_scenes_other_bin_with_pooled_offset"),
            "scenes", src, "gap-filled scenes whose 0.10 m bin (origin -30 m, DAHITI reference) differs if the pooled "
            "1992-2026 offset is used instead of the moving median")
        put(m, "levels.diag.n_scenes_above_smoothed_max", dg.get("n_scenes_above_smoothed_max"), "scenes", src,
            "scenes whose interpolated level exceeds the running-median maximum")
        put(m, "levels.diag.n_scenes_between_raw_and_smoothed_max", dg.get("n_scenes_between_raw_and_smoothed_max_dates"),
            "scenes", src, "scenes whose epoch (E1 or E2) depends on raw versus running-median maximum")
        nz = dg.get("dahiti_neighbour_z") or {}
        put_dp("levels.diag.neighbour_z_threshold", nz.get("z_threshold"), "sigma", src,
               "diagnostic threshold only; no record is removed", dp=1)
        put(m, "levels.diag.neighbour_z_n", nz.get("n"), "records", src)
        put(m, "levels.diag.neighbour_z_n_gt_2_5", nz.get("n_abs_gt_2_5"), "records", src,
            "records with |z| > 2.5 against their neighbours under the fitted noise and diffusion")
        put_dp("levels.diag.neighbour_z_expected", nz.get("expected_gaussian"), "records", src,
               "expected count of |z| > 2.5 under the Gaussian model", dp=1)
        put_dp("levels.diag.raw_max_record_z", nz.get("z_of_raw_max_record"), "sigma", src, dp=1)
        es = dg.get("e1_end_sensitivity") or {}
        for r in es.get("by_k", []):
            k_ = r["k"]
            nt = "single-record maximum" if k_ == 1 else f"maximum of the {k_}-record running median"
            put(m, f"levels.diag.e1_end.k{k_}.date", (r.get("time_of_max") or "")[:10] or None, "date", src,
                f"E1 end with the {nt} (sensitivity; applied k = {es.get('applied_k')})")
            put(m, f"levels.diag.e1_end.k{k_}.n_E1", r.get("n_E1_scenes"), "scenes", src,
                f"levelled E1 scenes with the {nt}")
        top = es.get("running_median_top_two_m")
        if top:
            put_dp("levels.diag.e1_end.top_two_margin_m", round(top[0] - top[1], 4), "m", src,
                   "margin between the two highest running-median values, which sets the applied E1 date")
        hq = dg.get("hydroweb_joint_free_q") or {}
        if hq:
            put(m, "levels.diag.hydroweb_free_q_m2_per_day", hq.get("q_m2_per_day"), "m2 per day", src,
                "Hydroweb's own joint (sigma, q) fit; not applied (DAHITI's q is used for the Hydroweb bridge)")
            put_dp("levels.diag.hydroweb_free_q_sigma_change_max_m", hq.get("filled_scenes_sigma_change_max_abs_m"),
                   "m", src, hq.get("note"))
        mdp = dg.get("offset_fill_year_max_departure") or {}
        if mdp:
            # calendar years are stored as text so that the macro prints 2023, not 2\,023
            put(m, "levels.gapfill.max_year_departure.year", str(mdp["year"]) if mdp.get("year") else None,
                "year", src, mdp.get("note"))
            put_dp("levels.gapfill.max_year_departure.m", mdp.get("departure_m"), "m", src, mdp.get("note"))
        c5 = dg.get("offset_check_5d_main_fill_year") or {}
        if c5:
            c5n = ("year with the most gap-filled scenes: median DAHITI - shifted Hydroweb at the +/-5-day epochs "
                   "(in-sample: a subset of the pairs that set the offset)")
            put(m, "levels.gapfill.check5.year", str(c5["year"]) if c5.get("year") else None, "year", src, c5n)
            put(m, "levels.gapfill.check5.n_filled_scenes", c5.get("n_filled_scenes"), "scenes", src, c5n)
            put(m, "levels.gapfill.check5.n_pairs", c5.get("n_pairs_5d"), "pairs", src, c5n)
            put_dp("levels.gapfill.check5.median_resid_m", c5.get("median_resid_m"), "m", src, c5n)
            put_dp("levels.gapfill.check5.offset_sigma_p50_m", c5.get("offset_sigma_p50_m"), "m", src, c5n)
    for e, rng_ in (sl.get("level_range_by_epoch_m") or {}).items():
        if rng_:
            put(m, f"levels.epoch_level_range.{e}.min_m", rng_[0], "m (DAHITI reference)", src)
            put(m, f"levels.epoch_level_range.{e}.max_m", rng_[1], "m (DAHITI reference)", src)
    for s, v in (sl.get("sigma_m_p50_by_source") or {}).items():
        put_dp(f"levels.scene_sigma_p50_by_source.{s}", v, "m", src)
    if sl.get("by_sensor_source"):
        src_tot = {}
        for sensor, d in sl["by_sensor_source"].items():
            for s, n in d.items():
                src_tot[s] = src_tot.get(s, 0) + int(n)
        for s, n in src_tot.items():
            put(m, f"levels.n_scenes_by_source.{s}", n, "scenes", src)
    if sl.get("by_sensor_epoch"):
        ep_tot = {}
        for sensor, d in sl["by_sensor_epoch"].items():
            for e, n in d.items():
                ep_tot[e] = ep_tot.get(e, 0) + int(n)
        for e, n in ep_tot.items():
            put(m, f"levels.n_scenes_by_epoch.{e}", n, "scenes", src)
    if sl.get("sigma_m_percentiles_p50_p90_p99"):
        p50, p90, p99 = sl["sigma_m_percentiles_p50_p90_p99"]
        put(m, "levels.scene_sigma_p50_m", p50, "m", src)
        put(m, "levels.scene_sigma_p90_m", p90, "m", src)
        put(m, "levels.scene_sigma_p99_m", p99, "m", src)

    # Values quoted from the literature (refs/literature_values.csv: value, unit, DOI,
    # location, verbatim quote); they enter the text through the same macros.
    import csv
    lit = ROOT / "refs" / "literature_values.csv"
    if lit.exists():
        for r in csv.DictReader(lit.open(encoding="utf-8")):
            raw = r["value"].strip()
            dp = len(raw.split(".")[1]) if "." in raw else 0
            v = float(raw) if dp else int(raw)
            put(m, r["key"], v, r["unit"], f"doi:{r['doi']} ({r['location']})")
            m[r["key"]]["decimals"] = dp  # keep the precision written in the source

    # Analysis parameters exported from the code constants (tools/export_params.py).
    # Identifiers that are integers but not quantities (the random seed) are stored as
    # text, so that the macro prints 20261020, not 20\,261\,020: make_macros puts
    # thousands separators on every number and has no per-key rule for integers.
    as_text = {"params.seed"}
    for k, rec in load("params.json").items():
        v = str(rec["value"]) if k in as_text and rec["value"] is not None else rec["value"]
        put(m, k, v, rec["unit"], rec["source"])
        if k in m and "decimals" in rec:
            m[k]["decimals"] = rec["decimals"]

    results_metrics(m)

    generality_metrics(m)

    out = RES / "metrics.json"
    out.write_text(json.dumps(dict(sorted(m.items())), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(m)} metrics to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
