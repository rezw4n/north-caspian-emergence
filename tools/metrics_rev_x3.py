"""Metrics of the journal-revision analyses (PREREG Amendment 20 items 1-8; D-161).

Loaded by tools/build_metrics.revision_metrics, which calls add(m); the file name sorts after
the earlier revision modules. Every key starts with "rev3."; each record is {"value", "unit",
"source", "note"?, "decimals"?}. Direct reads of results/revision3/ (scripts/rev3_marine.py,
rev3_nearterm.py, rev3_lidar.py, rev3_tables.py); nothing is recomputed here except minima,
maxima and counts over those tables. A missing input adds no key.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results" / "revision3"
REL = "results/revision3"
PK = {"gebco_2026": "gebco2026", "gebco_2022": "gebco2022", "etopo_2022": "etopo2022",
      "srtm15p_v2.7": "srtm15p", "etopo1": "etopo1"}
SK = {"west": "west", "volga": "volga", "interfluve": "interfluve", "ural": "ural", "northeast": "northeast",
      "komsomolets": "komsomolets", "open_shelf": "openshelf"}
CHECKED = ["west", "interfluve", "northeast", "open_shelf"]
FAILING = ["gebco_2026", "gebco_2022", "srtm15p_v2.7", "etopo1"]        # the four grids that fail
NOTE = "PREREG Amendment 20 (exploratory, post-review)"


def _put(m, key, value, unit, source, note=None, decimals=None):
    if value is None:
        return
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return
    rec = {"value": value, "unit": unit, "source": source}
    if note:
        rec["note"] = note
    if decimals is not None:
        rec["decimals"] = decimals
    m[f"rev3.{key}"] = rec


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _span(m, key, values, unit, source, note=None, decimals=None):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return
    _put(m, f"{key}.min", min(v), unit, source, note, decimals)
    _put(m, f"{key}.max", max(v), unit, source, note, decimals)


# --- item 1: the marine domain --------------------------------------------------------------------

def marine(m):
    mc = _json(R3 / "marine" / "marine_constants.json")
    if mc:
        src = f"{REL}/marine/marine_constants.json"
        _put(m, "marine.n_deltaic_sectors", len(mc["deltaic_sectors"]), "sectors", src, "the sectors registered as deltaic")
        _put(m, "marine.n_marine_sectors", len(mc["marine_sectors"]), "sectors", src)
        n, nm = mc["fit_region_cells"], mc["fit_region_cells_marine"]
        _put(m, "marine.fit_cells_deltaic_pct", 100.0 * (n - nm) / n, "%", src,
             "share of the known training-block cells of the area fit that lie in the deltaic sectors", 0)
        for p, v in mc["products"].items():
            _put(m, f"marine.{PK[p]}.c_marine_m", v["c_marine_m"], "m", src, "area-fitted offset on the marine domain", 2)
            _put(m, f"marine.{PK[p]}.c_change_m", v["c_change_m"], "m", src, "c_marine minus the primary offset", 2)
        ch = [v["c_change_m"] for v in mc["products"].values()]
        _put(m, "marine.c_change_min_m", min(ch), "m", src, "over the five grids", 2)
        _put(m, "marine.c_change_max_m", max(ch), "m", src, "over the five grids", 2)
        _put(m, "marine.c_change_abs_max_m", max(abs(x) for x in ch), "m", src, "largest |change| over the five grids", 2)
        _put(m, "marine.n_grids_unchanged", sum(1 for x in ch if x == 0), "grids", src, "grids whose offset does not change")
    s = _json(R3 / "marine" / "summary.json")
    if s and s.get("all_bins"):
        src = f"{REL}/marine/summary.json"
        _put(m, "marine.n_cases", s["n_cases"], "cases", src, "grid by sector")
        _put(m, "marine.n_changed", s["n_changed"], "cases", src, "verdicts that differ from the primary at c_marine")
        _put(m, "marine.n_cases_checked", s["n_cases_checked"], "cases", src, "grid by checked sector")
        _put(m, "marine.n_changed_checked", s["n_changed_checked"], "cases", src)
        _put(m, "marine.replica_tolerance_m", s["checks"]["replica_levels"]["tolerance_m"], "m", src,
             "the replica reproduces every per-level displacement and floor of the primary within this")
    vc = _csv(R3 / "marine" / "verdict_changes.csv")
    if vc is not None and s and s.get("all_bins"):
        src = f"{REL}/marine/verdict_changes.csv"
        for p, g in vc.groupby("product"):
            for cfg in ("shape", "marine"):
                col = g[f"mispredicts_{cfg}"].astype(bool)
                _put(m, f"marine.{PK[p]}.{cfg}.n_all7", int(col.sum()), "sectors", src,
                     "sectors in which the grid fails the shoreline test")
                _put(m, f"marine.{PK[p]}.{cfg}.n_checked4", int(col[g["checked"]].sum()), "sectors", src)
        for r in vc[vc["changed"]].itertuples():
            _put(m, f"marine.changed.{PK[r.product]}.{SK[r.sector]}",
                 "passes" if not bool(r.mispredicts_marine) else "fails", "", src, "the verdict at c_marine")
            _put(m, f"marine.changed.{PK[r.product]}.{SK[r.sector]}.n_exceeding", int(r.n_exceeding_marine), "levels", src)
            _put(m, f"marine.changed.{PK[r.product]}.{SK[r.sector]}.n_exceeding_primary", int(r.n_exceeding_shape),
                 "levels", src)
            _put(m, f"marine.changed.{PK[r.product]}.{SK[r.sector]}.n_evaluable", int(r.n_evaluable_marine), "levels", src)
    sm = _csv(R3 / "marine" / "sector_medians.csv")
    if sm is not None and s and s.get("all_bins"):
        src = f"{REL}/marine/sector_medians.csv"
        for cfg in ("shape", "marine"):
            d = sm[sm["variant"] == cfg]
            for sec in CHECKED:
                g = d[(d["sector"] == sec) & d["product"].isin(FAILING)]
                _put(m, f"marine.headline.{cfg}.{SK[sec]}.d_km_min", g["d_median_m"].min() / 1000, "km", src,
                     "sector-median displacement over the four failing grids", 2)
                _put(m, f"marine.headline.{cfg}.{SK[sec]}.d_km_max", g["d_median_m"].max() / 1000, "km", src, None, 2)
                _put(m, f"marine.headline.{cfg}.{SK[sec]}.dq_median_min", g["dq_median"].min(), "ratio", src, None, 1)
                _put(m, f"marine.headline.{cfg}.{SK[sec]}.dq_median_max", g["dq_median"].max(), "ratio", src, None, 1)
            for r in d.itertuples():
                _put(m, f"marine.{PK[r.product]}.{cfg}.{SK[r.sector]}.d_km", r.d_median_m / 1000, "km", src, None, 2)
        # the reporting rule of Amendment 20: the largest move of a failing grid's sector-median
        # displacement, in units of the sector's median floor
        w = sm[sm["product"].isin(FAILING)].pivot_table(index=["sector", "product"], columns="variant", values="d_median_m")
        q = sm[sm["variant"] == "shape"].set_index(["sector", "product"])["q_median_m"]
        move = ((w["marine"] - w["shape"]).abs() / q.reindex(w.index)).reset_index(name="r")
        for key, secs in (("westinterfluve", ["west", "interfluve"]), ("coasts3", ["west", "interfluve", "northeast"])):
            sub = move[move["sector"].isin(secs)]
            _put(m, f"marine.four.{key}.d_move_over_q_max", sub["r"].max(), "ratio", src,
                 "largest |change of the sector-median displacement| over the sector's median floor, four failing grids", 2)
            top = sub.loc[sub["r"].idxmax()]
            _put(m, f"marine.four.{key}.d_move_max_grid", PK[top["product"]], "", src, "the grid of that largest change")
            _put(m, f"marine.four.{key}.d_move_max_sector", SK[top["sector"]], "", src, "the sector of that largest change")
            _put(m, f"marine.four.{key}.d_move_max_from_km", w.loc[(top["sector"], top["product"]), "shape"] / 1000, "km",
                 src, "its sector-median displacement at the primary offset", 1)
            _put(m, f"marine.four.{key}.d_move_max_to_km", w.loc[(top["sector"], top["product"]), "marine"] / 1000, "km",
                 src, "and at the marine-domain offset", 1)
    ab = _csv(R3 / "marine" / "area_bias.csv")
    if ab is not None and s and s.get("all_bins"):      # never from a partial (test) pass
        src = f"{REL}/marine/area_bias.csv"
        note = "median over the (epoch, level) units of the relative total area bias"
        for (region, cfg), g in ab.groupby(["region", "config"]):
            for r in g.itertuples():
                _put(m, f"marine.area_bias.{region}.{cfg}.{PK[r.product]}.median_pct", r.median_pct, "%", src, note, 1)
            _put(m, f"marine.area_bias.{region}.{cfg}.median_min_pct", g["median_pct"].min(), "%", src, note, 0)
            _put(m, f"marine.area_bias.{region}.{cfg}.median_max_pct", g["median_pct"].max(), "%", src, note, 0)
            _put(m, f"marine.area_bias.{region}.{cfg}.median_absmax_pct", g["median_pct"].abs().max(), "%", src, note, 0)


def blocks(m):
    b = _csv(R3 / "marine" / "block_shares.csv")
    s = _json(R3 / "marine" / "summary.json")
    if b is None or not (s and s.get("all_bins")):
        return
    src = f"{REL}/marine/block_shares.csv"
    _put(m, "blocks.min_n", s["min_block_n"], "distances", f"{REL}/marine/summary.json",
         "least number of distances for a block to be evaluable (the block test's minimum)")
    note = "25 km blocks: D_b the pooled median displacement, Q_b the 95th percentile of the pooled floor distances"
    for scope, g in b.groupby("scope"):
        k = scope.replace("sector:", "sector.").replace("open_shelf", "openshelf")
        for r in g.itertuples():
            _put(m, f"blocks.{k}.{PK[r.product]}.n_blocks", int(r.n_blocks), "blocks", src, note)
            _put(m, f"blocks.{k}.{PK[r.product]}.n_exceeding", int(r.n_exceeding), "blocks", src, note)
            _put(m, f"blocks.{k}.{PK[r.product]}.share_pct", 100.0 * r.share_exceeding, "%", src, note, 0)
        four = g[g["product"].isin(FAILING)]
        if len(four):
            _put(m, f"blocks.{k}.four.share_min_pct", 100.0 * four["share_exceeding"].min(), "%", src,
                 "over the four failing grids", 0)
            _put(m, f"blocks.{k}.four.share_max_pct", 100.0 * four["share_exceeding"].max(), "%", src, None, 0)
            _put(m, f"blocks.{k}.four.n_blocks_min", int(four["n_blocks"].min()), "blocks", src)
            _put(m, f"blocks.{k}.four.n_blocks_max", int(four["n_blocks"].max()), "blocks", src)


def nearterm(m):
    c = _csv(R3 / "nearterm" / "marine_curves.csv")
    k = _csv(R3 / "nearterm" / "marine_constants.csv")
    run = _json(R3 / "nearterm" / "run.json")
    if c is None or k is None or not run:
        return
    src = f"{REL}/nearterm/marine_curves.csv"
    for reg, v in run["S_obs_km2"].items():
        _put(m, f"nearterm.S_obs.{reg}_km2", v, "km2", f"{REL}/nearterm/run.json", "observed 2026 sea")
    _put(m, "nearterm.S_obs.deltaic_pct", 100.0 * run["S_obs_km2"]["deltaic"] / run["S_obs_km2"]["domain"], "%",
         f"{REL}/nearterm/run.json", "share of the observed 2026 sea that lies in the deltaic sectors", 0)
    for const, g in k.groupby("constant"):
        _put(m, f"nearterm.{const}.c26_min_m", g["c26_m"].min(), "m", f"{REL}/nearterm/marine_constants.csv", None, 2)
        _put(m, f"nearterm.{const}.c26_max_m", g["c26_m"].max(), "m", f"{REL}/nearterm/marine_constants.csv", None, 2)
    e = c[c["quantity"] == "E"]
    for (const, reg, d), g in e.groupby(["constant", "region", "delta_m"]):
        dk = f"{d:g}".replace(".", "p")
        _put(m, f"nearterm.{const}.{reg}.E.{dk}.min_km2", g["km2"].min(), "km2", src, "over the five grids")
        _put(m, f"nearterm.{const}.{reg}.E.{dk}.max_km2", g["km2"].max(), "km2", src)
        for r in g.itertuples():
            _put(m, f"nearterm.{const}.{reg}.E.{dk}.{PK[r.product]}_km2", r.km2, "km2", src)
    mm = c[c["quantity"] == "misfit"]
    for (const, reg), g in mm.groupby(["constant", "region"]):
        _put(m, f"nearterm.{const}.{reg}.M.min_km2", g["km2"].min(), "km2", src, "misfit over the five grids")
        _put(m, f"nearterm.{const}.{reg}.M.max_km2", g["km2"].max(), "km2", src)
    # the share of the registered whole-domain first metre that lies in the deltaic sectors
    one = e[(e["constant"] == "registered") & (e["delta_m"] == 1.0)].pivot(index="product", columns="region", values="km2")
    sh = 100.0 * one["deltaic"] / one["domain"]
    _put(m, "nearterm.registered.deltaic_share_of_E1.min_pct", sh.min(), "%", src,
         "share of the registered first-metre increment that lies in the deltaic sectors", 0)
    _put(m, "nearterm.registered.deltaic_share_of_E1.max_pct", sh.max(), "%", src, None, 0)


def exposure(m):
    t = _csv(R3 / "exposure" / "exposure_split_first_metre.csv")
    if t is None:
        return
    src = f"{REL}/exposure/exposure_split_first_metre.csv"
    note = "first metre, re-referenced to the observed 2026 shoreline (registered constants), over the five grids"
    for r in t.itertuples():
        sub = "" if r.subset == "all" else f".{r.subset}"
        unit = "km2" if r.kind == "polygon" else ""
        dec = 0 if r.kind == "polygon" or r.layer in ("osm_ports", "osm_settlements", "osm_platforms") else 1
        for reg in ("domain", "deltaic", "marine"):
            _put(m, f"exposure.{r.layer}{sub}.{reg}.min", getattr(r, f"{reg}_min"), unit, src, note, dec)
            _put(m, f"exposure.{r.layer}{sub}.{reg}.max", getattr(r, f"{reg}_max"), unit, src, note, dec)
    # per grid: the share of the first-metre protected area that lies in the deltaic sectors
    g = _csv(R3 / "exposure" / "exposure_split.csv")
    if g is not None:
        w = g[(g["layer"] == "wdpa_protected_areas_noksrz") & (g["delta_m"] == 1.0) & (g["surface"] == "observed_reref")
              & (g["subset"] == "all") & (g["domain"] > 0)]
        if len(w) == len(PK):
            _span(m, "exposure.wdpa_protected_areas_noksrz.deltaic_share_pct", 100.0 * w["deltaic"] / w["domain"], "%",
                  f"{REL}/exposure/exposure_split.csv", "per grid, first metre: deltaic sectors over the whole domain", 0)


def band(m):
    t = _csv(R3 / "band" / "band_by_sector.csv")
    s = _json(R3 / "band" / "summary.json")
    if t is None or not s:
        return
    src = f"{REL}/band/band_by_sector.csv"
    for r in t.itertuples():
        _put(m, f"band.{SK[r.sector]}.band_km2", r.band_km2, "km2", src, "band of 1995-96 to 2025-26 (250 m extents)")
        _put(m, f"band.{SK[r.sector]}.inside_2000_km2", r.band_inside_2000_outline_km2, "km2", src)
        _put(m, f"band.{SK[r.sector]}.landward_km2", r.band_landward_of_2000_outline_km2, "km2", src)
        _put(m, f"band.{SK[r.sector]}.landward_pct", 100.0 * r.share_landward, "%", src,
             "share of the sector's band landward of the Caspian's February 2000 shoreline (HydroLAKES polygon)", 0)
        if "share_landward_glwd_raster" in t.columns:
            _put(m, f"band.raster.{SK[r.sector]}.landward_pct", 100.0 * r.share_landward_glwd_raster, "%", src,
                 "the same with the outline as the amendment named it (GLWD v2 raster, 15 arc-seconds)", 0)
    src = f"{REL}/band/summary.json"
    _put(m, "band.domain_km2", s["band_km2_domain"], "km2", src)
    _put(m, "band.deltaic_km2", s["band_km2_deltaic"], "km2", src)
    _put(m, "band.not_seabed_km2", s["band_km2_not_drawn_as_seabed"], "km2", src,
         "band of the deltaic sectors landward of the February 2000 shoreline (HydroLAKES polygon): not drawn as seabed")
    _put(m, "band.not_seabed_pct", 100.0 * s["share_of_domain_band_not_drawn_as_seabed"], "%", src, None, 0)
    if "band_km2_not_drawn_as_seabed_glwd_raster" in s:
        _put(m, "band.raster.not_seabed_km2", s["band_km2_not_drawn_as_seabed_glwd_raster"], "km2", src,
             "the same area with the outline as the amendment named it (GLWD v2 raster, 15 arc-seconds)")
        _put(m, "band.raster.not_seabed_pct", 100.0 * s["share_of_domain_band_not_drawn_as_seabed_glwd_raster"], "%",
             src, None, 0)
        _put(m, "band.raster.cell_arcsec", 15, "arc-seconds", src, "cell size of the GLWD v2 grids")
    o = _json(ROOT / "results" / "study_area" / "caspian_feb2000.json")
    if o:
        osrc = "results/study_area/caspian_feb2000.json"
        _put(m, "band.outline.hylak_id", o["lake"]["Hylak_id"], "", osrc, "HydroLAKES identifier of the Caspian Sea")
        _put(m, "band.outline.n_vertices", o["polygon"]["n_vertices_in_window"], "vertices", osrc,
             "vertices of the Caspian polygon inside the study window")
        _put(m, "band.outline.poly_src", o["lake"]["Poly_src"], "", osrc, "source of the polygon in HydroLAKES")
    rel = {k: v for k, v in s["swot_departure_rel_all_m"].items() if v is not None}
    _put(m, "band.swot.volga_rel_m", rel["volga"], "m", src,
         "SWOT water surface of the Volga delta sector above the all-sector surface", 2)
    _put(m, "band.swot.volga_n_passes", s["swot_n_passes"]["volga"], "passes", src)
    others = [abs(v) for k, v in rel.items() if k != "volga"]
    _put(m, "band.swot.others_abs_max_m", max(others), "m", src, "largest |between-sector departure| of the other sectors", 2)
    mon = s["volga_minus_all_by_month_m"]
    _put(m, "band.swot.volga_month_min_m", min(mon.values()), "m", src, "Volga sector minus all sectors, by calendar month", 2)
    _put(m, "band.swot.volga_month_max_m", max(mon.values()), "m", src, None, 2)
    npm = s["volga_passes_by_month"]
    _put(m, "band.swot.volga_passes_month_min", min(npm.values()), "passes", src)
    _put(m, "band.swot.volga_passes_month_max", max(npm.values()), "passes", src)
    st = s["static_share_E3"]
    _put(m, "band.static_e3.volga_pct", 100.0 * st["volga"], "%", src,
         "late-fall observed shoreline cells that do not move with level (static-contour rule)", 0)
    oth = [100.0 * v for k, v in st.items() if k != "volga"]
    _put(m, "band.static_e3.others_min_pct", min(oth), "%", src, None, 0)
    _put(m, "band.static_e3.others_max_pct", max(oth), "%", src, None, 0)


# --- item 3 -------------------------------------------------------------------------------------

def datum(m):
    d = _json(R3 / "datum" / "datum.json")
    if not d:
        return
    src = f"{REL}/datum/datum.json"
    for s, v in d["sources"].items():
        k = s.lower()
        _put(m, f"datum.{k}.mean_bias_range_m", v["mean_bias_range_m"], "m", src,
             "range of the checked sectors' all-block mean biases of the waterline model", 2)
        _put(m, f"datum.{k}.median_bias_range_m", v["median_bias_range_m"], "m", src, None, 2)
        _put(m, f"datum.{k}.largest_abs_mean_bias_m", v["largest_abs_mean_bias_m"], "m", src, None, 2)
    _put(m, "datum.swot.largest_abs_checked_m", d["swot"]["largest_abs_checked_m"], "m", src,
         "largest |between-sector SWOT departure| among the checked sectors", 2)


# --- items 4, 5 and 7 -----------------------------------------------------------------------------

def lidar(m):
    t = _csv(R3 / "lidar" / "strata.csv")
    s = _json(R3 / "lidar" / "summary.json")
    if t is None or not s:
        return
    src = f"{REL}/lidar/strata.csv"
    _put(m, "lidar.n_boot", 2000, "resamples", f"{REL}/lidar/summary.json", "block bootstrap of the strata")
    _put(m, "lidar.surface_edge_lo", s["surface_edges"][0], "", f"{REL}/lidar/summary.json",
         "emergent-vegetation share below which a cell is bare", 1)
    _put(m, "lidar.surface_edge_hi", s["surface_edges"][1], "", f"{REL}/lidar/summary.json",
         "emergent-vegetation share from which a cell is vegetated", 1)
    _put(m, "lidar.lowest_level_m", s["lowest_audited_level_m"], "m", f"{REL}/lidar/summary.json", None, 2)
    _put(m, "lidar.band_step_m", s["height_band_step_m"], "m", f"{REL}/lidar/summary.json", None, 1)
    _put(m, "lidar.lowest_band_top_m", s["lowest_audited_level_m"] + s["height_band_step_m"], "m",
         f"{REL}/lidar/summary.json", "top of the lowest height band", 2)
    lowest = f"{s['lowest_audited_level_m']:.2f}_to_"
    for r in t.itertuples():
        if not r.n_points:
            continue
        scope = "checked" if r.scope == "checked4" else SK[r.scope]
        if r.kind == "surface":
            st = r.stratum
        elif str(r.stratum).startswith(lowest):
            st = "lowband"
        else:
            continue
        b = f"lidar.{r.source.lower()}.{scope}.{st}"
        _put(m, f"{b}.n_points", int(r.n_points), "points", src)
        _put(m, f"{b}.n_blocks", int(r.n_blocks), "blocks", src)
        for col, dec in (("mean", 2), ("median", 2), ("median_lo", 2), ("median_hi", 2), ("nmad", 2), ("nmad_lo", 2),
                         ("nmad_hi", 2)):
            _put(m, f"{b}.{col}_m", getattr(r, col), "m", src, "waterline model minus ICESat-2", dec)
        _put(m, f"{b}.median_h_km", r.median_horizontal_km, "km", src,
             "median error divided by the sector's median shelf gradient", 1)
        _put(m, f"{b}.median_h_abs_km", abs(r.median_horizontal_km), "km", src, None, 1)
        _put(m, f"{b}.nmad_h_km", r.nmad_horizontal_km, "km", src, None, 1)
    # ranges over the two products, for the text
    for scope in ["checked"] + [SK[x] for x in CHECKED]:
        for st in ("bare", "mixed", "vegetated", "registered_bare", "lowband"):
            for q in ("median_m", "median_lo_m", "median_hi_m", "nmad_m", "median_h_abs_km", "nmad_h_km"):
                v = [m[k]["value"] for k in (f"rev3.lidar.atl06.{scope}.{st}.{q}", f"rev3.lidar.atl08.{scope}.{st}.{q}")
                     if k in m]
                if len(v) == 2:
                    dec = 1 if q.endswith("km") else 2
                    _put(m, f"lidar.both.{scope}.{st}.{q}.min", min(v), q[-2:].strip("_"), src, "over ATL06 and ATL08", dec)
                    _put(m, f"lidar.both.{scope}.{st}.{q}.max", max(v), q[-2:].strip("_"), src, None, dec)
                    _put(m, f"lidar.both.{scope}.{st}.{q}.absmax", max(abs(x) for x in v), q[-2:].strip("_"), src, None, dec)
                    _put(m, f"lidar.both.{scope}.{st}.{q}.absmin", min(abs(x) for x in v), q[-2:].strip("_"), src, None, dec)
    # the largest position bias of a stratum in each checked sector, against the smallest
    # sector-median displacement of the four failing grids there (rev.s.headline.*, an earlier module)
    worst = {}
    for sec in CHECKED:
        v = {(p, st): abs(m[k]["value"]) for p in ("atl06", "atl08") for st in ("bare", "mixed", "vegetated", "lowband")
             if (k := f"rev3.lidar.{p}.{SK[sec]}.{st}.median_h_km") in m}
        dmin = m.get(f"rev.s.headline.{sec}.d_km_min")
        if len(v) != 8 or dmin is None:
            continue
        (p, st), big = max(v.items(), key=lambda kv: kv[1])
        worst[sec] = (big, big / dmin["value"])
        b = f"lidar.worst.{SK[sec]}"
        _put(m, f"{b}.abs_h_km", big, "km", src, "largest |median position error| over the four strata and two products", 1)
        _put(m, f"{b}.stratum", {"lowband": "lowest band"}.get(st, st), "", src)
        _put(m, f"{b}.product", p.upper(), "", src)
        _put(m, f"{b}.over_d_min", big / dmin["value"], "ratio", src,
             "over the smallest sector-median displacement of the four failing grids in the sector", 1)
    coasts = [worst[s] for s in ("west", "interfluve", "northeast") if s in worst]
    if len(coasts) == 3:
        _span(m, "lidar.worst.coasts.abs_h_km", [a for a, _ in coasts], "km", src, "over the three checked coasts", 1)
        _span(m, "lidar.worst.coasts.over_d_min", [r for _, r in coasts], "ratio", src, "over the three checked coasts", 1)


def independence(m):
    t = _csv(R3 / "lidar" / "independence.csv")
    nf = _csv(R3 / "lidar" / "independence_near_far.csv")
    if t is None or nf is None:
        return
    src = f"{REL}/lidar/independence.csv"
    for s, g in t.groupby("source"):
        k = s.lower()
        for sub, key in (("all_test_points", "all"), ("at_least_10km", "buf10"), ("at_least_25km", "buf25")):
            d = g[g["subset"] == sub]
            dem = d[d["model"] == "dem"].iloc[0]
            _put(m, f"indep.{k}.{key}.n_points", int(dem["n_points"]), "points", src)
            _put(m, f"indep.{k}.{key}.n_blocks", int(dem["n_blocks"]), "blocks", src)
            if not dem["n_points"]:
                continue
            _put(m, f"indep.{k}.{key}.dem.nmad_m", dem["nmad"], "m", src, "waterline model", 2)
            _put(m, f"indep.{k}.{key}.dem.rmse_m", dem["rmse"], "m", src, None, 2)
            _put(m, f"indep.{k}.{key}.dem.mean_m", dem["mean"], "m", src, None, 2)
            gr = d[d["model"] != "dem"]
            _put(m, f"indep.{k}.{key}.grids.nmad_min_m", gr["nmad"].min(), "m", src, "over the five grids", 2)
            _put(m, f"indep.{k}.{key}.grids.nmad_max_m", gr["nmad"].max(), "m", src, None, 2)
            _put(m, f"indep.{k}.{key}.grids.rmse_min_m", gr["rmse"].min(), "m", src, None, 2)
            _put(m, f"indep.{k}.{key}.grids.rmse_max_m", gr["rmse"].max(), "m", src, None, 2)
        a, b = g[(g["subset"] == "all_test_points") & (g["model"] == "dem")].iloc[0], \
            g[(g["subset"] == "at_least_10km") & (g["model"] == "dem")].iloc[0]
        _put(m, f"indep.{k}.buf10.share_pct", 100.0 * b["n_points"] / a["n_points"], "%", src,
             "share of the test points at least 10 km from every training or calibration block", 0)
    src = f"{REL}/lidar/independence_near_far.csv"
    for r in nf.itertuples():
        k = r.source.lower()
        _put(m, f"indep.{k}.near_far.difference_m", r.difference_m, "m", src,
             "mean |error| within 10 km of a training or calibration block minus that farther away", 2)
        _put(m, f"indep.{k}.near_far.lo_m", r.difference_lo_m, "m", src, "block-bootstrap interval", 2)
        _put(m, f"indep.{k}.near_far.hi_m", r.difference_hi_m, "m", src, None, 2)
        _put(m, f"indep.{k}.max_distance_km", r.max_distance_km, "km", src,
             "largest distance of a test point from the nearest training or calibration block", 0)
    _put(m, "indep.buffer_km", 10, "km", f"{REL}/lidar/summary.json", "the smaller buffer")
    _put(m, "indep.buffer_far_km", 25, "km", f"{REL}/lidar/summary.json", "the larger buffer")


def support(m):
    t = _csv(R3 / "lidar" / "support.csv")
    if t is None:
        return
    src = f"{REL}/lidar/support.csv"
    for r in t.itertuples():
        if not r.n_points:
            continue
        b = f"support.{r.source.lower()}.{'overall' if r.sector == 'overall' else SK[r.sector]}"
        for col in ("n_points", "n_blocks", "n_blocks_train", "n_blocks_calib", "n_blocks_test", "n_segments", "n_dates"):
            _put(m, f"{b}.{col}", int(getattr(r, col)), col.split("_", 1)[1], src)
        _put(m, f"{b}.first_date", r.first_date, "date", src)
        _put(m, f"{b}.last_date", r.last_date, "date", src)
        _put(m, f"{b}.first_year", str(r.first_date)[:4], "year", src)      # a string: years take no thousands separator
        _put(m, f"{b}.last_year", str(r.last_date)[:4], "year", src)
        _put(m, f"{b}.icc_block", r.icc_block, "", src, "intra-block correlation of the error (one-way ANOVA)", 2)
        _put(m, f"{b}.icc_segment", r.icc_segment, "", src, "intra-track-segment correlation of the error", 2)
        _put(m, f"{b}.n_eff_block", r.n_eff_block, "points", src,
             "effective sample size, Kish design effect with blocks as clusters", 0)
        _put(m, f"{b}.n_eff_segment", r.n_eff_segment, "points", src, "with track segments as clusters", 0)
    for s, g in t[t["sector"].isin(CHECKED)].groupby("source"):
        k = s.lower()
        _span(m, f"support.{k}.checked.n_eff_block", g["n_eff_block"], "points", src, "over the checked sectors", 0)
        _span(m, f"support.{k}.checked.n_points", g["n_points"], "points", src, None, 0)
        _span(m, f"support.{k}.checked.n_blocks", g["n_blocks"], "blocks", src, None, 0)
        _span(m, f"support.{k}.checked.n_segments", g["n_segments"], "segments", src, None, 0)
        _span(m, f"support.{k}.checked.n_dates", g["n_dates"], "dates", src, None, 0)


# --- item 6 -------------------------------------------------------------------------------------

def tiers(m):
    t = _csv(R3 / "tiers" / "tiers.csv")
    s = _json(R3 / "tiers" / "summary.json")
    if t is None or not s:
        return
    src = f"{REL}/tiers/tiers.csv"
    word = {"pass": "passes", "resolution": "resolution-scale", "material": "material"}
    for r in t.to_dict("records"):          # records, not itertuples: "class" is not an identifier
        b = f"tier.{PK[r['product']]}.{SK[r['sector']]}"
        _put(m, f"{b}.class", word[r["class"]], "", src, "class of the grid in the sector (post hoc overlay)")
        for col in ("n_evaluable", "n_above_floor", "n_material", "n_resolution"):
            _put(m, f"{b}.{col}", int(r[col]), "levels", src)
        _put(m, f"{b}.d_minus_q_km", r["d_minus_q_median_m"] / 1000, "km", src, "median over levels of D - Q", 1)
        _put(m, f"{b}.d_minus_q_lo_km", r["d_minus_q_lo_m"] / 1000, "km", src, "moving-block bootstrap interval", 1)
        _put(m, f"{b}.d_minus_q_hi_km", r["d_minus_q_hi_m"] / 1000, "km", src, None, 1)
        _put(m, f"{b}.p_material", r["p_material_majority"], "probability", src,
             "share of level resamples with a material majority", 2)
    src = f"{REL}/tiers/summary.json"
    _put(m, "tier.four.n_cases", s["four_failing_grids_checked_cases"], "cases", src, "four failing grids by checked sector")
    by = s["four_failing_grids_by_class"]
    _put(m, "tier.four.n_material", by.get("material", 0), "cases", src)
    _put(m, "tier.four.n_resolution", by.get("resolution", 0), "cases", src)
    _put(m, "tier.four.n_pass", by.get("pass", 0), "cases", src)
    _put(m, "tier.height_m", s["height_m"], "m", src, "the checked-sector tolerance", 1)
    _put(m, "tier.cell_15s_km", s["native_cell_m"]["gebco_2026"] / 1000, "km", src,
         "longer side of a 15 arc-second cell", 2)
    _put(m, "tier.cell_1m_km", s["native_cell_m"]["etopo1"] / 1000, "km", src, "longer side of a 1 arc-minute cell", 2)
    hd = s["half_metre_distance_m"]
    for sec in SK:
        _put(m, f"tier.half_metre_km.{SK[sec]}", hd[sec] / 1000, "km", src,
             "distance that 0.5 m of height spans at the sector's median gradient", 1)
    _span(m, "tier.half_metre_km.coasts", [hd[x] / 1000 for x in ("west", "interfluve", "northeast")], "km", src,
          "over the three checked coasts", 1)
    _put(m, "tier.n_boot", s["n_boot"], "resamples", src)
    _put(m, "tier.mbb_block_levels", s["mbb_block_levels"], "levels", src)
    coasts = t[t["product"].isin(FAILING) & t["sector"].isin(["west", "interfluve"])]
    _span(m, "tier.four.westinterfluve.p_material", coasts["p_material_majority"], "probability", src,
          "over the four failing grids on the two headline coasts", 2)
    _span(m, "tier.four.westinterfluve.d_minus_q_lo_km", coasts["d_minus_q_lo_m"] / 1000, "km", src,
          "lower interval bounds of the median D - Q", 1)
    ne = t[t["product"].isin(FAILING) & (t["sector"] == "northeast")]
    _span(m, "tier.four.northeast.p_material", ne["p_material_majority"], "probability", src, None, 2)
    os_ = t[t["product"].isin(FAILING) & (t["sector"] == "open_shelf")]
    _span(m, "tier.four.openshelf.n_material", os_["n_material"], "levels", src, None, 0)


# --- item 8 -------------------------------------------------------------------------------------

def setup(m):
    t = _csv(R3 / "setup" / "setup_by_sector.csv")
    w = _csv(R3 / "setup" / "wind_sampling.csv")
    s = _json(R3 / "setup" / "summary.json")
    if t is None or w is None or not s:
        return
    src = f"{REL}/setup/setup_by_sector.csv"
    _put(m, "setup.n_acquisitions", s["n_acquisitions"], "acquisitions", f"{REL}/setup/summary.json",
         "Landsat acquisitions of the main processing plan")
    _put(m, "setup.n_units", s["n_units"], "units", f"{REL}/setup/summary.json", "epoch and level-bin units")
    import inspect

    from caspian import rev3 as _rev3
    thr = max(inspect.signature(_rev3.abs_summary).parameters["thresholds"].default)
    if abs(thr - 0.20) > 1e-12:      # the table column and the keys below are named for 0.20 m
        raise SystemExit("metrics_rev_x3: the setup threshold of caspian.rev3.abs_summary is no longer 0.20 m")
    _put(m, "setup.threshold_m", thr, "m", "caspian.rev3.abs_summary (the larger default threshold)",
         "|predicted setup| above which an acquisition is counted in share_above_0p20", 2)
    for r in t.itertuples():
        b = f"setup.{SK[r.sector]}"
        _put(m, f"{b}.sd_m", r.sd_m, "m", src, "standard deviation of the predicted setup at the acquisition times", 2)
        _put(m, f"{b}.abs_p95_m", r.abs_p95_m, "m", src, "95th percentile of |predicted setup|", 2)
        _put(m, f"{b}.share_above_0p20_pct", 100.0 * r.share_above_0p20, "%", src,
             "acquisitions with |predicted setup| above 0.20 m", 0)
        _put(m, f"{b}.share_above_0p10_pct", 100.0 * r.share_above_0p10, "%", src, None, 0)
        _put(m, f"{b}.total_sd_m", r.total_sd_m, "m", src, "predicted setup and calibration residual in quadrature", 2)
        _put(m, f"{b}.total_sd_h_km", r.total_sd_horizontal_m / 1000, "km", src, "at the sector's median gradient", 1)
        _put(m, f"{b}.total_sd_over_floor", r.total_sd_over_floor, "ratio", src,
             "shoreline equivalent of the total SD over the sector's median floor", 1)
        _put(m, f"{b}.unit_mean_abs_max_m", r.unit_mean_abs_max_m, "m", src,
             "largest |mean predicted setup| of a level bin", 2)
        _put(m, f"{b}.unit_mean_abs_median_m", r.unit_mean_abs_median_m, "m", src, None, 2)
        _put(m, f"{b}.unit_mean_abs_max_h_km", r.unit_mean_abs_max_horizontal_m / 1000, "km", src, None, 1)
        _put(m, f"{b}.floor_km", r.floor_median_m / 1000, "km", src, "the sector's median floor", 2)
        _put(m, f"{b}.sigma_b_m", r.sigma_budget_m, "m", src, "the sector's systematic level-error budget (median over levels)", 2)
        _put(m, f"{b}.unit_mean_abs_max_over_sigma_b", r.unit_mean_abs_max_over_sigma, "ratio", src,
             "largest |mean predicted setup| of a level bin over the sector's level-error budget", 1)
    c = t[t["sector"].isin(["west", "interfluve", "northeast"])]
    _span(m, "setup.coasts.unit_mean_abs_max_over_sigma_b", c["unit_mean_abs_max_over_sigma"], "ratio", src,
          "over the three checked coasts", 1)
    _span(m, "setup.coasts.unit_mean_abs_median_m", c["unit_mean_abs_median_m"], "m", src,
          "median over level bins of |mean predicted setup|, over the three checked coasts", 2)
    _span(m, "setup.coasts.sigma_b_m", c["sigma_budget_m"], "m", src, "over the three checked coasts", 2)
    for col, key, unit, dec, scale in (("share_above_0p20", "coasts.share_above_0p20_pct", "%", 0, 100.0),
                                       ("total_sd_m", "coasts.total_sd_m", "m", 2, 1.0),
                                       ("total_sd_horizontal_m", "coasts.total_sd_h_km", "km", 1, 1e-3),
                                       ("total_sd_over_floor", "coasts.total_sd_over_floor", "ratio", 1, 1.0),
                                       ("unit_mean_abs_max_m", "coasts.unit_mean_abs_max_m", "m", 2, 1.0),
                                       ("unit_mean_abs_max_horizontal_m", "coasts.unit_mean_abs_max_h_km", "km", 1, 1e-3),
                                       ("abs_p95_m", "coasts.abs_p95_m", "m", 2, 1.0)):
        _span(m, f"setup.{key}", c[col] * scale, unit, src, "over the three checked coasts", dec)
    _span(m, "setup.all.share_above_0p20_pct", t["share_above_0p20"] * 100.0, "%", src, "over the seven sectors", 0)
    src = f"{REL}/setup/wind_sampling.csv"
    _span(m, "setup.wind.ratio_same_hours", w["median_ratio_same_hours"], "ratio", src,
          "median wind speed at the acquisition hours over that of the same hours of the day, months and years", 2)
    _span(m, "setup.wind.ratio_all_hours", w["median_ratio"], "ratio", src,
          "over that of all hours of the same months and years", 2)
    _span(m, "setup.wind.acq_median_m_s", w["acq_median_m_s"], "m/s", src, "sector-mean 10 m wind speed", 1)


def sector_table(m):
    """Item 2a: values of the sector table that no earlier key holds."""
    g = _csv(ROOT / "results" / "revision" / "r" / "budget" / "gradients.csv")
    if g is not None:
        for r in g.itertuples():
            _put(m, f"sector.{SK[r.sector]}.gradient_m_per_km", r.g_median_m_per_km, "m per km",
                 "results/revision/r/budget/gradients.csv", "median seabed gradient of the emerged band", 2)
    v = _json(ROOT / "results" / "levels" / "vegetation_fraction.json")
    if v:
        for sec, f in v["fraction"].items():
            _put(m, f"sector.{SK[sec]}.vegetated_pct", 100.0 * f, "%", "results/levels/vegetation_fraction.json",
                 "share of the band's cells dominated by emergent vegetation", 0)


def text_claims(m) -> None:
    """Relations between numbers that the article and Supplementary Section S11 state in words
    ("smaller than", "most of", "none", "nearly twofold"). The number trace checks every numeral
    but cannot check a comparison, so each is asserted here: if a later run changes a result so
    that a sentence no longer holds, the metrics build stops and names the sentence."""
    if "rev3.marine.n_changed_checked" not in m:        # the revision analyses have not been run
        return

    def v(key):
        if key not in m:
            raise SystemExit(f"metrics_rev_x3.text_claims: {key} is missing")
        return m[key]["value"]

    four = ("gebco2026", "gebco2022", "srtm15p", "etopo1")
    coasts = ("west", "interfluve", "northeast")
    n_checked = v("rev.s.checked.n_sectors")
    claims = {
        "no verdict in a checked sector changes on the marine domain": v("rev3.marine.n_changed_checked") == 0,
        "the reporting rule of Amendment 20 is not triggered (no move of a floor or more on the two headline coasts)":
            v("rev3.marine.four.westinterfluve.d_move_over_q_max") < 1,
        "the largest move of one grid on the three coasts is GEBCO_2022's on the northeastern coast":
            (v("rev3.marine.four.coasts3.d_move_max_grid"), v("rev3.marine.four.coasts3.d_move_max_sector"))
            == ("gebco2022", "northeast"),
        "each of the two changed verdicts changes by a single level": all(
            abs(v(f"rev3.marine.changed.{k}.n_exceeding") - v(f"rev3.marine.changed.{k}.n_exceeding_primary")) == 1
            for k in ("etopo2022.ural", "srtm15p.komsomolets")),
        "ETOPO 2022 now passes in the Ural delta, SRTM15+ now fails in Komsomolets Bay":
            v("rev3.marine.changed.etopo2022.ural") == "passes" and v("rev3.marine.changed.srtm15p.komsomolets") == "fails",
        "the pattern persists in interior blocks (more than half exceed for each failing grid)":
            v("rev3.blocks.interior.four.share_min_pct") > 50,
        "each checked sector's departure is smaller than the budget's standard deviation": max(
            v("rev3.datum.atl06.largest_abs_mean_bias_m"), v("rev3.datum.atl08.largest_abs_mean_bias_m"),
            v("rev3.datum.swot.largest_abs_checked_m")) < v("rev.w.r.budget.sigma_median_min"),
        "the largest stratum error is in vegetated cells on the western and interfluve coasts, in the lowest band "
        "on the northeastern coast": (v("rev3.lidar.worst.west.stratum"), v("rev3.lidar.worst.interfluve.stratum"),
                                      v("rev3.lidar.worst.northeast.stratum")) == ("vegetated", "vegetated", "lowest band"),
        "on the three coasts the largest stratum error is less than half of the grids' smallest miss":
            v("rev3.lidar.worst.coasts.over_d_min.max") < 0.5,
        "on the open shelf the largest stratum error exceeds the grids' misses":
            v("rev3.lidar.worst.openshelf.abs_h_km") > v("rev.s.headline.open_shelf.d_km_max"),
        "the vegetated-cell bias is positive (the Landsat shoreline lies seaward) for both products":
            v("rev3.lidar.both.checked.vegetated.median_m.min") > 0,
        "the lowest band is landward on the interfluve and northeastern coasts for both products": all(
            v(f"rev3.lidar.{p}.{s}.lowband.median_m") < 0 for p in ("atl06", "atl08") for s in ("interfluve", "northeast")),
        "the reed bias is larger than the level-error budget":
            v("rev3.lidar.both.checked.vegetated.median_m.min") > v("rev.w.r.budget.sigma_median_max"),
        "with vegetation as dry and as water each of the four grids fails every checked sector": all(
            v(f"rev2.a.i11.{var}.{g}.n_validated4") == n_checked for var in ("vegasdry", "vegaswater") for g in four),
        "the near-far interval of the buffered accuracy test includes zero": all(
            v(f"rev3.indep.{p}.near_far.lo_m") < 0 < v(f"rev3.indep.{p}.near_far.hi_m") for p in ("atl06", "atl08")),
        "accuracy at a distance is the same (NMAD within 0.05 m)": all(
            abs(v(f"rev3.indep.{p}.buf10.dem.nmad_m") - v(f"rev3.indep.{p}.all.dem.nmad_m")) < 0.05 for p in ("atl06", "atl08")),
        "material failures lie on the three coasts and resolution-scale ones on the open shelf": all(
            v(f"rev3.tier.{g}.{s}.class") == "material" for g in four for s in coasts) and all(
            v(f"rev3.tier.{g}.openshelf.class") == "resolution-scale" for g in four),
        "the acquisitions are not unusually calm (wind ratio within 5% of one)":
            0.95 < v("rev3.setup.wind.ratio_same_hours.min") and v("rev3.setup.wind.ratio_same_hours.max") < 1.05,
        "both setup variants of robustness check 4 are judged robust":
            v("robustness.check04.tau015_R.verdict") == "robust" == v("robustness.check04.setup_correction.verdict"),
        "the grids disagree nearly twofold on the first-metre area": 1.7 <= v(
            "rev2.n.reref.area.main.E.1.domain.max_km2") / v("rev2.n.reref.area.main.E.1.domain.min_km2") <= 2.0,
        "most of the first-metre protected area lies in the deltaic sectors, for every grid":
            v("rev3.exposure.wdpa_protected_areas_noksrz.deltaic_share_pct.min") > 50,
        "no first-metre settlement lies in the deltaic sectors": v("rev3.exposure.osm_settlements.deltaic.max") == 0,
        "the exposure counts differ several-fold between grids": min(
            v("rev3.exposure.wdpa_protected_areas_noksrz.domain.max") / v("rev3.exposure.wdpa_protected_areas_noksrz.domain.min"),
            v("rev3.exposure.osm_channels.domain.max") / v("rev3.exposure.osm_channels.domain.min"),
            v("rev3.exposure.osm_ports.domain.max") / v("rev3.exposure.osm_ports.domain.min")) >= 3,
    }
    failed = [k for k, ok in claims.items() if not ok]
    if failed:
        raise SystemExit("metrics_rev_x3.text_claims: the text states what the results no longer show:\n  - "
                         + "\n  - ".join(failed))


def add(m: dict) -> None:
    for f in (marine, blocks, nearterm, exposure, band, datum, lidar, independence, support, tiers, setup, sector_table,
              text_claims):
        f(m)
