"""Revision, workstream V: each sector's departure of the water surface from the basin
level (PREREG Amendment 18 item 9, first bullet; D-120).

1. SWOT block means before de-meaning, exactly as the setup model builds them
   (scripts/a5_setup_model.sector_anomalies: the same pass grouping, quality screens
   and block-pass means, and the same basin level at the pass time, i.e. the scene-level
   code path: DAHITI, or Hydroweb shifted onto DAHITI in DAHITI gaps over 60 days).
   anom = block-mean wse - level.
2. To the project's datum chain (caspian.watersurface.swot_to_working): SWOT's geoid
   is EGM2008 in the mean tide system (JPL D-56416 Rev C). Primary: SWOT's EGM2008 is
   taken as NGA's realisation (with its -0.41 m zero-degree term; the realisation the
   project uses for EGM2008 heights, D-074), the working datum as tide-free. Per block
   the conversion is evaluated at the block centre from
   results/study_area/geoid_egm2008_minus_eigen6c4.npz.
3. Per sector: the median over April-October passes of the pass's mean over the
   sector's blocks; IQR over passes. A sector with fewer than 2 blocks or 20 passes is
   "too thin" and gets no value.

Sensitivities (extra columns; never chosen among): SWOT geoid frame (no conversion),
ICGEM realisation of EGM2008 (no -0.41 m), mean-tide working datum, passes levelled by
DAHITI only, training and calibration blocks only, all months, and the median over
block-passes. A per-sector, per-month table goes to sector_departures_by_month.csv.

Outputs (results/revision/v/): sector_departures.csv, sector_departures_by_month.csv,
sector_departures.json.
Usage: python scripts/rev_v_sector_departures.py
"""

from __future__ import annotations

import importlib.util
import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
from scipy.interpolate import RegularGridInterpolator

from caspian import watersurface as ws
from caspian.config import CRS_EQUAL_AREA, RESULTS
from caspian.geography import SECTOR_ORDER

OUT = RESULTS / "revision" / "v"
MONTHS = (4, 5, 6, 7, 8, 9, 10)     # April-October: the months of the hindcast scenes
MIN_BLOCKS, MIN_PASSES = 2, 20      # D-120: below either, the sector is "too thin"
GEOID = RESULTS / "study_area" / "geoid_egm2008_minus_eigen6c4.npz"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, RESULTS.parent / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def block_conversions(blocks: pd.DataFrame) -> pd.DataFrame:
    """Per block (centre): the geoid terms and the three conversions SWOT -> working datum."""
    inv = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(blocks["xc"].to_numpy(float), blocks["yc"].to_numpy(float))
    g = np.load(GEOID)
    pts = np.column_stack([lat, lon])
    f = {k: RegularGridInterpolator((g["lat"], g["lon"]), g[k].astype(float), bounds_error=False, fill_value=np.nan)(pts)
         for k in ("N_egm2008_nga_m", "N_egm2008_icgem_m", "N_eigen6c4_icgem_m")}
    out = blocks[["i", "j", "sector", "split"]].copy()
    out["lon"], out["lat"] = lon, lat
    out["conv_primary_m"] = ws.swot_to_working(0.0, f["N_egm2008_nga_m"], f["N_eigen6c4_icgem_m"], lat, "tide_free")
    out["conv_icgem_m"] = ws.swot_to_working(0.0, f["N_egm2008_icgem_m"], f["N_eigen6c4_icgem_m"], lat, "tide_free")
    out["conv_meantide_m"] = ws.swot_to_working(0.0, f["N_egm2008_nga_m"], f["N_eigen6c4_icgem_m"], lat, "mean_tide")
    return out


def table(bp: pd.DataFrame, value: str) -> pd.DataFrame:
    t = ws.sector_departures(bp, value, min_blocks=MIN_BLOCKS, min_passes=MIN_PASSES)
    al = ws.sector_departures(bp.assign(sector="all"), value, min_blocks=MIN_BLOCKS, min_passes=MIN_PASSES)
    return pd.concat([t, al], ignore_index=True)


def main() -> int:
    a5, a4 = _load("a5_setup_model"), _load("a4_scene_levels")
    prep = a4.prepare()
    sw = pd.read_csv(RESULTS / "levels" / "swot_block_levels.csv")
    blocks = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    sec, bp = a5.sector_anomalies(sw, blocks, prep, a4)
    pass_t = sec.drop_duplicates("pass_id").set_index("pass_id")["time"]
    lv, _ = a4.levels_at(prep, pass_t.values)
    src = dict(zip(pass_t.index, lv["level_source"].to_numpy()))
    conv = block_conversions(blocks)
    bp = bp.merge(conv, left_on=["block_i", "block_j", "sector", "split"], right_on=["i", "j", "sector", "split"],
                  how="left")
    if bp["conv_primary_m"].isna().any():
        raise SystemExit("a block centre falls outside the geoid grid")
    bp["time"] = bp["pass_id"].map(pass_t)
    bp["month"] = pd.to_datetime(bp["time"]).dt.month
    bp["level_source"] = bp["pass_id"].map(src)
    bp["dep_m"] = bp["anom"] + bp["conv_primary_m"]
    bp["dep_swot_geoid_m"] = bp["anom"]
    bp["dep_icgem_m"] = bp["anom"] + bp["conv_icgem_m"]
    bp["dep_meantide_m"] = bp["anom"] + bp["conv_meantide_m"]
    warm = bp[bp["month"].isin(MONTHS)]

    prim = table(warm, "dep_m")
    sens = {"departure_swot_geoid_m": table(warm, "dep_swot_geoid_m"),
            "departure_icgem_realisation_m": table(warm, "dep_icgem_m"),
            "departure_meantide_working_m": table(warm, "dep_meantide_m"),
            "departure_dahiti_level_only_m": table(warm[warm["level_source"] == "dahiti"], "dep_m"),
            "departure_train_calib_blocks_m": table(warm[warm["split"].isin(["train", "calib"])], "dep_m"),
            "departure_all_months_m": table(bp, "dep_m")}
    out = prim.copy()
    for k, t in sens.items():
        out[k] = out["sector"].map(dict(zip(t["sector"], t["departure_m"])))
        out["n_passes_" + k.removeprefix("departure_").removesuffix("_m")] = out["sector"].map(
            dict(zip(t["sector"], t["n_passes"])))
    dep_all = float(out.loc[out["sector"] == "all", "departure_m"].iloc[0])
    out.insert(6, "departure_rel_all_m", out["departure_m"] - dep_all)   # relative to the all-block value
    bpm = warm.groupby("sector")["dep_m"].median()
    out["departure_blockpass_median_m"] = out["sector"].map(bpm.to_dict() | {"all": float(warm["dep_m"].median())})
    out["first_pass"] = out["sector"].map(warm.groupby("sector")["time"].min().astype(str).to_dict()
                                          | {"all": str(warm["time"].min())})
    out["last_pass"] = out["sector"].map(warm.groupby("sector")["time"].max().astype(str).to_dict()
                                         | {"all": str(warm["time"].max())})
    out["share_hydroweb_levelled_passes"] = out["sector"].map(
        (warm.drop_duplicates(["sector", "pass_id"]).assign(h=lambda d: d["level_source"] != "dahiti")
         .groupby("sector")["h"].mean()).to_dict()
        | {"all": float(warm.drop_duplicates("pass_id")["level_source"].ne("dahiti").mean())})
    order = {s: k for k, s in enumerate(SECTOR_ORDER + ["all"])}
    missing = [s for s in SECTOR_ORDER if s not in set(out["sector"])]
    for s in missing:      # sectors with no screened SWOT block at all
        out = pd.concat([out, pd.DataFrame([{"sector": s, "departure_m": np.nan, "n_passes": 0, "n_blocks": 0,
                                             "iqr_m": np.nan, "n_block_passes": 0,
                                             "note": "no SWOT open-sea block passes the A5 screens; no value"}])],
                        ignore_index=True)
    out = out.sort_values("sector", key=lambda s: s.map(order)).reset_index(drop=True)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    out.loc[out["sector"] == "all", "note"] = (
        "all blocks pooled. The common part of every departure depends on undocumented constants (which EGM2008 "
        "realisation SWOT uses: +0.405 m without NGA's -0.41 m term; DAHITI's own reference, cf. the ICESat-2 "
        "constant); a common shift cancels in the shape variant, so departure_rel_all_m is the between-sector part")
    first = ["sector", "departure_m", "n_passes", "n_blocks", "iqr_m", "note"]
    out = out[first + [c for c in out.columns if c not in first]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "sector_departures.csv", index=False)

    by_month = (pd.concat([warm, warm.assign(sector="all")], ignore_index=True)
                .groupby(["sector", "month", "pass_id"], as_index=False)["dep_m"].mean()
                .groupby(["sector", "month"])["dep_m"].agg(departure_m="median", n_passes="size",
                                                         q25_m=lambda v: np.percentile(v, 25),
                                                         q75_m=lambda v: np.percentile(v, 75)).reset_index())
    by_month.round(4).to_csv(OUT / "sector_departures_by_month.csv", index=False)

    conv_s = conv[conv.set_index(["i", "j"]).index.isin(bp.set_index(["block_i", "block_j"]).index.unique())]
    rec = {"rule": "A18 item 9 (D-120): per sector, the median over April-October SWOT passes of the pass's mean over "
                   "the sector's blocks of (block-mean wse - basin level at the pass time), on the working datum",
           "screens": "scripts/a5_setup_model.sector_anomalies (MIN_PASS_PX, MAX_BLOCK_NMAD, MIN_BLOCK_PASSES, "
                      "MAX_ROBUST_Z), unchanged",
           "level": "scripts/a4_scene_levels.levels_at (DAHITI; Hydroweb shifted onto DAHITI in DAHITI gaps > 60 d)",
           "months": list(MONTHS), "min_blocks": MIN_BLOCKS, "min_passes": MIN_PASSES,
           "datum_chain": {
               "swot_geoid": "EGM2008, mean tide system (JPL D-56416 Rev C, geoid variable)",
               "primary": "NGA realisation of EGM2008 (zero-degree term -0.41 m), working datum tide-free: "
                          "H = wse + N_EGM2008(NGA) - N_EIGEN-6C4 + g(lat) - e(lat)",
               "conversion_at_used_blocks_m": {k: [round(float(conv_s[k].min()), 4), round(float(conv_s[k].max()), 4)]
                                               for k in ("conv_primary_m", "conv_icgem_m", "conv_meantide_m")}},
           "n_block_passes_warm": int(len(warm)), "n_passes_warm": int(warm["pass_id"].nunique()),
           "n_blocks_warm": int(warm[["block_i", "block_j"]].drop_duplicates().shape[0]),
           "n_passes_all_months": int(bp["pass_id"].nunique()),
           "pass_period": [str(warm["time"].min()), str(warm["time"].max())],
           "level_source_of_warm_passes": warm.drop_duplicates("pass_id")["level_source"].value_counts().to_dict(),
           "where_measured": "open-sea SWOT pixels only (water_frac >= 0.9 inside the D-029 sea mask): the "
                             "departure is that of the sector's offshore water, not of its shoreline",
           "table": json.loads(out.to_json(orient="records"))}
    (OUT / "sector_departures.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(out[["sector", "departure_m", "n_passes", "n_blocks", "iqr_m", "note"]].to_string(index=False))
    print(json.dumps({k: rec[k] for k in ("n_block_passes_warm", "n_passes_warm", "n_blocks_warm", "pass_period",
                                          "level_source_of_warm_passes")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
