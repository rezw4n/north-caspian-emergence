"""Robustness: the Earth Engine export mode for the block-sample alternatives of
docs/ROBUSTNESS_PLAN.md §2.3 (D-080), with a dry run that prints the plan and its cost.
The robustness agent wrote this and did NOT run it against Earth Engine; running it
needs the lead's go-ahead, a budget in results/inventory/ee_budgets.json and the guard.

Design
- Region: the seeded robustness sample R (14 test blocks of results/study_area/
  blocks_25km.csv, 2 per sector, >= 100 km2 of stage-2 cover; robustness.block_sample),
  each the stage-2 cover within the 25 km square, with no sector clip; or the 100 km
  main tiles (region "cover") for the cheap whole-cover items.
- Groups: the main plan's (epoch, bin, parity) groups, so the counts are summed into or
  substituted for the main counts by scripts/r00_counts.py (--substitute for a full
  alternative on R; --add / --sub for a delta). 0.05 m bins use their own group labels.
- Graph: scripts/a6_stage2_export.build_image and interactive, imported (not edited).
  Per alternative this process patches, before any graph is built:
  pipeline_ee.water_valid (AWEI_sh, NDVI threshold), masks.CLASSIFY_BANDS (AWEI_sh reads
  blue and SWIR2), masks.NDSI_MONTHS (snow tests in every month) and a6_stage2_export.
  targets (the R blocks). Each invocation is its own process, so nothing leaks into
  other runs.
- Execution: interactive computePixels, 512 px sub-blocks x 16 groups per request (the
  D-079 setting), metered by ee.profilePrinting against --budget (EECU-s).
- Output: data/interim/stage2/<tag>/tile_<i>_<j>.npz and results/inventory/
  stage2_<tag>.json (transform, dims, sector_code None per block), the format
  r00_counts.block_regions and --substitute read.

Alternatives (--alt) and what they change
  repro            the main plan and classifier unchanged: the pilot, whose counts must
                   equal the main counts in the block (a check of the export mode)
  fixed_mndwi0     every scene, MNDWI > 0 instead of the acquisition threshold (check 1)
  awei_sh          every scene, AWEI_sh > 0 (Feyisa et al. 2014 form, caspian.water.awei_sh)
  ndvi020, ndvi040 every scene, emergent vegetation at NDVI > 0.20 / 0.40 (check 9)
  snow_all_months  every scene, both snow tests in every month (Amendment 9)
  november         ADD: the November scenes (D-062 5% bar); needs their stage-1
                   thresholds (--november-plan CSV with scene, sensor, acq, threshold,
                   epoch, bin, parity); the dry run costs them from scene_levels.csv
  tau015           SUB: the scene-sector pairs with |setup| > 0.15 m (main tau = inf, D-077)
  sub_tm, sub_etm, sub_slcoff, sub_oli   SUB: one Landsat sensor class (check 2)
  bins005          every scene regrouped in 0.05 m bins (PREREG section 4); analysed inside
                   R only (its groups differ from the main ones)
  e1_end_k1, e1_end_k9   whole cover: the E1-end window scenes under the E2 groups
                   (ADD) and under their E1 groups (SUB), Amendment 4
  s2_add           ADD: the Sentinel-2 scenes of the main plan on R (D-079, if the quota allows)

Cost (dry run): rate x area x the alternative's share of the plan's scene-weighted zone
area (results/inventory/stage2_cost_model.json units) x a band factor (AWEI_sh 1.5;
Sentinel-2 4.5, the D-079 per-km2 ratio). Rate: 5.8 EECU-s per km2 of region for the full
Landsat plan (D-079, 512 px x 16 groups), with 6.8 as the high bound.

Usage:
  python scripts/r18_rblock_export.py plan --alt ALL|<alt>[,<alt>] [--dry-run]
  python scripts/r18_rblock_export.py run --alt <alt> --tag <tag> --budget <EECU-s> --i-have-approval
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import design, levels, robustness as R
from caspian.config import CRS_EQUAL_AREA, INTERIM, RESULTS
from caspian.geography import SECTOR_ORDER

PLAN_L = INTERIM / "stage2" / "plan_main_tau9900cm_landsat.csv"
PLAN_S = INTERIM / "stage2" / "plan_main_tau9900cm_s2.csv"
RATE = {"central": 5.8, "high": 6.8}            # EECU-s per km2 of region, full Landsat plan (D-079)
S2_FACTOR = 4.5                                  # Sentinel-2 / Landsat cost per km2 (D-079)
AWEI_FACTOR = 1.5                                # blue and SWIR2 read in addition to 4 bands (assumed)
ALTS = ("repro", "fixed_mndwi0", "awei_sh", "ndvi020", "ndvi040", "snow_all_months", "november", "tau015",
        "sub_tm", "sub_etm", "sub_slcoff", "sub_oli", "bins005", "e1_end_k1", "e1_end_k9", "s2_add")
E1_ENDS = {"k1": pd.Timestamp("1995-07-06 17:31:11"), "k3": pd.Timestamp("1995-07-16 15:29:42"),
           "k9": pd.Timestamp("1995-05-28 01:37:03")}      # scene_levels_summary.json, e1_end_sensitivity


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- geometry -------------------------------------------------------------------------------

def r_blocks() -> pd.DataFrame:
    """The robustness sample R with its cover area (km2)."""
    from shapely.geometry import box
    cov = R.load_cover(RESULTS / "study_area" / "zone_v6_cover_buffered.geojson", CRS_EQUAL_AREA)
    b = pd.read_csv(RESULTS / "study_area" / "blocks_25km.csv")
    size = float(design.BLOCK_SIZE_M)
    b["cover_km2"] = [cov.intersection(box(x, y, x + size, y + size)).area / 1e6 for x, y in zip(b["x0"], b["y0"])]
    return R.block_sample(b, b["cover_km2"], n_per_sector=2)


# --- plans ------------------------------------------------------------------------------------

def _scene_km2(plan: pd.DataFrame) -> pd.Series:
    cm = json.loads((RESULTS / "inventory" / "stage2_cost_model.json").read_text(encoding="utf-8"))
    zone = {u["unit"]: u["zone_km2"] for u in cm["units"]}
    unit = [s.split("_")[-1] if se == "MSI_S2" else s.split("_")[1] for s, se in zip(plan["scene"], plan["sensor"])]
    return pd.Series([zone.get(u, 0.0) for u in unit], index=plan.index, dtype=float)


def _export_cols(d: pd.DataFrame) -> pd.DataFrame:
    return d[["scene", "sensor", "acq", "threshold", "group", "sectors", "mask_mode", "epoch", "bin", "parity",
              "level_m", "t"]].copy()


def build_plan(alt: str, plan: pd.DataFrame, november_plan: str | None = None) -> dict:
    """{rows, region, factor, kind, classifier} of one alternative (see the docstring)."""
    d = plan.copy()
    d["sectors"] = d["sectors"].fillna("").astype(str)
    full = {"rows": _export_cols(d), "region": "R", "factor": 1.0, "kind": "full"}
    if alt == "repro":
        return {**full, "classifier": "main"}
    if alt == "fixed_mndwi0":
        r = _export_cols(d)
        r["threshold"] = 0.0
        return {**full, "rows": r, "classifier": "mndwi_fixed0"}
    if alt == "awei_sh":
        return {**full, "factor": AWEI_FACTOR, "classifier": "awei_sh"}
    if alt in ("ndvi020", "ndvi040"):
        return {**full, "classifier": f"ndvi{alt[-3:]}"}
    if alt == "snow_all_months":
        return {**full, "classifier": "snow_all_months"}
    if alt == "tau015":
        delta = R.screening_delta(d, R.exclusion(d, np.inf), R.exclusion(d, 0.15))
        return {"rows": _export_cols(delta["sub"]), "region": "R", "factor": 1.0, "kind": "sub",
                "classifier": "main", "n_pairs": delta["n_sub_pairs"], "mixed": delta["n_mixed_acq_sector"]}
    if alt.startswith("sub_"):
        cls = {"sub_tm": ["TM"], "sub_etm": ["E7on", "E7off"], "sub_slcoff": ["E7off"], "sub_oli": ["OLI"]}[alt]
        sc = R.sensor_class(d["sensor"], pd.to_datetime(d["t"], format="ISO8601"))
        r = _export_cols(d[np.isin(sc, cls)])
        r["sectors"], r["mask_mode"] = "", "exclude"
        return {"rows": r, "region": "R", "factor": 1.0, "kind": "sub", "classifier": "main"}
    if alt == "bins005":
        r = _export_cols(d)
        med = r.groupby(["epoch", "bin", "acq"])["level_m"].transform("median")
        fine = levels.bin_index(med.to_numpy(), 0.05, design.BIN_ORIGIN_M)
        r["group"] = [R.group_name(e, b, p) for e, b, p in zip(r["epoch"], fine, r["parity"])]
        return {**full, "rows": r, "classifier": "main", "kind": "full_regrouped", "bin_width_m": 0.05}
    if alt in ("e1_end_k1", "e1_end_k9"):
        end = E1_ENDS[alt.split("_")[-1]]
        t = pd.to_datetime(d["t"], format="ISO8601")
        win = d[(d["epoch"] == "E1") & (t > end)]
        add = _export_cols(win)
        add["group"] = [R.group_name("E2", b, p) for b, p in zip(add["bin"], add["parity"])]
        return {"rows": _export_cols(win), "rows_add": add, "region": "cover", "factor": 1.0, "kind": "sub+add",
                "classifier": "main", "n_window_scenes": int(len(win))}
    if alt == "november":
        if november_plan:
            r = pd.read_csv(november_plan)
            r["group"] = [R.group_name(e, b, p) for e, b, p in zip(r["epoch"], r["bin"], r["parity"])]
            r["sectors"], r["mask_mode"] = "", "exclude"
            return {"rows": r, "region": "R", "factor": 1.0, "kind": "add", "classifier": "main"}
        return {"rows": None, "region": "R", "factor": 1.0, "kind": "add", "classifier": "main",
                "note": "needs stage-1 thresholds of the November scenes (--november-plan); costed from scene_levels"}
    if alt == "s2_add":
        s = pd.read_csv(PLAN_S, low_memory=False)
        s["sectors"] = s["sectors"].fillna("").astype(str)
        return {"rows": _export_cols(s), "region": "R", "factor": S2_FACTOR, "kind": "add", "classifier": "main",
                "s2": True}
    raise SystemExit(f"unknown alternative {alt}")


def november_scene_km2() -> tuple[int, float]:
    lv = pd.read_csv(INTERIM / "levels" / "scene_levels.csv", low_memory=False)
    nov = lv[lv["epoch"].isin(["E1", "E2", "E3"]) & (lv["month"] == 11) & (lv["cloud_scene_pct"] < 60)
             & (lv["sensor"] != "MSI_S2")].copy()
    unit = (nov["WRS_PATH"].fillna(0).astype(int).astype(str).str.zfill(3)
            + nov["WRS_ROW"].fillna(0).astype(int).astype(str).str.zfill(3))
    cm = json.loads((RESULTS / "inventory" / "stage2_cost_model.json").read_text(encoding="utf-8"))
    zone = {u["unit"]: u["zone_km2"] for u in cm["units"]}
    km2 = unit.map(zone).fillna(0.0)
    return int((km2 > 0).sum()), float(km2.sum())


def cost(alt: str, p: dict, plan: pd.DataFrame, total_km2: float, area_km2: dict) -> dict:
    """EECU-h (central and high rate) of one alternative, with the arithmetic."""
    area = area_km2[p["region"]]
    if alt == "november":
        n, km2 = november_scene_km2()
        share = km2 / total_km2
        stage1_h = n * 3.25 / 3600            # stage-1 run 3 Landsat rate, EECU-s per scene (cost_plan.json)
    elif p.get("s2"):
        share, stage1_h = 1.0, 0.0
    else:
        rows = p["rows"]
        share = float(_scene_km2(rows).sum() / total_km2) if len(rows) else 0.0
        if "rows_add" in p:
            share *= 2                        # SUB and ADD of the same scenes
        stage1_h = 0.0
    out = {}
    for k, rate in RATE.items():
        h = rate * area * share * p["factor"] / 3600 + stage1_h
        out[f"eecu_h_{k}"] = round(h, 2)
    out["arithmetic"] = (f"{RATE['central']}-{RATE['high']} EECU-s/km2 x {area:.0f} km2 ({p['region']}) x "
                         f"{share:.3f} (scene-km2 share) x {p['factor']:g}" + (f" + {stage1_h:.2f} h stage 1"
                                                                                  if stage1_h else ""))
    out["share"], out["region"], out["kind"], out["classifier"] = share, p["region"], p["kind"], p["classifier"]
    out["n_scenes"] = None if p["rows"] is None else int(p["rows"]["scene"].nunique())
    return out


# --- Earth Engine (not run by the robustness agent) ----------------------------------------

def patch_classifier(classifier: str):
    """Patch the Earth Engine graph functions for one alternative (this process only)."""
    from caspian import masks, pipeline_ee, water
    import ee
    if classifier in ("main", "mndwi_fixed0"):
        return
    if classifier.startswith("ndvi"):
        veg = int(classifier[4:]) / 100.0
        orig = pipeline_ee.water_valid
        pipeline_ee.water_valid = lambda img, thr, ndvi_veg=veg, observed=False: orig(img, thr, ndvi_veg, observed)
        return
    if classifier == "snow_all_months":
        masks.NDSI_MONTHS = tuple(range(1, 13))
        return
    if classifier == "awei_sh":
        masks.CLASSIFY_BANDS = list(masks.COMMON)

        def awei_valid(img, thr, ndvi_veg=water.NDVI_VEG, observed=False):
            a = (img.select("blue").add(img.select("green").multiply(2.5))
                 .subtract(img.select("nir").add(img.select("swir1")).multiply(1.5))
                 .subtract(img.select("swir2").multiply(0.25)))
            n = water.ee_nd(img, "nir", "red")
            w = a.gt(0)
            v = w.Or(n.lte(ndvi_veg).unmask(0))
            known = w.Or(n.mask())
            bands, names = ([w, v, w.Or(w.Not())], ["w", "v", "o"]) if observed else ([w, v], ["w", "v"])
            return ee.Image.cat(bands).rename(names).uint16().updateMask(a.mask()).updateMask(known)
        pipeline_ee.water_valid = awei_valid
        return
    raise ValueError(classifier)


def r_targets_factory(a6s, blocks: pd.DataFrame):
    """A replacement for a6_stage2_export.targets yielding the R blocks (cover within the
    25 km square, no sector clip; main groups; w and v only)."""
    from shapely.geometry import box
    from shapely.ops import transform
    size = float(design.BLOCK_SIZE_M)

    def targets(st, blocks_=0, only=None):
        cov = a6s.cover_laea()
        for i, j in zip(blocks["i"], blocks["j"]):
            key = f"{int(i)}_{int(j)}"
            if only and key not in only:
                continue
            part = a6s.clean_part(cov.intersection(box(i * size, j * size, (i + 1) * size, (j + 1) * size)))
            if part.is_empty:
                continue
            region_ll = a6s.ee_planar(transform(a6s.INV, part))
            x0, y0, x1, y1 = part.bounds
            x0, y1 = math.floor(x0 / a6s.RES) * a6s.RES, math.ceil(y1 / a6s.RES) * a6s.RES
            dims = (int(math.ceil((x1 - x0) / a6s.RES)), int(math.ceil((y1 - y0) / a6s.RES)))
            yield key, part, region_ll, [a6s.RES, 0, x0, 0, -a6s.RES, y1], dims, st["groups"], None, False
    return targets


def run(alt: str, tag: str, budget: float, only=None, resume: bool = False) -> int:
    a6s = _load("a6_stage2_export")
    plan = pd.read_csv(PLAN_L, low_memory=False)
    p = build_plan(alt, plan)
    if p["rows"] is None:
        raise SystemExit(p["note"])
    if p.get("rows_add") is not None:
        raise SystemExit("run the SUB and ADD parts as two tags: --alt e1_end_* exports the SUB rows; build the ADD "
                         "plan from build_plan(...)['rows_add'] under a second tag")
    plan_csv = INTERIM / "stage2" / f"plan_r18_{alt}.csv"
    p["rows"].to_csv(plan_csv, index=False)
    st_path = a6s.STATE / f"stage2_{tag}.json"
    if st_path.exists() and not resume:
        raise SystemExit(f"{st_path} exists: choose a new tag, or --resume (blocks already written are skipped)")
    if not st_path.exists():
        st_path.write_text(json.dumps({"tag": tag, "plan": str(plan_csv), "n_rows": len(p["rows"]),
                                   "groups": sorted(p["rows"]["group"].unique()), "chunks": [], "tiles": {},
                                   "robustness_alt": alt, "classifier": p["classifier"], "region": p["region"]},
                                      indent=1), encoding="utf-8")
    patch_classifier(p["classifier"])
    if p["region"] == "R":
        a6s.targets = r_targets_factory(a6s, r_blocks())
    return a6s.interactive(tag, budget, only=only, workers=24, px=512, groups_per_req=16)


def month_total(live: bool = True) -> dict:
    """Month-to-date EECU: every interactive log in results/inventory (profiled requests),
    plus, with ``live``, the batch operations of this month from ee.data.listOperations
    (batchEecuUsageSeconds; metadata only, no computation)."""
    import glob
    inter = 0.0
    per = {}
    for f in sorted(glob.glob(str(RESULTS / "inventory" / "*interactive_log.jsonl"))):
        s = 0.0
        for line in open(f, encoding="utf-8"):
            try:
                s += float(json.loads(line).get("eecu_s") or 0)
            except ValueError:
                continue
        per[Path(f).name] = round(s / 3600, 2)
        inter += s
    out = {"interactive_logged_h": round(inter / 3600, 2), "by_log_h": per}
    if live:
        import ee
        from caspian.config import EE_PROJECT
        ee.Initialize(project=EE_PROJECT)
        month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
        b = 0.0
        for op in ee.data.listOperations():
            md = op.get("metadata", {})
            if str(md.get("createTime", "")).startswith(month):
                b += float(md.get("batchEecuUsageSeconds") or 0)
        out["batch_live_h"] = round(b / 3600, 2)
    out["total_h"] = round(out["interactive_logged_h"] + out.get("batch_live_h", 0.0), 2)
    return out


def plan_cmd(alts, dry_run: bool, november_plan=None) -> int:
    plan = pd.read_csv(PLAN_L, low_memory=False)
    total = float(_scene_km2(plan).sum())
    rb = r_blocks()
    a6s_cover = R.load_cover(RESULTS / "study_area" / "zone_v6_cover_buffered.geojson", CRS_EQUAL_AREA).area / 1e6
    area = {"R": float(rb["cover_km2"].sum()), "cover": a6s_cover}
    rows = {}
    for alt in alts:
        p = build_plan(alt, plan, november_plan)
        rows[alt] = cost(alt, p, plan, total, area)
        if alt in ("tau015",):
            rows[alt]["n_sub_pairs"], rows[alt]["mixed_acq_sector"] = p["n_pairs"], p["mixed"]
        if "n_window_scenes" in p:
            rows[alt]["n_window_scenes"] = p["n_window_scenes"]
    tot = {k: round(sum(r[k] for r in rows.values()), 1) for k in ("eecu_h_central", "eecu_h_high")}
    rec = {"R_blocks": [f"{int(i)}_{int(j)}" for i, j in zip(rb["i"], rb["j"])], "areas_km2": area,
           "rate_eecu_s_per_km2": RATE, "alternatives": rows, "total": tot,
           "note": "dry run: nothing was sent to Earth Engine" if dry_run else "plan only"}
    out = RESULTS / "robustness" / "r18_rblock_export"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    for alt, r in rows.items():
        print(f"{alt:16s} {r['kind']:15s} {r['region']:6s} share {r['share']:.3f}  "
              f"{r['eecu_h_central']:6.2f}-{r['eecu_h_high']:6.2f} EECU-h  scenes {r['n_scenes']}")
    print("total", tot)
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["plan", "run", "budget"])
    ap.add_argument("--alt", default="ALL")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--november-plan", default=None)
    ap.add_argument("--tag")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--i-have-approval", action="store_true")
    ap.add_argument("--blocks", default=None, help="run: only these R block keys (pilot)")
    ap.add_argument("--resume", action="store_true", help="run: continue an existing tag")
    a = ap.parse_args(argv)
    alts = list(ALTS) if a.alt == "ALL" else a.alt.split(",")
    if a.cmd == "plan":
        return plan_cmd(alts, a.dry_run, a.november_plan)
    if a.cmd == "budget":
        print(json.dumps(month_total(live=not a.dry_run), indent=1))
        return 0
    if not a.i_have_approval or a.budget <= 0 or not a.tag or len(alts) != 1:
        raise SystemExit("run needs one --alt, a new --tag, --budget (EECU-s) and --i-have-approval (Earth Engine "
                         "costs quota; the lead approves each item and its budget)")
    return run(alts[0], a.tag, a.budget, a.blocks.split(",") if a.blocks else None, a.resume)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
