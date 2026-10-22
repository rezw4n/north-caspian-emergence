"""Robustness verdicts per check (01-11) and per named variant, from the
summaries the robustness runs wrote (results/robustness/*/summary.json), by the rule
below. Writes results/robustness/verdicts.json, which tools/build_metrics.py
(robustness_metrics) turns into metrics.json keys. Reads nothing else.

Rule
- Headline decisions compared:
  - H2: the overall status (supported / not supported) per source (ATL06, ATL08) and
    product variant (matched, total), validation.h2_decision on the paired tables;
  - H1: per product (shape variant decides; total for the total-variant check),
    whether it mispredicts in a majority (>= 4) of the 7 sectors; and the deltaic-or-
    vegetated block test (significant at 0.05) where both runs have it.
- Detail: the per-sector H1 outcomes and the per-sector H2 support.
- Magnitude: the DEM's overall NMAD changes by more than 10% (relative), or its overall
  bias by more than 0.10 m, against the primary on the same points.
- Verdict of a variant: "robust" (no headline change, no detail change, no magnitude
  change), "partly sensitive" (details or magnitudes change, every headline decision
  holds), "sensitive" (a headline decision flips). A check takes the worst verdict of its variants. A check with no variant run
  is "not run: <reason>" or "pending EE". Method comparisons without a decision rule
  (TIN vs kriging) use the H2 headline ordering: the DEM's NMAD below every product's
  NMAD on the same points. Prior work (check 11): "robust" if the published value lies
  within 10% of our estimate under every datum reading, "partly sensitive" under some,
  "sensitive" under none.

Each variant record lists in "why" the criteria that fired (empty when robust).

Usage: python scripts/r20_verdicts.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import caspian  # noqa: F401
import numpy as np

from caspian.config import RESULTS

ROB = RESULTS / "robustness"
ORDER = {"robust": 0, "partly sensitive": 1, "sensitive": 2}
# what a check's verdict does not cover (stated with it)
NOTES = {"check06": "per-sector datum constants (the lead's red-team run, H1) and the validation side (c and the "
                    "product offsets at +-1 bootstrap sigma, both mean-tide recipes; H2 on the sources listed per "
                    "variant, sources_evaluated); the D-074 total-variant hindcast was not run; the DAHITI -> BHS-77 "
                    "offset (D-093) cancels in every metric (reporting only)",
         "check07": "kriging against TIN on a seeded subsample of 8 blocks (same points); 0.20 m bins and the "
                    "static-contour rule on the whole DEM (H2 side); 0.05 m bins on the block sample R; "
                    "natural-neighbour interpolation not run (no SciPy implementation)",
         "check08": "block size (H2 at 25 and 12.5 km against 50 km) and the calibration split (20 seeded "
                    "reassignments of the 50 km blocks as one variant, worst over the seeds, and the calibration/"
                    "test swap); each moves the test points, so the magnitude criterion does not apply; the split "
                    "variants are post-hoc sensitivities that reuse test blocks for fitting",
         "check09": "NDVI 0.20 / 0.40 on the block sample R; the as-dry / as-water bounds (o band) on the whole "
                    "DEM, H2 side; the H1 side of the bounds needs a full hindcast and was not run",
         "check10": "H1 on E2 + E3 and the morphology table decide; the E1 minus E2 + E3 limb-difference map "
                    "(r10_morphology, two-limb band) is descriptive"}
N_SECTORS = 7


def _load(name: str) -> dict | None:
    p = ROB / name / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def worst(vs):
    vs = [v for v in vs if v in ORDER]
    return max(vs, key=ORDER.get) if vs else None


def h2_variant(h2: dict, magnitude: bool = True) -> tuple[str, dict, list]:
    """Verdict, numbers and the criteria that fired ("why") of one H2 variant
    (both sources where present). ``magnitude=False`` when the variant is evaluated on
    other test points (block-size checks), where the magnitude criterion, defined
    on the same points, does not apply; the numbers are still reported."""
    nums, head, detail, why = {}, False, False, []
    for src, r in h2.items():
        s = src.lower()
        for q in ("nmad", "rmse", "bias", "n"):
            v = r["dem"].get(f"overall|{q}")
            if v:
                nums[f"{s}.dem_{q}{'_m' if q != 'n' else ''}_primary"] = v[0]
                nums[f"{s}.dem_{q}{'_m' if q != 'n' else ''}"] = v[1]
        nm, bs = r["dem"].get("overall|nmad"), r["dem"].get("overall|bias")
        if magnitude and nm and nm[0] and abs(nm[1] - nm[0]) / nm[0] > 0.10:
            detail = True
            why.append(f"{s}: DEM NMAD {nm[0]:.3f} -> {nm[1]:.3f} m ({(nm[1] - nm[0]) / nm[0]:+.0%})")
        if magnitude and bs and abs(bs[1] - bs[0]) > 0.10:
            detail = True
            why.append(f"{s}: DEM bias {bs[0]:+.3f} -> {bs[1]:+.3f} m")
        cov = r.get("coverage", {}).get("overall")
        if cov:
            nums[f"{s}.coverage_primary"], nums[f"{s}.coverage"] = cov
        for var in ("matched", "total"):
            d = r.get(f"h2_{var}")
            if not d:
                continue
            nums[f"{s}.h2_{var}_supported_primary"] = int(d["primary"] == "supported")
            nums[f"{s}.h2_{var}_supported"] = int(d["variant"] == "supported")
            if d["primary"] != d["variant"]:
                head = True
                why.append(f"{s}: H2 ({var}) {d['primary']} -> {d['variant']}")
            a, b = sorted(d["sectors_supported_primary"]), sorted(d["sectors_supported_variant"])
            if a != b:
                detail = True
                why.append(f"{s}: sectors supporting H2 ({var}) {a} -> {b}")
    nums["sources_evaluated"] = len(h2)
    return ("sensitive" if head else "partly sensitive" if detail else "robust"), nums, why


def h1_variant(h1: dict, variant: str = "shape") -> tuple[str, dict, list]:
    nums, head, detail, why = {}, False, False, []
    for key, r in h1["by_product"].items():
        prod, var = key.split("|")
        if var != variant or "mispredicts_in_primary" not in r:
            continue
        a, b = len(r["mispredicts_in_primary"]), len(r["mispredicts_in_variant"])
        nums[f"{prod}.n_sectors_mispredict_primary"], nums[f"{prod}.n_sectors_mispredict"] = a, b
        if (a >= 4) != (b >= 4):
            head = True
            why.append(f"{prod}: majority rule flips ({a} -> {b} of {N_SECTORS} sectors mispredict)")
        if r["changed_sectors"]:
            detail = True
            why.append(f"{prod}: sector outcomes change in {r['changed_sectors']} ({a} -> {b} mispredict)")
        bt_v, bt_m = r.get("block_test_alt"), r.get("block_test_main_fixed") or r.get("block_test_main_fix")
        if bt_v:
            nums[f"{prod}.block_test_p"] = bt_v["p"]
        if bt_v and bt_m:
            nums[f"{prod}.block_test_p_primary_fix"] = bt_m["p"]
            if bt_v["significant"] != bt_m["significant"]:
                head = True
                why.append(f"{prod}: block test p {bt_m['p']:.3g} -> {bt_v['p']:.3g} (significance changes)")
    return ("sensitive" if head else "partly sensitive" if detail else "robust"), nums, why


RBLOCK = {"fixed_mndwi0": "check01", "awei_sh": "check01", "sub_tm": "check02", "sub_etm": "check02",
          "sub_slcoff": "check02", "sub_oli": "check02", "s2_add": "check02", "snow_all_months": "check03",
          "november": "check03", "tau015": "check04", "bins005": "check07", "ndvi020": "check09",
          "ndvi040": "check09"}


def rlocal_variant(s: dict) -> tuple[str, dict, list]:
    """R-local comparison (scripts/r19_local_runs.py rblock): the H2 headline ordering
    (DEM NMAD below every product's NMAD on the same R points) overall decides; a
    per-sector ordering change is a detail."""
    nums, head, detail, why = {"eecu_h_export": s.get("eecu_s_export", 0) / 3600}, False, False, []
    for src, rec in s["h2_rlocal"].items():
        k = src.lower()
        for scope, r in rec.items():
            if scope == "nmad_diff_ci95_m":
                nums[f"{k}.nmad_diff_ci95_low_m"], nums[f"{k}.nmad_diff_ci95_high_m"] = r
                continue
            if scope == "overall":
                nums[f"{k}.dem_nmad_m_primary"], nums[f"{k}.dem_nmad_m"] = r["dem_primary_nmad_m"], r["dem_variant_nmad_m"]
                nums[f"{k}.dem_rmse_m_primary"], nums[f"{k}.dem_rmse_m"] = r["dem_primary_rmse_m"], r["dem_variant_rmse_m"]
                nums[f"{k}.min_product_nmad_m"], nums[f"{k}.n_points"] = r["min_product_nmad_m"], r["n"]
                if r["ordering_primary"] != r["ordering_variant"]:
                    head = True
                    why.append(f"{k}: H2 ordering {r['ordering_primary']} -> {r['ordering_variant']}")
                nums[f"{k}.dem_bias_m_primary"], nums[f"{k}.dem_bias_m"] = r["dem_primary_bias_m"], r["dem_variant_bias_m"]
                a, b = r["dem_primary_nmad_m"], r["dem_variant_nmad_m"]
                if abs(b - a) / a > 0.10:
                    detail = True
                    why.append(f"{k}: DEM NMAD {a:.3f} -> {b:.3f} m ({(b - a) / a:+.0%})")
                a, b = r["dem_primary_bias_m"], r["dem_variant_bias_m"]
                if abs(b - a) > 0.10:
                    detail = True
                    why.append(f"{k}: DEM bias {a:+.3f} -> {b:+.3f} m")
            elif r["ordering_primary"] != r["ordering_variant"]:
                detail = True
                why.append(f"{k} {scope}: H2 ordering {r['ordering_primary']} -> {r['ordering_variant']}")
    if "contour_offset_median_m" in s:
        nums["contour_offset_median_m"] = s["contour_offset_median_m"]
        nums["share_bins_offset_above_noise"] = s["share_bins_offset_above_noise"]
    return ("sensitive" if head else "partly sensitive" if detail else "robust"), nums, why


def main() -> int:
    out: dict = {"created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "rule": "DECISIONS D-092", "checks": {}, "named": {}}
    C = out["checks"]

    def add(check, variant, verdict, nums, source, why=()):
        C.setdefault(check, {"variants": {}})["variants"][variant] = {"verdict": verdict, "metrics": nums,
                                                                      "source": source, "why": list(why)}

    # check 04: setup correction (first order), Komsomolets screening, tau = 0.15 on R
    s = _load("r04_screening_setupcorr")
    if s and "h2" in s:
        v, n, w = h2_variant(s["h2"])
        n["shift_p05_m"], n["shift_p95_m"] = s["shift_m"]["p05"], s["shift_m"]["p95"]
        add("check04", "setup_correction", v, n, "results/robustness/r04_screening_setupcorr/summary.json", w)
    s = _load("r04_screening_komsomolets")
    if s:
        n = {}
        for r in s["table"]:
            key = f"tau{int(round(r['tau_compared_m'] * 100)):03d}cm.{r['screening']}"
            n[f"{key}.scatter_median_m"], n[f"{key}.scatter_p90_m"] = r["S_median_m"], r["p90_m"]
        unscr = [r for r in s["table"] if r["screening"] == "unscreened"][0]
        worse = all(r["p90_m"] >= unscr["p90_m"] - 1e-9 for r in s["table"] if r["screening"] == "screened")
        w = [f"screening lowers the p90 scatter at tau {r['tau_compared_m']:.2f} m: {unscr['p90_m']:.1f} -> "
             f"{r['p90_m']:.1f} m" for r in s["table"]
             if r["screening"] == "screened" and r["p90_m"] < unscr["p90_m"] - 1e-9]
        add("check04", "komsomolets_screening", "robust" if worse else "partly sensitive", n,
            "results/robustness/r04_screening_komsomolets/summary.json", w)
    # R-block alternatives (Earth Engine exports on the sample R, scripts/r18 + r19 rblock)
    for alt, check in RBLOCK.items():
        s = _load(f"rblock_{alt}")
        if s and "h2_rlocal" in s:
            v, n, w = rlocal_variant(s)
            add(check, f"{alt}_R", v, n, f"results/robustness/rblock_{alt}/summary.json", w)
    # check 05: level series
    s = _load("r05_levels")
    if s:
        for name, r in s["alternatives"].items():
            n = {"shift_p05_m": r["shift_m"]["p05"], "shift_p95_m": r["shift_m"]["p95"],
                 "share_acq_rebin": r["share_acq_rebin"]}
            if "h2" in r:
                v, n2, w = h2_variant(r["h2"])
                n.update(n2)
                n["dem_diff_rms_m"] = r["dem_difference_overall"]["rms_m"]
            else:
                v = "robust" if max(abs(r["shift_m"]["p05"]), abs(r["shift_m"]["p95"])) < 0.02 else None
                n["note_not_rebuilt"] = 1
                w = []
            if v:
                add("check05", name, v, n, "results/robustness/r05_levels/summary.json", w)
    # check 06: total error without the -0.41 m constant
    s = _load("r06_datums")
    if s and "by_product" in s:
        v, n, w = h1_variant(s, "total")
        n["correction_difference_mean_m"] = s["primary_minus_variant_correction_m"]["mean"]
        add("check06", "total_without_nga_constant", v, n, "results/robustness/r06_datums/summary.json", w)
    # check 06, validation side: c and product offsets at +-1 sigma, mean-tide recipes
    s = _load("r06_datums_validation")
    if s:
        for name, h2 in sorted(s.get("variants", {}).items()):
            if h2:
                v, n, w = h2_variant(h2)
                add("check06", name, v, n, "results/robustness/r06_datums_validation/summary.json", w)
    # check 06: the per-sector datum run (red team; results/hindcast/main_sectorshape)
    s = _load("r06_datums_sectorshape")
    if s and "by_product" in s:
        v, n, w = h1_variant(s, "shape")
        add("check06", "sector_shape_constants", v, n, "results/robustness/r06_datums_sectorshape/summary.json", w)
    # check 08: block sizes 25 and 12.5 km (sensitivities; other test points)
    s = _load("r08_blocking")
    if s and s.get("h2"):
        for name, h2 in sorted(s["h2"].items()):
            if h2:
                v, n, w = h2_variant(h2, magnitude=not s.get("same_points") is False)
                add("check08", name, v, n, "results/robustness/r08_blocking/summary.json", w)
    # check 08: the calibration split (scripts/r08_blocking.py splits). Each reassignment
    # moves the test points, so the magnitude criterion (same points) does not apply, as for
    # the block sizes. The seeded reassignments are one variant whose verdict is the worst
    # over the seeds (the check's verdict is the same as listing each seed), with the
    # number of seeds per verdict and per H2 status; the swap is its own variant.
    if s and s.get("splits"):
        src_ = "results/robustness/r08_blocking/summary.json"
        seeds = {k: h for k, h in sorted(s["splits"].items()) if k.startswith("split_seed") and h}
        if seeds:
            res = {k: h2_variant(h, magnitude=False) for k, h in seeds.items()}
            n = {"n_seeds": len(res)}
            for verdict in ORDER:
                n[f"n_seeds_{verdict.replace(' ', '_')}"] = sum(r[0] == verdict for r in res.values())
            for key in sorted({k for r in res.values() for k in r[1] if k.endswith("_supported")}):
                src, rest = key.split(".", 1)
                n[f"{src}.n_seeds_{rest}"] = sum(int(r[1].get(key, 0)) for r in res.values())
                prim = [r[1].get(f"{key}_primary") for r in res.values() if f"{key}_primary" in r[1]]
                if prim:
                    n[f"{key}_primary"] = prim[0]
            for q, lo, hi in (("dem_nmad_m", "dem_nmad_min_m", "dem_nmad_max_m"), ("coverage", "coverage_min", "coverage_max")):
                for src in sorted({k.split(".")[0] for r in res.values() for k in r[1] if "." in k}):
                    vals = [r[1][f"{src}.{q}"] for r in res.values() if f"{src}.{q}" in r[1]]
                    if vals:
                        n[f"{src}.{lo}"], n[f"{src}.{hi}"] = float(min(vals)), float(max(vals))
            w = [f"{k}: {x}" for k, r in res.items() for x in r[2] if ": H2 (" in x]   # headline flips only
            add("check08", "split_reassignment", worst([r[0] for r in res.values()]), n, src_, w)
        sw = s["splits"].get("split_swap_calib_test")
        if sw:
            v, n, w = h2_variant(sw, magnitude=False)
            add("check08", "split_swap_calib_test", v, n, src_, w)
    # check 09 bounds: reed beds counted as dry or as water (o band)
    s = _load("r09_vegetation")
    if s:
        for mode, r in sorted(s.get("alternatives", {}).items()):
            if r.get("h2"):
                v, n, w = h2_variant(r["h2"])
                n["dem_diff_mean_m"] = r["dem_difference_overall"]["mean_m"]
                n["dem_diff_rms_m"] = r["dem_difference_overall"]["rms_m"]
                if r.get("share_observations_vegetation") is not None:
                    n["share_observations_vegetation"] = r["share_observations_vegetation"]
                add("check09", f"vegetation_{mode}", v, n, "results/robustness/r09_vegetation/summary.json", w)
    # check 07: kriging, 0.20 m bins, no static exclusion
    s = _load("r07_interpolation")
    if s:
        n, ok, big, w = {}, [], [], []
        for r in s["overall"]:
            if r["scope"] == "split:test":
                src = r["source"].lower()
                for q in ("nmad", "rmse", "bias"):
                    n[f"{src}.tin_{q}_m"], n[f"{src}.gp_{q}_m"] = r[f"tin_{q}"], r[f"gp_{q}"]
                if "min_product_nmad" in r:
                    n[f"{src}.min_product_nmad_m"] = r["min_product_nmad"]
                    ok.append((r["tin_nmad"] < r["min_product_nmad"]) == (r["gp_nmad"] < r["min_product_nmad"]))
                    if not ok[-1]:
                        w.append(f"{src}: H2 ordering differs between TIN and kriging")
                big.append(abs(r["gp_nmad"] - r["tin_nmad"]) / r["tin_nmad"] > 0.10
                           or abs(r["gp_bias"] - r["tin_bias"]) > 0.10)
                if big[-1]:
                    w.append(f"{src}: NMAD {r['tin_nmad']:.3f} (TIN) vs {r['gp_nmad']:.3f} m (kriging); bias "
                             f"{r['tin_bias']:+.3f} vs {r['gp_bias']:+.3f} m")
        if not ok:
            w.append("no product NMAD on the subsample: ordering not evaluable")
        v = ("sensitive" if not all(ok) else "partly sensitive" if any(big) else "robust") if ok else "partly sensitive"
        add("check07", "kriging_subsample", v, n, "results/robustness/r07_interpolation/summary.json", w)
    for name, folder in (("bins_020m", "r07_interpolation_bw020"), ("no_static_exclusion", "r12_static_rule")):
        s = _load(folder)
        if s and "h2" in s:
            v, n, w = h2_variant(s["h2"])
            n["dem_diff_mean_m"] = s["dem_difference_overall"]["mean_m"]
            n["dem_diff_rms_m"] = s["dem_difference_overall"]["rms_m"]
            add("check07", name, v, n, f"results/robustness/{folder}/summary.json", w)
    # check 10: H1 on E2 + E3 only; morphology table
    s = _load("h1_e2e3")
    if s and "by_product" in s:
        v, n, w = h1_variant(s, "shape")
        add("check10", "h1_e2e3_only", v, n, "results/robustness/h1_e2e3/summary.json", w)
        v2, n2, w2 = h1_variant(s, "total")
        add("check10", "h1_e2e3_only_total", v2, n2, "results/robustness/h1_e2e3/summary.json", w2)
    s = _load("r10_morphology")
    if s:
        n = {}
        for r in s["summary_shape"]:
            n[f"{r['product']}.{r['sector']}.n_levels_e1_better"] = r["n_e1_better"]
            n[f"{r['product']}.{r['sector']}.n_levels_shared"] = r["n_bins"]
        tot_b = sum(r["n_e1_better"] for r in s["summary_shape"])
        tot_n = sum(r["n_bins"] for r in s["summary_shape"])
        n["share_levels_e1_better"] = tot_b / tot_n if tot_n else None
        # morphology would explain the errors if products fit the older (E1) seabed better
        ok = bool(tot_n) and tot_b / tot_n < 0.5
        w = [] if ok else [f"products fit the E1 seabed better at {tot_b} of {tot_n} shared levels"]
        add("check10", "morphology_table", "robust" if ok else "partly sensitive", n,
            "results/robustness/r10_morphology/summary.json", w)
    # check 11: prior work
    s = _load("r11_prior_work")
    if s:
        pub = s["published"]["value_km2_per_m"]
        ests = {k: v["km2_per_m_emerged_band"] for k, v in s["dem_hypsometry_km2_per_m"].items() if k.startswith("range")}
        within = [abs(e - pub) / pub <= 0.10 for e in ests.values()]
        n = {"published_km2_per_m": pub, **{f"dem_band_{k}_km2_per_m": v for k, v in ests.items()}}
        w = [f"datum reading {k}: {e:.0f} vs published {pub:.0f} km2 per m ({(e - pub) / pub:+.0%})"
             for k, e in ests.items() if abs(e - pub) / pub > 0.10]
        add("check11", "akbari2020_area_slope", "robust" if all(within) else "partly sensitive" if any(within)
            else "sensitive", n, "results/robustness/r11_prior_work/summary.json", w)
    # named variants outside the 11 checks
    for name, folder in (("emergence_lag_365d", "r13_emergence_lag"), ("products_nearest", "r14_product_resampling")):
        s = _load(folder)
        if s and "h2" in s:
            v, n, w = h2_variant(s["h2"])
            rec = {"verdict": v, "metrics": n, "source": f"results/robustness/{folder}/summary.json", "why": w}
            if "h1" in s:
                v1, n1, w1 = h1_variant(s["h1"], "shape")
                rec["verdict"] = worst([v, v1])
                rec["metrics"].update({f"h1.{k}": x for k, x in n1.items()})
                rec["why"] += [f"H1 {x}" for x in w1]
            out["named"][name] = rec
    # verdicts per check, and the checks with nothing run
    reasons = {
        "check01": "pending EE",
        "check02": "pending EE",
        "check03": "pending EE",
        "check08": "not run: no block-size table and no split reassignment",
        "check09": "not run: vegetated population reported by the lead (D-088 addendum); as-dry/as-water bounds not run",
    }
    for k in [f"check{i:02d}" for i in range(1, 12)]:
        rec = C.setdefault(k, {"variants": {}})
        vs = [v["verdict"] for v in rec["variants"].values()]
        rec["verdict"] = worst(vs) or reasons.get(k, "not run: no variant completed")
        if k in NOTES and vs:
            rec["note"] = NOTES[k]
    (ROB / "verdicts.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for k, rec in C.items():
        print(k, rec["verdict"], {n: v["verdict"] for n, v in rec["variants"].items()})
        for n, v in rec["variants"].items():
            for x in v.get("why", []):
                print(f"    {n}: {x}")
    for k, rec in out["named"].items():
        print(k, rec["verdict"])
        for x in rec.get("why", []):
            print(f"    {x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
