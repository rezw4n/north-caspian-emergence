"""A4: water level and 1-sigma for every inventoried scene (D-018, D-020, D-021 and
their review corrections).

Primary: DAHITI target 39. Record noise sigma and the Brownian-bridge diffusion q are
estimated jointly from cross-validated linear-interpolation residuals
(caspian.levels.estimate_noise_and_diffusion); per-record 1-sigma =
sqrt(wse_u^2 + sigma^2).

Gap-fill: Hydroweb L_caspian, used for scenes outside the DAHITI record and where
DAHITI's bracketing gap exceeds MAX_GAP_DAYS (60 days: the default of caspian.levels,
fixed in commit aabacee before PREREG v1.0 and before the DAHITI data arrived; not in
PREREG v1.0, stated for the record in PREREG Amendment 4 item 2; D-038). Hydroweb is shifted onto the DAHITI reference by a moving median
of the +/-20-day DAHITI - Hydroweb pair differences within +/-OFFSET_HALF_WINDOW_DAYS
of each scene, because the difference drifts (period medians 0.14 -> 0.08 m). Per-record
random 1-sigma = sqrt(noise^2 + NMAD^2 of the pair residuals about the moving median);
the offset error is common to the bracketing records and is added after interpolation.
It combines the SE of the moving median with tau(t), the year-to-year variability of
the offset beyond sampling noise in the years within the same +/-2-year window
(at least TAU_MIN_YEARS years; caspian.levels.local_excess_sd), floored at the
record-wide tau. A single record-wide tau would average years in which the offset is
stable within its sampling noise with the more variable gap-fill years (second D-020
correction). The two parts of each scene's 1-sigma are exported (sigma_random_m,
sigma_offset_m; sigma_m^2 = sum of squares) so that A6 can treat the offset part as
common to all gap-filled scenes of a period.

Hydroweb's record noise is estimated with q fixed to DAHITI's, and the Hydroweb bridge
uses that q. This is a choice, not a property of the sea: Hydroweb's own joint fit gives
a much smaller q (reported under diagnostics, with the effect of using it instead).

Epochs (PREREG section 4): E1 from the start of the record to the date of the maximum,
read as the maximum of the 3-point running median of DAHITI (a single noisy record
cannot set its level); E2 to 2010-12-31; E3 after. The top of the running median is
flat, so its date is not sharper than the raw one: the E1 end under k = 1, 3, 5, 7, 9
is reported under diagnostics (e1_end_sensitivity). Scenes before the altimetry record
(1984 - Sep 1992) get no level here (PREREG section 2 fallback).

Inputs:  results/qc/level_crosscheck_w20d.json and level_crosscheck_pairs_w20d.csv
         (scripts/a4_level_crosscheck.py 20)
Outputs: data/interim/levels/scene_levels.csv (git-ignored; one row per scene)
         results/levels/scene_levels_summary.json (committed)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import levels
from caspian.config import INTERIM, RAW, RESULTS
from caspian.io import read_dahiti, read_hydroweb

MAX_GAP_DAYS = 60.0
OFFSET_HALF_WINDOW_DAYS = 730.5   # +/-2 years (D-020 correction)
OFFSET_MIN_PAIRS = 20
TAU_MIN_YEARS = 5                 # years in the local tau window (second D-020 correction)
SMOOTH_K = 3                      # running median for the date of the maximum (D-021 correction)
SMOOTH_K_SENSITIVITY = (1, 3, 5, 7, 9)   # E1-end sensitivity, reported only
N_BOOT = 200                      # moving-block bootstrap for the SE of sigma and q
SEED = 0
BIN_WIDTH, BIN_ORIGIN = 0.10, -30.0   # PREREG section 4 bins, used here only for diagnostics
Z_THRESHOLD = 2.5                     # neighbour-residual diagnostic only (no record is removed)


def prepare() -> dict:
    """Level series, noise model and Hydroweb offset inputs shared by every level
    assignment (scenes here, SWOT passes in scripts/a5_setup_model.py)."""
    d = read_dahiti(RAW / "dahiti" / "dahiti_39_caspian_sea_water_level.csv").data
    hw = read_hydroweb(RAW / "hydroweb" / "extracted" / "HYDROWEB_LAKES_OPE" / "hydroprd_L_caspian.txt").data
    xc = json.loads((RESULTS / "qc" / "level_crosscheck_w20d.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(RESULTS / "qc" / "level_crosscheck_pairs_w20d.csv", parse_dates=["time"])
    if len(pairs) != xc["n_pairs"]:
        raise RuntimeError("pair table and crosscheck summary disagree; rerun a4_level_crosscheck.py 20")

    # DAHITI: joint record noise and diffusion
    nd = levels.estimate_noise_and_diffusion(d["time"], d["level_m"], max_span_days=MAX_GAP_DAYS,
                                             n_boot=N_BOOT, seed=SEED)
    floor_d, q_d = nd.sigma, nd.q
    sig_d = np.sqrt(np.nan_to_num(d["sigma_m"].to_numpy()) ** 2 + floor_d ** 2)

    # Hydroweb: record noise given DAHITI's q (the bridge term is removed from its residuals).
    # q is a property of each series' interpolation error, not of the sea (caspian.levels);
    # DAHITI's larger q is used for the Hydroweb bridge as the conservative choice. The
    # effect of Hydroweb's own joint (sigma, q) fit instead is reported under diagnostics.
    q_h = q_d
    nh = levels.estimate_noise_and_diffusion(hw["time"], hw["level_m"], max_span_days=MAX_GAP_DAYS, q=q_h,
                                             n_boot=N_BOOT, seed=SEED)
    floor_h = nh.sigma
    noise_h = np.maximum(np.nan_to_num(hw["sigma_m"].to_numpy()), floor_h)  # per-record random noise

    # Offset onto the DAHITI reference: moving median of the pair differences
    off_pairs = levels.moving_median_offset(pairs["time"], pairs["diff"], pairs["time"],
                                            OFFSET_HALF_WINDOW_DAYS, OFFSET_MIN_PAIRS)
    resid = pairs["diff"].to_numpy() - off_pairs["offset_m"].to_numpy()
    resid_nmad = levels._nmad(resid)
    excess = levels.offset_excess_sd(pairs["time"], pairs["diff"], OFFSET_HALF_WINDOW_DAYS, OFFSET_MIN_PAIRS)
    tau_record = excess["tau_m"]
    sig_h = np.sqrt(noise_h ** 2 + resid_nmad ** 2)
    return {"d": d, "hw": hw, "xc": xc, "pairs": pairs, "nd": nd, "nh": nh, "floor_d": floor_d, "q_d": q_d,
            "sig_d": sig_d, "q_h": q_h, "floor_h": floor_h, "noise_h": noise_h, "resid_nmad": resid_nmad,
            "excess": excess, "tau_record": tau_record, "sig_h": sig_h}


def levels_at(prep: dict, times) -> pd.DataFrame:
    """Level, 1-sigma (with its random and common-offset parts), gap and source at
    arbitrary times: DAHITI where its bracketing records are at most MAX_GAP_DAYS
    apart, otherwise Hydroweb shifted onto the DAHITI reference (D-036)."""
    d, hw, pairs, excess = prep["d"], prep["hw"], prep["pairs"], prep["excess"]
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    out = pd.DataFrame(index=times.index)
    off_sc = levels.moving_median_offset(pairs["time"], pairs["diff"], times, OFFSET_HALF_WINDOW_DAYS,
                                         OFFSET_MIN_PAIRS)
    tau_loc = levels.local_excess_sd(excess["by_year"], times, OFFSET_HALF_WINDOW_DAYS, TAU_MIN_YEARS)
    tau_sc = np.maximum(tau_loc["tau_m"].to_numpy(), prep["tau_record"])
    off_sigma = np.sqrt(off_sc["se_m"].to_numpy() ** 2 + tau_sc ** 2)
    ld = levels.interpolate(d["time"], d["level_m"], prep["sig_d"], times, q=prep["q_d"], max_gap_days=MAX_GAP_DAYS)
    lh = levels.interpolate_shifted(hw["time"], hw["level_m"], prep["sig_h"], times, off_sc["offset_m"],
                                    off_sigma, q=prep["q_h"], max_gap_days=MAX_GAP_DAYS)
    use_d = ~ld["in_gap"].to_numpy()
    use_h = ~use_d & ~lh["in_gap"].to_numpy()
    out["level_m"] = np.where(use_d, ld["level_m"], np.where(use_h, lh["level_m"], np.nan))
    out["sigma_m"] = np.where(use_d, ld["sigma_m"], np.where(use_h, lh["sigma_m"], np.nan))
    # 1-sigma parts: records and bridge (random between dates) and the Hydroweb offset
    # error (common to the gap-filled scenes of a period); sigma_m^2 = random^2 + offset^2.
    out["sigma_random_m"] = np.where(use_d, ld["sigma_m"], np.where(use_h, lh["sigma_random_m"], np.nan))
    out["sigma_offset_m"] = np.where(use_d, 0.0, np.where(use_h, lh["sigma_offset_m"], np.nan))
    out["gap_days"] = np.where(use_d, ld["gap_days"], np.where(use_h, lh["gap_days"], np.nan))
    out["level_source"] = np.select([use_d, use_h], ["dahiti", "hydroweb_shifted"], "none")
    out["hydroweb_offset_m"] = np.where(use_h, off_sc["offset_m"], np.nan)
    out["hydroweb_offset_tau_m"] = np.where(use_h, tau_sc, np.nan)
    return out, {"ld": ld, "lh": lh, "use_d": use_d, "use_h": use_h, "off_sc": off_sc, "off_sigma": off_sigma,
                 "tau_loc": tau_loc}


def main(out_dir=None) -> int:
    prep = prepare()
    d, hw, xc, pairs, nd, nh = prep["d"], prep["hw"], prep["xc"], prep["pairs"], prep["nd"], prep["nh"]
    floor_d, q_d, q_h, floor_h = prep["floor_d"], prep["q_d"], prep["q_h"], prep["floor_h"]
    noise_h, resid_nmad, excess, tau_record = prep["noise_h"], prep["resid_nmad"], prep["excess"], prep["tau_record"]
    scenes = pd.read_csv(INTERIM / "inventory" / "scenes.csv", parse_dates=["time"])
    lv, parts = levels_at(prep, scenes["time"])
    for c in lv.columns:
        scenes[c] = lv[c].to_numpy()
    ld, lh, use_d, use_h = parts["ld"], parts["lh"], parts["use_d"], parts["use_h"]
    off_sc, off_sigma, tau_loc = parts["off_sc"], parts["off_sigma"], parts["tau_loc"]   # tau_loc: make-all fix
    tau_sc = scenes["hydroweb_offset_tau_m"].to_numpy()

    sm = levels.smoothed_extremes(d["time"], d["level_m"], k=SMOOTH_K)
    t_max = sm["time_of_max"]
    t_max_raw = d.loc[d["level_m"].idxmax(), "time"]
    scenes["epoch"] = np.select([scenes["time"] <= t_max, scenes["time"] <= pd.Timestamp("2010-12-31 23:59:59")],
                                ["E1", "E2"], "E3")
    scenes.loc[scenes["level_source"] == "none", "epoch"] = np.where(
        scenes.loc[scenes["level_source"] == "none", "time"] < d["time"].min(), "pre_altimetry", "no_level")
    out = Path(out_dir) if out_dir else INTERIM / "levels"
    out.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(out / "scene_levels.csv", index=False)

    # --- diagnostics -----------------------------------------------------------------
    ok = scenes["level_source"] != "none"
    pooled = float(xc["offset_median_m"])
    lev_h = lh["level_m"].to_numpy()
    lev_pooled = lev_h - off_sc["offset_m"].to_numpy() + pooled
    bin_change = int((levels.bin_index(lev_h[use_h], BIN_WIDTH, BIN_ORIGIN)
                      != levels.bin_index(lev_pooled[use_h], BIN_WIDTH, BIN_ORIGIN)).sum())
    z = levels.neighbour_zscores(d["time"], d["level_m"], floor_d, q_d)
    zf = z[np.isfinite(z)]
    p_tail = math.erfc(Z_THRESHOLD / math.sqrt(2))
    two_step_floor = levels.empirical_sigma(d["time"], d["level_m"])
    two_step_q = levels.estimate_diffusion(
        d["time"], d["level_m"], np.sqrt(np.nan_to_num(d["sigma_m"].to_numpy()) ** 2 + two_step_floor ** 2),
        max_gap_days=MAX_GAP_DAYS)
    nh_free = levels.estimate_noise_and_diffusion(hw["time"], hw["level_m"], max_span_days=MAX_GAP_DAYS)
    # Effect of using Hydroweb's own joint (sigma, q) fit for its bridge instead of DAHITI's q
    noise_h_alt = np.maximum(np.nan_to_num(hw["sigma_m"].to_numpy()), nh_free.sigma)
    lh_alt = levels.interpolate_shifted(hw["time"], hw["level_m"], np.sqrt(noise_h_alt ** 2 + resid_nmad ** 2),
                                        scenes["time"], off_sc["offset_m"], off_sigma, q=nh_free.q,
                                        max_gap_days=MAX_GAP_DAYS)
    dsig_alt = lh_alt["sigma_m"].to_numpy()[use_h] - lh["sigma_m"].to_numpy()[use_h]
    # E1-end sensitivity to the running-median length (k = 1 is the single-record maximum)
    e1_sens = []
    for k in SMOOTH_K_SENSITIVITY:
        sk = levels.smoothed_extremes(d["time"], d["level_m"], k=k)
        e1_sens.append({"k": int(k), "time_of_max": str(sk["time_of_max"]), "max_m": sk["max_m"],
                        "range_m": sk["range_m"],
                        "n_E1_scenes": int(((scenes["time"] <= sk["time_of_max"]) & ok).sum())})
    rm = levels.running_median(d.sort_values("time")["level_m"].to_numpy(), SMOOTH_K)
    top2 = np.sort(rm)[-2:]
    # Offset error: the applied local tau against the record-wide and scene-weighted values
    fill_year = scenes.loc[use_h, "time"].dt.year.value_counts()
    yr_terms = {r["year"]: r["term_m2"] for r in excess["by_year"]}
    wy = [(int(fill_year[y]), yr_terms[y]) for y in fill_year.index if y in yr_terms]
    tau_scene_weighted = math.sqrt(max(sum(n * t for n, t in wy) / sum(n for n, _ in wy), 0.0)) if wy else None
    # In-sample check: DAHITI minus shifted Hydroweb at the +/-5-day epochs, by gap-fill year
    p5 = levels.pair_series(d["time"], d["level_m"], hw["time"], hw["level_m"], 5.0)
    o5 = levels.moving_median_offset(pairs["time"], pairs["diff"], p5["time"], OFFSET_HALF_WINDOW_DAYS,
                                     OFFSET_MIN_PAIRS)
    r5 = p5["diff"].to_numpy() - o5["offset_m"].to_numpy()
    y5 = pd.to_datetime(p5["time"]).dt.year.to_numpy()
    sc_year = scenes["time"].dt.year.to_numpy()
    check5 = []
    for y in sorted(fill_year.index):
        if fill_year[y] < 100:
            continue
        m5 = y5 == y
        check5.append({"year": int(y), "n_filled_scenes": int(fill_year[y]), "n_pairs_5d": int(m5.sum()),
                       "median_resid_m": float(np.median(r5[m5])) if m5.any() else None,
                       "offset_sigma_p50_m": float(np.median(off_sigma[use_h & (sc_year == y)]))})
    main_fill_year = max(check5, key=lambda r: r["n_filled_scenes"]) if check5 else None
    dep = [(r["year"], r["median_m"] - r["moving_median_others_m"]) for r in excess["by_year"]
           if r["year"] in fill_year.index and fill_year[r["year"]] >= 100]
    max_dep = max(dep, key=lambda x: abs(x[1])) if dep else None

    def lev_range(mask):
        v = scenes.loc[mask & ok, "level_m"]
        return [float(v.min()), float(v.max())] if len(v) else None

    tab = scenes.pivot_table(index="sensor", columns="level_source", values="time", aggfunc="count", fill_value=0)
    ep = scenes.pivot_table(index="sensor", columns="epoch", values="time", aggfunc="count", fill_value=0)
    fill_off = scenes.loc[use_h, "hydroweb_offset_m"]
    summary = {
        "dahiti_noise_floor_m": round(floor_d, 4), "dahiti_noise_floor_se_m": round(nd.sigma_se, 4),
        "q_dahiti_m2_per_day": q_d, "q_dahiti_se_m2_per_day": nd.q_se,
        "hydroweb_noise_floor_m": round(floor_h, 4), "q_hydroweb_m2_per_day": q_h,
        "noise_diffusion_method": ("joint IRLS fit of binned NMAD^2 of cross-validated linear-interpolation "
                                   f"residuals (span <= {MAX_GAP_DAYS:g} d), caspian.levels."
                                   f"estimate_noise_and_diffusion; SE: moving-block bootstrap, {N_BOOT} reps, "
                                   f"seed {SEED}; Hydroweb sigma estimated with q fixed to DAHITI's"),
        "noise_diffusion_n_residuals": nd.n_residuals,
        "hydroweb_offset": {
            "method": (f"moving median of the {xc['n_pairs']} DAHITI - Hydroweb pairs within +/-"
                       f"{xc['max_half_gap_days']:g} d, window +/-{OFFSET_HALF_WINDOW_DAYS:g} d around each scene "
                       f"(>= {OFFSET_MIN_PAIRS} pairs, else the nearest {OFFSET_MIN_PAIRS})"),
            "half_window_days": OFFSET_HALF_WINDOW_DAYS, "half_window_years": round(OFFSET_HALF_WINDOW_DAYS / 365.25, 2),
            "min_pairs": OFFSET_MIN_PAIRS,
            "pooled_median_m": pooled,
            "filled_scenes_offset_median_m": float(fill_off.median()),
            "filled_scenes_offset_min_m": float(fill_off.min()),
            "filled_scenes_offset_max_m": float(fill_off.max()),
            "filled_scenes_offset_sigma_p50_m": float(np.median(off_sigma[use_h])),
            "pair_residual_nmad_m": resid_nmad,
            "excess_sd_tau_m": tau_record, "excess_sd_n_years": excess["n_years"],
            "tau_rule": (f"tau(t) = max(local, record-wide): local = method-of-moments excess over the years whose "
                         f"middle pair time is within +/-{OFFSET_HALF_WINDOW_DAYS:g} d of the scene (at least "
                         f"{TAU_MIN_YEARS} years, else the nearest {TAU_MIN_YEARS}); offset 1-sigma = "
                         "sqrt(SE_moving_median^2 + tau^2), common to the gap-filled scenes, added after interpolation"),
            "tau_min_years": TAU_MIN_YEARS,
            "filled_scenes_tau_p50_m": float(np.median(tau_sc[use_h])),
            "filled_scenes_tau_min_m": float(tau_sc[use_h].min()),
            "filled_scenes_tau_max_m": float(tau_sc[use_h].max()),
            "n_filled_scenes_tau_at_record_floor": int((tau_loc["tau_m"].to_numpy()[use_h] <= tau_record).sum()),
            "filled_scenes_offset_sigma_p50_with_record_tau_m": float(np.median(
                np.sqrt(off_sc["se_m"].to_numpy()[use_h] ** 2 + tau_record ** 2))),
        },
        "max_gap_days": MAX_GAP_DAYS,
        "max_gap_rule_provenance": ("caspian.levels default (commit aabacee, before PREREG v1.0 and the DAHITI "
                                    "data); not in PREREG v1.0; stated for the record in PREREG Amendment 4 "
                                    "item 2 (D-038)"),
        "epoch_E1_end": str(t_max),
        "epoch_E1_end_rule": f"maximum of the {SMOOTH_K}-point running median of DAHITI",
        "dahiti_smoothed_extremes": {"k": sm["k"], "max_m": sm["max_m"], "time_of_max": str(sm["time_of_max"]),
                                     "min_m": sm["min_m"], "time_of_min": str(sm["time_of_min"]),
                                     "range_m": sm["range_m"]},
        "dahiti_raw_max_record": {"time": str(t_max_raw), "level_m": float(d["level_m"].max())},
        "n_scenes": int(len(scenes)), "by_sensor_source": tab.to_dict(orient="index"),
        "by_sensor_epoch": ep.to_dict(orient="index"),
        "sigma_m_percentiles_p50_p90_p99": [float(scenes.loc[ok, "sigma_m"].quantile(q)) for q in (0.5, 0.9, 0.99)],
        "sigma_m_p50_by_source": {s: float(g["sigma_m"].median()) for s, g in scenes[ok].groupby("level_source")},
        "level_range_m": [float(scenes.loc[ok, "level_m"].min()), float(scenes.loc[ok, "level_m"].max())],
        "level_range_by_epoch_m": {e: lev_range(scenes["epoch"] == e) for e in ("E1", "E2", "E3")},
        "diagnostics": {
            "n_scenes_above_smoothed_max": int((scenes.loc[ok, "level_m"] > sm["max_m"]).sum()),
            "n_scenes_between_raw_and_smoothed_max_dates": int(((scenes["time"] > min(t_max, t_max_raw))
                                                                 & (scenes["time"] <= max(t_max, t_max_raw))).sum()),
            "n_hydroweb_scenes_other_bin_with_pooled_offset": bin_change,
            "bins_for_diagnostic": {"width_m": BIN_WIDTH, "origin_m": BIN_ORIGIN,
                                    "reference": "DAHITI (analysis-datum offset not applied)"},
            "dahiti_neighbour_z": {"z_threshold": Z_THRESHOLD, "n": int(zf.size),
                                   "n_abs_gt_2_5": int((np.abs(zf) > Z_THRESHOLD).sum()),
                                   "expected_gaussian": round(zf.size * p_tail, 1),
                                   "z_of_raw_max_record": float(z[int(d["level_m"].idxmax())])},
            "superseded_two_step_estimate": {"dahiti_floor_m": round(two_step_floor, 4), "q_m2_per_day": two_step_q},
            "hydroweb_joint_free_q": {"sigma_m": round(nh_free.sigma, 4), "q_m2_per_day": nh_free.q,
                                      "filled_scenes_sigma_change_p50_m": float(np.median(dsig_alt)),
                                      "filled_scenes_sigma_change_max_abs_m": float(np.max(np.abs(dsig_alt))),
                                      "note": "change in gap-filled scene 1-sigma if Hydroweb's own joint fit "
                                              "(sigma, q) replaced DAHITI's q (not applied)"},
            "e1_end_sensitivity": {"applied_k": SMOOTH_K, "by_k": e1_sens,
                                   "running_median_top_two_m": [float(top2[1]), float(top2[0])],
                                   "note": "k = 1 is the single-record maximum; the date of the maximum of a "
                                           "running median is set by the top two values, which may differ by "
                                           "less than the record noise"},
            "offset_tau_scene_weighted_m": tau_scene_weighted,
            "offset_fill_year_max_departure": (
                {"year": int(max_dep[0]), "departure_m": float(max_dep[1]),
                 "note": "largest |year median - leave-year-out moving median| among years with >= 100 "
                         "gap-filled scenes"} if max_dep else None),
            "offset_check_5d_main_fill_year": main_fill_year,
            "offset_check_5d_in_sample": {"rows": check5,
                                          "note": "median DAHITI - shifted Hydroweb at the +/-5-day epochs (a subset "
                                                  "of the +/-20-day pairs that set the offset, so not independent), "
                                                  "by gap-fill year with >= 100 filled scenes"},
            "offset_by_year": excess["by_year"],
        },
    }
    (RESULTS / "levels").mkdir(parents=True, exist_ok=True)
    (RESULTS / "levels" / "scene_levels_summary.json").write_text(json.dumps(summary, indent=1, default=int),
                                                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "diagnostics"}, indent=1, default=int))
    print(json.dumps({k: v for k, v in summary["diagnostics"].items() if k != "offset_by_year"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
