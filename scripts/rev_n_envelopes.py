"""A18 item 10(f): hindcast error envelopes for near-term emergence, replacing the "90%
intervals"; the 33 versus 34 pairs (R3 minor 4); and a paired difference for the H3
comparison (R3 minor 5). DECISIONS D-124. Exploratory, except the lowest-epoch scores,
which are the registered sensitivity (D-081).

Pairs: the registered calibration pairs of scripts/a8_emergence.py (each epoch's highest
unit u* against every other unit of the epoch); the scored set (observed levels, anchors
excluded) is read from results/emergence/main/loeo_errors.csv. Counts over the cells known
at both units (caspian.nearterm.pair_counts) on the uncorrected surface (c = 0) and the
shifted product (F - c, D-070). Errors per surface (D-124):
  self-referenced (uncorrected, shifted): (p_top - p_u) - (w_top - w_u)
  common baseline, area version:          w_u - p_u
  common baseline, location version:      w_both - w_top_p_u
Scores: |e| / (w_top - w_u) (scale-aware, as written; +inf where the observed change is
0), |e| / (L_top - L_u) (km2 per m of decline; logged deviation), and |e| in km2
(registered style, a check against emergence.csv). Each over all scored pairs and over
the E3 pairs alone (the registered lowest-epoch sensitivity).

Outputs (results/revision/n/): pairs.csv, scores.csv, envelopes.csv,
paired_difference.csv, run_envelopes.json.

Usage: python scripts/rev_n_envelopes.py [--tile 2048]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as C  # noqa: E402

from caspian import conformal  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian.config import INTERIM, RESULTS  # noqa: E402

ALPHA = 0.10
SURF_PAIR = {"uncorrected": ("self", "uncorrected"), "shifted": ("self", "shifted"),
             "common_area": ("area", "uncorrected"), "common_location": ("location", "uncorrected"),
             "shifted_common_area": ("area", "shifted"), "shifted_common_location": ("location", "shifted")}


def pair_table(S, tile) -> pd.DataFrame:
    a7, st, tr, shape = S["a7"], S["static"], S["transform"], S["shape"]
    a7.ENVELOPE = st["envelope"]
    units = a7.stage2_index(INTERIM / "stage2" / "main")
    ulist = [(ep, b) for b in sorted(units) for ep in sorted(units[b])]
    rows = []
    for e in sorted({ep for ep, _ in ulist}):
        eu = sorted(b for ep, b in ulist if ep == e)
        top = eu[-1]
        raw, _ = a7.observed_raw(a7.groups_for(units, top, [e]), tr, shape)
        code_top = a7.classify(raw, st, tile)
        del raw
        for b in eu[:-1]:
            raw, _ = a7.observed_raw(a7.groups_for(units, b, [e]), tr, shape)
            code_u = a7.classify(raw, st, tile)
            del raw
            Lt, Lu = a7._bin_level(top), a7._bin_level(b)
            for p in S["names"]:
                pc = nt.pair_counts(S["F"][p], Lt, Lu, code_top, code_u, st["outside"], S["finite"][p],
                                    S["lab_in"], S["K"], shifts=(0.0, S["c"][p]), tile=tile)
                obs = {k: C.to_regions(pc[k]) for k in ("n_common", "w_top", "w_u", "w_both")}
                for i, surf in enumerate(("uncorrected", "shifted")):
                    sur = {k: C.to_regions(pc[k][i]) for k in ("p_top", "p_u", "w_top_p_u")}
                    for r, reg in enumerate(C.REGIONS):
                        rows.append({"product": p, "surface": surf, "epoch": e, "bin": b, "anchor_bin": top,
                                     "level_top_m": round(Lt, 4), "level_u_m": round(Lu, 4), "region": reg,
                                     **{k: int(v[r]) for k, v in obs.items()}, **{k: int(v[r]) for k, v in sur.items()}})
            del code_u
            C.log("pairs", e, top, b)
        del code_top
    return pd.DataFrame(rows)


def errors(pt: pd.DataFrame, cell: float) -> pd.DataFrame:
    out = []
    for surf, (fn, base) in SURF_PAIR.items():
        x = pt[pt["surface"] == base].copy()
        if fn == "self":
            est, true = x["p_top"] - x["p_u"], x["w_top"] - x["w_u"]
        elif fn == "area":
            est, true = x["w_top"] - x["p_u"], x["w_top"] - x["w_u"]
        else:
            est, true = x["w_top"] - x["w_top_p_u"], x["w_top"] - x["w_both"]
        x = x.assign(surface=surf, pred_km2=est * cell, obs_km2=true * cell, error_km2=(est - true) * cell,
                     observed_change_km2=(x["w_top"] - x["w_u"]) * cell,
                     decline_m=(x["level_top_m"] - x["level_u_m"]).round(4))
        out.append(x)
    return pd.concat(out, ignore_index=True)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--from-pairs", action="store_true",
                    help="reuse results/revision/n/pairs.csv instead of recounting the pairs")
    a = ap.parse_args(argv)
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=False)
    cell = S["a7"].CELL_KM2
    if a.from_pairs:
        pt = pd.read_csv(C.OUT / "pairs.csv")
    else:
        pt = pair_table(S, a.tile)
        C.OUT.mkdir(parents=True, exist_ok=True)
        pt.to_csv(C.OUT / "pairs.csv", index=False)
    err = errors(pt, cell)
    # scored set and the check against the registered errors
    lo = pd.read_csv(RESULTS / "emergence" / "main" / "loeo_errors.csv")
    lo = lo[(lo["surface"] == "uncorrected") & (lo["functional"] == "emergence")]
    chk = err[err["surface"] == "uncorrected"].merge(
        lo[["product", "fold", "bin", "region", "error_km2", "score"]].rename(columns={"fold": "epoch"}),
        on=["product", "epoch", "bin", "region"], suffixes=("", "_reg"), how="inner")
    worst = float((chk["error_km2"] - chk["error_km2_reg"]).abs().max())
    n_chk = int(len(chk))
    n_ref = int((~lo["is_anchor"].astype(bool)).sum())          # the anchors pair with themselves
    if worst > 1e-6 or n_chk != n_ref:
        raise SystemExit(f"check failed: errors differ from loeo_errors.csv by {worst} km2 ({n_chk} of {n_ref} rows)")
    C.log("check passed: uncorrected errors equal loeo_errors.csv", worst, n_chk)
    flag = lo[["product", "fold", "bin", "region", "score"]].rename(columns={"fold": "epoch"})
    flag_dom = flag[flag["region"] == "domain"]
    # validated sectors: a pair is scored if it is scored in any validated sector (as "domain" is the OR)
    fv = flag[flag["region"].isin(C.VALIDATED)].groupby(["product", "epoch", "bin"], as_index=False)["score"].any()
    flag = pd.concat([flag, fv.assign(region="validated")], ignore_index=True)
    err = err.merge(flag, on=["product", "epoch", "bin", "region"], how="left")
    err["score"] = err["score"].fillna(False).astype(bool)
    err["s_scale"] = nt.scale_scores(err["error_km2"], err["observed_change_km2"])
    err["s_decline"] = err["error_km2"].abs() / err["decline_m"]
    err["s_abs"] = err["error_km2"].abs()
    err.to_csv(C.OUT / "scores.csv", index=False)
    sc = err[err["score"]]
    n_by_epoch = sc[(sc["region"] == "domain") & (sc["surface"] == "uncorrected") &
                    (sc["product"] == S["names"][0])].groupby("epoch").size().to_dict()
    dmax = float(sc[sc["region"] == "domain"]["decline_m"].max())
    # registered q check (absolute scores, uncorrected, domain)
    em = pd.read_csv(RESULTS / "emergence" / "main" / "emergence.csv")
    emd = em[(em["functional"] == "emergence") & (em["surface"] == "uncorrected") & (em["region"] == "domain")]
    qchk = 0.0
    for p in S["names"]:
        s = sc[(sc["product"] == p) & (sc["surface"] == "uncorrected") & (sc["region"] == "domain")]
        q = conformal.conformal_quantile(s["s_abs"].to_numpy(float), ALPHA)
        qe = conformal.conformal_quantile(s[s["epoch"] == "E3"]["s_abs"].to_numpy(float), ALPHA)
        r = emd[emd["product"] == p].iloc[0]
        qchk = max(qchk, abs(q - float(r["q_km2"])), abs(qe - float(r["q_extrap_km2"])))
    if qchk > 1e-6:
        raise SystemExit(f"check failed: the registered q-hat is not reproduced ({qchk} km2)")
    C.log("check passed: registered q-hat reproduced")
    # envelopes on the curves; the composite observed baseline (D-123 addendum 2) takes the scores of
    # the same hindcast functional (the observed anchor is the baseline there), under its own name
    cv = pd.read_csv(C.OUT / "curves.csv")
    cv = cv[cv["sensitivity"] == "main"]
    if (C.OUT / "curves_composite.csv").exists():
        cc = pd.read_csv(C.OUT / "curves_composite.csv")
        cc = cc[(cc["sensitivity"] == "main") & cc["surface"].isin(["common_area", "common_location",
                                                                    "shifted_common_area", "shifted_common_location"])]
        cv = pd.concat([cv, cc.assign(surface=cc["surface"] + "_composite")], ignore_index=True)
        comp = sc[sc["surface"].isin(["common_area", "common_location", "shifted_common_area",
                                      "shifted_common_location"])]
        sc = pd.concat([sc, comp.assign(surface=comp["surface"] + "_composite")], ignore_index=True)
    env = []
    qrows = []
    for (p, surf, reg), s in sc.groupby(["product", "surface", "region"]):
        cur = cv[(cv["product"] == p) & (cv["surface"] == surf) & (cv["region"] == reg)]
        keep_all = reg in ("domain", "validated")
        for scope, ss in (("all_pairs", s), ("E3_pairs", s[s["epoch"] == "E3"])):
            for kind in ("scale", "decline", "abs"):
                v = ss[f"s_{kind}"].to_numpy(float)
                q = conformal.conformal_quantile(v, ALPHA) if v.size else float("inf")
                qrows.append({"product": p, "surface": surf, "region": reg, "pairs": scope, "score": kind,
                              "n_scores": int(v.size), "n_infinite": int(np.isinf(v).sum()), "q_hat": q,
                              "k": int(np.ceil((v.size + 1) * (1 - ALPHA))),
                              "max_decline_m": float(ss["decline_m"].max()) if len(ss) else np.nan})
                for r in cur.itertuples():
                    D = float(r.delta_m)
                    if D <= 0 or (not keep_all and not np.isclose(D, [0.5, 1, 2, 3, 5]).any()):
                        continue
                    calibrated = D <= dmax + 1e-9
                    if kind == "scale":
                        lo_, hi_ = nt.relative_envelope(float(r.E_km2), q)
                    elif kind == "decline":
                        lo_, hi_ = nt.decline_envelope(float(r.E_km2), q, D)
                    else:
                        lo_, hi_ = max(float(r.E_km2) - q, 0.0), float(r.E_km2) + q
                    if not calibrated:
                        lo_, hi_ = np.nan, np.nan
                    env.append({"product": p, "surface": surf, "region": reg, "pairs": scope, "score": kind,
                                "delta_m": D, "estimate_km2": float(r.E_km2), "lo_km2": lo_, "hi_km2": hi_,
                                "calibrated": bool(calibrated), "q_hat": q})
    pd.DataFrame(qrows).to_csv(C.OUT / "envelope_q.csv", index=False)
    pd.DataFrame(env).to_csv(C.OUT / "envelopes.csv", index=False)
    # paired difference (R3 minor 5), descriptive
    prow = []
    for tag, form in (("main", "linear_hold"), ("main_constant", "constant")):
        le = pd.read_csv(RESULTS / "emergence" / tag / "loeo_errors.csv")
        le = le[(le["region"] == "domain") & (le["functional"] == "emergence") & le["score"]]
        u = le[le["surface"] == "uncorrected"][["product", "fold", "bin", "error_km2"]]
        c = le[(le["surface"] == "corrected") & (le["form"] == form)][["product", "fold", "bin", "error_km2"]]
        m = c.merge(u, on=["product", "fold", "bin"], suffixes=("_corr", "_unc"))
        m["d"] = m["error_km2_corr"] - m["error_km2_unc"]
        e2 = pd.read_csv(RESULTS / "emergence" / tag / "emergence.csv")
        e2 = e2[(e2["functional"] == "emergence") & (e2["region"] == "domain")]
        dec = pd.read_csv(RESULTS / "emergence" / tag / "h3_decision.csv")
        dec = dec[(dec["region"] == "domain") & (dec["functional"] == "emergence")]
        for p, g in m.groupby("product"):
            qd = conformal.conformal_quantile(np.abs(g["d"].to_numpy(float)), ALPHA)
            for D in (0.5, 1.0, 2.0):
                ef = float(e2[(e2["product"] == p) & (e2["surface"] == "fused") & np.isclose(e2["scenario_m"], D)]["estimate_km2"].iloc[0])
                eu = float(e2[(e2["product"] == p) & (e2["surface"] == "uncorrected") & np.isclose(e2["scenario_m"], D)]["estimate_km2"].iloc[0])
                dr = dec[(dec["product"] == p) & np.isclose(dec["scenario_m"], D)]
                if len(dr) != 1:
                    raise SystemExit(f"{tag} {p} {D}: expected one h3_decision.csv row")
                prow.append({"run": tag, "form": form, "product": p, "delta_m": D, "n_pairs": int(len(g)),
                             "q_hat_d_km2": qd, "fused_km2": ef, "uncorrected_km2": eu, "difference_km2": ef - eu,
                             "outside_paired_envelope": bool(abs(ef - eu) > qd),
                             "registered_relevant": bool(dr["decision_relevant"].iloc[0])})
    pd.DataFrame(prow).to_csv(C.OUT / "paired_difference.csv", index=False)
    C.write_json(C.OUT / "run_envelopes.json", {
        "script": "scripts/rev_n_envelopes.py", "decision": "D-124", "alpha": ALPHA,
        "n_scored_pairs_by_epoch_domain": n_by_epoch, "max_pair_decline_m": dmax,
        "checks": {"uncorrected_errors_vs_loeo_errors_csv": {"max_abs_diff_km2": worst, "n_rows": n_chk},
                   "registered_q_hat_reproduced_max_abs_diff_km2": qchk},
        "scored_flag_source": "results/emergence/main/loeo_errors.csv (uncorrected, emergence); validated = any of the four",
        "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
