"""A19 item 1 (D-146): near-term emergence from the observed 2026 shoreline, re-referenced at
2026, with the three-lowest-bins and L2026 +- 0.3 m sensitivities; and, for A19 item 3, the
observed five-grid agreement on the first metre at 60 m for the re-referenced and the
shifted grids. Exploratory (post-review).

For each product, F (hindcast flood-level map, total variant) less the D-070 shape constant
c less c26: a cell is sea at level L when F < L + c + c26. c26 is found by bisection on a
0.01 m grid so that the product's domain sea area at L2026 equals |S_obs|, the composite
observed 2026 sea (second D-123 addendum). Then, with T = L2026 + c + c26:
  M      = |S_obs ∩ {F >= T}|                         (misfit)
  E(D)   = |S_obs ∩ {T - D <= F < T}|, D = 0.5, 1, 2, 3 m (and 0-5 m at 0.01 m)
Sensitivities: c26 fitted to the three lowest audited units (E3 bins 12-14, summed absolute
area error over known domain cells, D-070 ties); L2026 +- 0.3 m with each c26 held. Check:
the area-match c26 refitted at L2026 +- 0.3 m gives back the same T.

Inputs: the registered caches through scripts/rev_n_common.setup (hindcast flood maps,
static layers, record-low envelope, stage-2 counts) and data/interim/revision_n/
composite_baseline.npy.

Outputs (results/revision2/n/): reref_constants.csv, reref_curves.csv,
reref_curves_fine.csv.gz, reref_bins_fit.csv, agreement_60m.csv, run_reref.json; the 60 m
count rasters in data/interim/revision2/n/firstmetre_counts_60m.npz.

Usage: python scripts/rev2_n_reref.py [--tile 2048]
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

from caspian import exposure as ex  # noqa: E402
from caspian import hindcast_tiled as ht  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian import products  # noqa: E402
from caspian import rev2_n as r2  # noqa: E402
from caspian.config import INTERIM, RESULTS  # noqa: E402

OUT2 = RESULTS / "revision2" / "n"
WORK2 = INTERIM / "revision2" / "n"
DELTAS = (0.5, 1.0, 2.0, 3.0)
FINE = np.round(np.arange(0, 501) / 100.0, 2)          # 0-5 m at 0.01 m
BINS = (12, 13, 14)                                    # the three lowest audited units (E3)
K_RANGE = 500                                          # c26 on [-5, 5] m
SENS = {"main": 0, "L26+0.3": 30, "L26-0.3": -30}      # level shift in 0.01 m steps
BELOW, ABOVE = 16.0, 5.0


def level_index(lv, level) -> int:
    i = np.flatnonzero(np.isclose(lv, round(level, 6), rtol=0, atol=1e-6))
    if i.size != 1:
        raise SystemExit(f"level {level} not on the level grid")
    return int(i[0])


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=True, fused_form="composite")
    a7, st, tr, shape, lab, K, L26 = S["a7"], S["static"], S["transform"], S["shape"], S["lab_in"], S["K"], S["L26"]
    cell, names = a7.CELL_KM2, S["names"]
    Gc = S["Ffused"][names[0]]                       # one composite map for every product
    # S_obs (60 m) and its cells per sector
    sobs = np.zeros(shape, bool)
    nS = np.zeros(K + 1, np.int64)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        m = np.asarray(Gc[r0:r1, c0:c1], np.float64) < L26
        sobs[r0:r1, c0:c1] = m
        nS += np.bincount(np.asarray(lab[r0:r1, c0:c1]).astype(np.int64)[m], minlength=K + 1)
    nS = nS[1:]
    target = int(nS.sum())
    C.log("S_obs", round(target * cell, 1), "km2 in the domain")
    lv = ex.level_grid(L26, BELOW, above=ABOVE)
    # the three lowest audited units, classified as audited (record-low envelope on)
    a7.ENVELOPE = st["envelope"]
    units = a7.stage2_index(INTERIM / "stage2" / "main")
    codes = {}
    for b in BINS:
        raw, _ = a7.observed_raw(a7.groups_for(units, b, ["E3"]), tr, shape)
        codes[b] = a7.classify(raw, st, a.tile)
        del raw
    C.log("bins classified", BINS)
    cgrid = np.asarray(a7.C_GRID, float)
    k_grid = np.round(cgrid / 0.01).astype(int)
    const_rows, curve_rows, fine_rows, bins_rows = [], [], [], []
    T_of, checks = {}, {"refit_at_shifted_level_same_T": {}, "bisection_equals_full_evaluation": {}}
    for p in names:
        t1 = time.time()
        F, c = S["F"][p], S["c"][p]
        cnt_all = nt.region_counts(F, lab, K, lv, tile=a.tile)
        cnt_S = nt.region_counts(F, lab, K, lv, mask=sobs, tile=a.tile)
        dom_all = cnt_all.sum(axis=0)
        i_c = level_index(lv, L26 + c)
        if i_c - 2 * K_RANGE - 30 < 0 or i_c + K_RANGE + 30 >= lv.size:
            raise SystemExit("the level grid is too short")
        # area match at L2026 (and refit at L2026 +- 0.3 m as a check)
        fits = {}
        for sens, sh in SENS.items():
            i0 = i_c + sh
            fits[sens] = r2.bisect_offset(lambda k, i0=i0: float(dom_all[i0 + k]), float(target), -K_RANGE, K_RANGE)
        full = r2.closest_offset(dom_all[i_c - K_RANGE:i_c + K_RANGE + 1].astype(float),
                                 np.arange(-K_RANGE, K_RANGE + 1), float(target))
        checks["bisection_equals_full_evaluation"][p] = bool(full["k"] == fits["main"]["k"])
        if full["k"] != fits["main"]["k"]:
            raise SystemExit(f"{p}: bisection and full evaluation disagree")
        iT_area = i_c + fits["main"]["k"]
        same = all(i_c + SENS[s] + fits[s]["k"] == iT_area for s in SENS)
        checks["refit_at_shifted_level_same_T"][p] = bool(same)
        # three lowest bins: summed absolute area error over known domain cells
        bias = np.zeros((len(BINS), cgrid.size), np.int64)
        for j, b in enumerate(BINS):
            T_b = products.flood_level(a7._bin_level(b), 0.0, 0.0, c + cgrid)
            u = ex.unit_region_counts(F, codes[b], st["outside"], S["finite"][p], lab, K, T_b, tile=a.tile)
            bias[j] = u["below"].sum(axis=0) - int(u["water"].sum())
            bins_rows.append({"product": p, "bin": b, "level_m": a7._bin_level(b),
                              "known_km2": float(u["known"].sum() * cell), "water_km2": float(u["water"].sum() * cell),
                              "_bias": bias[j]})
        bf = a7.fit_offset(bias, ["E3"] * len(BINS))
        k_bins = int(round(bf["offset_m"] / 0.01))
        for rr in bins_rows[-len(BINS):]:
            bb = rr.pop("_bias")
            rr["error_at_c26_bins_km2"] = float(bb[int(np.flatnonzero(k_grid == k_bins)[0])] * cell)
            rr["error_at_c26_area_km2"] = float(bb[int(np.flatnonzero(k_grid == fits["main"]["k"])[0])] * cell)
            rr["error_at_c26_0_km2"] = float(bb[int(np.flatnonzero(k_grid == 0)[0])] * cell)
        iT_bins = i_c + k_bins
        T_of[p] = {"area_match": float(lv[iT_area]), "bins": float(lv[iT_bins])}
        for variant, iT0, k0, extra in (("area_match", iT_area, fits["main"]["k"],
                                         {"n_tied_minima": full["n_tied"]}),
                                        ("bins", iT_bins, k_bins,
                                         {"n_tied_minima": bf["n_tied_minima"],
                                          "objective_km2": bf["objective_km2_at_offset"]})):
            for sens, sh in SENS.items():
                iT = iT0 + sh
                prod_sea = C.to_regions(cnt_all[:, iT])
                in_S = C.to_regions(cnt_S[:, iT])
                Sreg = C.to_regions(nS)
                M = Sreg - in_S
                outside = prod_sea - in_S
                const_rows.append({"product": p, "variant": variant, "sensitivity": sens, "level_2026_m": L26 + sh / 100,
                                   "c_shape_m": c, "c26_m": round(k0 * 0.01, 2), "c_total_m": round(c + k0 * 0.01, 2),
                                   "T_m": float(lv[iT]), "S_obs_domain_km2": float(target * cell),
                                   "product_sea_domain_km2": float(prod_sea[-2] * cell),
                                   "area_error_domain_km2": float((prod_sea[-2] - target) * cell),
                                   "misfit_domain_km2": float(M[-2] * cell),
                                   "outside_S_obs_domain_km2": float(outside[-2] * cell),
                                   **({} if sens != "main" else extra)})
                rows_q = {"S_obs": Sreg, "product_sea": prod_sea, "misfit": M, "outside_S_obs": outside}
                for q, v in rows_q.items():
                    for i, reg in enumerate(C.REGIONS):
                        curve_rows.append({"product": p, "variant": variant, "sensitivity": sens, "region": reg,
                                           "quantity": q, "delta_m": np.nan, "km2": float(v[i] * cell)})
                for D in (0.0,) + DELTAS:
                    e = C.to_regions(cnt_S[:, iT] - cnt_S[:, iT - int(round(100 * D))])
                    for i, reg in enumerate(C.REGIONS):
                        curve_rows.append({"product": p, "variant": variant, "sensitivity": sens, "region": reg,
                                           "quantity": "E", "delta_m": D, "km2": float(e[i] * cell)})
        # fine curve, main
        for j, D in enumerate(FINE):
            e = C.to_regions(cnt_S[:, iT_area] - cnt_S[:, iT_area - j])
            for reg in ("domain", "validated"):
                fine_rows.append({"surface": "observed_reref", "product": p, "region": reg, "delta_m": float(D),
                                  "E_km2": float(e[C.REGIONS.index(reg)] * cell)})
        C.log(p, f"c={c} c26_area={fits['main']['offset_m']} c26_bins={bf['offset_m']}", round(time.time() - t1, 1), "s")
        del cnt_all, cnt_S
    del codes
    # 60 m five-grid agreement on the first metre: re-referenced and shifted
    cnt_r = np.zeros(shape, np.uint8)
    cnt_s = np.zeros(shape, np.uint8)
    per = {(s, p): 0 for s in ("reref", "shifted") for p in names}
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        dm = np.asarray(lab[r0:r1, c0:c1]) > 0
        so = sobs[r0:r1, c0:c1] & dm
        for p in names:
            f = np.asarray(S["F"][p][r0:r1, c0:c1], np.float64)
            T, cs = T_of[p]["area_match"], L26 + S["c"][p]
            br = so & (f < T) & ~(f < T - 1.0)
            bs = dm & (f < cs) & ~(f < cs - 1.0)
            cnt_r[r0:r1, c0:c1] += br.astype(np.uint8)
            cnt_s[r0:r1, c0:c1] += bs.astype(np.uint8)
            per[("reref", p)] += int(br.sum())
            per[("shifted", p)] += int(bs.sum())
    cur = pd.DataFrame(curve_rows)
    ref = pd.read_csv(C.OUT / "curves.csv")
    worst = 0.0
    for p in names:
        e1 = cur[(cur["product"] == p) & (cur["variant"] == "area_match") & (cur["sensitivity"] == "main")
                 & (cur["region"] == "domain") & (cur["quantity"] == "E") & np.isclose(cur["delta_m"], 1.0)]["km2"].iloc[0]
        worst = max(worst, abs(per[("reref", p)] * cell - e1))
        s1 = ref[(ref["surface"] == "shifted") & (ref["sensitivity"] == "main") & (ref["product"] == p)
                 & (ref["region"] == "domain") & np.isclose(ref["delta_m"], 1.0)]["E_km2"].iloc[0]
        worst = max(worst, abs(per[("shifted", p)] * cell - s1))
    checks["band_rasters_vs_counts_max_abs_diff_km2"] = worst
    if worst > 1e-6:
        raise SystemExit(f"band rasters differ from the counts by {worst} km2")
    ag_rows = []
    for surf, cnt in (("observed_reref", cnt_r), ("shifted", cnt_s)):
        h = np.zeros(len(names) + 1, np.int64)
        for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
            dm = np.asarray(lab[r0:r1, c0:c1]) > 0
            h += np.bincount(cnt[r0:r1, c0:c1][dm], minlength=len(names) + 1)
        anyc, allc, maj = h[1:].sum(), h[len(names)], h[3:].sum()
        rec = {"surface": surf, "delta_m": 1.0, "resolution_m": 60, "any_km2": float(anyc * cell),
               "all_km2": float(allc * cell), "majority_3of5_km2": float(maj * cell),
               "ratio_all_over_any": float(allc / anyc) if anyc else np.nan}
        rec.update({f"count_{k}_km2": float(h[k] * cell) for k in range(len(names) + 1)})
        ag_rows.append(rec)
    disp = RESULTS / "display" / "erl_nearterm_map.json"
    if disp.exists():
        import json
        d = json.loads(disp.read_text(encoding="utf-8"))["surfaces"]["shifted"]
        checks["shifted_60m_vs_figure5_json"] = {"all_km2": d["area_km2_all"], "any_km2": d["area_km2_any"],
                                                 "abs_diff_all_km2": abs(d["area_km2_all"] - ag_rows[1]["all_km2"]),
                                                 "abs_diff_any_km2": abs(d["area_km2_any"] - ag_rows[1]["any_km2"])}
    WORK2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK2 / "firstmetre_counts_60m.npz", observed_reref=cnt_r, shifted=cnt_s,
                        transform=np.asarray(tr[:6], float), note=np.array(
                            "per 60 m cell, the number of the five grids placing it in the first metre "
                            "(observed_reref: S_obs and T-1 <= F < T; shifted: L26+c-1 <= F < L26+c); all cells"))
    del cnt_r, cnt_s
    OUT2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(const_rows).to_csv(OUT2 / "reref_constants.csv", index=False)
    cur.to_csv(OUT2 / "reref_curves.csv", index=False)
    pd.DataFrame(fine_rows).to_csv(OUT2 / "reref_curves_fine.csv.gz", index=False)
    pd.DataFrame(bins_rows).to_csv(OUT2 / "reref_bins_fit.csv", index=False)
    pd.DataFrame(ag_rows).to_csv(OUT2 / "agreement_60m.csv", index=False)
    C.write_json(OUT2 / "run_reref.json", {
        "script": "scripts/rev2_n_reref.py", "decision": "D-146", "amendment": "PREREG Amendment 19 items 1 and 3",
        "level_2026_m": L26, "shape_constants_m": S["c"], "S_obs_domain_km2": target * cell,
        "S_obs_file": "data/interim/revision_n/composite_baseline.npy", "bins": list(BINS),
        "bin_levels_m": [a7._bin_level(b) for b in BINS], "c26_grid": {"step_m": 0.01, "range_m": [-5, 5]},
        "thresholds_T_m": T_of, "level_grid": {"below_m": BELOW, "above_m": ABOVE, "step_m": 0.01},
        "checks": checks, "regions": C.REGIONS, "cell_km2": cell, "seconds": round(time.time() - t0, 1)})
    C.log("checks", checks)
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
