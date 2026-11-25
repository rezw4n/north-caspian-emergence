"""Journal revision, PREREG Amendment 20 item 1d (D-161): near-term emergence from the
observed 2026 shoreline on the marine domain (the study domain less the deltaic sectors).
Exploratory, post-review; the Amendment 19 item 1 estimate on the whole domain stands as
it was reported and is reproduced here first.

As scripts/rev2_n_reref.py: for each grid, F (hindcast flood-level map) less the D-070 shape
constant c less a second constant c26; a cell is sea at level L when F < L + c + c26. Here
c26 is found by the same bisection on a 0.01 m grid, but so that the grid's sea area at
L2026 inside the marine domain equals the observed 2026 sea inside the marine domain. With
T = L2026 + c + c26:
  M    = |S_obs ∩ marine ∩ {F >= T}|                       (misfit)
  E(D) = |S_obs ∩ marine ∩ {T - D <= F < T}|, D = 0.5, 1, 2, 3 m
Beside it, the registered (whole-domain) constant restricted to the marine domain and to
the deltaic sectors.

Checks: the whole-domain bisection reproduces the registered c26 of
results/revision2/n/reref_constants.csv, and E on the whole domain reproduces
results/revision2/n/reref_curves.csv.

Outputs (results/revision3/nearterm/): marine_constants.csv, marine_curves.csv, run.json.
Usage: python scripts/rev3_nearterm.py [--tile 2048]
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
from caspian import rev2_n as r2  # noqa: E402
from caspian import rev3  # noqa: E402
from caspian.config import RESULTS  # noqa: E402
from caspian.geography import SECTOR_ORDER  # noqa: E402

OUT = RESULTS / "revision3" / "nearterm"
REF = RESULTS / "revision2" / "n"
DELTAS = (0.5, 1.0, 2.0, 3.0)
K_RANGE = 500
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
    a7, shape, lab, K, L26, names = S["a7"], S["shape"], S["lab_in"], S["K"], S["L26"], S["names"]
    cell = a7.CELL_KM2
    labels = pd.read_csv(RESULTS / "levels" / "sector_labels.csv").set_index("sector")
    delt = [s for s in SECTOR_ORDER if bool(labels.loc[s, "deltaic"])]
    mar = np.array(rev3.marine_codes(SECTOR_ORDER, delt)) - 1          # row indices of the marine sectors
    dlt = np.array([SECTOR_ORDER.index(s) for s in delt])
    regions = {"domain": np.arange(K), "marine": mar, "deltaic": dlt}
    Gc = S["Ffused"][names[0]]                       # the composite observed baseline (one map)
    sobs = np.zeros(shape, bool)
    nS = np.zeros(K + 1, np.int64)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        m = np.asarray(Gc[r0:r1, c0:c1], np.float64) < L26
        sobs[r0:r1, c0:c1] = m
        nS += np.bincount(np.asarray(lab[r0:r1, c0:c1]).astype(np.int64)[m], minlength=K + 1)
    nS = nS[1:]
    reg_const = pd.read_csv(REF / "reref_constants.csv")
    reg_const = reg_const[(reg_const["variant"] == "area_match") & (reg_const["sensitivity"] == "main")].set_index("product")
    reg_cur = pd.read_csv(REF / "reref_curves.csv")
    reg_cur = reg_cur[(reg_cur["variant"] == "area_match") & (reg_cur["sensitivity"] == "main")
                      & (reg_cur["region"] == "domain")]
    if abs(float(nS.sum() * cell) - float(reg_const["S_obs_domain_km2"].iloc[0])) > 1e-6:
        raise SystemExit("S_obs differs from the registered run")
    C.log("S_obs km2", {k: round(float(nS[i].sum() * cell), 1) for k, i in regions.items()})
    lv = ex.level_grid(L26, BELOW, above=ABOVE)
    const_rows, curve_rows, worst = [], [], 0.0
    for p in names:
        t1 = time.time()
        F, c = S["F"][p], S["c"][p]
        cnt_all = nt.region_counts(F, lab, K, lv, tile=a.tile)
        cnt_S = nt.region_counts(F, lab, K, lv, mask=sobs, tile=a.tile)
        i_c = level_index(lv, L26 + c)
        if i_c - K_RANGE - 300 < 0 or i_c + K_RANGE >= lv.size:
            raise SystemExit("the level grid is too short")
        fits = {}
        for name in ("domain", "marine"):
            area = cnt_all[regions[name]].sum(axis=0)
            fits[name] = r2.bisect_offset(lambda k, area=area: float(area[i_c + k]), float(nS[regions[name]].sum()),
                                          -K_RANGE, K_RANGE)
        if abs(fits["domain"]["offset_m"] - float(reg_const.loc[p, "c26_m"])) > 1e-9:
            raise SystemExit(f"{p}: the whole-domain c26 {fits['domain']['offset_m']} differs from the registered "
                             f"{reg_const.loc[p, 'c26_m']}")
        for constant, fit in (("registered", fits["domain"]), ("marine", fits["marine"])):
            iT = i_c + fit["k"]
            const_rows.append({"product": p, "constant": constant, "c_shape_m": c, "c26_m": fit["offset_m"],
                               "c_total_m": round(c + fit["offset_m"], 2), "T_m": float(lv[iT]),
                               **{f"S_obs_{r}_km2": float(nS[i].sum() * cell) for r, i in regions.items()},
                               **{f"grid_sea_{r}_km2": float(cnt_all[i, iT].sum() * cell) for r, i in regions.items()}})
            for r, i in regions.items():
                curve_rows.append({"product": p, "constant": constant, "region": r, "quantity": "misfit",
                                   "delta_m": np.nan, "km2": float((nS[i].sum() - cnt_S[i, iT].sum()) * cell)})
                for D in DELTAS:
                    e = float((cnt_S[i, iT].sum() - cnt_S[i, iT - int(round(100 * D))].sum()) * cell)
                    curve_rows.append({"product": p, "constant": constant, "region": r, "quantity": "E",
                                       "delta_m": D, "km2": e})
                    if constant == "registered" and r == "domain":
                        w = reg_cur[(reg_cur["product"] == p) & (reg_cur["quantity"] == "E")
                                    & np.isclose(reg_cur["delta_m"], D)]["km2"].iloc[0]
                        worst = max(worst, abs(e - float(w)))
        C.log(p, f"c={c} c26 registered={fits['domain']['offset_m']} marine={fits['marine']['offset_m']}",
              round(time.time() - t1, 1), "s")
        del cnt_all, cnt_S
    if worst > 1e-6:
        raise SystemExit(f"E on the whole domain differs from the registered curves by {worst} km2")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(const_rows).to_csv(OUT / "marine_constants.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "marine_curves.csv", index=False)
    C.write_json(OUT / "run.json", {
        "script": "scripts/rev3_nearterm.py", "decision": "D-161", "amendment": "PREREG Amendment 20 item 1d",
        "level_2026_m": L26, "deltaic_sectors": delt, "marine_sectors": [SECTOR_ORDER[i] for i in mar],
        "S_obs_km2": {r: float(nS[i].sum() * cell) for r, i in regions.items()},
        "checks": {"registered_c26_reproduced": True, "registered_E_domain_max_abs_diff_km2": worst},
        "deltas_m": list(DELTAS), "cell_km2": cell, "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
