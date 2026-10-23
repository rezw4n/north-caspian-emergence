"""A19 item 1, exposure (D-146): the D-126 exposure counts recomputed on the increment
re-referenced to the observed 2026 shoreline: the emerged area inside the polygon layers
(WDPA protected areas with and without record 342651, Ramsar, OSM wetlands) and its share
of E(D), and the OSM feature layers (platforms, pipelines, channels, ports with the
port-harbour subset, settlements) under the a8 rules. No location envelope. Exploratory.

Surface: on S_obs (the composite observed 2026 sea) the flood level F, elsewhere +inf, read
at the levels T (= L2026 + c + c26, scripts/rev2_n_reref.py) and T - D, so the emerged set is
exactly the increment |S_obs ∩ {T - D <= F < T}|. Misfit cells (S_obs, F >= T) are land in
both states; the check variant "misfit_sea" holds them as sea in both states (the 2026 sea
is S_obs itself), which changes only the access rules and is run for ports and
settlements only.

Inputs: scripts/rev_n_common.setup, data/interim/revision_n/composite_baseline.npy,
results/revision2/n/run_reref.json (T per product), the layers of scripts/rev_n_exposure.py.

Outputs (results/revision2/n/): exposure2_polygons.csv, exposure2_features.csv,
run_exposure2.json.

Usage: python scripts/rev2_n_exposure.py [--tile 2048]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_n_common as C  # noqa: E402
import rev_n_exposure as X  # noqa: E402

from caspian import exposure as ex  # noqa: E402
from caspian import hindcast_tiled as ht  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian import rev2_n as r2  # noqa: E402
from caspian.config import CRS_EQUAL_AREA, RAW, RESULTS  # noqa: E402

OUT2 = RESULTS / "revision2" / "n"
DELTAS = (0.5, 1.0, 2.0, 3.0)
ACCESS_LAYERS = ("osm_ports", "osm_settlements")


def main(argv) -> int:
    import geopandas as gpd
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    a = ap.parse_args(argv)
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=False)
    a8 = C.load_script("a8_emergence")
    tr, shape, lab, K, L26, names = S["transform"], S["shape"], S["lab_in"], S["K"], S["L26"], S["names"]
    cell, res = S["a7"].CELL_KM2, float(S["a7"].RES)
    S.pop("static", None)
    rr = json.loads((OUT2 / "run_reref.json").read_text(encoding="utf-8"))
    if abs(rr["level_2026_m"] - L26) > 1e-9:
        raise SystemExit("run_reref.json: another 2026 level")
    T_of = {p: float(rr["thresholds_T_m"][p]["area_match"]) for p in names}
    cur = pd.read_csv(OUT2 / "reref_curves.csv")
    Gc = np.load(C.WORK / "composite_baseline.npy", mmap_mode="r")
    sobs = np.zeros(shape, bool)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        sobs[r0:r1, c0:c1] = np.asarray(Gc[r0:r1, c0:c1], np.float64) < L26
    # layers, as rev_n_exposure
    lay = gpd.read_file(RAW.parent / "interim" / "wdpa" / "layers.gpkg")
    if lay.crs is None or not lay.crs.equals(CRS_EQUAL_AREA):
        lay = lay.to_crs(CRS_EQUAL_AREA)
    osm = a8.osm_layers(RAW / "osm")
    polys = {n: list(lay.loc[lay["layer"] == n].geometry) for n in X.POLY_LAYERS}
    polys["osm_wetlands"] = osm["osm_wetlands"]["geoms"]
    pmask = {n: X.rasterise(g, tr, shape) & sobs for n, g in polys.items()}
    C.log("polygon layers rasterised (within S_obs)", {n: int(m.sum()) for n, m in pmask.items()})
    feats = {n: osm[n] for n in X.FEATURE_LAYERS}
    samp = {n: ex.feature_samples(v["geoms"], tr, shape) for n, v in feats.items()}
    freg = {n: ex.feature_regions(samp[n], lab) for n in feats}
    ports_g = gpd.read_file(RAW / "osm" / "osm_ports.gpkg").to_crs(CRS_EQUAL_AREA)
    if len(ports_g) != len(feats["osm_ports"]["geoms"]):
        raise SystemExit("osm_ports.gpkg: feature count differs from a8_emergence.osm_layers")
    ptype = np.array([nt.port_type(t) for t in ports_g["tags"]])
    poly_rows, feat_rows = [], []
    worst = 0.0
    for p in names:
        F, T = S["F"][p], T_of[p]
        lvl = np.array(sorted({round(T - D, 6) for D in DELTAS} | {round(T, 6)}))
        idx = {round(v, 6): i for i, v in enumerate(lvl)}
        allS = C.to_regions(nt.region_counts(F, lab, K, lvl, mask=sobs, tile=a.tile))
        # check: E(D) from these counts equals scripts/rev2_n_reref.py
        for D in DELTAS:
            e = (allS[:, idx[round(T, 6)]] - allS[:, idx[round(T - D, 6)]]) * cell
            ref = cur[(cur["product"] == p) & (cur["variant"] == "area_match") & (cur["sensitivity"] == "main")
                      & (cur["quantity"] == "E") & np.isclose(cur["delta_m"], D)].set_index("region")["km2"]
            worst = max(worst, float(np.max(np.abs(e - ref.reindex(C.REGIONS).to_numpy(float)))))
        for name, m in pmask.items():
            lc = C.to_regions(nt.region_counts(F, lab, K, lvl, mask=m, tile=a.tile))
            for D in DELTAS:
                kt, kl = idx[round(T, 6)], idx[round(T - D, 6)]
                em_l, em_all = lc[:, kt] - lc[:, kl], allS[:, kt] - allS[:, kl]
                for i, reg in enumerate(C.REGIONS):
                    poly_rows.append({"layer": name, "product": p, "surface": "observed_reref", "region": reg,
                                      "delta_m": D, "emerged_in_layer_km2": float(em_l[i] * cell),
                                      "E_km2": float(em_all[i] * cell),
                                      "share": float(em_l[i] / em_all[i]) if em_all[i] > 0 else np.nan})
        C.log(p, "polygon layers done")
        for surf, misfit in (("observed_reref", "land"), ("observed_reref_misfit_sea", "sea")):
            obj = r2.ReRefSurface(F, Gc, L26, 0.0, misfit=misfit, sea_level=T)
            for n, spec in feats.items():
                rule, thr = spec["rule"], spec.get("threshold_m")
                if misfit == "sea" and n not in ACCESS_LAYERS:
                    continue
                s = samp[n]
                radius = max(float(spec.get("radius_m") or 0.0), float(thr or 0.0))
                fv = ex.sample_values(obj, s["row"], s["col"])
                prof = ex.distance_profiles(obj, s["row"], s["col"], res, radius) if rule == "access" else None
                sets = X.region_sets(freg[n])
                sets_port = None
                if n == "osm_ports":
                    sets_port = {k_: v[np.isin(v, np.flatnonzero(ptype == "port_harbour"))] for k_, v in sets.items()}
                for D in DELTAS:
                    per = ex.exposure_at(rule, s, fv, T, round(T - D, 6), profiles=prof, threshold_m=thr).set_index("fid")
                    for reg, fids in sets.items():
                        sm = ex.summarise_exposure(per.loc[fids], rule)
                        row = {"layer": n, "rule": rule, "product": p, "surface": surf, "region": reg, "delta_m": D,
                               "subset": "all", "n_features": int(len(fids)), **sm}
                        feat_rows.append(row)
                        if sets_port is not None:
                            sp = sets_port[reg]
                            feat_rows.append({**row, "subset": "port_harbour_tags", "n_features": int(len(sp)),
                                              **ex.summarise_exposure(per.loc[sp], rule)})
            C.log(p, surf, "feature layers done", round(time.time() - t0, 1), "s")
    if worst > 1e-6:
        raise SystemExit(f"check failed: E differs from reref_curves.csv by {worst} km2")
    OUT2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(poly_rows).to_csv(OUT2 / "exposure2_polygons.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(OUT2 / "exposure2_features.csv", index=False)
    C.write_json(OUT2 / "run_exposure2.json", {
        "script": "scripts/rev2_n_exposure.py", "decision": "D-146 (D-126 rules)",
        "amendment": "PREREG Amendment 19 item 1 (exposure)", "level_2026_m": L26, "thresholds_T_m": T_of,
        "deltas_m": list(DELTAS), "polygon_cells_within_S_obs": {n: int(m.sum()) for n, m in pmask.items()},
        "n_samples": {n: int(len(s)) for n, s in samp.items()},
        "checks": {"E_vs_reref_curves_max_abs_diff_km2": worst},
        "misfit_rule": "misfit cells (S_obs, F >= T) land in both states; variant observed_reref_misfit_sea "
                       "(ports and settlements only) holds them as sea in both states",
        "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
