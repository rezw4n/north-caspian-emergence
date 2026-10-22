"""A18 item 10(i): exposure on the common baseline (a) and the shifted products (b), with
no count intervals; the protected-area share with and without WDPA record 342651; a
location envelope from each product's band dilated and eroded by its validated-sector
median displacement; and the OSM ports classified into seaports or harbours and river
piers or landings (R2.M10(d)). DECISIONS D-126. Exploratory.

Surfaces (D-123): uncorrected (registered, a check against results/emergence/main/
exposure.csv), shifted (F - c), common_location (S_base and the uncorrected product),
shifted_common_location (S_base and the shifted product).

Polygon layers (WDPA protected areas, the same without 342651, Ramsar, OSM wetlands):
the emerged area inside the domain's sector cells, from the layer rasterised by cell
centre. Point and line layers (OSM platforms, pipelines, channels, ports, settlements):
the a8 rules through caspian.exposure, features assigned to regions as there.

Outputs (results/revision/n/): exposure_polygons.csv, exposure_features.csv,
exposure_envelope.csv, ports_summary.csv, run_exposure.json. The per-feature port table
stays in data/interim/revision_n/ports_classified.csv (OpenStreetMap, ODbL).

Usage: python scripts/rev_n_exposure.py [--tile 2048] [--envelope-deltas 1,2]
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

from caspian import exposure as ex  # noqa: E402
from caspian import hindcast_tiled as ht  # noqa: E402
from caspian import nearterm as nt  # noqa: E402
from caspian.config import CRS_EQUAL_AREA, RAW, RESULTS  # noqa: E402
from caspian.geography import SECTOR_ORDER  # noqa: E402

POLY_LAYERS = ("wdpa_protected_areas", "wdpa_protected_areas_noksrz", "wdpa_ramsar")
FEATURE_LAYERS = ("osm_platforms", "osm_pipelines", "osm_channels", "osm_ports", "osm_settlements")
SURFACES = ("uncorrected", "shifted", "common_location", "shifted_common_location", "common_location_composite",
            "shifted_common_location_composite")
# D-126: displacement variant of the radius; the composite baseline is D-123 addendum 2
ENVELOPE_SURFACES = {"shifted": "shape", "common_location": "total", "common_location_composite": "total"}
DELTA_CLASSES_KM = (10.0, 20.0, 50.0)
FAR_KM = 20.0


def tops(S, p, surf):
    """(top level on F, the baseline set used: None, "registered" or "composite")."""
    c = S["c"][p]
    return {"uncorrected": (S["L26"], None), "shifted": (S["L26"] + c, None),
            "common_location": (S["L26"], "registered"), "shifted_common_location": (S["L26"] + c, "registered"),
            "common_location_composite": (S["L26"], "composite"),
            "shifted_common_location_composite": (S["L26"] + c, "composite")}[surf]


def surface_obj(S, p, surf):
    F, Ff, c, L26 = S["F"][p], S["Ffused"][p], S["c"][p], S["L26"]
    if surf == "uncorrected":
        return F, L26
    if surf == "shifted":
        return nt.ShiftedSurface(F, c), L26
    if surf == "common_location":
        return nt.BaselineSurface(Ff, F, L26), L26
    if surf == "shifted_common_location":
        return nt.BaselineSurface(Ff, nt.ShiftedSurface(F, c), L26), L26
    if surf == "common_location_composite":
        return nt.BaselineSurface(S["Gc"], F, L26), L26
    return nt.BaselineSurface(S["Gc"], nt.ShiftedSurface(F, c), L26), L26


def rasterise(geoms, transform, shape) -> np.ndarray:
    from affine import Affine
    from rasterio.features import geometry_mask
    if not len(geoms):
        return np.zeros(shape, bool)
    return geometry_mask(list(geoms), out_shape=shape, transform=Affine(*[float(v) for v in transform[:6]]),
                         invert=True)


def radii(names) -> dict:
    """D-126: per product and variant, the median of the per-level median displacements
    over the evaluable levels of the validated sectors (pooled epochs), m."""
    d = pd.read_csv(RESULTS / "hindcast" / "main" / "h1_mispredicts_detail.csv")
    d = d[(d["epoch"] == "pooled") & d["sector"].isin(C.VALIDATED) & d["evaluable"].astype(bool)]
    out = {}
    for p in names:
        for var in ("shape", "total"):
            v = d[(d["product"] == p) & (d["variant"] == var)]["median"].to_numpy(float)
            v = v[np.isfinite(v)]
            out[(p, var)] = {"radius_m": float(np.median(v)), "n_levels": int(v.size)}
    return out


def region_sets(freg: pd.Series) -> dict:
    vcodes = [SECTOR_ORDER.index(s) + 1 for s in C.VALIDATED]
    out = {s: freg.index[freg == k + 1] for k, s in enumerate(SECTOR_ORDER)}
    out["domain"] = freg.index[freg >= 1]
    out["validated"] = freg.index[freg.isin(vcodes)]
    return out


def main(argv) -> int:
    import geopandas as gpd
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--envelope-deltas", default="1,2", help="empty: no location envelope (D-126 addendum)")
    ap.add_argument("--surfaces", default=",".join(SURFACES))
    ap.add_argument("--polygons-only", action="store_true",
                    help="only the polygon layers (protected areas, Ramsar, OSM wetlands); D-126 addendum 2")
    a = ap.parse_args(argv)
    t0 = time.time()
    S = C.setup(tile=a.tile, fused=True)
    a8 = C.load_script("a8_emergence")
    tr, shape, lab, K, L26, names = S["transform"], S["shape"], S["lab_in"], S["K"], S["L26"], S["names"]
    cell, res = S["a7"].CELL_KM2, float(S["a7"].RES)
    S.pop("static", None)                     # not used below; frees about 375 MB
    deltas = S["scenarios"]
    env_deltas = [float(x) for x in a.envelope_deltas.split(",") if x]
    surfs = [s for s in a.surfaces.split(",") if s]
    # layers
    lay = gpd.read_file(RAW.parent / "interim" / "wdpa" / "layers.gpkg")
    if lay.crs is None or not lay.crs.equals(CRS_EQUAL_AREA):
        lay = lay.to_crs(CRS_EQUAL_AREA)
    osm = a8.osm_layers(RAW / "osm")
    polys = {n: list(lay.loc[lay["layer"] == n].geometry) for n in POLY_LAYERS}
    polys["osm_wetlands"] = osm["osm_wetlands"]["geoms"]
    pmask = {n: rasterise(g, tr, shape) for n, g in polys.items()}
    C.log("polygon layers rasterised", {n: int(m.sum()) for n, m in pmask.items()})
    feats = {} if a.polygons_only else {n: osm[n] for n in FEATURE_LAYERS}
    samp = {n: ex.feature_samples(v["geoms"], tr, shape) for n, v in feats.items()}
    freg = {n: ex.feature_regions(samp[n], lab) for n in feats}
    C.log("feature samples", {n: len(s) for n, s in samp.items()})
    # ports: classification
    ports_g = gpd.read_file(RAW / "osm" / "osm_ports.gpkg").to_crs(CRS_EQUAL_AREA)
    if not a.polygons_only and len(ports_g) != len(feats["osm_ports"]["geoms"]):
        raise SystemExit("osm_ports.gpkg: feature count differs from a8_emergence.osm_layers")
    ptype = np.array([nt.port_type(t) for t in ports_g["tags"]])
    rad = radii(names)
    S["Gc"] = np.load(C.WORK / "composite_baseline.npy", mmap_mode="r")
    base_c = np.zeros(shape, bool)
    for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
        base_c[r0:r1, c0:c1] = np.asarray(S["Gc"][r0:r1, c0:c1], np.float64) < L26
    poly_rows, feat_rows, env_rows = [], [], []
    checks = {}
    ref = pd.read_csv(RESULTS / "emergence" / "main" / "exposure.csv")
    worst_feat = 0.0
    for p in names:
        Ff = S["Ffused"][p]
        # S_base mask (60 m) for envelopes and the port distances
        base = np.zeros(shape, bool)
        for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
            base[r0:r1, c0:c1] = np.asarray(Ff[r0:r1, c0:c1], np.float64) < L26
        bases = {"registered": (base, Ff), "composite": (base_c, S["Gc"])}
        # --- polygon layers: counts on the needed levels
        c = S["c"][p]
        need = sorted({round(L26 - D, 6) for D in deltas} | {round(L26 + c - D, 6) for D in deltas}
                      | {round(L26, 6), round(L26 + c, 6)})
        F = S["F"][p]
        lvl = np.array(need)
        allc = C.to_regions(nt.region_counts(F, lab, K, lvl, tile=a.tile))
        basec = {b: C.to_regions(nt.region_counts(F, lab, K, lvl, mask=bm, tile=a.tile)) for b, (bm, _) in bases.items()}
        nbase = {b: C.to_regions(nt.region_counts(bf, lab, K, [L26], tile=a.tile))[:, 0] for b, (_, bf) in bases.items()}
        idx = {L: i for i, L in enumerate(need)}
        for name, m in pmask.items():
            lc = C.to_regions(nt.region_counts(F, lab, K, lvl, mask=m, tile=a.tile))
            lbc = {b: C.to_regions(nt.region_counts(F, lab, K, lvl, mask=m & bm, tile=a.tile)) for b, (bm, _) in bases.items()}
            nlb = {b: C.to_regions(nt.region_counts(bf, lab, K, [L26], mask=m, tile=a.tile))[:, 0]
                   for b, (_, bf) in bases.items()}
            for surf in surfs:
                top, use_base = tops(S, p, surf)
                for D in deltas:
                    kt, kl = idx[round(top, 6)], idx[round(top - D, 6)]
                    if use_base:
                        em_l, em_all = nlb[use_base] - lbc[use_base][:, kl], nbase[use_base] - basec[use_base][:, kl]
                    else:
                        em_l, em_all = lc[:, kt] - lc[:, kl], allc[:, kt] - allc[:, kl]
                    for i, reg in enumerate(C.REGIONS):
                        poly_rows.append({"layer": name, "product": p, "surface": surf, "region": reg, "delta_m": D,
                                          "emerged_in_layer_km2": float(em_l[i] * cell),
                                          "E_km2": float(em_all[i] * cell),
                                          "share": float(em_l[i] / em_all[i]) if em_all[i] > 0 else np.nan})
        C.log(p, "polygon layers done")
        # --- feature layers
        cache = {}
        for surf in surfs:
            obj, _ = surface_obj(S, p, surf)
            for n, spec in feats.items():
                s, rule, thr = samp[n], spec["rule"], spec.get("threshold_m")
                radius = max(float(spec.get("radius_m") or 0.0), float(thr or 0.0))
                fv = ex.sample_values(obj, s["row"], s["col"])
                prof = ex.distance_profiles(obj, s["row"], s["col"], res, radius) if rule == "access" else None
                if surf in ENVELOPE_SURFACES:
                    cache[(surf, n)] = (fv, prof)
                sets = region_sets(freg[n])
                sets_port = None
                if n == "osm_ports":
                    sets_port = {k: v[np.isin(v, np.flatnonzero(ptype == "port_harbour"))] for k, v in sets.items()}
                for D in deltas:
                    L = round(L26 - D, 6)
                    per = ex.exposure_at(rule, s, fv, L26, L, profiles=prof, threshold_m=thr).set_index("fid")
                    for reg, fids in sets.items():
                        sm = ex.summarise_exposure(per.loc[fids], rule)
                        row = {"layer": n, "rule": rule, "product": p, "surface": surf, "region": reg, "delta_m": D,
                               "subset": "all", "n_features": int(len(fids)), **sm}
                        feat_rows.append(row)
                        if surf == "uncorrected" and reg in SECTOR_ORDER + ["domain"]:
                            r = ref[(ref["layer"] == n) & (ref["product"] == p) & (ref["surface"] == "uncorrected")
                                    & (ref["region"] == reg) & np.isclose(ref["scenario_m"], D)]
                            if len(r) != 1:
                                raise SystemExit(f"check: no single exposure.csv row for {n} {p} {reg} {D}")
                            worst_feat = max(worst_feat, abs(float(r["amount"].iloc[0]) - sm["amount"]),
                                             abs(float(r["n_changed"].iloc[0]) - sm["n_changed"]))
                        if sets_port is not None:
                            sp = sets_port[reg]
                            feat_rows.append({**row, "subset": "port_harbour_tags", "n_features": int(len(sp)),
                                              **ex.summarise_exposure(per.loc[sp], rule)})
            C.log(p, surf, "feature layers done")
        # --- location envelope
        for surf, var in ([] if not env_deltas else [(s_, v_) for s_, v_ in ENVELOPE_SURFACES.items() if s_ in surfs]):
            r_m = rad[(p, var)]["radius_m"]
            top, use_base = tops(S, p, surf)
            obj, _ = surface_obj(S, p, surf)
            for D in env_deltas:
                band = np.zeros(shape, bool)
                for r0, r1, c0, c1 in ht.tiles(shape, a.tile):
                    f = np.asarray(F[r0:r1, c0:c1], np.float64)
                    b = ~(f < top - D)
                    b &= bases[use_base][0][r0:r1, c0:c1] if use_base else (f < top)
                    band[r0:r1, c0:c1] = b
                dil, ero = nt.distance_masks(band, r_m, res, tile=a.tile)
                if use_base:
                    dil &= bases[use_base][0]
                for tag, B in (("lo", ero), ("est", band), ("hi", dil)):
                    for name, m in pmask.items():
                        cnt = C.to_regions(np.array([np.count_nonzero(B & m & (lab == k + 1)) for k in range(K)]))
                        for i, reg in enumerate(C.REGIONS):
                            env_rows.append({"layer": name, "product": p, "surface": surf, "region": reg,
                                             "delta_m": D, "radius_m": r_m, "bound": tag,
                                             "amount": float(cnt[i] * cell), "n_changed": np.nan})
                    for n, spec in feats.items():
                        s, rule, thr = samp[n], spec["rule"], spec.get("threshold_m")
                        sets = region_sets(freg[n])
                        rows_, cols_ = s["row"].to_numpy(), s["col"].to_numpy()
                        fv, prof = cache[(surf, n)]
                        if rule == "access":
                            if tag == "est":
                                continue
                            dr, dl = ex.distance_at(prof, L26), ex.distance_at(prof, round(L26 - D, 6))
                            g = pd.DataFrame({"fid": s["fid"], "dr": dr, "dl": dl}).groupby("fid").min()
                            t_eff = thr - r_m if tag == "hi" else thr + r_m
                            changed = (g["dr"] <= thr) & (g["dl"] > t_eff)
                            for reg, fids in sets.items():
                                env_rows.append({"layer": n, "product": p, "surface": surf, "region": reg,
                                                 "delta_m": D, "radius_m": r_m, "bound": tag,
                                                 "amount": float(changed.reindex(fids).fillna(False).sum()),
                                                 "n_changed": float(changed.reindex(fids).fillna(False).sum())})
                            continue
                        inB = B[rows_, cols_]
                        wet_ref = (fv < L26) | inB
                        wet = (fv < round(L26 - D, 6)) & ~inB
                        per = nt.exposure_from_states(rule, s, wet_ref, wet).set_index("fid")
                        for reg, fids in sets.items():
                            sm = ex.summarise_exposure(per.loc[fids], rule)
                            env_rows.append({"layer": n, "product": p, "surface": surf, "region": reg, "delta_m": D,
                                             "radius_m": r_m, "bound": tag, "amount": sm["amount"],
                                             "n_changed": sm["n_changed"]})
                del band, dil, ero
            C.log(p, surf, "envelope done")
        del base
    checks["uncorrected_features_vs_exposure_csv_max_abs_diff"] = worst_feat
    if worst_feat > 1e-6:
        raise SystemExit(f"check failed: uncorrected feature exposure differs from exposure.csv by {worst_feat}")
    C.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(poly_rows).to_csv(C.OUT / "exposure_polygons.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(C.OUT / "exposure_features.csv", index=False)
    pd.DataFrame(env_rows).to_csv(C.OUT / "exposure_envelope.csv", index=False)
    # ports: distance of each feature to the observed 2026 sea (the composite observed
    # baseline, D-123 addendum 2), on 240 m blocks (any cell), by the Euclidean distance
    # transform; the feature's distance is the smallest over its samples
    from scipy.ndimage import distance_transform_edt
    k4 = 4
    sea240 = nt.block_region_counts(base_c.astype(np.uint8), k4, 1)[0] > 0
    del base_c
    d240 = distance_transform_edt(~sea240) * (k4 * res)
    sp = samp["osm_ports"] if "osm_ports" in samp else ex.feature_samples(osm["osm_ports"]["geoms"], tr, shape)
    rr, cc = np.minimum(sp["row"].to_numpy() // k4, d240.shape[0] - 1), np.minimum(sp["col"].to_numpy() // k4,
                                                                                    d240.shape[1] - 1)
    pr = pd.Series(d240[rr, cc], index=sp["fid"].to_numpy()).groupby(level=0).min()
    fr = freg["osm_ports"] if "osm_ports" in freg else ex.feature_regions(sp, lab)
    sec = {k + 1: s_ for k, s_ in enumerate(SECTOR_ORDER)}
    tags = ports_g["tags"].map(lambda t: json.loads(t) if isinstance(t, str) else {})
    pt = pd.DataFrame({"fid": np.arange(len(ports_g)), "osm_type": ports_g["osm_type"], "osm_id": ports_g["osm_id"],
                       "type": ptype, "sector": [sec.get(int(fr.get(i, 0)), "outside") for i in range(len(ports_g))],
                       "distance_to_obs_sea_km": [float(pr.get(i, np.inf)) / 1000 for i in range(len(ports_g))],
                       "harbour_tag": tags.map(lambda t: t.get("harbour")),
                       "key_tags": tags.map(lambda t: ";".join(f"{k}={t[k]}" for k in
                                                               ("industrial", "landuse", "harbour", "seamark:type",
                                                                "man_made", "amenity") if k in t))})
    pt["class"] = np.where((pt["type"] == "port_harbour") & (pt["distance_to_obs_sea_km"] <= FAR_KM),
                           "seaport_harbour", "river_pier_landing")
    pt["far_up_delta"] = pt["sector"].isin(["volga", "ural"]) & (pt["distance_to_obs_sea_km"] > FAR_KM)
    C.WORK.mkdir(parents=True, exist_ok=True)
    pt.to_csv(C.WORK / "ports_classified.csv", index=False)
    summ = []
    for col, vals in (("type", ["port_harbour", "pier_landing"]), ("class", ["seaport_harbour", "river_pier_landing"])):
        for v in vals:
            q = pt[pt[col] == v]
            summ.append({"by": col, "value": v, "n": int(len(q)), "n_in_domain": int((q["sector"] != "outside").sum()),
                         **{f"n_within_{int(k)}km": int((q["distance_to_obs_sea_km"] <= k).sum())
                            for k in DELTA_CLASSES_KM},
                         "n_far_up_delta": int(q["far_up_delta"].sum())})
    pd.DataFrame(summ).to_csv(C.OUT / "ports_summary.csv", index=False)
    C.write_json(C.OUT / "run_exposure.json", {
        "script": "scripts/rev_n_exposure.py", "decision": "D-126", "level_2026_m": L26, "deltas_m": deltas,
        "envelope_deltas_m": env_deltas,
        "radii": {f"{p}|{v}": r for (p, v), r in rad.items()},
        "polygon_cells": {n: int(m.sum()) for n, m in pmask.items()},
        "n_samples": {n: int(len(s)) for n, s in samp.items()}, "checks": checks,
        "port_distance_note": "distance from the feature's cells to the nearest 240 m block holding a cell of the "
                              "composite observed 2026 sea (D-123 addendum 2; Euclidean, block centres)",
        "far_up_delta_km": FAR_KM, "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
