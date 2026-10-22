"""A18 item 10 (a)-(e) and (h): near-term emergence on the common baseline, the shifted
products and the uncorrected surfaces, with the reference-level and datum sensitivities,
the already-dry share, the observed band, dry seabed against isolated water, and the
integer-contour diagnostics. Exploratory, except the common baseline
and the running-median reference level, which are registered sensitivities.

Inputs: the registered caches only (scripts/rev_n_common.py). The EGM2008-constant
sensitivity needs five uncorrected flood maps on the no-constant chain; they are built
once with scripts/a7_hindcast.flood_map_cached and cached under data/interim/revision_n/.

Outputs (results/revision/n/):
  curves.csv            E(D) at 0.1 m steps 0-5 m: surface x sensitivity x product x
                        region (sensitivities: domain and validated sectors only), with
                        the dry and isolated parts for the main reference level
  curves_fine.csv.gz    the same at 0.01 m steps, main reference level, domain and
                        validated sectors (input of rev_n_pathways.py)
  registered_check.json the common-baseline (area) values against the sea_area rows of
                        results/emergence/main/emergence.csv
  already_dry.csv       share of each band dry in the observed 2025-26 extent (c)
  observed_band.csv     the observed seabed between L2026 and the lowest audited unit (e)
  contour_classes.csv   band area by distributed height class (whole-metre terraces)
  contour_positions.csv where distributed whole-metre contours fall on the DAHITI datum
  shared_soundings.json native-lattice comparison of the products with GEBCO_2022
  run_curves.json       inputs, levels, checks, timing

Option --fused-form constant: the common baseline from the constant-form
fused maps; --fused-form composite: the composite observed baseline of
scripts/rev_n_baseline.py, not checked (no registered counterpart). Outputs carry the
suffix _constfused or _composite (curves, curves_fine, already_dry,
run_curves), checked against results/emergence/main_constant/emergence.csv; the
diagnostics that do not depend on the baseline are not rewritten.

Usage: python scripts/rev_n_curves.py [--tile 2048] [--fused-form linear_hold|constant]
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
from caspian.config import INTERIM, RESULTS  # noqa: E402
from caspian.geography import DOMAIN  # noqa: E402

DELTAS = np.round(np.arange(0, 51) / 10.0, 2)          # 0-5 m, 0.1 m
DELTAS_FINE = np.round(np.arange(0, 501) / 100.0, 2)   # 0-5 m, 0.01 m
BELOW, ABOVE = 7.5, 0.8                                # level grid around the reference level
SURFACES = ("uncorrected", "common_area", "common_location", "shifted", "shifted_common_area",
            "shifted_common_location")


def mask_fn(*parts):
    """Window callable: the AND of window callables / arrays (None skipped)."""
    parts = [p for p in parts if p is not None]

    def f(r0, r1, c0, c1):
        m = None
        for p in parts:
            v = p(r0, r1, c0, c1) if callable(p) else np.asarray(p[r0:r1, c0:c1]).astype(bool)
            m = v if m is None else (m & v)
        return m
    return f


def below_fn(F, L):
    return lambda r0, r1, c0, c1: np.asarray(F[r0:r1, c0:c1], np.float64) < L


def counts_for_product(S, p, lv, lv_rm, Lb, dry, valid_zv, Fnc, tile):
    lab, K = S["lab_in"], S["K"]
    F, Ff, z, c = S["F"][p], S["Ffused"][p], S["z"][p], S["c"][p]
    L26 = S["L26"]
    rc = lambda A, grid, m=None: nt.region_counts(A, lab, K, grid, mask=m, tile=tile)  # noqa: E731
    out = {"F_all": rc(F, lv), "F_all_rm": rc(F, lv_rm), "Ff_all": rc(Ff, lv), "Ff_all_rm": rc(Ff, lv_rm)}
    for name, L in Lb.items():
        grid = lv_rm if name == "running_median" else lv
        out[f"F_base[{name}]"] = rc(F, grid, below_fn(Ff, L))
    base = below_fn(Ff, L26)
    out["z_selfu"] = rc(z, lv, below_fn(F, L26))
    out["z_selfs"] = rc(z, lv, below_fn(F, L26 + c))
    out["z_base"] = rc(z, lv, base)
    out["F_dry"] = rc(F, lv, dry)
    out["F_validzv"] = rc(F, lv, valid_zv)
    out["F_base_dry"] = rc(F, lv, mask_fn(base, dry))
    out["F_base_validzv"] = rc(F, lv, mask_fn(base, valid_zv))
    out["Ff_dry"] = rc(Ff, lv, dry)
    out["Ff_validzv"] = rc(Ff, lv, valid_zv)
    out["Fnc_all"] = rc(Fnc, lv)
    out["Fnc_base"] = rc(Fnc, lv, base)
    return out


def at(cnt, grid, L):
    return nt.interp_counts(cnt, grid, np.atleast_1d(L))


def curves(cnt, lv, lv_rm, Lb, c, cell, deltas):
    """{(surface, sensitivity): [7, nD] km2}."""
    out = {}
    for sens, L in list(Lb.items()) + [("egm2008_constant_0", Lb["main"])]:
        rm = sens == "running_median"
        grid = lv_rm if rm else lv
        Fall = cnt["F_all_rm"] if rm else (cnt["Fnc_all"] if sens.startswith("egm") else cnt["F_all"])
        Fff = cnt["Ff_all_rm"] if rm else cnt["Ff_all"]
        Fbase = cnt["Fnc_base"] if sens.startswith("egm") else cnt[f"F_base[{sens}]"]
        base_area = at(Fff, grid, L)[:, 0]
        d = np.asarray(deltas, float)
        out[("uncorrected", sens)] = (at(Fall, grid, L) - at(Fall, grid, L - d)) * cell
        out[("common_area", sens)] = (base_area[:, None] - at(Fall, grid, L - d)) * cell
        out[("common_location", sens)] = (base_area[:, None] - at(Fbase, grid, L - d)) * cell
        if sens.startswith("egm"):
            continue                    # the shape constant absorbs the datum constant
        out[("shifted", sens)] = (at(Fall, grid, L + c) - at(Fall, grid, L + c - d)) * cell
        out[("shifted_common_area", sens)] = (base_area[:, None] - at(Fall, grid, L + c - d)) * cell
        out[("shifted_common_location", sens)] = (base_area[:, None] - at(Fbase, grid, L + c - d)) * cell
    return out


def isolated(cnt, lv, L26, c, cell, deltas):
    d = np.asarray(deltas, float)
    L = L26 - d
    return {"uncorrected": nt.isolated_from_counts(cnt["z_selfu"], cnt["F_all"], lv, L) * cell,
            "shifted": nt.isolated_from_counts(cnt["z_selfs"], cnt["F_all"], lv, L + c) * cell,
            "common_location": nt.isolated_from_counts(cnt["z_base"], cnt["F_base[main]"], lv, L) * cell,
            "shifted_common_location": nt.isolated_from_counts(cnt["z_base"], cnt["F_base[main]"], lv, L + c) * cell}


def already_dry(cnt, lv, L26, c, cell, deltas=(0.5, 1.0, 2.0)):
    """Band area, its part dry in the zone-v6 2025-26 extent (and the part with no value
    there), and its part outside the 60 m observed baseline, per region [7]."""
    rows = []
    d = np.asarray(deltas, float)
    for surf in ("uncorrected", "shifted", "common_location"):
        if surf == "common_location":
            band = at(cnt["Ff_all"], lv, L26)[:, :1] - at(cnt["F_base[main]"], lv, L26 - d)
            b_dry = at(cnt["Ff_dry"], lv, L26)[:, :1] - at(cnt["F_base_dry"], lv, L26 - d)
            b_val = at(cnt["Ff_validzv"], lv, L26)[:, :1] - at(cnt["F_base_validzv"], lv, L26 - d)
            b_nobase = np.zeros_like(band)
        else:
            top = L26 + (c if surf == "shifted" else 0.0)
            lo = top - d
            band = at(cnt["F_all"], lv, top) - at(cnt["F_all"], lv, lo)
            b_dry = at(cnt["F_dry"], lv, top) - at(cnt["F_dry"], lv, lo)
            b_val = at(cnt["F_validzv"], lv, top) - at(cnt["F_validzv"], lv, lo)
            b_base = at(cnt["F_base[main]"], lv, top) - at(cnt["F_base[main]"], lv, lo)
            b_nobase = band - b_base
        for j, D in enumerate(d):
            rows.append({"surface": surf, "delta_m": float(D), "band": band[:, j] * cell, "dry_zv6": b_dry[:, j] * cell,
                         "nodata_zv6": (band[:, j] - b_val[:, j]) * cell, "valid_zv6": b_val[:, j] * cell,
                         "outside_obs_baseline": b_nobase[:, j] * cell})
    return rows


def observed_band(S, tile):
    """(e): counts per sector of the observed units E3 bins 12, 13, 14 and of the fused
    baselines (km2)."""
    a7, st, tr, shape, lab, K = S["a7"], S["static"], S["transform"], S["shape"], S["lab_in"], S["K"]
    a7.ENVELOPE = st["envelope"]
    units = a7.stage2_index(INTERIM / "stage2" / "main")
    codes = {}
    for b in (12, 13, 14):
        raw, _ = a7.observed_raw(a7.groups_for(units, b, ["E3"]), tr, shape)
        codes[b] = a7.classify(raw, st, tile)
        del raw
    nR = K + 1
    acc = {k: np.zeros(nR, np.int64) for k in (
        "known12", "sea12", "known13", "sea13", "known14", "sea14", "both_12_13", "band_13_12", "unknown12_sea13",
        "both_12_14", "band_14_12", "unknown12_sea14")}
    e2 = {p: np.zeros(nR, np.int64) for p in S["names"]}
    base = {p: np.zeros(nR, np.int64) for p in S["names"]}
    for r0, r1, c0, c1 in ht.tiles(shape, tile):
        o = np.asarray(st["outside"][r0:r1, c0:c1])
        lb = np.asarray(lab[r0:r1, c0:c1]).astype(np.int64)
        km = {b: ex.known_masks(np.asarray(codes[b][r0:r1, c0:c1]), o) for b in codes}
        k12, w12 = km[12]
        s12 = k12 & w12
        put = {"known12": k12, "sea12": s12}
        for b in (13, 14):
            kb, wb = km[b]
            sb = kb & wb
            put[f"known{b}"], put[f"sea{b}"] = kb, sb
            put[f"both_12_{b}"] = k12 & kb
            put[f"band_{b}_12"] = k12 & kb & wb & ~w12
            put[f"unknown12_sea{b}"] = sb & ~k12
        for k, m in put.items():
            acc[k] += np.bincount(lb[m], minlength=nR)
        for p in S["names"]:
            sb_ = np.asarray(S["Ffused"][p][r0:r1, c0:c1], np.float64) < S["L26"]
            base[p] += np.bincount(lb[sb_], minlength=nR)
            e2[p] += np.bincount(lb[sb_ & ~s12], minlength=nR)
    del codes
    cell = S["a7"].CELL_KM2
    reg = C.REGIONS
    rows = []
    for k, v in acc.items():
        rr = C.to_regions(v[1:])
        rows += [{"quantity": k, "product": "observed", "region": reg[i], "km2": float(rr[i] * cell)} for i in range(len(reg))]
    for p in S["names"]:
        for name, v in (("e2_baseline_less_sea12", e2[p]), ("baseline_sea_L2026", base[p])):
            rr = C.to_regions(v[1:])
            rows += [{"quantity": name, "product": p, "region": reg[i], "km2": float(rr[i] * cell)} for i in range(len(reg))]
    return pd.DataFrame(rows)


def contour_classes(S, tile):
    """Band area (domain) by the distributed height class of the 60 m (bilinear) raw heights:
    an exact whole metre k (terrace) or between k and k + 1 (slope)."""
    lab, L26 = S["lab_in"], S["L26"]
    lo_k, hi_k = -45, -15
    nk = hi_k - lo_k + 1
    rows = []
    for p in S["names"]:
        F, raw, c = S["F"][p], S["raw"][p], S["c"][p]
        bands = {("uncorrected", 0.5): (L26 - 0.5, L26), ("uncorrected", 1.0): (L26 - 1.0, L26),
                 ("uncorrected", 2.0): (L26 - 2.0, L26), ("shifted", 1.0): (L26 + c - 1.0, L26 + c),
                 ("shifted", 2.0): (L26 + c - 2.0, L26 + c), ("domain", 0.0): None}
        acc = {k: np.zeros(2 * nk + 1, np.int64) for k in bands}
        hb = np.round(np.arange(-50, 51) / 10.0, 1)                  # difference bins vs GEBCO_2022, 0.1 m
        dif = {k: np.zeros(hb.size + 2, np.int64) for k in bands}
        eq = {k: 0 for k in bands}
        g22 = S["raw"]["gebco_2022"]
        for r0, r1, c0, c1 in ht.tiles(S["shape"], tile):
            inside = np.asarray(lab[r0:r1, c0:c1]) > 0
            v = np.asarray(raw[r0:r1, c0:c1], np.float64)
            f = np.asarray(F[r0:r1, c0:c1], np.float64)
            dd = v - np.asarray(g22[r0:r1, c0:c1], np.float64)
            di = np.clip(np.searchsorted(hb, np.round(dd, 1) - 1e-9), 0, hb.size + 1)
            di[~np.isfinite(dd)] = hb.size + 1
            rv = np.round(v)
            integer = np.abs(v - rv) < 1e-3
            k = np.where(integer, rv, np.floor(v))
            code = np.where(np.isfinite(v) & (k >= lo_k) & (k <= hi_k),
                            2 * (k - lo_k) + (~integer).astype(int), 2 * nk).astype(np.int64)
            for key, rng_ in bands.items():
                m = inside if rng_ is None else inside & (f >= rng_[0]) & (f < rng_[1])
                acc[key] += np.bincount(code[m], minlength=2 * nk + 1)
                dif[key] += np.bincount(di[m], minlength=hb.size + 2)
                eq[key] += int((np.abs(dd[m]) < 1e-3).sum())
        for (surf, D), a in acc.items():
            for j in range(2 * nk):
                if a[j]:
                    rows.append({"product": p, "band": surf, "delta_m": D, "k": lo_k + j // 2,
                                 "class": "slope" if j % 2 else "terrace", "area_km2": float(a[j] * S["a7"].CELL_KM2)})
            if a[2 * nk]:
                rows.append({"product": p, "band": surf, "delta_m": D, "k": np.nan, "class": "other",
                             "area_km2": float(a[2 * nk] * S["a7"].CELL_KM2)})
            h = dif[(surf, D)]
            n = int(h[:hb.size + 1].sum())
            if n and p != "gebco_2022":
                cum = np.cumsum(h[:hb.size + 1])
                med = float(hb[min(int(np.searchsorted(cum, n / 2)), hb.size - 1)])
                rows.append({"product": p, "band": surf, "delta_m": D, "k": np.nan, "class": "vs_gebco_2022",
                             "area_km2": float(n * S["a7"].CELL_KM2), "share_equal_gebco_2022": eq[(surf, D)] / n,
                             "median_difference_gebco_2022_m": med,
                             "share_within_0p5m_gebco_2022": float(h[(np.abs(hb) <= 0.5).nonzero()[0]].sum() / n)})
        C.log("contour classes", p)
    return pd.DataFrame(rows)


def contour_positions(S, Lb, tile):
    """dN over the domain cells, and where distributed whole-metre contours k fall on the
    DAHITI datum (k + dN, and k + dN - c for the shifted products) against each reference
    level."""
    lab, corr = S["lab_in"], S["corr"]
    lo, hi, s, n = np.inf, -np.inf, 0.0, 0
    for r0, r1, c0, c1 in ht.tiles(S["shape"], tile):
        m = np.asarray(lab[r0:r1, c0:c1]) > 0
        v = np.asarray(corr[r0:r1, c0:c1], np.float64)[m]
        if v.size:
            lo, hi, s, n = min(lo, v.min()), max(hi, v.max()), s + v.sum(), n + v.size
    rows = []
    for p in S["names"]:
        for surf, shift in (("uncorrected", 0.0), ("shifted", -S["c"][p])):
            for k in range(-32, -25):
                for sens, L in Lb.items():
                    rows.append({"product": p, "surface": surf, "k_distributed_m": k, "sensitivity": sens,
                                 "level_2026_m": L, "dahiti_min_m": k + lo + shift, "dahiti_max_m": k + hi + shift,
                                 "below_L2026_min_m": L - (k + hi + shift), "below_L2026_max_m": L - (k + lo + shift)})
    return pd.DataFrame(rows), {"dN_min_m": float(lo), "dN_max_m": float(hi), "dN_mean_m": float(s / n), "n_cells": int(n)}


def shared_soundings(names):
    """Native-lattice comparison of each product with GEBCO_2022 over the native cells
    whose centre lies in the domain (all, and with GEBCO_2022 between -34 and -26 m)."""
    import shapely
    ref = products.load_native("gebco_2022")
    lon_g, lat_g = np.meshgrid(ref.lon, ref.lat)
    inside = shapely.contains_xy(DOMAIN, lon_g, lat_g)
    out = {"reference": "gebco_2022", "lattice": "15 arc-second cell centres", "products": {}}
    for p in names:
        if p == "gebco_2022":
            continue
        q = products.load_native(p)
        same = q.lon.shape == ref.lon.shape and q.lat.shape == ref.lat.shape and \
            np.allclose(q.lon, ref.lon, atol=1e-9) and np.allclose(q.lat, ref.lat, atol=1e-9)
        rec = {"same_lattice": bool(same)}
        if same:
            for scope, m in (("domain", inside), ("domain_-34_to_-26m", inside & (ref.z >= -34) & (ref.z <= -26))):
                a, b = q.z[m].astype(np.float64), ref.z[m].astype(np.float64)
                ok = np.isfinite(a) & np.isfinite(b)
                d = a[ok] - b[ok]
                vals, cnts = np.unique(np.round(d, 2), return_counts=True)
                mode = float(vals[np.argmax(cnts)])
                rec[scope] = {"n_cells": int(ok.sum()), "share_equal": float(np.mean(np.abs(d) < 1e-3)),
                              "modal_difference_m": mode,
                              "share_within_0p01_of_mode": float(np.mean(np.abs(d - mode) <= 0.01)),
                              "median_difference_m": float(np.median(d)),
                              "p05_difference_m": float(np.percentile(d, 5)),
                              "p95_difference_m": float(np.percentile(d, 95)),
                              "share_product_integer": float(np.mean(np.abs(a[ok] - np.round(a[ok])) < 1e-3)),
                              "share_reference_integer": float(np.mean(np.abs(b[ok] - np.round(b[ok])) < 1e-3))}
        out["products"][p] = rec
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--fused-form", default="linear_hold", choices=["linear_hold", "constant", "composite"])
    a = ap.parse_args(argv)
    t0 = time.time()
    sfx = {"linear_hold": "", "constant": "_constfused", "composite": "_composite"}[a.fused_form]
    S = C.setup(tile=a.tile, heights=True, fused_form=a.fused_form)
    a7, tr, shape, K, L26, cell = S["a7"], S["transform"], S["shape"], S["K"], S["L26"], S["a7"].CELL_KM2
    rm = C.level_2026_running_median()
    L26rm = round(float(rm["level_m"]), 4)
    Lb = {"main": L26, "L26+0.3": round(L26 + 0.3, 4), "L26-0.3": round(L26 - 0.3, 4), "running_median": L26rm}
    lv = ex.level_grid(L26, BELOW, above=ABOVE)
    lv_rm = ex.level_grid(L26rm, BELOW, above=ABOVE)
    for L in (L26 + 0.3, L26 - 0.3):
        if not np.isclose(lv, round(L, 6), atol=1e-6).any():
            raise SystemExit("the level grid does not hold L2026 +- 0.3 m")
    for p in S["names"]:
        if not np.isclose(lv, round(L26 + S["c"][p], 6), atol=1e-6).any():
            raise SystemExit(f"{p}: the level grid does not hold L2026 + c")
    # observed 2025-26 extent (zone v6), nearest 250 m cell
    zv = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    a7.check_crs(str(zv["crs"]), "extents_v6_250m.npz")
    low = a7.nearest(zv["low"], zv["transform"], tr, shape, 255)
    dry, valid_zv = low == 0, low != 255
    del low
    # EGM2008 constant set to 0: the no-constant chain, flood maps cached
    corr_nc, corr_nc_rec = products.correction(C.CORR_NO_CONSTANT, tr, shape, cache=C.WORK / "correction")
    seed = S["static"]["sea_seed"]
    rows, fine_rows, dry_rows, check = [], [], [], {}
    em = pd.read_csv(RESULTS / "emergence" / ("main" if not sfx else "main_constant") / "emergence.csv")
    em = em[(em["functional"] == "sea_area") & (em["surface"] == "uncorrected")]
    worst = 0.0
    nc_files = {}
    for p in S["names"]:
        h = S["raw"][p]
        t1 = time.time()
        Fnc = a7.flood_map_cached(p, products.Corrected(h, corr_nc), seed,
                                  {"heights": Path(h.filename).name, "correction": corr_nc_rec["sha256"]}, a.tile,
                                  cache=C.WORK / "flood")
        nc_files[p] = Path(Fnc.filename).name
        C.log(p, "no-constant flood map", round(time.time() - t1, 1), "s")
        cnt = counts_for_product(S, p, lv, lv_rm, Lb, dry, valid_zv, Fnc, a.tile)
        C.log(p, "counts done")
        c = S["c"][p]
        cv = curves(cnt, lv, lv_rm, Lb, c, cell, DELTAS)
        iso = isolated(cnt, lv, L26, c, cell, DELTAS)
        for (surf, sens), arr in cv.items():
            rr = C.to_regions(arr)
            regs = range(len(C.REGIONS)) if sens == "main" else [C.REGIONS.index("domain"), C.REGIONS.index("validated")]
            ii = C.to_regions(iso[surf]) if (sens == "main" and surf in iso) else None
            for i in regs:
                for j, D in enumerate(DELTAS):
                    row = {"surface": surf, "sensitivity": sens, "product": p, "region": C.REGIONS[i],
                           "delta_m": float(D), "level_2026_m": Lb.get(sens, L26), "E_km2": float(rr[i, j])}
                    if ii is not None:
                        row["isolated_km2"] = float(ii[i, j])
                        row["dry_km2"] = float(rr[i, j] - ii[i, j])
                    rows.append(row)
        cvf = curves(cnt, lv, lv_rm, {"main": L26}, c, cell, DELTAS_FINE)
        for (surf, sens), arr in cvf.items():
            if sens != "main":
                continue
            rr = C.to_regions(arr)
            for i in (C.REGIONS.index("domain"), C.REGIONS.index("validated")):
                fine_rows += [{"surface": surf, "product": p, "region": C.REGIONS[i], "delta_m": float(D),
                               "E_km2": float(rr[i, j])} for j, D in enumerate(DELTAS_FINE)]
        # registered check: common baseline (area) at the scenarios against emergence.csv sea_area rows
        ca = cv[("common_area", "main")]
        car = C.to_regions(ca)
        for r in em[em["product"] == p].itertuples():
            if r.region not in C.REGIONS:
                continue
            j = int(np.flatnonzero(np.isclose(DELTAS, r.scenario_m))[0])
            worst = max(worst, abs(float(car[C.REGIONS.index(r.region), j]) - float(r.estimate_km2)))
        for rec in already_dry(cnt, lv, L26, c, cell):
            for name in ("band", "dry_zv6", "nodata_zv6", "valid_zv6", "outside_obs_baseline"):
                rec[name] = C.to_regions(rec[name])
            for i, reg in enumerate(C.REGIONS):
                band, val = float(rec["band"][i]), float(rec["valid_zv6"][i])
                dry_rows.append({"product": p, "surface": rec["surface"], "delta_m": rec["delta_m"], "region": reg,
                                 "band_km2": band, "dry_zv6_km2": float(rec["dry_zv6"][i]),
                                 "nodata_zv6_km2": float(rec["nodata_zv6"][i]),
                                 "share_dry_zv6": float(rec["dry_zv6"][i]) / val if val > 0 else np.nan,
                                 "outside_obs_baseline_km2": float(rec["outside_obs_baseline"][i]),
                                 "share_outside_obs_baseline": float(rec["outside_obs_baseline"][i]) / band
                                 if band > 0 else np.nan})
        del cnt, Fnc
    check["common_area_vs_emergence_csv_sea_area"] = {"max_abs_diff_km2": worst, "n": int(len(em))}
    if a.fused_form == "composite":
        check = {"common_area_vs_emergence_csv_sea_area": "not applicable: no registered counterpart (D-123 addendum 2)"}
    elif worst > 1e-6:
        raise SystemExit(f"registered check failed: common baseline differs from emergence.csv by {worst} km2")
    C.log("registered check", check["common_area_vs_emergence_csv_sea_area"])
    C.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.OUT / f"curves{sfx}.csv", index=False)
    pd.DataFrame(fine_rows).to_csv(C.OUT / f"curves_fine{sfx}.csv.gz", index=False)
    pd.DataFrame(dry_rows).to_csv(C.OUT / f"already_dry{sfx}.csv", index=False)
    del dry, valid_zv
    if sfx:
        C.write_json(C.OUT / f"run_curves{sfx}.json", {
            "script": f"scripts/rev_n_curves.py --fused-form {a.fused_form}", "decision": "D-123 addenda",
            "level_2026_m": L26, "reference_levels_m": Lb, "fused_maps": {p: f.name for p, f in S["fused_files"].items()},
            "checks": check, "seconds": round(time.time() - t0, 1)})
        C.log("done", round(time.time() - t0, 1), "s")
        return 0
    ob = observed_band(S, a.tile)
    ob.to_csv(C.OUT / "observed_band.csv", index=False)
    C.log("observed band done")
    cc = contour_classes(S, a.tile)
    cc.to_csv(C.OUT / "contour_classes.csv", index=False)
    cp, dn = contour_positions(S, Lb, a.tile)
    cp.to_csv(C.OUT / "contour_positions.csv", index=False)
    ss = shared_soundings(S["names"])
    C.write_json(C.OUT / "shared_soundings.json", ss)
    C.write_json(C.OUT / "run_curves.json", {
        "script": "scripts/rev_n_curves.py", "decision": "D-123", "level_2026_m": L26,
        "level_2026_running_median": {**rm, "rounded_m": L26rm}, "reference_levels_m": Lb,
        "shape_constants_m": S["c"], "level_grid": {"below_m": BELOW, "above_m": ABOVE, "step_m": 0.01},
        "fused_maps": {p: f.name for p, f in S["fused_files"].items()},
        "no_constant_correction": corr_nc_rec, "no_constant_flood_maps": nc_files,
        "dN_over_domain": dn, "checks": check, "zone_v6": "results/study_area/extents_v6_250m.npz layer low",
        "regions": C.REGIONS, "validated": list(C.VALIDATED), "cell_km2": cell,
        "seconds": round(time.time() - t0, 1)})
    C.log("done", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
