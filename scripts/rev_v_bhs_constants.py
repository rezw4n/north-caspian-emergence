"""Revision, workstream V: the fitted product constants expressed in BHS-77 (R2.M3
request (i)), and the mean depth of the North Caspian at the 2026 level (R2 minor 5);
D-122.

Constants in BHS-77. The shape variant floods z + dN - c, with z the product height as
distributed, dN = N(EGM2008, NGA) - N(EIGEN-6C4) per cell (the primary correction) and c
the D-070 constant on the DAHITI datum. A level H_D on the DAHITI datum is H_D + delta in
BHS-77 (D-093). So the product's heights as distributed become BHS-77 heights by adding
    k = mean(dN) - c + delta,
with mean(dN) over the cells the constant was fitted on (the domain's training-block
cells for the domain constant; a sector's training cells for the R2 sector constants).
A product built from chart depths shifted by 28 m (GEBCO's Caspian grid, GEBCO_2014
documentation Annex A.1; ETOPO1, Amante and Eakins 2009 section 3.2.1) puts the charts'
zero at -28 m; in BHS-77 that zero then sits at -28 + k. This is what the fitted
constant implies if all of k were a datum offset; it is not a documented chart datum.

Mean depth at the 2026 level L (results/emergence/main/run.json, DAHITI datum): over
the study domain (the North Caspian north of the Chechen Island - Cape Tyub-Karagan
line), the cells connected to the open sea below the level (flood-level map F < level),
their area and mean depth, per product, for the shape variant (level L + c) and the
total variant (level L).

Output: results/revision/v/bhs_constants.json. Usage: python scripts/rev_v_bhs_constants.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import hindcast_tiled as ht
from caspian import products
from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER, sectors

SCRIPTS = Path(__file__).resolve().parent
OUT = RESULTS / "revision" / "v"
CORRECTION = RESULTS / "study_area" / "geoid_correction_primary.npz"
CHART_SHIFT_DOCUMENTED = {
    "gebco_2026": "GEBCO_2014 documentation (BODC Doc. 301801) Annex A.1: Hall's Caspian grid from 107 Russian "
                  "charts, 28 m added to the depths; GEBCO_2026 codes the North Caspian TID 70 (pre-generated grid) "
                  "and does not re-describe it",
    "gebco_2022": "as GEBCO_2026",
    "etopo_2022": "GEBCO 2022 is the lake-bathymetry base in the Caspian (MacFerrin et al. 2025); workstream G finds "
                  "the lake-depth cells equal min(GEBCO_2022, -29 m), an undocumented cap "
                  "(results/revision/g/etopo/lake_clamp.json), so k is not a pure chart-datum offset",
    "etopo1": "Amante and Eakins (2009) sections 3.1.2 and 3.2.1: contours of the 1987 USSR navigation map "
              "referenced to local lake level, shifted 28 m; NOAA NCEI, ETOPO1 Data Sources (2023), p. 2: "
              "'shifted 28-m vertically to be consistent with mean sea level'",
    "srtm15p_v2.7": None}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    a7 = _load("a7_hindcast")
    run_rec = json.loads((RESULTS / "hindcast" / "main" / "run.json").read_text(encoding="utf-8"))
    tile = int(run_rec["tile"])
    w, s, e, n = DOMAIN.bounds
    transform, shape = a7.grid((w - a7.MARGIN_DEG, s - a7.MARGIN_DEG, e + a7.MARGIN_DEG, n + a7.MARGIN_DEG))
    dom = a7.burn([(a7.to_laea(DOMAIN), 1)], transform, shape) > 0
    sec = a7.burn([(a7.to_laea(p), k + 1) for k, (_, p) in enumerate(sectors().values())], transform, shape)
    sec[~dom] = 0
    blocks = pd.read_csv(a7.BLOCKS)
    fit = dom & a7.train_mask(blocks, transform, shape)
    corr, corr_rec = products.correction(CORRECTION, transform, shape)
    if corr_rec["sha256"] != run_rec["vertical_correction"]["sha256"]:
        raise SystemExit("the correction is not the primary run's")
    off = json.loads((RESULTS / "levels" / "dahiti_bhs_offset.json").read_text(encoding="utf-8"))
    delta, sig_delta = float(off["primary"]["delta_m"]), float(off["posts_and_years_bootstrap"]["sigma_m"])
    prim = json.loads((RESULTS / "hindcast" / "main" / "shape_offsets.json").read_text(encoding="utf-8"))["used_in_this_run"]
    r2 = json.loads((RESULTS / "hindcast" / "main_sectorshape" / "sector_offsets.json").read_text(encoding="utf-8"))
    l26 = float(json.loads((RESULTS / "emergence" / "main" / "run.json").read_text(encoding="utf-8"))["level_2026"]["level_m"])

    # mean dN over the fit cells, overall and per sector (cells with a finite correction)
    K = len(SECTOR_ORDER)
    sdn, cnt, sdn2 = np.zeros(K + 1), np.zeros(K + 1), 0.0
    for r0, r1, c0, c1 in ht.tiles(shape, tile):
        m = fit[r0:r1, c0:c1]
        v = np.asarray(corr[r0:r1, c0:c1], float)
        ok = m & np.isfinite(v)
        sk = sec[r0:r1, c0:c1][ok].astype(np.int64)
        sdn += np.bincount(sk, weights=v[ok], minlength=K + 1)
        cnt += np.bincount(sk, minlength=K + 1)
        sdn2 += float((v[ok] ** 2).sum())
    mean_dn = float(sdn.sum() / cnt.sum())
    sd_dn = float(np.sqrt(sdn2 / cnt.sum() - mean_dn ** 2))
    mean_dn_sec = {SECTOR_ORDER[k - 1]: float(sdn[k] / cnt[k]) for k in range(1, K + 1) if cnt[k] > 0}

    rec = {"delta_m": delta, "delta_sigma_posts_years_m": sig_delta, "mean_dN_fit_m": mean_dn, "sd_dN_fit_m": sd_dn,
           "mean_dN_fit_by_sector_m": mean_dn_sec, "level_2026_dahiti_m": l26, "level_2026_bhs_m": l26 + delta,
           "rule": "k = mean(dN) - c + delta: added to the product heights as distributed, they become BHS-77 heights",
           "products": {}}
    for p, so in prim.items():
        c = float(so["offset_m"])
        k = mean_dn - c + delta
        by = {}
        for nm, v in r2["products"][p]["sectors"].items():
            if nm in mean_dn_sec:
                by[nm] = {"c_m": float(v["offset_m"]), "k_bhs_m": mean_dn_sec[nm] - float(v["offset_m"]) + delta}
        rec["products"][p] = {"c_domain_m": c, "k_bhs_m": k, "chart_zero_bhs_if_28m_shift_m": -28.0 + k,
                              "chart_shift_documented": CHART_SHIFT_DOCUMENTED.get(p),
                              "sector_constants_R2": by,
                              "k_bhs_range_over_sectors_m": [min(b["k_bhs_m"] for b in by.values()),
                                                             max(b["k_bhs_m"] for b in by.values())]}

    # mean depth at the 2026 level
    cache = INTERIM / "hindcast" / "flood"
    for p, so in prim.items():
        c = float(so["offset_m"])
        raw = products.heights(p, transform, shape, run_rec["resampling"])
        h = products.Corrected(raw, corr)
        F = _primary_flood_map(p, cache)
        res = {}
        for var, lev in (("shape", l26 + c), ("total", l26)):
            area, sdep = 0, 0.0
            for r0, r1, c0, c1 in ht.tiles(shape, tile):
                m = dom[r0:r1, c0:c1] & (np.asarray(F[r0:r1, c0:c1], float) < lev)
                z = np.asarray(h[r0:r1, c0:c1], float)[m]
                area += int(m.sum())
                sdep += float((lev - z).sum())
            res[var] = {"flood_level_m": lev, "sea_area_km2": area * a7.CELL_KM2,
                        "mean_depth_m": sdep / area if area else None}
        rec["products"][p]["mean_depth_2026"] = res
        print(p, json.dumps(res), flush=True)
    vals = [rec["products"][p]["mean_depth_2026"]["shape"]["mean_depth_m"] for p in prim]
    rec["mean_depth_2026_shape_range_m"] = [min(vals), max(vals)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bhs_constants.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps({p: {k: v for k, v in r.items() if k in ("c_domain_m", "k_bhs_m", "chart_zero_bhs_if_28m_shift_m",
                                                              "k_bhs_range_over_sectors_m")}
                      for p, r in rec["products"].items()}, indent=1))
    return 0


def _primary_flood_map(p, cache):
    """The primary run's cached flood-level map of product p. Its key needs the static
    sea seed, which is not rebuilt here; the single cached map named <p>_<12-hex key>.npy
    (not a _sectorshape or _planar map) is used, and more than one stops the script."""
    cands = [f for f in cache.glob(f"{p}_*.npy") if f.stem[len(p) + 1:].isalnum() and len(f.stem) == len(p) + 13]
    if len(cands) != 1:
        raise SystemExit(f"{p}: {len(cands)} candidate primary flood-level maps")
    return np.load(cands[0], mmap_mode="r")


if __name__ == "__main__":
    sys.exit(main())
