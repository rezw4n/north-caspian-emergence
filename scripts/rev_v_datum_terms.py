"""Revision, workstream V: decomposition of the ICESat-2 constant c_ref (R1 minor 7,
R2.M3(b); D-122).

c = median over training points of (DEM - H), with the DEM on the DAHITI datum and
H = h_WGS84 - N(EIGEN-6C4, ICGEM, tide-free, GM zero-degree term on). So c measures how
far DAHITI heights sit above ICGEM EIGEN-6C4 normal heights of the same surface, plus
any bias of the waterline DEM. Candidate terms, each as its contribution to c (m):

documented or computable:
- GM zero-degree term (ICGEM FAQ Q16 Eq. 24; GM of EIGEN-6C4 and of WGS84 from the
  ICGEM headers): enters c only if DGFI's geoid omits it;
- permanent tide: +0.030 to +0.042 m if DAHITI heights are mean-tide (ICESat-2 R007
  parameters; DATUMS), 0 if tide-free;
- frame: ITRF2014 versus ITRF2020, +1.6 to +2.5 mm (Comparison Guide Appendix C; DATUMS);
undocumented for DAHITI, sized from documents that were read:
- altimetric heights over the TOPEX/Poseidon ellipsoid reduced by a geoid over WGS84
  (T/P is about 0.70 m smaller; Comparison Guide R007, section 3.2.3 and Table 4.1):
  h_TP - h_WGS84 at the domain's latitudes;
- a W0-type zero-degree term in DGFI's geoid: -(W0 - U0)/gamma with U0 of WGS84 (ICGEM
  header) and W0 = 62636853.4 m2 s-2 (IERS Conventions, Table 1.1, 2017 update; IAG
  2015), or NGA's -0.41 m (GeographicLib's HeightOffset for EGM2008).
Independent check without the DEM: the SWOT water surface on the same chain against
DAHITI (results/revision/v/sector_departures.csv, all blocks): -departure is the same
kind of constant as c.

Output: results/revision/v/datum_terms.json. Usage: python scripts/rev_v_datum_terms.py
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian import watersurface as ws
from caspian.config import RESULTS
from caspian.geography import DOMAIN

OUT = RESULTS / "revision" / "v"
GM_EIGEN6C4 = 3.986004415e14     # ICGEM EIGEN-6C4 .gfc header, earth_gravity_constant
GM_WGS84 = 3.986004418e14        # ICGEM .gdf header, gmrefpot (WGS84)
U0_WGS84 = 6.263685171456948e07  # ICGEM .gdf header, normal_potential (WGS84)
W0_IERS = 62636853.4             # IERS Conventions Table 1.1 (2017/11/06 update), IAG 2015 Resolution 1
W0_OLD = 62636856.0              # the pre-2015 conventional value (IERS Conventions 2010, original Table 1.1)
NGA_HEIGHT_OFFSET = -0.41        # GeographicLib egm2008 model file, HeightOffset
FRAME_M = (0.0016, 0.0025)       # docs/DATUMS.md (Comparison Guide Appendix C pipeline)


def constants() -> dict:
    out = {}
    for name, path in {"ATL06": "validation/main/ATL06_datum_constant.json",
                       "ATL08": "validation/main/ATL08_datum_constant.json",
                       "GEDI": "validation/main_gedi/GEDI_datum_constant.json",
                       "GLO30": "validation/main_glo30/GLO30_datum_constant.json"}.items():
        p = RESULTS / path
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))["constant"]
            out[name] = {"c_m": float(c["constant_m"]), "sigma_m": float(c["sigma_boot_m"]),
                         "n_points": int(c.get("n_points", 0)), "source": f"results/{path}"}
    return out


def main() -> int:
    w, s, e, n = DOMAIN.bounds
    lats = np.linspace(s, n, 7)
    _, h_tp = ws.ellipsoid_height(lats, np.full(lats.size, 50.0), np.full(lats.size, -40.0))
    tp = h_tp - (-40.0)
    g, e_ = ws.free2mean(lats)
    tide = e_ - g                        # H_mean - H_free of a surface point
    gm = ws.zero_degree_term(GM_EIGEN6C4, GM_WGS84)
    w0_iers = (W0_IERS - U0_WGS84) / 9.80665
    w0_old = (W0_OLD - U0_WGS84) / 9.80665
    terms = {
        "gm_zero_degree_if_dgfi_omits_m": gm,
        "tide_if_dahiti_mean_tide_m": [float(tide.min()), float(tide.max())],
        "frame_itrf2014_to_2020_m": list(FRAME_M),
        "tp_ellipsoid_if_unreduced_m": [float(tp.min()), float(tp.max())],
        "w0_term_iers2017_m": float(w0_iers), "w0_term_pre2015_m": float(w0_old),
        "w0_term_nga_egm2008_m": -NGA_HEIGHT_OFFSET,
        "signs": "each value is the contribution to c = H_DAHITI - H(ICGEM EIGEN-6C4) if the stated condition "
                 "holds; the GM term enters only if DGFI's geoid omits it (ICGEM includes it)",
        "latitudes_deg": [float(v) for v in lats]}
    cs = constants()
    dep = pd.read_csv(OUT / "sector_departures.csv").set_index("sector")
    swot = {"nga_realisation_m": -float(dep.loc["all", "departure_m"]),
            "icgem_realisation_m": -float(dep.loc["all", "departure_icgem_realisation_m"]),
            "meantide_working_m": -float(dep.loc["all", "departure_meantide_working_m"]),
            "note": "minus the all-block SWOT departure: how far DAHITI sits above the SWOT water surface on the "
                    "working chain; no DEM involved"}
    tp_mid = float(np.median(tp))
    tide_mid = float(np.median(tide))
    resid = {}
    for k, v in cs.items():
        c = v["c_m"]
        resid[k] = {
            "documented_terms_only": {"explained_max_m": float(max(tide.max(), 0) + FRAME_M[1] + abs(gm)),
                                      "residual_min_m": float(c - (tide.max() + FRAME_M[1] + abs(gm)))},
            "tp_ellipsoid": {"tide_free": c - tp_mid, "mean_tide": c - tp_mid - tide_mid},
            "w0_iers2017": c - w0_iers, "w0_pre2015": c - w0_old, "w0_nga": c + NGA_HEIGHT_OFFSET,
            "sigma_m": v["sigma_m"]}
    for k in ("nga_realisation_m", "icgem_realisation_m"):
        resid[f"SWOT_{k}"] = {"tp_ellipsoid": swot[k] - tp_mid, "w0_nga": swot[k] + NGA_HEIGHT_OFFSET}
    rec = {"constants": cs, "terms": terms, "swot_check": swot, "residuals_m": resid,
           "reading": "documented terms (GM, tide, frame) explain at most about 0.05 m of c; DGFI's ellipsoid and "
                      "zero-degree handling are not documented in what was read"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datum_terms.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
