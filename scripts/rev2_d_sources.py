"""Second-review analysis D, depth sources (PREREG Amendment 19 item 14). Exploratory,
post-review; defaults in D-149. No download: the GEBCO_2026 and GEBCO_2022 type-identifier
(TID) layers are part of the CEDA range-request subsets already held
(scripts/a7_get_gebco.py; results/provenance/gebco_2026.json, gebco_2022.json), and the
Great Salt Lake subset holds its own TID layer (scripts/g1_lineage.py).

1. TID shares of GEBCO_2026 (GEBCO_2022 beside it):
   - domain: the 15" cells whose centre lies in caspian.geography.DOMAIN (cell counts);
   - emerged band: the 250 m equal-area cells of results/study_area/extents_v6_250m.npz
     with high == 1 and low == 0 (water at the 1995-96 high stand, not at the 2025-26 low
     stand), each taking the nearest 15" cell (as scripts/a7_level_bands.py), in all and
     per sector (sector polygons of caspian.geography; cells outside the domain apart).
2. Shared values:
   - GEBCO_2026 = GEBCO_2022 (stored int16 elevations, exact) in the domain and the band;
   - ETOPO 2022 = GEBCO_2022 (|difference| < 0.01 m, the rule of scripts/rev_g_etopo.py
     clamp) at ETOPO 2022 cells with source ID 11 (GEBCO Lake Depths, the cells inside
     ETOPO 2022's lake outline) in the domain and the band; source ID 10 (FABDEM) beside it.
3. Great Salt Lake: the TID values of the GEBCO_2026 subset of the basin.

Class names: the TID variable's flag_meanings attribute in the file. It lists 20 names for
21 flag values; codes 0-44 are matched in order from the start and 70-72 from the end, which
is unambiguous for the codes present here (0, 40, 44, 70).

Outputs: results/revision2/d/sources/{tid_shares.csv, identical_cells.json, summary.json}
Usage: python scripts/rev2_d_sources.py
"""

from __future__ import annotations

import json
import sys

import caspian  # noqa: F401
import numpy as np
import pandas as pd
import pyproj
import shapely
import xarray as xr

from caspian.config import CRS_EQUAL_AREA, RAW, RESULTS
from caspian.geography import DOMAIN, SECTOR_ORDER, sectors

OUT = RESULTS / "revision2" / "d" / "sources"
BATHY = RAW / "bathymetry"
GSL = RAW / "generality" / "great_salt_lake" / "bathymetry" / "gebco_2026.nc"
TOL_EQUAL_M = 0.01


def tid_names(attrs: dict) -> dict[int, str]:
    """Code -> name from the file's attributes. The codes below 70 take names in order from
    the start, those from 70 up in order from the end; codes left without a name when the
    file lists fewer names than codes get "unnamed in file"."""
    vals = [int(v) for v in str(attrs["flag_values"]).strip("[]").split(",")]
    names = str(attrs["flag_meanings"]).split()
    head = [v for v in vals if v < 70]
    tail = [v for v in vals if v >= 70]
    n_head = min(len(head), len(names) - len(tail))
    out = {v: "unnamed in file" for v in vals}
    out.update(zip(head[:n_head], names[:n_head]))
    out.update(zip(tail, names[len(names) - len(tail):]))
    return out


def shares(codes, names) -> dict:
    u, n = np.unique(np.asarray(codes), return_counts=True)
    tot = int(n.sum())
    return {"n_cells": tot, "shares": {f"{int(k)} {names.get(int(k), '?')}": round(float(c / tot), 6)
                                       for k, c in zip(u, n)}}


def main() -> int:
    g26 = xr.open_dataset(BATHY / "gebco_2026_ncaspian.nc")
    g22 = xr.open_dataset(BATHY / "gebco_2022_ncaspian.nc")
    et = xr.open_dataset(BATHY / "etopo_2022_ncaspian.nc")
    lon, lat = g26["lon"].values, g26["lat"].values
    for name, ds in (("gebco_2022", g22), ("etopo_2022", et)):
        if not (np.allclose(ds["lon"].values, lon, atol=1e-6) and np.allclose(ds["lat"].values, lat, atol=1e-6)):
            raise SystemExit(f"{name} is not on the GEBCO_2026 cell centres")
    names = tid_names(g26["tid"].attrs)
    T26 = g26["tid"].transpose("lat", "lon").values.astype(int)
    T22 = g22["tid"].transpose("lat", "lon").values.astype(int)
    Z26 = g26["elevation"].transpose("lat", "lon").values
    Z22 = g22["elevation"].transpose("lat", "lon").values
    ZE = et["elevation"].transpose("lat", "lon").values.astype(float)
    SE = et["source_id"].transpose("lat", "lon").values.astype(int)

    # 1. domain cells
    LON, LAT = np.meshgrid(lon, lat)
    dom = shapely.contains_xy(DOMAIN, LON, LAT)
    rows = []
    for which, T in (("gebco_2026", T26), ("gebco_2022", T22)):
        s = shares(T[dom], names)
        for k, v in s["shares"].items():
            rows.append({"grid": which, "region": "domain", "sector": "all", "tid": k, "share": v, "n_cells": s["n_cells"],
                         "unit": "15 arc-second cells"})

    # emerged band: 250 m cells -> nearest 15" cell
    ex = np.load(RESULTS / "study_area" / "extents_v6_250m.npz")
    band = (ex["high"] == 1) & (ex["low"] == 0)
    tr = ex["transform"]
    r, c = np.nonzero(band)
    xs, ys = tr[2] + (c + 0.5) * tr[0], tr[5] + (r + 0.5) * tr[4]
    blon, blat = pyproj.Transformer.from_crs(CRS_EQUAL_AREA, "EPSG:4326", always_xy=True).transform(xs, ys)
    gi = g26.indexes["lat"].get_indexer(blat, method="nearest")
    gj = g26.indexes["lon"].get_indexer(blon, method="nearest")
    if (gi < 0).any() or (gj < 0).any():
        raise SystemExit("band cells outside the GEBCO subset")
    sec = np.full(blon.size, "outside_domain", dtype=object)
    for k, (_, poly) in sectors().items():
        sec[shapely.contains_xy(poly, blon, blat)] = k
    for which, T in (("gebco_2026", T26), ("gebco_2022", T22)):
        for sname in ["all"] + SECTOR_ORDER + ["outside_domain"]:
            m = np.ones(blon.size, bool) if sname == "all" else (sec == sname)
            if not m.any():
                continue
            s = shares(T[gi[m], gj[m]], names)
            for k, v in s["shares"].items():
                rows.append({"grid": which, "region": "emerged_band", "sector": sname, "tid": k, "share": v,
                             "n_cells": s["n_cells"], "unit": "250 m equal-area cells (nearest 15 arc-second cell)"})
    tid = pd.DataFrame(rows)

    # 2. shared values
    eq26 = Z26 == Z22
    beq26 = eq26[gi, gj]
    lake, fab = SE == 11, SE == 10
    eqE = np.abs(ZE - Z22.astype(float)) < TOL_EQUAL_M
    bse = SE[gi, gj]
    beqE = eqE[gi, gj]
    by_tid_domain = {}
    for k in np.unique(T26[dom]):
        m = dom & (T26 == k)
        by_tid_domain[f"{int(k)} {names.get(int(k), '?')}"] = {"n_cells": int(m.sum()), "share_equal": round(float(eq26[m].mean()), 6)}
    ident = {
        "gebco2026_eq_gebco2022": {
            "domain": {"n_cells": int(dom.sum()), "share_equal": round(float(eq26[dom].mean()), 6),
                       "by_gebco2026_tid": by_tid_domain},
            "emerged_band": {"n_cells_250m": int(band.sum()), "share_equal": round(float(beq26.mean()), 6),
                             "by_sector": {s: round(float(beq26[sec == s].mean()), 6) for s in SECTOR_ORDER if (sec == s).any()}},
            "tid_identical_domain_share": round(float((T26[dom] == T22[dom]).mean()), 6),
            "rule": "stored int16 elevations equal (m)"},
        "etopo2022_eq_gebco2022": {
            "lake_depth_cells_domain": {"n_cells": int((dom & lake).sum()), "share_equal": round(float(eqE[dom & lake].mean()), 6),
                                        "share_etopo_lower": round(float((ZE < Z22 - TOL_EQUAL_M)[dom & lake].mean()), 6)},
            "fabdem_cells_domain": {"n_cells": int((dom & fab).sum()), "share_equal": round(float(eqE[dom & fab].mean()), 6)},
            "lake_depth_cells_emerged_band": {"n_cells_250m": int((bse == 11).sum()),
                                              "share_equal": round(float(beqE[bse == 11].mean()), 6)},
            "fabdem_cells_emerged_band": {"n_cells_250m": int((bse == 10).sum()),
                                          "share_equal": round(float(beqE[bse == 10].mean()), 6)},
            "all_domain_cells": {"n_cells": int(dom.sum()), "share_equal": round(float(eqE[dom].mean()), 6)},
            "etopo_source_share_domain": {str(int(k)): round(float(n / dom.sum()), 6) for k, n in
                                          zip(*np.unique(SE[dom], return_counts=True))},
            "rule": f"|ETOPO 2022 - GEBCO_2022| < {TOL_EQUAL_M} m at the same cell centre; source ID 11 = GEBCO "
                    "Lake Depths (inside ETOPO 2022's lake outline), 10 = FABDEM (MacFerrin et al. 2025, Table 1)"},
    }

    # 3. Great Salt Lake
    gs = xr.open_dataset(GSL)
    gt = gs["tid"].values.astype(int)
    gsl = shares(gt.ravel(), tid_names(gs["tid"].attrs) if "flag_values" in gs["tid"].attrs else names)
    lineage = json.loads((RESULTS / "generality" / "great_salt_lake" / "lineage.json").read_text(encoding="utf-8"))
    gsl_band = {arm: v.get("gebco_2026_tid_shares") for arm, v in lineage["arms"].items()}
    summary = {
        "rule": "A19 item 14 (D-149)",
        "tid_source": "TID layer of data/raw/bathymetry/gebco_2026_ncaspian.nc (CEDA subset, results/provenance/gebco_2026.json)",
        "tid_names_from_file": {str(k): v for k, v in sorted(names.items())},
        "domain_cells": int(dom.sum()), "band_cells_250m": int(band.sum()),
        "band_cells_outside_domain": int((sec == "outside_domain").sum()),
        "great_salt_lake": {"subset_cells": gsl["n_cells"], "subset_tid_shares": gsl["shares"],
                            "band_tid_shares_from_lineage_json": gsl_band,
                            "note": "every cell of the basin subset has TID 0, so the emerged-band shares are 100% TID 0"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tid.to_csv(OUT / "tid_shares.csv", index=False)
    (OUT / "identical_cells.json").write_text(json.dumps(ident, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(tid[(tid["grid"] == "gebco_2026") & tid["sector"].isin(["all"])].to_string(index=False))
    print(json.dumps(ident, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
