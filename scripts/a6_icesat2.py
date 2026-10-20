"""A6 validation input: ICESat-2 ATL06 and ATL08 heights over the stage-2 cover
(PREREG section 2; brief section 8).

NASA Earthdata (CMR search, HTTPS range reads through earthaccess and h5py). The
granules over the domain total about 515 GB (2,003 ATL06 + 1,990 ATL08, 2018-10 to
2026-10), so each granule is read in place:
- each beam's latitude array is read in full;
- only the slice inside the domain's latitude band is read for the other variables;
- points outside the buffered stage-2 cover (results/study_area/
  zone_v6_cover_buffered.geojson) are dropped.

Heights are ellipsoidal (WGS84, ITRF2014). The ATL06 file also carries the geoid
height (dem/geoid_h, EGM2008). Fill values (> 1e30) become NaN. The time is the ATLAS
SDP epoch 2018-01-01T00:00:00 plus delta_time; leap seconds are ignored, which
matters only at the second level.

Outputs (git-ignored): data/interim/icesat2/<product>/<granule>.csv.gz
Provenance (committed): results/provenance/icesat2_<product>.json

Usage: python scripts/a6_icesat2.py ATL06|ATL08 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import pandas as pd

from caspian.config import INTERIM, RESULTS
from caspian.geography import DOMAIN

BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
EPOCH = np.datetime64("2018-01-01T00:00:00")
VARS = {
    "ATL06": ("land_ice_segments", {"lat": "latitude", "lon": "longitude", "dt": "delta_time", "h": "h_li",
                                     "h_sigma": "h_li_sigma", "quality": "atl06_quality_summary",
                                     "geoid_h": "dem/geoid_h", "dem_h": "dem/dem_h",
                                     "n_photons": "fit_statistics/n_fit_photons",
                                     "snr_sig": "fit_statistics/snr_significance"}),
    "ATL08": ("land_segments", {"lat": "latitude", "lon": "longitude", "dt": "delta_time",
                                 "h": "terrain/h_te_best_fit", "h_sigma": "terrain/h_te_uncertainty",
                                 "h_std": "terrain/h_te_std", "n_photons": "terrain/n_te_photons",
                                 "watermask": "segment_watermask", "landcover": "segment_landcover",
                                 "cloud": "cloud_flag_atm", "msw": "msw_flag", "night": "night_flag",
                                 "dem_h": "dem_h", "terrain_flg": "terrain_flg"}),
}
_COVER = None


def _init():
    global _COVER
    import json as _j

    import shapely
    from shapely.geometry import shape

    from caspian import earthdata
    earthdata.login()
    g = _j.loads((RESULTS / "study_area" / "zone_v6_cover_buffered.geojson").read_text(encoding="utf-8"))
    _COVER = shape(g["features"][0]["geometry"])
    shapely.prepare(_COVER)


def read_granule(granule, product: str) -> tuple[str, int]:
    import earthaccess
    import h5py
    from shapely import contains_xy

    grp, var = VARS[product]
    w, s, e, n = DOMAIN.bounds
    name = granule["umm"]["GranuleUR"]
    out = INTERIM / "icesat2" / product / f"{name}.csv.gz"
    frames = []
    f = earthaccess.open([granule])[0]
    with h5py.File(f, "r") as h:
        for gt in BEAMS:
            p = f"{gt}/{grp}"
            if p not in h:
                continue
            lat = h[f"{p}/latitude"][:]
            sel = np.flatnonzero((lat >= s) & (lat <= n))
            if sel.size == 0:
                continue
            a, b = int(sel.min()), int(sel.max()) + 1
            cols = {}
            for k, v in var.items():
                if f"{p}/{v}" in h:
                    x = h[f"{p}/{v}"][a:b]
                    if x.ndim > 1:      # e.g. per-segment arrays of 5 sub-segments: keep the first
                        x = x[:, 0]
                    cols[k] = x
            d = pd.DataFrame(cols)
            inside = (d["lon"] >= w) & (d["lon"] <= e)
            d = d[inside]
            if d.empty:
                continue
            d = d[contains_xy(_COVER, d["lon"].to_numpy(), d["lat"].to_numpy())]
            if d.empty:
                continue
            for c in ("h", "h_sigma", "h_std", "geoid_h", "dem_h"):
                if c in d:
                    d.loc[np.abs(d[c]) > 1e30, c] = np.nan
            d["time"] = EPOCH + (d["dt"].to_numpy() * 1e9).astype("timedelta64[ns]")
            d["beam"] = gt
            frames.append(d.drop(columns=["dt"]))
        # sc_orient 0 (backward): left beams strong; 1 (forward): right beams strong
        sc = h["orbit_info/sc_orient"][0] if "orbit_info/sc_orient" in h else -1
    f.close()
    if frames:
        d = pd.concat(frames, ignore_index=True)
        d["sc_orient"] = int(sc)
        d["granule"] = name
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, index=False, float_format="%.9g")   # 9 digits: lat/lon to about 0.1 m
        return name, len(d)
    return name, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("product", choices=list(VARS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import earthaccess

    from caspian import earthdata
    earthdata.login()
    w, s, e, n = DOMAIN.bounds
    res = earthaccess.search_data(short_name=a.product, bounding_box=(w, s, e, n),
                                  temporal=("2018-10-01", "2026-12-31"), count=-1)
    prov_path = RESULTS / "provenance" / f"icesat2_{a.product.lower()}.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else {"granules": {}}
    todo = [r for r in res if r["umm"]["GranuleUR"] not in prov["granules"]]
    print(f"{a.product}: {len(res)} granules, {len(todo)} to read", flush=True)
    t0, k = time.time(), 0
    with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        futs = {pool.submit(read_granule, r, a.product): r["umm"]["GranuleUR"] for r in todo}
        for fut in as_completed(futs):
            g = futs[fut]
            try:
                name, n_pts = fut.result()
                prov["granules"][name] = n_pts
            except Exception as ex:  # leave the granule for a rerun
                print("error", g, str(ex)[:160], flush=True)
                continue
            k += 1
            if k % 25 == 0:
                prov_path.write_text(json.dumps(prov, indent=0), encoding="utf-8")
                print(f"{k}/{len(todo)} granules, {k / (time.time() - t0) * 60:.1f}/min", flush=True)
    prov.update(product=a.product, search_bbox=[w, s, e, n], temporal=["2018-10-01", "2026-12-31"],
                n_granules_found=len(res))
    prov_path.write_text(json.dumps(prov, indent=0), encoding="utf-8")
    print("done", k, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
