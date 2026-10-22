"""DEM validation diagnostic: where ETOPO 2022's heights come from at the test-block points
of an additional independent source, and the error of the DEM and of ETOPO 2022 by that
source.

ETOPO 2022 fills the emerged band from FABDEM (source ID 10: the Copernicus DEM with
forests and buildings removed) and GEBCO Lake Depths (ID 11) (MacFerrin et al. 2025,
Table 1). Against GLO-30, the FABDEM part of ETOPO 2022 is not an independent
comparison: both rest on the same TanDEM-X acquisitions. The table says how much of the
GLO-30 comparison that affects, and whether ETOPO 2022's advantage there is also seen
against GEDI, which is independent of both. Descriptive; nothing is refitted (the
training-block constant c and the matched ETOPO 2022 offset are read from the evaluate
outputs).

Output (committed): results/validation/<tag>_<source lower>/<SOURCE>_etopo2022_sources.json
Usage: python scripts/a6_etopo_sources.py <tag> <GLO30|GEDI>
"""

import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np
import xarray as xr

from caspian import validation
from caspian.config import RAW, RESULTS

ETOPO_SOURCES = {10: "FABDEM", 11: "GEBCO Lake Depths"}   # MacFerrin et al. 2025, Table 1 (as scripts/a7_level_bands.py)
ROOT = Path(__file__).resolve().parents[1]


def main(tag: str, source: str) -> int:
    spec = importlib.util.spec_from_file_location("a6_validate_dem", ROOT / "scripts" / "a6_validate_dem.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    res = RESULTS / "validation" / f"{tag}_{source.lower()}"
    size = float(json.loads((res / "block_size_check.json").read_text(encoding="utf-8"))["final_block_size_m"])
    pts = validation.attach_blocks(drv._load_points(tag, source, "bare"), drv._blocks(size), size)
    te = validation.split_rows(pts, "test")
    c = json.loads((res / f"{source}_datum_constant.json").read_text(encoding="utf-8"))["constant"]["constant_m"]
    off = json.loads((res / f"{source}_product_offsets.json").read_text(encoding="utf-8"))["product_offsets"]
    ds = xr.open_dataset(RAW / "bathymetry" / "etopo_2022_ncaspian.nc")
    sid = ds["source_id"].sel(lon=xr.DataArray(te["lon"].to_numpy(float)), lat=xr.DataArray(te["lat"].to_numpy(float)),
                              method="nearest").values.astype(float)
    y = te["h_orth"].to_numpy(float) + c
    e_dem = te["dem_h"].to_numpy(float) - y
    e_et = te["prod:etopo_2022"].to_numpy(float) + off["etopo_2022"]["offset_m"] - y
    ok = np.isfinite(e_dem) & np.isfinite(e_et)
    out = {"tag": tag, "source": source, "split": "test", "population": "bare (D-084)", "n_points": int(ok.sum()),
           "source_id_sampling": "nearest ETOPO 2022 cell (products are compared bilinearly)",
           "note": "descriptive; c and the matched offset from the evaluate outputs", "by_etopo_source": {}}
    for k in sorted(set(np.unique(sid[ok]).tolist())):
        m = ok & (sid == k)
        name = ETOPO_SOURCES.get(int(k), f"other ({int(k)})")
        out["by_etopo_source"][name] = {
            "share": round(float(m.sum() / ok.sum()), 4), "n_points": int(m.sum()),
            "n_blocks": int(np.unique(te["block"].to_numpy()[m]).size),
            "dem": validation.error_metrics(e_dem[m]), "etopo_2022_matched": validation.error_metrics(e_et[m])}
    (res / f"{source}_etopo2022_sources.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, r in out["by_etopo_source"].items():
        print(source, name, r["share"], "DEM nmad", round(r["dem"]["nmad"], 3), "ETOPO nmad",
              round(r["etopo_2022_matched"]["nmad"], 3), "DEM rmse", round(r["dem"]["rmse"], 3), "ETOPO rmse",
              round(r["etopo_2022_matched"]["rmse"], 3))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
