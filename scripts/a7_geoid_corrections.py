"""A7: the two vertical-correction rasters for the hindcast's total error (D-074), in
the input format of caspian.products.load_correction (npz: lon, lat, dN).

- geoid_correction_primary.npz: dN = N(EGM2008, NGA/PROJ realisation, with its
  -0.41 m HeightOffset) - N(EIGEN-6C4, ICGEM) (`dN_ngaEGM2008_m`), the primary total
  error;
- geoid_correction_no_constant.npz: dN = N(EGM2008, ICGEM) - N(EIGEN-6C4, ICGEM)
  (`dN_m`), the sensitivity without the constant.

Input: results/study_area/geoid_egm2008_minus_eigen6c4.npz (scripts/a7_geoid_difference.py).
Usage: python scripts/a7_geoid_corrections.py
"""

import sys

import caspian  # noqa: F401
import numpy as np

from caspian.config import RESULTS


def main() -> int:
    z = np.load(RESULTS / "study_area" / "geoid_egm2008_minus_eigen6c4.npz")
    for name, key in (("geoid_correction_primary.npz", "dN_ngaEGM2008_m"), ("geoid_correction_no_constant.npz", "dN_m")):
        out = RESULTS / "study_area" / name
        np.savez_compressed(out, lon=z["lon"], lat=z["lat"], dN=z[key].astype(np.float32), source_key=np.array(key))
        print(out.name, key, f"mean {np.nanmean(z[key]):+.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
