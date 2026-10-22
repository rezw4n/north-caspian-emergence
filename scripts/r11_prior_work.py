"""Robustness check 11: consistency with published Caspian results.
No Earth Engine.

Quantitative comparison where a published value is directly comparable and recorded
with its verbatim quote in refs/literature_values.csv:
- Akbari et al. (2020): Northern Caspian flooded area against
  sea level, 7,103 km2 per m for -29 < SWL < -26 m (Table 1;
  lit.akbari2020.nc_area_per_m_km2, range lit.akbari2020.fit_swl_lo_m / _hi_m). Our
  slope: observed sea area against level over the same range of the main run's units
  (robustness.area_level_slope on results/hindcast/<main>/observed.csv, domain-wide row
  "all", full samples). A constant offset between DAHITI and BHS-77 does not change a
  slope. Their region is their own "Northern Caspian" and their area the flooded part
  of the 1977-1995 band, so a difference can come from the
  regions as well as from the methods; both are stated.
Every other prior shoreline result enters the paper qualitatively until its numbers are
recorded in refs/literature_values.csv with quotes.

Usage:
  python scripts/r11_prior_work.py --main <tag>
"""

from __future__ import annotations

import argparse
import json
import sys

import caspian  # noqa: F401
import pandas as pd

from caspian import robustness as R
from caspian.config import ROOT, RESULTS

CHECK = "r11_prior_work"


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", required=True)
    a = ap.parse_args(argv)
    lit = pd.read_csv(ROOT / "refs" / "literature_values.csv").set_index("key")["value"]
    lo, hi = float(lit["lit.akbari2020.fit_swl_lo_m"]), float(lit["lit.akbari2020.fit_swl_hi_m"])
    obs = pd.read_csv(RESULTS / "hindcast" / a.main / "observed.csv")
    ours = R.area_level_slope(obs, lo=lo, hi=hi)
    rec = {"main": a.main, "published": {"key": "lit.akbari2020.nc_area_per_m_km2",
                                         "value_km2_per_m": float(lit["lit.akbari2020.nc_area_per_m_km2"]),
                                         "range_m": [lo, hi]},
           "ours": ours,
           "caveats": ["different regions (their Northern Caspian, our domain)",
                       "their area is the flooded part of the 1977-1995 band; ours the sea-connected observed water",
                       "levels: theirs BHS (instrumental), ours DAHITI; a constant offset does not change a slope"]}
    out = RESULTS / "robustness" / CHECK
    out.mkdir(parents=True, exist_ok=True)
    (out / "akbari2020_area_slope.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
