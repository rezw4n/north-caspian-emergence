"""Robustness check 6 (brief section 10): sensitivity to each datum offset estimate.
No Earth Engine (docs/ROBUSTNESS_PLAN.md; docs/DATUMS.md).

Commands
  corrections  Write the two vertical-correction files of the total error (PREREG
               section 5a; D-074): data/interim/robustness/correction_dN_ngaEGM2008_m.npz
               (the main choice) and correction_dN_m.npz (without NGA's -0.41 m height
               offset), from results/study_area/geoid_egm2008_minus_eigen6c4.npz, with
               the mean and sd of their difference. If that sd is below the 0.01 m grid
               of the shape constant, the shape variant cannot change (it fits its own
               constant), and only the total variant is rerun.
  hindcast     The total variant with correction_dN_m.npz (no shape fit), compared with
               the main run (scripts/r00_pipeline.py hindcast_step).
  validation   The H2 and coverage headline with (a) the ICESat-2 -> DEM constant c at
               +-1 bootstrap sigma (robustness.run_fixed), (b) the product offsets at
               +-1 sigma, (c) the ICESat-2 heights in a mean-tide system by either recipe
               of D-073 (robustness.tide_system_shift; the constant part is absorbed by c,
               the north-south tilt remains). The DAHITI -> BHS-77 offset cancels in
               every metric (D-070) and is not varied.

Usage:
  python scripts/r06_datums.py corrections
  python scripts/r06_datums.py hindcast --main <tag>
  python scripts/r06_datums.py validation --main <tag> [--sources ATL06,ATL08]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import caspian  # noqa: F401
import numpy as np

from caspian import robustness as R, validation
from caspian.config import INTERIM, RESULTS

CHECK = "r06_datums"
GEOID = RESULTS / "study_area" / "geoid_egm2008_minus_eigen6c4.npz"
CORR = INTERIM / "robustness"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def corrections() -> int:
    recs = {layer: R.write_correction(GEOID, layer, CORR / f"correction_{layer}.npz")
            for layer in ("dN_ngaEGM2008_m", "dN_m")}
    with np.load(GEOID) as z:
        diff = z["dN_ngaEGM2008_m"].astype(float) - z["dN_m"].astype(float)
    recs["difference_nga_minus_icgem_m"] = {"mean": float(np.nanmean(diff)), "sd": float(np.nanstd(diff)),
                                            "min": float(np.nanmin(diff)), "max": float(np.nanmax(diff))}
    recs["shape_variant_unchanged"] = bool(np.nanstd(diff) < 0.01)
    out = RESULTS / "robustness" / CHECK
    out.mkdir(parents=True, exist_ok=True)
    (out / "corrections.json").write_text(json.dumps(recs, indent=1), encoding="utf-8")
    print(json.dumps(recs, indent=1))
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["corrections", "hindcast", "validation"])
    if argv and argv[0] == "corrections":
        return corrections()
    P = _load("r00_pipeline")
    P.add_common(ap)
    a = ap.parse_args(argv)
    if a.cmd == "hindcast":
        path = CORR / "correction_dN_m.npz"
        if not path.exists():
            raise SystemExit("run `python scripts/r06_datums.py corrections` first")
        P.hindcast_step(CHECK, "total_dN_m", a.main, stage2_tag=a.main, vertical_correction=str(path),
                        correction_is_variant=True, shape="none",
                        note="total error with dN_m (no NGA -0.41 m offset), D-074 sensitivity")
        return 0
    sources = a.sources.split(",")
    for n_sigma in (-1.0, 1.0):
        def run_c(p, pr, n_sigma=n_sigma):
            c = R.shifted_constant(validation.fit_datum_constant(p, a.n_boot), n_sigma)
            return R.run_fixed(p, pr, constant=c, n_boot=a.n_boot)

        def run_o(p, pr, n_sigma=n_sigma):
            c = validation.fit_datum_constant(p, a.n_boot)
            off = validation.fit_product_offsets(validation.apply_datum_constant(p, c), pr, a.n_boot)
            return R.run_fixed(p, pr, constant=c, product_offsets=R.shifted_offsets(off, n_sigma), n_boot=a.n_boot)
        P.compare_validation(CHECK, f"c_{n_sigma:+g}sigma", a.main, sources=sources, alt_run=run_c, n_boot=a.n_boot,
                             note="ICESat-2 -> DEM constant shifted")
        P.compare_validation(CHECK, f"offsets_{n_sigma:+g}sigma", a.main, sources=sources, alt_run=run_o,
                             n_boot=a.n_boot, note="product offsets shifted")
    for recipe in ("both_mean", "geoid_mean"):
        def tide(p, source, recipe=recipe):
            return p.assign(h_orth=p["h_orth"].to_numpy(float) + R.tide_system_shift(p["lat"], recipe))
        P.compare_validation(CHECK, f"tide_{recipe}", a.main, sources=sources, alt_points=tide, n_boot=a.n_boot,
                             note="ICESat-2 heights in a mean-tide system (D-073)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
