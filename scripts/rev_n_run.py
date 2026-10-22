"""Run the rev_n revision steps in order, each as its own process, stopping at the
first failure (A18 item 10). Meant to be run as one job so that dependent steps run in
turn instead of queueing separately.

Steps: curves (rev_n_curves.py), envelopes (rev_n_envelopes.py, reads curves.csv),
pathways (rev_n_pathways.py, reads curves_fine.csv.gz), exposure (rev_n_exposure.py),
simulate (rev_n_simulate.py, reads curves.csv for the 60 m values).

Usage: python scripts/rev_n_run.py curves,envelopes,pathways
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {"curves": "rev_n_curves.py", "envelopes": "rev_n_envelopes.py", "pathways": "rev_n_pathways.py",
         "exposure": "rev_n_exposure.py", "simulate": "rev_n_simulate.py"}


def main(argv) -> int:
    names = (argv[0] if argv else "curves,envelopes,pathways").split(",")
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}; choose from {list(STEPS)}")
    for n in names:
        t0 = time.time()
        print(time.strftime("%H:%M:%S"), "step", n, "start", flush=True)
        r = subprocess.run([sys.executable, str(HERE / STEPS[n])], cwd=HERE.parent)
        print(time.strftime("%H:%M:%S"), "step", n, "exit", r.returncode, round(time.time() - t0, 1), "s", flush=True)
        if r.returncode:
            return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
