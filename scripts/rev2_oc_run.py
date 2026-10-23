"""Driver of the Amendment 19 item 5 and item 9 runs (D-148), meant to run inside one heavy slot
(tools/heavy_slot.py), so the steps do not each queue for a slot:

1. item 9: all five grids (scripts/rev2_oc_item9.py run --workers 2), then summarise;
2. item 5: the sigma = 0 check, then the timed pilot realisation;
3. waits until data/interim/revision2/oc/go_item5.json exists ({"R": .., "workers": ..}), which is
   written only after the run-time decision is logged in DECISIONS (D-148 addendum);
4. item 5: the run with that R, then summarise.

Usage: python scripts/rev2_oc_run.py [--skip-item9]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev2_oc_common as C  # noqa: E402

GO = C.WORK / "go_item5.json"


def step(*args) -> None:
    cmd = [sys.executable, *map(str, args)]
    C.log("step:", " ".join(cmd[1:]))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"step failed ({rc}): {' '.join(cmd[1:])}")


def main(argv) -> int:
    if "--skip-item9" not in argv:
        step(HERE / "rev2_oc_item9.py", "run", "--workers", "2")
        step(HERE / "rev2_oc_item9.py", "summarise")
    step(HERE / "rev2_oc_item5.py", "check")
    step(HERE / "rev2_oc_item5.py", "pilot")
    C.log(f"waiting for {GO}")
    while not GO.exists():
        time.sleep(20)
    go = json.loads(GO.read_text(encoding="utf-8"))
    C.log("go:", go)
    step(HERE / "rev2_oc_item5.py", "run", "--R", go["R"], "--workers", go["workers"])
    step(HERE / "rev2_oc_item5.py", "summarise")
    C.log("all steps done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
