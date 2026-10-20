"""Wait for Earth Engine batch tasks to finish; print state and EECU-seconds.
Usage: python tools/ee_wait.py TASK_ID [TASK_ID ...]"""

import sys
import time

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[1].joinpath("src").as_posix())
import caspian  # noqa: E402,F401
import ee  # noqa: E402

from caspian.config import EE_PROJECT  # noqa: E402

ee.Initialize(project=EE_PROJECT)
ids = sys.argv[1:]
while True:
    ops = {i: ee.data.getOperation(f"projects/{EE_PROJECT}/operations/{i}") for i in ids}
    states = {i: o["metadata"].get("state") for i, o in ops.items()}
    if all(s in ("SUCCEEDED", "FAILED", "CANCELLED") for s in states.values()):
        break
    time.sleep(60)
for i, o in ops.items():
    md = o["metadata"]
    print(i, md.get("description"), md.get("state"), "EECU-s:", md.get("batchEecuUsageSeconds"),
          o.get("error", {}).get("message", ""))
