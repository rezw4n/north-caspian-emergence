"""Pin every top-level dependency in environment.yml to the version installed in
the active environment (read from <prefix>/conda-meta/*.json)."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
installed = {}
for f in (Path(sys.prefix) / "conda-meta").glob("*.json"):
    meta = json.loads(f.read_text(encoding="utf-8"))
    installed[meta["name"]] = meta["version"]

env_path = ROOT / "environment.yml"
lines = env_path.read_text(encoding="utf-8").splitlines()
out, in_deps = [], False
for line in lines:
    if line.strip() == "dependencies:":
        in_deps = True
    m = re.match(r"^(\s*-\s+)([A-Za-z0-9_.\-]+)\s*(=.*)?$", line)
    if in_deps and m and m.group(2) in installed:
        line = f"{m.group(1)}{m.group(2)}={installed[m.group(2)]}"
    elif in_deps and m and m.group(2) not in installed:
        sys.exit(f"{m.group(2)} is listed in environment.yml but not installed")
    out.append(line)
env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
pinned = sum(1 for line in out if re.match(r"^\s*-\s+\S+=", line))
print(f"pinned {pinned} dependencies")
