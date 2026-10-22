"""Stamp files for the `make all` steps whose outputs are many git-ignored files.

A step such as the waterline DEM writes dozens of tiles under data/ and no single file
that could stand for all of them, so its make target is a stamp file. The stamp is
written only after the step's recipe succeeded, and only if every output folder it names
holds at least one matching file (a step that wrote nothing leaves no stamp). It lists
each file's name and size, so a later reader can see what the step produced.

  python tools/make_stamp.py STAMP DIR [DIR ...] [--pattern GLOB]

Some steps reuse outputs that already exist (scripts/g1_hindcast.py writes each masked
stage-2 tile once; scripts/r19_local_runs.py keeps some variant counts and DEMs). Before
such a step, its derived folders are removed, so that a rebuild recomputes them from its
current inputs:

  python tools/make_stamp.py --clean DIR [DIR ...]

--clean removes a folder only if its path lies under data/interim or data/processed of
this repository (derived data; never data/raw or results/), the folder itself is not a
link, and nothing inside it is a link or junction (so the removal never reaches outside
the folder). The path is checked as written, not resolved: in a fresh verification clone,
data/interim/<name> can be a junction to the working copy's folder, and the folders a
step writes are reached through it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEANABLE = tuple(os.path.normcase(os.path.abspath(ROOT / "data" / d)) for d in ("interim", "processed"))


def is_link(p: Path) -> bool:
    """A symlink, or (Windows) any reparse point such as a directory junction."""
    try:
        st = os.lstat(p)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & 0x400)


def stamp(path: Path, dirs: list[Path], pattern: str) -> int:
    lines = []
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"make_stamp: {d} is not a folder (the step did not write it)")
        files = sorted(p for p in d.glob(pattern) if p.is_file())
        if not files:
            raise SystemExit(f"make_stamp: no file matching {pattern} in {d} (the step wrote nothing)")
        lines += [f"{p.relative_to(d).as_posix()}\t{p.stat().st_size}" for p in files]
        print(f"{d}: {len(files)} files", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0


def clean(dirs: list[Path]) -> int:
    for d in dirs:
        a = os.path.normcase(os.path.abspath(d))
        if not any(a.startswith(c + os.sep) for c in CLEANABLE):
            raise SystemExit(f"make_stamp --clean: {d} is not under data/interim or data/processed; not removed")
        if not os.path.lexists(a):
            continue
        if is_link(Path(a)) or not os.path.isdir(a):
            raise SystemExit(f"make_stamp --clean: {d} is a link or not a folder; not removed")
        inner = [os.path.join(r, n) for r, ds, fs in os.walk(a, followlinks=False) for n in ds + fs
                 if is_link(Path(r) / n)]
        if inner:
            raise SystemExit(f"make_stamp --clean: {d} holds links ({inner[:2]}); not removed")
        shutil.rmtree(a)
        print(f"removed {d}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean", action="store_true", help="remove the named derived folders (see above)")
    ap.add_argument("--pattern", default="*", help="files counted in each folder (default every file)")
    ap.add_argument("paths", nargs="+", type=Path)
    a = ap.parse_args(argv)
    if a.clean:
        return clean(a.paths)
    if len(a.paths) < 2:
        ap.error("STAMP and at least one DIR")
    return stamp(a.paths[0], a.paths[1:], a.pattern)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
