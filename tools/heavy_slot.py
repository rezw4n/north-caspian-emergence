"""Two shared slots for heavy local jobs (D-112), so parallel workstreams do not run
the machine out of memory.

Usage, from Python:
    from tools.heavy_slot import heavy_slot
    with heavy_slot("R: quantisation control"):
        run()

or wrapping a command:
    python tools/heavy_slot.py "R: quantisation control" -- python scripts/x.py args

A slot is the lock file data/interim/.heavy_slot_<k>, created exclusively and
holding the owner, PID and start time. A slot whose PID is no longer running is
stale and is taken over. A job also waits until at least MIN_FREE_GB of RAM is
available before it starts.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "data" / "interim"
N_SLOTS = 2
MIN_FREE_GB = 3.5
POLL_S = 60


def free_ram_gb() -> float:
    class MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    st = MemStatus()
    st.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / 2**30


def pid_alive(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def _try_take(path: Path, owner: str, pid: int) -> bool:
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
            if not pid_alive(int(held["pid"])):
                path.unlink()
                print(f"heavy_slot: removed stale {path.name} of {held.get('owner')}", flush=True)
        except (OSError, ValueError, KeyError):
            pass
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"owner": owner, "pid": pid, "start": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    return True


@contextlib.contextmanager
def heavy_slot(owner: str, pid: int | None = None):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    pid = pid or os.getpid()
    slot = None
    waited = False
    while slot is None:
        if free_ram_gb() >= MIN_FREE_GB:
            for k in range(1, N_SLOTS + 1):
                p = LOCK_DIR / f".heavy_slot_{k}"
                if _try_take(p, owner, pid):
                    slot = p
                    break
        if slot is None:
            if not waited:
                print(f"heavy_slot: waiting for a slot or {MIN_FREE_GB} GB free RAM ({owner})", flush=True)
                waited = True
            time.sleep(POLL_S)
    print(f"heavy_slot: {owner} holds {slot.name}", flush=True)
    try:
        yield slot
    finally:
        with contextlib.suppress(FileNotFoundError):
            slot.unlink()


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") != 1:
        print(__doc__)
        return 2
    owner, cmd = argv[0], argv[2:]
    with heavy_slot(owner):
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
