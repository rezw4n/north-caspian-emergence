"""Minimal reader for classic netCDF-3 (CDF-1/CDF-2) files over HTTP range requests:
parse the header, then fetch only the needed rows of a 2-D variable. Used for the
GEBCO Type Identifier grid, which CEDA distributes as a 3.7 GB netCDF-3 file."""

from __future__ import annotations

import struct

import numpy as np
import requests

_TYPES = {1: ("i1", 1), 2: ("S1", 1), 3: (">i2", 2), 4: (">i4", 4), 5: (">f4", 4), 6: (">f8", 8)}


class _Buf:
    def __init__(self, b: bytes, version: int):
        self.b, self.p, self.v = b, 0, version

    def i4(self):
        x = struct.unpack(">i", self.b[self.p:self.p + 4])[0]
        self.p += 4
        return x

    def off(self):
        if self.v == 2:
            x = struct.unpack(">q", self.b[self.p:self.p + 8])[0]
            self.p += 8
            return x
        return self.i4()

    def name(self):
        n = self.i4()
        s = self.b[self.p:self.p + n].decode()
        self.p += n + (-n % 4)
        return s

    def values(self, nc_type, n):
        dt, size = _TYPES[nc_type]
        raw = self.b[self.p:self.p + n * size]
        self.p += n * size + (-(n * size) % 4)
        return raw.decode(errors="replace") if nc_type == 2 else np.frombuffer(raw, dt).tolist()

    def atts(self):
        tag, n = self.i4(), self.i4()
        out = {}
        for _ in range(n if tag else 0):
            name = self.name()
            t = self.i4()
            out[name] = self.values(t, self.i4())
        return out


def _get(url, start, stop):
    r = requests.get(url, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=600)
    r.raise_for_status()
    if r.status_code != 206 and start > 0:
        raise IOError("server ignored the Range header")
    return r.content


def header(url: str, nbytes: int = 1 << 16) -> dict:
    b = _get(url, 0, nbytes)
    if b[:3] != b"CDF":
        raise ValueError("not a classic netCDF file")
    buf = _Buf(b, b[3])
    buf.p = 4
    buf.i4()  # numrecs
    tag, ndims = buf.i4(), buf.i4()
    dims = [(buf.name(), buf.i4()) for _ in range(ndims if tag else 0)]
    gatts = buf.atts()
    tag, nvars = buf.i4(), buf.i4()
    variables = {}
    for _ in range(nvars if tag else 0):
        name = buf.name()
        ids = [buf.i4() for _ in range(buf.i4())]
        atts = buf.atts()
        t, vsize, begin = buf.i4(), buf.i4(), buf.off()
        variables[name] = {"dims": [dims[i] for i in ids], "type": t, "begin": begin, "atts": atts}
    return {"dims": dims, "attrs": gatts, "variables": variables}


def read_1d(url: str, hdr: dict, var: str) -> np.ndarray:
    v = hdr["variables"][var]
    dt, size = _TYPES[v["type"]]
    n = v["dims"][0][1]
    return np.frombuffer(_get(url, v["begin"], v["begin"] + n * size), dt)


def read_rows(url: str, hdr: dict, var: str, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    v = hdr["variables"][var]
    dt, size = _TYPES[v["type"]]
    ncols = v["dims"][1][1]
    start = v["begin"] + r0 * ncols * size
    block = np.frombuffer(_get(url, start, start + (r1 - r0) * ncols * size), dt)
    return block.reshape(r1 - r0, ncols)[:, c0:c1].copy()
