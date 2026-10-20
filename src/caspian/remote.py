"""Read a lat/lon window of a 2-D variable from a remote netCDF-4/HDF5 file by HTTP
range requests (fsspec + h5py): only the chunks that intersect the window are
transferred."""

from __future__ import annotations

import numpy as np


def h5_window(url: str, var: str, bounds, margin: float = 0.1, lat: str = "lat", lon: str = "lon"):
    import fsspec
    import h5py

    w, s, e, n = bounds
    with fsspec.open(url, "rb", block_size=4 * 2**20) as f, h5py.File(f, "r") as h:
        la = h[lat][:]
        lo = h[lon][:]
        i0, i1 = np.searchsorted(la, s - margin), np.searchsorted(la, n + margin)
        j0, j1 = np.searchsorted(lo, w - margin), np.searchsorted(lo, e + margin)
        data = h[var][i0:i1, j0:j1]
        attrs = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in h[var].attrs.items()
                 if k != "DIMENSION_LIST" and not k.startswith("_")}
        chunks = h[var].chunks
    return la[i0:i1], lo[j0:j1], data, attrs, chunks


def contiguous_window(url: str, offset: int, ncols: int, dtype, r0: int, r1: int, c0: int, c1: int,
                      workers: int = 8, tries: int = 4) -> np.ndarray:
    """Window [r0:r1, c0:c1] of a contiguous row-major 2-D array stored at byte
    ``offset`` of a remote file: one range request per row, fetched in parallel,
    transferring only the needed columns."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    dt = np.dtype(dtype)
    size = dt.itemsize
    session = requests.Session()

    def row(i: int) -> bytes:
        start = offset + (i * ncols + c0) * size
        stop = offset + (i * ncols + c1) * size - 1
        for k in range(tries):
            try:
                r = session.get(url, headers={"Range": f"bytes={start}-{stop}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == stop - start + 1:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 * (k + 1))
        raise IOError(f"row {i}: range request failed")

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(row, range(r0, r1)))
    return np.frombuffer(b"".join(rows), dt).reshape(r1 - r0, c1 - c0)
