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
