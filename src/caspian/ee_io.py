"""Fetch Earth Engine image assets as NumPy arrays on a fixed equal-area grid.

Uses ee.data.computePixels in tiles (each request stays well under the 48 MB
response limit), so assets of any size are downloaded block by block without
Google Drive. Reading a stored asset costs little compute.
"""

from __future__ import annotations

import numpy as np


def grid(transform, dims, crs):
    """computePixels grid dictionary for an affine ``transform`` [a, b, c, d, e, f]
    (x = a*col + b*row + c, y = d*col + e*row + f) and ``dims`` (width, height).
    ``crs`` is an EPSG code ("EPSG:4326"; None means EPSG:4326) or a WKT1 string."""
    a, b, c, d, e, f = transform
    g = {"dimensions": {"width": int(dims[0]), "height": int(dims[1])},
         "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                             "shearY": d, "scaleY": e, "translateY": f}}
    if crs is None or str(crs).upper().startswith("EPSG:"):
        g["crsCode"] = crs or "EPSG:4326"
    else:
        g["crsWkt"] = crs
    return g


def tiles(dims, block: int):
    """(col0, row0, width, height) blocks covering a (width, height) raster."""
    w, h = dims
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            yield c0, r0, min(block, w - c0), min(block, h - r0)


MAX_REQUEST_BYTES = 40 * 2**20     # computePixels refuses responses over 48 MiB
MAX_REQUEST_BANDS = 250            # under 1,024 bands, and the client np.load header limit (10 kB; ~30 B per band name)


def band_chunks(bands, width: int, height: int, bytes_per_px: int,
                max_bytes: int = MAX_REQUEST_BYTES, max_bands: int = MAX_REQUEST_BANDS):
    """Split ``bands`` into consecutive groups whose width x height x bytes_per_px
    response stays within max_bytes and max_bands."""
    per_band = max(1, width * height * bytes_per_px)
    n = max(1, min(max_bands, max_bytes // per_band))
    bands = list(bands)
    return [bands[k:k + n] for k in range(0, len(bands), n)]


def fetch(image, bands, transform, dims, crs, block: int = 1024, dtype=None, bytes_per_px: int = 8) -> dict:
    """Download ``bands`` of an ee.Image on the given grid. Returns {band: 2-D array}
    with rows from north to south (row 0 at translateY). Masked pixels come back as 0
    unless the image was unmasked to a sentinel before the call. Each spatial block is
    requested in band groups (band_chunks) sized for ``bytes_per_px`` per band."""
    import ee

    a, b, c, d, e, f = transform
    out = {}
    img = ee.Image(image)
    for c0, r0, w, h in tiles(dims, block):
        sub = [a, b, c + a * c0 + b * r0, d, e, f + d * c0 + e * r0]
        # Earth Engine counts one mask byte per pixel and band on top of the value
        # (measured: 62,750,160 bytes for 120 uint16 bands of 417 x 417 pixels)
        for chunk in band_chunks(bands, w, h, bytes_per_px + 1):
            arr = ee.data.computePixels({"expression": img.select(chunk), "fileFormat": "NUMPY_NDARRAY",
                                         "grid": grid(sub, (w, h), crs)})
            for band in chunk:
                if band not in out:
                    out[band] = np.zeros((dims[1], dims[0]), dtype=dtype or arr[band].dtype)
                out[band][r0:r0 + h, c0:c0 + w] = arr[band]
    return out
