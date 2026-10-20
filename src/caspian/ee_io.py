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


def fetch(image, bands, transform, dims, crs, block: int = 1024, dtype=None) -> dict:
    """Download ``bands`` of an ee.Image on the given grid. Returns {band: 2-D array}
    with rows from north to south (row 0 at translateY). Masked pixels come back as 0
    unless the image was unmasked to a sentinel before the call."""
    import ee

    a, b, c, d, e, f = transform
    out = {}
    img = ee.Image(image).select(list(bands))
    for c0, r0, w, h in tiles(dims, block):
        sub = [a, b, c + a * c0 + b * r0, d, e, f + d * c0 + e * r0]
        arr = ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY",
                                     "grid": grid(sub, (w, h), crs)})
        for band in bands:
            if band not in out:
                out[band] = np.zeros((dims[1], dims[0]), dtype=dtype or arr[band].dtype)
            out[band][r0:r0 + h, c0:c0 + w] = arr[band]
    return out
