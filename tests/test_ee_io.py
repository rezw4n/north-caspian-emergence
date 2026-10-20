import numpy as np

from caspian import ee_io


def test_tiles_cover_raster_exactly():
    dims = (2500, 1100)
    cover = np.zeros((dims[1], dims[0]), int)
    for c0, r0, w, h in ee_io.tiles(dims, 1024):
        cover[r0:r0 + h, c0:c0 + w] += 1
    assert (cover == 1).all()


def test_grid_dictionary():
    g = ee_io.grid([60, 0, 1000, 0, -60, 5000], (10, 20), "WKT")
    assert g["crsWkt"] == "WKT"
    assert g["dimensions"] == {"width": 10, "height": 20}
    assert g["affineTransform"]["scaleY"] == -60 and g["affineTransform"]["translateX"] == 1000


def test_grid_crs_code_or_wkt():
    assert ee_io.grid([1, 0, 0, 0, -1, 0], (1, 1), None)["crsCode"] == "EPSG:4326"
    assert ee_io.grid([1, 0, 0, 0, -1, 0], (1, 1), "EPSG:32639")["crsCode"] == "EPSG:32639"
    assert "crsWkt" in ee_io.grid([1, 0, 0, 0, -1, 0], (1, 1), 'PROJCS["x"]')
