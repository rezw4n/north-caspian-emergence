"""Sub-pixel waterline extraction and contour-to-grid interpolation (synthetic)."""

import numpy as np
import pytest

from caspian import dem, waterlines

# North-up 1 m grid with top-left corner at (0, 100): x = col + 0.5, y = 100 - (row + 0.5) at centres.
TRANSFORM = (1.0, 0.0, 0.0, 0.0, -1.0, 100.0)


def test_pixel_to_map_centres():
    x, y = waterlines.pixel_to_map(np.array([0]), np.array([0]), TRANSFORM)
    assert (x[0], y[0]) == (0.5, 99.5)


def test_straight_shoreline_subpixel_position():
    # Frequency ramps linearly from 1 (west) to 0 (east): f = 1 - (col - 10)/10 for col in [10, 20].
    cols = np.arange(40)
    f = np.clip(1 - (cols - 10) / 10, 0, 1)
    freq = np.tile(f, (30, 1))
    lines = waterlines.extract(freq, TRANSFORM)
    assert len(lines) == 1
    xs = np.array(lines[0].coords)[:, 0]
    assert np.allclose(xs, 15.5, atol=1e-9)  # f = 0.5 at col 15 -> x = 15.5


def test_invalid_pixels_break_contours():
    cols = np.arange(40)
    freq = np.tile(np.clip(1 - (cols - 10) / 10, 0, 1), (30, 1))
    valid = np.ones_like(freq, bool)
    valid[14:16, :] = False
    lines = waterlines.extract(freq, TRANSFORM, valid=valid)
    assert len(lines) == 2


def test_tin_reproduces_plane_and_distance():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, (400, 2))
    z = 0.01 * pts[:, 0] - 0.02 * pts[:, 1] + 3
    gx, gy = dem.grid_centers(10, 90, 80, 80, 1.0)
    zi = dem.tin(pts, z, gx, gy)
    ok = np.isfinite(zi)
    assert ok.mean() > 0.95
    assert np.allclose(zi[ok], 0.01 * gx[ok] - 0.02 * gy[ok] + 3, atol=1e-9)
    d = dem.interpolation_distance(pts, gx, gy)
    assert d.min() >= 0 and d.max() < 15


def test_contour_points_spacing():
    from shapely.geometry import LineString

    pts = waterlines.contour_points([LineString([(0, 0), (10, 0)])], spacing=1.0)
    assert len(pts) == 11 and pts[-1, 0] == pytest.approx(10)
