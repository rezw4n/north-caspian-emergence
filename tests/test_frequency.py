import numpy as np
import pytest

from caspian import frequency


def test_combine_sums_counts_not_frequencies():
    # odd half: 1 of 1 water; even half: 0 of 9 -> combined 1/10, not mean(1.0, 0.0) = 0.5
    counts = {"odd": (np.array([[1]]), np.array([[1]])), "even": (np.array([[0]]), np.array([[9]]))}
    w, v = frequency.combine(counts, ["odd", "even"])
    assert w[0, 0] == 1 and v[0, 0] == 10
    assert frequency.water_frequency(w, v)[0, 0] == pytest.approx(0.1)


def test_min_valid_rule():
    f = frequency.water_frequency(np.array([4, 4, 0]), np.array([4, 5, 0]), min_valid=5)
    assert np.isnan(f[0]) and f[1] == pytest.approx(0.8) and np.isnan(f[2])


def test_combine_rejects_inconsistent_counts():
    with pytest.raises(ValueError):
        frequency.combine({"g": (np.array([3]), np.array([2]))}, ["g"])
    with pytest.raises(ValueError):
        frequency.combine({}, [])
