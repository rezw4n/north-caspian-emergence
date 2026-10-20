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


def test_stage2_combine_sums_counts(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("comb", "scripts/a6_stage2_combine.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    tr = np.array([60.0, 0, 0, 0, -60.0, 0])
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez_compressed(a, transform=tr, w_E3_b010_p0=np.array([[1, 0]], np.uint16), v_E3_b010_p0=np.array([[2, 1]], np.uint16))
    np.savez_compressed(b, transform=tr, w_E3_b010_p0=np.array([[3, 1]], np.uint16), v_E3_b010_p0=np.array([[3, 2]], np.uint16),
                        w_E3_b011_p1=np.array([[1, 1]], np.uint16), v_E3_b011_p1=np.array([[1, 1]], np.uint16))
    out = tmp_path / "c.npz"
    assert m.combine_tile([a, b], out) == 4
    z = np.load(out)
    assert z["w_E3_b010_p0"].tolist() == [[4, 1]] and z["v_E3_b010_p0"].tolist() == [[5, 3]]
    assert z["w_E3_b011_p1"].tolist() == [[1, 1]]          # missing in one source counts as zero there
    bad = tmp_path / "d.npz"
    np.savez_compressed(bad, transform=tr * 2, w_E3_b010_p0=np.array([[1, 0]], np.uint16))
    with pytest.raises(ValueError):
        m.combine_tile([a, bad], tmp_path / "e.npz")


def test_vegetation_share():
    from caspian import frequency
    o = np.array([10, 10, 3, 0])
    v = np.array([10, 4, 1, 0])
    s = frequency.vegetation_share(o, v, min_valid=5)
    assert s[0] == 0 and s[1] == pytest.approx(0.6) and np.isnan(s[2]) and np.isnan(s[3])
    with pytest.raises(ValueError):
        frequency.vegetation_share(np.array([1]), np.array([2]))
