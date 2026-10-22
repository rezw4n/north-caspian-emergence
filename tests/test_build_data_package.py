"""Release rules of the data package (tools/build_data_package.py)."""
import importlib.util
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_data_package", ROOT / "tools" / "build_data_package.py")
bdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bdp)

POSTS = bdp.post_regex(["BAKU", "MAKH", "JILOY (Chilov)", "F_KENAR"])


def test_metric_exclusion_keeps_basin_wide_offset_and_drops_single_posts():
    keep = {"datum.bhs.delta_m": {"source": "results/levels/dahiti_bhs_offset.json",
                                  "note": "mean over the CASPCOM century posts of the per-post mean annual difference"},
            "datum.bhs.between_post_sd_m": {"source": "results/levels/dahiti_bhs_offset.json"},
            "h1.west.d_km": {"source": "results/hindcast/main/displacement.csv"}}
    drop = {"rev.v.posts.baku.offset_m": {"source": "results/revision/v/caspcom_posts.json"},
            "datum.bhs.post_max_m": {"source": "results/levels/dahiti_bhs_offset.json"},
            "x.makh.value": {"source": "elsewhere.json"},
            "h3.exposure.wdpa.n_sites": {"source": "results/emergence/wdpa/layers.json"},
            "erl.text.h3.ramsar.1.max": {"source": "results/metrics.json"},
            "h3.other": {"source": "emergence/wdpa/exposure_main.csv"}}
    for k, v in keep.items():
        assert bdp.metric_exclusion(k, v, POSTS) is None, k
    for k, v in drop.items():
        assert bdp.metric_exclusion(k, v, POSTS) is not None, k


def test_content_exclusion():
    assert bdp.content_exclusion("results/levels/dahiti_bhs_offset.json", "{}", POSTS)
    assert bdp.content_exclusion("a.csv", "layer\nwdpa_protected_areas\n", POSTS) == bdp.REASON_WDPA
    assert bdp.content_exclusion("a.csv", "post,offset\nBAKU,-0.4\n", POSTS) == bdp.REASON_GAUGE
    assert bdp.content_exclusion("a.csv", "sector,median_m\nwest,640\n", POSTS) is None
    # post codes match as upper-case words only
    assert POSTS.search("makh") is None and POSTS.search("BAKUX") is None
    assert re.search(POSTS, "F_KENAR") is not None


def test_zip_is_byte_reproducible(tmp_path):
    members = [("results/b.csv", b"x\n1\n"), ("results/a.json", b"{}")]
    bdp.write_zip(tmp_path / "1.zip", members)
    bdp.write_zip(tmp_path / "2.zip", list(reversed(members)))
    assert (tmp_path / "1.zip").read_bytes() == (tmp_path / "2.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "1.zip") as z:
        assert z.namelist() == ["results/a.json", "results/b.csv"]


def test_licence_text_is_cc_by_4():
    t = bdp.LICENCE_SRC.read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0 International Public License" in t
    assert bdp.sha256_bytes(bdp.LICENCE_SRC.read_bytes().replace(b"\r\n", b"\n")) == bdp.LICENCE_SHA256
