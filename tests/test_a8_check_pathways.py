"""scripts/a8_check_pathways.py on a synthetic reading record (no real note is read),
and the relation assembly of scripts/a8_osm_exposure.py on a synthetic Overpass
element."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


chk = _load("a8_check_pathways")


def test_numbers_and_values_in_quotes():
    assert chk.numbers_in("−1.8 m (−6.7 to 3.1) by 2100") == [1.8, 6.7, 3.1, 2100.0]
    assert chk.value_in_quote(-8.0, "a drop of about 8 (inter-model range from 2 to 15) m")
    assert chk.value_in_quote(-0.057, "which corresponds to 5.7 cm/year")
    assert not chk.value_in_quote(-9.0, "about 8 m")


def test_check_rows_against_a_note_abstract(tmp_path):
    notes = tmp_path / "refs" / "notes"
    notes.mkdir(parents=True)
    (notes / "SOURCES.csv").write_text("key,path,sha256,url,retrieved,kind\n", encoding="utf-8")
    (notes / "foo2030.md").write_text("# foo2030\n\n## Abstract (verbatim as retrieved; source: test)\n\n"
                                      "> The level falls by about 8 m by 2100 relative to 2020.\n", encoding="utf-8")
    df = pd.DataFrame({"pathway_id": ["a", "b", "c"], "source": ["foo2030", "foo2030", "bar"],
                       "doi": [None, None, None], "value_m": [-8.0, -9.0, -1.0], "location": ["Abstract"] * 3,
                       "verbatim_quote": ["falls by about 8 m by 2100", "falls by about 8 m by 2100",
                                          "anything 1 m"]})
    r = chk.check_rows(df, tmp_path).set_index("pathway_id")
    assert r.loc["a", "quote_found"] and r.loc["a", "value_in_quote"]
    assert r.loc["b", "quote_found"] and not r.loc["b", "value_in_quote"]
    assert not r.loc["c", "quote_found"]


def test_osm_relation_polygon():
    osm = _load("a8_osm_exposure")
    sq = lambda x0, y0, x1, y1: [{"lon": x0, "lat": y0}, {"lon": x1, "lat": y0}, {"lon": x1, "lat": y1},  # noqa: E731
                                 {"lon": x0, "lat": y1}, {"lon": x0, "lat": y0}]
    outer = sq(0, 0, 4, 4)
    el = {"type": "relation", "id": 1, "tags": {"boundary": "protected_area"},
          "members": [{"type": "way", "role": "outer", "geometry": outer[:3]},
                      {"type": "way", "role": "outer", "geometry": outer[2:]},
                      {"type": "way", "role": "inner", "geometry": sq(1, 1, 2, 2)},
                      {"type": "node", "role": "label", "geometry": None}]}
    g = osm.relation_polygon(el)
    assert abs(g.area - 15.0) < 1e-9
    assert osm.relation_polygon({"members": [{"type": "way", "role": "outer", "geometry": outer[:3]}]}) is None
