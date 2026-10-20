"""Tests for the DAHITI reader. All inputs are small synthetic fixtures that mimic
the documented DAHITI layouts; they are not real observations."""

import math

import pandas as pd
import pytest

from caspian.io import find_dahiti_file, read_dahiti

LEGACY_CSV = "date,water level,error\n2008-07-23,-27.512,0.050\n2008-08-12,-27.530,0.041\n"

V2_CSV = (
    "datetime;wse;wse_u;mission;phase;timeliness\n"
    "2008-07-23 23:22:08;-27.512;0.050;J2;NO;NTC;002;052\n"
    "2026-02-25 07:28:59;-29.101;0.034;S6A;NO;NTC;195;052\n"
)

ASCII_LEGACY = (
    "# DAHITI-ID       : 99999\n"
    "# Target name     : Synthetic, Test\n"
    "# column 1        : date [yyyy-mm-dd]\n"
    "# column 2        : water level from altimetry [m]\n"
    "# column 3        : error [m]\n"
    "2008-08-12 -27.530 0.041\n"
    "2008-07-23 -27.512 0.050\n"
)

ASCII_V2 = (
    "2008-07-23 23:22:08 -27.512 0.050 J2 NO NTC 002 052\n"
    "2026-02-25 07:28:59 -29.101 0.034 S6A NO NTC 195 052\n"
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_legacy_csv(tmp_path):
    s = read_dahiti(_write(tmp_path, "a.csv", LEGACY_CSV))
    assert s.n == 2
    assert list(s.data.columns[:3]) == ["time", "level_m", "sigma_m"]
    assert s.data["level_m"].iloc[0] == pytest.approx(-27.512)
    assert s.data["sigma_m"].iloc[1] == pytest.approx(0.041)


def test_v2_csv_with_extra_fields(tmp_path):
    s = read_dahiti(_write(tmp_path, "b.csv", V2_CSV))
    assert s.layout == "csv-v2"
    assert s.n == 2
    assert s.data["time"].iloc[1] == pd.Timestamp("2026-02-25 07:28:59")
    assert s.data["mission"].tolist() == ["J2", "S6A"]
    assert s.data["cycle"].tolist() == ["002", "195"]
    assert s.data["pass_nr"].tolist() == ["052", "052"]


def test_ascii_legacy_metadata_and_sorting(tmp_path):
    s = read_dahiti(_write(tmp_path, "c.txt", ASCII_LEGACY))
    assert s.layout == "ascii-legacy"
    assert s.meta["DAHITI-ID"] == "99999"
    assert "[m]" in s.meta["column 2"]
    assert s.data["time"].is_monotonic_increasing
    assert s.data["level_m"].iloc[0] == pytest.approx(-27.512)


def test_ascii_v2(tmp_path):
    s = read_dahiti(_write(tmp_path, "d.txt", ASCII_V2))
    assert s.layout == "ascii-v2"
    assert s.data["sigma_m"].tolist() == pytest.approx([0.050, 0.034])
    assert s.data["mission"].tolist() == ["J2", "S6A"]


def test_nonnumeric_level_dropped_and_counted(tmp_path):
    text = LEGACY_CSV + "2008-09-01,nan,0.1\n2008-09-11,,0.1\n"
    s = read_dahiti(_write(tmp_path, "e.csv", text))
    assert s.n == 2
    assert s.meta["rows_dropped_nonnumeric_level"] == "2"


def test_missing_error_column(tmp_path):
    s = read_dahiti(_write(tmp_path, "f.csv", "date,water level\n2008-07-23,-27.5\n"))
    assert math.isnan(s.data["sigma_m"].iloc[0])


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError):
        read_dahiti(_write(tmp_path, "g.csv", "# only comments\n"))


def test_find_file(tmp_path):
    assert find_dahiti_file(tmp_path / "missing") is None
    assert find_dahiti_file(tmp_path) is None
    _write(tmp_path, "series.csv", LEGACY_CSV)
    assert find_dahiti_file(tmp_path).name == "series.csv"


HYDROWEB_TXT = (
    "#\n# surface of reference: GGMO2C; high resolution global gravity model\n"
    "# https://doi.org/10.24400/329360/HYDROWEB_WATER_LEVEL\n"
    "lake=synthetic;country=X;basin=Y;lat=42;lon=50;type=operational;id=1 \n"
    "1992.73799901 ; 1992/09/27 ; 02:35 ;   -26.48 ;     0.07 ; 9999.999 ; 9999.999 ;                ; \n"
    "2026.78713470 ; 2026/10/15 ; 07:18 ;    -28.7 ;     0.03 ; 9999.999 ; 9999.999 ; sentinel3A-212 ; \n"
    "2026.79 ; 2026/10/16 ; 07:18 ;    9999.999 ;     0.03 ; 9999.999 ; 9999.999 ; x ; \n"
)


def test_hydroweb_reader(tmp_path):
    from caspian.io import read_hydroweb

    s = read_hydroweb(_write(tmp_path, "h.txt", HYDROWEB_TXT))
    assert s.n == 2
    assert s.data["level_m"].tolist() == pytest.approx([-26.48, -28.7])
    assert s.data["sigma_m"].tolist() == pytest.approx([0.07, 0.03])
    assert s.data["time"].iloc[1] == pd.Timestamp("2026-10-15 07:18")
    assert s.data["mission"].iloc[1] == "sentinel3A-212"
    assert s.meta["lake"] == "synthetic"
    assert s.meta["surface_of_reference"].startswith("GGMO2C")
    assert s.meta["rows_dropped_missing_level"] == "1"
