"""scripts/README.md is the current rendering of tools/scripts_index.py, and every script has a docstring."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import scripts_index  # noqa: E402


def test_scripts_readme_is_current():
    assert (ROOT / "scripts" / "README.md").read_text(encoding="utf-8") == scripts_index.render(), \
        "run tools/scripts_index.py"


def test_every_script_has_a_docstring():
    missing = [p.name for p in (ROOT / "scripts").glob("*.py") if scripts_index.first_line(p) == "(no docstring)"]
    assert not missing, missing
