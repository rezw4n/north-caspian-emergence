"""Word count of the Environmental Research Letters Letter (manuscript/main_erl.tex).

ERL: a Letter has at most 4,000 words, excluding the title, abstract, figures, tables, captions
and references; the abstract has at most 300 words. The count uses the manuscript-check
helpers of tools/texcheck.py, so the two cannot disagree:

- the total is everything ERL does not exclude: the body without title and authors, abstract,
  floats (figures, tables, their captions) and references; section headings, keywords,
  statements and acknowledgements included;
- a displayed equation, an inline formula, a \\metric{} value or a cross-reference counts as one
  word, a citation as none, and a \\todo note as its words (it is text until resolved).

Per section, the same rules are applied to each section file, comments removed. Exit status 1
if a limit is exceeded.

Usage: python tools/word_count_erl.py [--json]
"""

from __future__ import annotations

import caspian  # noqa: F401  (project rule: the package is imported first)
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
sys.path.insert(0, str(ROOT / "tools"))
import texcheck as tc  # noqa: E402

MAIN_LIMIT = 4000
ABSTRACT_LIMIT = 300
PARTS = {"erl_intro": "Introduction", "erl_methods": "Methods", "erl_results": "Results",
         "erl_discussion": "Discussion", "erl_conclusion": "Conclusion", "erl_statements": "Statements"}


def file_words(stem: str) -> int:
    p = MS / "sections" / f"{stem}.tex"
    return tc.word_count(tc.strip_comments(p.read_text(encoding="utf-8"))) if p.exists() else 0


def main() -> int:
    text, missing = tc.expand_inputs(MS / "main_erl.tex")
    total = tc.word_count(tc.main_text_for_count(text))
    ab = tc.abstract(tc.document_body(text))
    abstract = tc.word_count(ab) if ab else 0
    parts = {name: file_words(stem) + (1 if stem != "erl_statements" else 0) for stem, name in PARTS.items()}
    report = {"total_words": total, "limit": MAIN_LIMIT, "by_part": parts,
              "sections_without_statements": sum(v for k, v in parts.items() if k != "Statements"),
              "abstract_words": abstract, "abstract_limit": ABSTRACT_LIMIT, "missing_inputs": missing}
    if "--json" in sys.argv:
        print(json.dumps(report, indent=1))
    else:
        for name, n in parts.items():
            print(f"{name:<13} {n:>5}")
        print(f"{'Sections':<13} {report['sections_without_statements']:>5}  (Introduction to Conclusion, headings included)")
        print(f"{'Total':<13} {total:>5}  (limit {MAIN_LIMIT}; everything but title, abstract, "
              "floats, captions and references, statements and keywords included)")
        print(f"{'Abstract':<13} {abstract:>5}  (limit {ABSTRACT_LIMIT})")
        if missing:
            print("missing inputs:", missing)
    return 1 if total > MAIN_LIMIT or abstract > ABSTRACT_LIMIT else 0


if __name__ == "__main__":
    sys.exit(main())
