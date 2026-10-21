"""Word count of the Environmental Research Letters Letter (manuscript/main_erl.tex; D-107).

ERL: a Letter has at most 4,000 words, excluding the title, abstract, figures, tables, captions
and references; the abstract has at most 300 words (docs/journal_specs.md). This script counts:

- main text: the section files that main_erl.tex inputs between \\section{Introduction} and the
  statements (sections/erl_intro, erl_methods, erl_results, erl_discussion, erl_conclusion),
  headings included;
- the abstract (sections/erl_abstract), against its own limit;
- the statements (data and code availability, AI use, acknowledgements), reported separately
  because ERL's count does not name them.

Counting rules: LaTeX comments are dropped; \\todo{...} notes are dropped (a citation todo becomes
a citation, which does not count); citations, labels and figure/table environments are dropped;
each \\metric{...} and each \\ref{...} counts as one word, as does each inline formula; a word is a
whitespace-separated token with at least one letter or digit, so a range such as "0.5--1" is one
word. Exit status 1 if a limit is exceeded.

Usage: python tools/word_count_erl.py [--json]
"""

from __future__ import annotations

import caspian  # noqa: F401  (project rule: the package is imported first)
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
BS = chr(92)

MAIN_LIMIT = 4000
ABSTRACT_LIMIT = 300
MAIN_PARTS = ("erl_intro", "erl_methods", "erl_results", "erl_discussion", "erl_conclusion")
HEADINGS = {"erl_intro": "Introduction", "erl_methods": "Methods", "erl_results": "Results",
            "erl_discussion": "Discussion", "erl_conclusion": "Conclusion"}


def _drop_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        i = 0
        while True:
            j = line.find("%", i)
            if j < 0:
                break
            if j > 0 and line[j - 1] == BS:
                i = j + 1
                continue
            line = line[:j]
            break
        out.append(line)
    return "\n".join(out)


def _drop_command(text: str, name: str, replacement: str = "") -> str:
    """Remove every \\name[opt]{balanced arg} (one mandatory argument), replacing it."""
    pat = re.compile(re.escape(BS + name) + r"\*?(\[[^\]]*\])?\{")
    while True:
        m = pat.search(text)
        if not m:
            return text
        depth, k = 1, m.end()
        while k < len(text) and depth:
            if text[k] == "{" and text[k - 1] != BS:
                depth += 1
            elif text[k] == "}" and text[k - 1] != BS:
                depth -= 1
            k += 1
        text = text[:m.start()] + replacement + text[k:]


def _drop_environments(text: str, names: tuple[str, ...]) -> str:
    for n in names:
        text = re.sub(re.escape(BS + "begin{" + n) + r"\*?\}.*?" + re.escape(BS + "end{" + n) + r"\*?\}",
                      " ", text, flags=re.S)
    return text


def clean(text: str) -> str:
    text = _drop_comments(text)
    text = _drop_environments(text, ("figure", "table", "abstract"))
    for name in ("todo", "label", "citep", "citet", "cite", "citenum", "bibliography", "bibliographystyle"):
        text = _drop_command(text, name, " " if name != "citet" else " Author ")
    text = _drop_command(text, "metric", "N")
    for name in ("ref", "eqref"):
        text = _drop_command(text, name, "1")
    text = re.sub(r"\$[^$]*\$", " X ", text)                    # inline formula = one word
    text = text.replace(BS + "SIdem", "S2")                         # supplementary section macros
    text = re.sub(re.escape(BS) + r"SI[a-z]+(\{\})?", "S1", text)
    text = re.sub(re.escape(BS) + r"[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)   # other commands: keep their text
    text = re.sub(r"[{}~]", " ", text)
    text = text.replace(BS, " ")
    return text


def count_words(text: str) -> int:
    return sum(1 for tok in clean(text).split() if re.search(r"[A-Za-z0-9]", tok))


def section_file(stem: str) -> Path:
    return MS / "sections" / f"{stem}.tex"


def main() -> int:
    parts = {}
    for stem in MAIN_PARTS:
        p = section_file(stem)
        parts[stem] = count_words(p.read_text(encoding="utf-8")) + len(HEADINGS[stem].split()) if p.exists() else 0
    main_total = sum(parts.values())
    abstract_text = section_file("erl_abstract").read_text(encoding="utf-8")
    abstract_body = re.search(re.escape(BS + "begin{abstract}") + r"(.*?)" + re.escape(BS + "end{abstract}"),
                              abstract_text, flags=re.S)
    abstract = count_words(abstract_body.group(1)) if abstract_body else 0
    statements = count_words(section_file("erl_statements").read_text(encoding="utf-8"))
    report = {"main_text_words": main_total, "main_text_limit": MAIN_LIMIT, "by_section": parts,
              "abstract_words": abstract, "abstract_limit": ABSTRACT_LIMIT,
              "statements_words": statements, "main_plus_statements": main_total + statements}
    if "--json" in sys.argv:
        print(json.dumps(report, indent=1))
    else:
        for stem, n in parts.items():
            print(f"{HEADINGS[stem]:<12} {n:>5}")
        print(f"{'Main text':<12} {main_total:>5}  (limit {MAIN_LIMIT}; excludes title, abstract, captions, "
              "tables, references)")
        print(f"{'Abstract':<12} {abstract:>5}  (limit {ABSTRACT_LIMIT})")
        print(f"{'Statements':<12} {statements:>5}  (data, code, AI use, acknowledgements; reported separately)")
        print(f"{'Main + stmts':<12} {main_total + statements:>5}")
    return 1 if main_total > MAIN_LIMIT or abstract > ABSTRACT_LIMIT else 0


if __name__ == "__main__":
    sys.exit(main())
