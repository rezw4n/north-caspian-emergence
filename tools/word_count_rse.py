"""Word counts and front-matter limits of the article (manuscript/rse_main.tex) against the
Guide for Authors of Remote Sensing Applications: Society and Environment (read 2026-11-25,
docs/journal_specs.md).

Limits checked (exit status 1 if one is exceeded):
- abstract: at most 250 words;
- keywords: 1 to 7;
- highlights (manuscript/rse_highlights.tex): 3 to 5 bullets of at most 85 characters each,
  spaces included, as printed (every \\metric{} macro replaced by its formatted value).
Reported, not limited: the words of each section file and of the whole article. The guide
gives 6,000 words for a research paper and accepts longer ones "on an occasional basis"; the
article was reviewed at about 15,000, so the revision is held to no net growth (D-161). Pass
--baseline <git rev> to print the same counts at an earlier commit and the difference.

The count uses the helpers of tools/texcheck.py, as tools/word_count_erl.py does: a
displayed equation, an inline formula, a \\metric{} value or a cross-reference counts as one
word, a citation as none; floats and their captions are counted separately.

Usage: python tools/word_count_rse.py [--json] [--baseline <rev>]
"""

from __future__ import annotations

import caspian  # noqa: F401  (project rule: the package is imported first)
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
sys.path.insert(0, str(ROOT / "tools"))
import texcheck as tc  # noqa: E402

ABSTRACT_LIMIT = 250
KEYWORDS = (1, 7)
HIGHLIGHTS = (3, 5)
HIGHLIGHT_CHARS = 85
PARTS = {"rse_intro": "1 Introduction", "rse_data": "2 Study areas and data", "rse_methods": "3 Methods",
         "rse_results": "4 Results", "rse_discussion": "5 Discussion", "rse_conclusions": "6 Conclusions",
         "rse_declarations": "Declarations"}
FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.S)
INPUT = re.compile(r"\\input\{([^}]+)\}")


def _read(rel: str, rev: str | None) -> str | None:
    if rev is None:
        p = ROOT / rel
        return p.read_text(encoding="utf-8") if p.exists() else None
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{rel}"], capture_output=True)
    return r.stdout.decode("utf-8") if r.returncode == 0 else None


def _expand(text: str, rev: str | None) -> str:
    """Inline the \\input files of a section file (the figure wrappers in manuscript/rse/)."""
    def sub(m):
        name = m.group(1)
        t = _read(f"manuscript/{name if name.endswith('.tex') else name + '.tex'}", rev)
        return t if t is not None else ""
    return INPUT.sub(sub, text)


def section_counts(stem: str, rev: str | None) -> dict:
    t = _read(f"manuscript/sections/{stem}.tex", rev)
    if t is None:
        return {"text": 0, "floats": 0}
    t = tc.strip_comments(_expand(tc.strip_comments(t), rev))
    floats = " ".join(m.group(0) for m in FLOAT.finditer(t))
    return {"text": tc.word_count(FLOAT.sub(" ", t)), "floats": tc.word_count(floats)}


def metric_values() -> dict:
    """{key: printed value} from manuscript/numbers.tex, for the highlights' character count."""
    out = {}
    p = MS / "numbers.tex"
    if p.exists():
        for m in re.finditer(r"\\@namedef\{metricraw@([^}]+)\}\{([^}]*)\}", p.read_text(encoding="utf-8")):
            out[m.group(1)] = float(m.group(2))
    return out


def _sig(v: float, n: int) -> str:
    from decimal import ROUND_HALF_UP, Decimal
    if v == 0:
        return "0"
    d = Decimal(repr(v))
    q = Decimal(1).scaleb(d.adjusted() - n + 1)
    s = format(d.quantize(q, rounding=ROUND_HALF_UP), "f")
    return s.rstrip("0").rstrip(".") if "." in s and d.adjusted() - n + 1 >= 0 else s


def highlights(rev: str | None = None) -> list[str]:
    """The bullets as printed: macros replaced by their values, TeX markup removed."""
    t = _read("manuscript/rse_highlights.tex", rev)
    if t is None:
        return []
    vals = metric_values()
    t = tc.strip_comments(t)
    body = t.split("\\begin{itemize}", 1)[1].split("\\end{itemize}", 1)[0]
    out = []
    for item in body.split("\\item")[1:]:
        s = item.strip()
        s = re.sub(r"\\metricsf\{([^}]+)\}\{(\d+)\}", lambda m: _sig(vals[m.group(1)], int(m.group(2))), s)
        s = re.sub(r"\\metricdp\{([^}]+)\}\{(\d+)\}", lambda m: f"{vals[m.group(1)]:.{int(m.group(2))}f}", s)
        if "\\metric{" in s:
            raise SystemExit("highlights: \\metric{} without a rounding macro cannot be measured here")
        s = s.replace("--", "–").replace("~", " ").replace("\\,", " ").replace("\\_", "_").replace("\\%", "%")
        s = re.sub(r"\\textsuperscript\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        if "\\" in s:
            raise SystemExit(f"highlights: unhandled TeX in {s!r}")
        out.append(s)
    return out


def report(rev: str | None = None) -> dict:
    main = _read("manuscript/rse_main.tex", rev)
    parts = {name: section_counts(stem, rev) for stem, name in PARTS.items()}
    ab = _read("manuscript/sections/rse_abstract.tex", rev)
    ab_text = tc.abstract(tc.strip_comments(ab)) if ab else None
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tc.strip_comments(main or ""), re.S)
    keywords = [k.strip() for k in kw.group(1).split("\\sep")] if kw else []
    hl = highlights(rev)
    body = sum(v["text"] for k, v in parts.items() if k != "Declarations")
    return {"by_part": parts, "body_words": body, "float_words": sum(v["floats"] for v in parts.values()),
            "declaration_words": parts["Declarations"]["text"],
            "abstract_words": tc.word_count(ab_text) if ab_text else 0, "abstract_limit": ABSTRACT_LIMIT,
            "keywords": keywords, "highlights": [{"text": h, "chars": len(h)} for h in hl]}


def main() -> int:
    rep = report()
    base = None
    if "--baseline" in sys.argv:
        base = report(sys.argv[sys.argv.index("--baseline") + 1])
    if "--json" in sys.argv:
        print(json.dumps({"now": rep, "baseline": base}, indent=1, ensure_ascii=False))
    else:
        print(f"{'part':<26} {'text':>6} {'tables and captions':>20}" + ("   change in text" if base else ""))
        for name, v in rep["by_part"].items():
            extra = f"   {v['text'] - base['by_part'][name]['text']:+d}" if base else ""
            print(f"{name:<26} {v['text']:>6} {v['floats']:>20}{extra}")
        extra = f"   {rep['body_words'] - base['body_words']:+d}" if base else ""
        print(f"{'Sections 1 to 6':<26} {rep['body_words']:>6} {rep['float_words']:>20}{extra}")
        print(f"abstract: {rep['abstract_words']} words (limit {ABSTRACT_LIMIT})"
              + (f"; baseline {base['abstract_words']}" if base else ""))
        print(f"keywords: {len(rep['keywords'])} (limit {KEYWORDS[0]} to {KEYWORDS[1]}): " + "; ".join(rep["keywords"]))
        for h in rep["highlights"]:
            print(f"highlight ({h['chars']:>2} of {HIGHLIGHT_CHARS} characters): {h['text']}")
    ok = (0 < rep["abstract_words"] <= ABSTRACT_LIMIT and KEYWORDS[0] <= len(rep["keywords"]) <= KEYWORDS[1]
          and HIGHLIGHTS[0] <= len(rep["highlights"]) <= HIGHLIGHTS[1]
          and all(h["chars"] <= HIGHLIGHT_CHARS for h in rep["highlights"]))
    if not ok:
        print("FAIL: a front-matter limit is exceeded", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
