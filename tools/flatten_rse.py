"""Flat LaTeX source of the article for the publisher (first revision, D-161).

Elsevier asks for editable source files at revision ("A PDF is not an acceptable source
file") and its submission system cannot process subfolders: "All submission files -
including figures, tables, and style and bibliographic files - must be stored at the same
folder level" (docs/journal_specs.md). The working manuscript (manuscript/rse_main.tex) is
not that: it reads sections/*.tex and rse/fig_*.tex, takes every number from the generated
numbers.tex (3 MB of macros), finds its figures in ../figures/out and its references in
../refs. This tool writes the same article as one self-contained folder:

  manuscript.tex    one file: every \\input inlined, every \\metric{}, \\metricsf{}{} and
                    \\metricdp{}{} written out as the number it prints, the revision-marking
                    wrappers (\\rev{}, revblock) removed, comments removed
  Figure_<n>.pdf    the figures, numbered in the order they appear
  references.bib    the cited entries only
  manuscript.bbl    the reference list as BibTeX built it (written by the build below)

The preamble loads fontspec under XeTeX or LuaTeX and fontenc/inputenc under pdfTeX, so the
file does not depend on the engine. Line 1 is the directive "%!TEX TS-program = xelatex":
the publisher's submission system (Editorial Manager; TeX Live 2024) builds with LaTeX, then
pdfLaTeX, unless that line asks for XeLaTeX (Aries Systems, "LaTeX Guide, FAQs, and Resources
for EM/PM", updated March 2025, read 2026-11-25), and XeTeX is the engine this tool builds
and checks the file with (tectonic). The pdfTeX branch is kept for readers who compile with
pdfLaTeX; it has not been run.

Checks (exit status 1 if one fails):
  - no \\input, \\metric, \\rev or path separator is left in manuscript.tex, and every
    character is ASCII or one that both engines read from UTF-8 (listed in ALLOWED);
  - with --build: tectonic builds the folder on its own, and the built PDF has the same
    number of pages and prints the same tokens the same number of times as the reference
    PDF (PyMuPDF), so every word and number of the flat source is that of the article built
    from the working files (build_and_compare).

Usage: python tools/flatten_rse.py <output folder> [--build] [--reference manuscript/rse_main.pdf]
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
sys.path.insert(0, str(ROOT / "tools"))
import make_macros  # noqa: E402
import texcheck as tc  # noqa: E402

# Characters beyond ASCII that both pdfTeX (inputenc utf8, T1) and XeTeX read from UTF-8. Written
# as escapes so that no tool that mishandles the file's encoding can alter the set.
ALLOWED = set("\u2013\u2014\u2212\u00b1\u00d7\u00b0\u00b7\u2019\u2018\u201c\u201d"
              "\u00e9\u00e8\u00f6\u00fc\u00e4\u00e5\u00e7\u00f1\u00f3\u00e1\u00ed\u00fa\u00f8\u00c5\u00d6\u00dc\u00c9\u00a7")
BEGIN_MARK, END_MARK = "% >>> revision marking", "% <<< revision marking"
# Under pdfTeX the reference list (references.bib, manuscript.bbl) is read as UTF-8 too. Its
# characters are Latin letters with accents, dashes and quotation marks, which inputenc knows,
# and the Unicode hyphen U+2010 that Crossref writes in some titles, which older LaTeX kernels
# do not: it is declared here as an ordinary hyphen.
ENGINE_BLOCK = ("\\usepackage{iftex}\n\\ifPDFTeX\n  \\usepackage[T1]{fontenc}\n  \\usepackage[utf8]{inputenc}\n"
                "  \\usepackage{lmodern}\n  \\DeclareUnicodeCharacter{2010}{-}\n\\else\n  \\usepackage{fontspec}\n\\fi")
# Line 1 of the flat source: the engine directive that Editorial Manager and TeX editors read.
# It must stay the first line of the file.
ENGINE_DIRECTIVE = "%!TEX TS-program = xelatex"
HEADER = (ENGINE_DIRECTIVE + "\n"
          "% Built and checked with XeLaTeX; the line above selects that engine in Editorial Manager and\n"
          "% in TeX editors. The preamble also carries a pdfLaTeX branch (\\ifPDFTeX).\n")
# Characters of the reference list that pdfTeX reads through inputenc (T1) or the declaration above.
BIB_ALLOWED = ALLOWED | set("\u2010\u00df\u00eb\u00ef\u0142\u00e3\u00ea\u00f4\u00e0\u00f9\u00ee\u00e2\u00fd\u010d\u0161\u017e\u0107\u0144\u015b\u017a\u017c")
# Packages that only the working copy needs, each with the commands that would need it.
OPTIONAL_PACKAGES = (("siunitx", r"\\(?:num|qty|SI|si|unit|ang|numrange|qtyrange|SIrange|numlist|tablenum|numproduct)\b"),
                     ("lineno", r"\\(?:linenumbers|nolinenumbers|modulolinenumbers|linelabel)\b"))
FIG_OPTS = "height=0.62\\textheight,keepaspectratio"       # what the working preamble adds to every figure


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _comment_start(line: str) -> int:
    """Index of the % that starts a comment (preceded by an even number of backslashes), or -1."""
    i = line.find("%")
    while i >= 0:
        j = i
        while j > 0 and line[j - 1] == "\\":
            j -= 1
        if (i - j) % 2 == 0:
            return i
        i = line.find("%", i + 1)
    return -1


def strip_comments(tex: str) -> str:
    """Remove comments the way TeX reads them. A % swallows the rest of its line, the line
    end and the indentation of the next line, so:
    - a line that holds only a comment disappears (it must not become an empty line, which
      would end the paragraph: a comment between a sentence and its display equation);
    - a comment after text joins that text to the next line, unless the next line is empty
      (then the empty line still ends the paragraph)."""
    out: list[str] = []
    join = False                       # the previous kept line ended in a comment
    for line in tex.split("\n"):
        k = _comment_start(line)
        content = line if k < 0 else line[:k]
        if k >= 0 and not content.strip():
            continue                   # comment-only line: gone; a pending join stays pending
        if join and line.strip():
            out[-1] += content.lstrip()
        else:
            out.append(content)
        join = k >= 0
    return "\n".join(out)


def cut_marking(tex: str) -> str:
    i, j = tex.find(BEGIN_MARK), tex.find(END_MARK)
    if i < 0 or j < 0 or j < i:
        raise SystemExit("rse_main.tex: the revision-marking block is not delimited")
    return tex[:i] + tex[j + len(END_MARK):]


def inline_inputs(tex: str, base: Path, seen: tuple = ()) -> str:
    """Every \\input{...} replaced by its file (comments removed first), numbers.tex dropped."""
    def sub(m):
        name = m.group(1).strip()
        if Path(name).stem == "numbers":
            return ""
        p = base / (name if name.endswith(".tex") else name + ".tex")
        if not p.exists():
            raise SystemExit(f"\\input file not found: {p}")
        if p in seen:
            raise SystemExit(f"circular \\input: {p}")
        return inline_inputs(strip_comments(read(p)), base, seen + (p,)).strip("\n")
    return re.sub(r"\\input\s*\{([^{}]+)\}", sub, tex)


def unwrap(tex: str, command: str) -> str:
    """\\command{...} replaced by its argument (balanced braces)."""
    out, i = [], 0
    pat = re.compile(r"\\" + command + r"\s*\{")
    while True:
        m = pat.search(tex, i)
        if not m:
            out.append(tex[i:])
            return "".join(out)
        out.append(tex[i:m.start()])
        arg, j = tc.read_group(tex, m.end() - 1)
        out.append(unwrap(arg, command))
        i = j


def _group(digits: str) -> str:
    """Thin-space thousands separators for an integer part of four or more digits (siunitx
    group-minimum-digits=4, group-separator={\\,}, group-digits=integer)."""
    if len(digits) < 4:
        return digits
    parts = []
    while digits:
        parts.append(digits[-3:])
        digits = digits[:-3]
    return "\\,".join(reversed(parts))


def siunitx(raw: str, mode: str, n: int) -> str:
    """What \\num[round-mode=<mode>,round-precision=n,...]{raw} prints in text mode (the
    working preamble sets \\sisetup{mode=text}): half-up rounding of the decimal string, zero
    padding to the precision, grouped integer part, a text minus."""
    d = Decimal(raw)
    if mode == "places":
        q = Decimal(1).scaleb(-n)
    else:
        if d == 0:
            return "0"
        q = Decimal(1).scaleb(d.adjusted() - n + 1)
    r = d.quantize(q, rounding=ROUND_HALF_UP)
    s = format(abs(r), "f")
    whole, _, frac = s.partition(".")
    body = _group(whole) + ("." + frac if frac else "")
    neg = r < 0 and any(c in "123456789" for c in s)
    return ("\\textminus{}" if neg else "") + body


def write_out_metrics(tex: str, metrics: dict) -> str:
    def sf(m):
        k, n = m.group(2), int(m.group(3))
        if k not in metrics:
            raise SystemExit(f"\\metric{m.group(1)}{{{k}}}: unknown key")
        v = metrics[k]["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise SystemExit(f"\\metric{m.group(1)}{{{k}}}: not numeric")
        return siunitx(repr(v), "figures" if m.group(1) == "sf" else "places", n)

    def plain(m):
        k = m.group(1)
        if k not in metrics:
            raise SystemExit(f"\\metric{{{k}}}: unknown key")
        rec = metrics[k]
        return make_macros.fmt(k, rec["value"], rec.get("decimals"), rec.get("unit"))

    tex = re.sub(r"\\metric(sf|dp)\s*\{([^{}]+)\}\s*\{(\d+)\}", sf, tex)
    return re.sub(r"\\metric\s*\{([^{}]+)\}", plain, tex)


def place_figures(tex: str, out: Path) -> tuple[str, list[tuple[str, str]]]:
    """Figures copied as Figure_<n>.pdf in order of appearance; paths replaced."""
    figs: list[tuple[str, str]] = []

    def sub(m):
        opts, path = m.group(1) or "", m.group(2).strip()
        src = (MS / path).resolve()
        if not src.exists():
            raise SystemExit(f"figure file not found: {src}")
        name = f"Figure_{len(figs) + 1}{src.suffix}"
        shutil.copyfile(src, out / name)
        figs.append((name, str(src.relative_to(ROOT)).replace("\\", "/")))
        o = [x.strip() for x in opts.split(",") if x.strip()]
        if not any(x.startswith("height=") for x in o):
            o += FIG_OPTS.split(",")
        return f"\\includegraphics[{','.join(o)}]{{{name}}}"

    tex = re.sub(r"\\includegraphics\s*(?:\[([^\]]*)\])?\s*\{([^{}]+)\}", sub, tex)
    return tex, figs


def prune_bib(keys: list[str], out: Path) -> int:
    bib = read(ROOT / "refs" / "references.bib")
    entries, i = {}, 0
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", bib):
        if m.start() < i:
            continue
        body, j = tc.read_group(bib, bib.index("{", m.start()))
        entries[m.group(2)] = f"@{m.group(1)}{{{body}}}"
        i = j
    missing = [k for k in keys if k not in entries]
    if missing:
        raise SystemExit(f"cited but not in refs/references.bib: {missing}")
    (out / "references.bib").write_text("\n\n".join(entries[k] for k in sorted(set(keys))) + "\n", encoding="utf-8")
    return len(set(keys))


def drop_unused_packages(doc: str) -> str:
    """Remove from the preamble the packages that only the working copy needs (siunitx, once
    every number is plain text; lineno, when no line numbers are set), each only if the
    document uses none of its commands. The publisher's TeX then needs no particular version
    of them."""
    head, body = doc.split("\\begin{document}", 1)
    for name, commands in OPTIONAL_PACKAGES:
        if re.search(commands, body):
            continue
        head = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{" + name + r"\}[ \t]*\n?", "", head)
        if name == "siunitx":
            while True:
                k = head.find("\\sisetup")
                if k < 0:
                    break
                _, end = tc.read_group(head, head.index("{", k))
                head = head[:k] + head[end:].lstrip(" \t")
    return head + "\\begin{document}" + body


def flatten(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.iterdir():
        if p.is_dir():
            raise SystemExit(f"{out} holds a subfolder ({p.name}); the output folder must be flat")
        p.unlink()
    metrics = json.loads(read(ROOT / "results" / "metrics.json"))
    tex = cut_marking(read(MS / "rse_main.tex"))
    head, body = tex.split("\\begin{document}", 1)
    # preamble: comments out, the working-copy helpers replaced
    head = strip_comments(head)
    head = head.replace("\\usepackage{fontspec}", ENGINE_BLOCK)
    head = re.sub(r"\\input\s*\{numbers(?:\.tex)?\}\s*", "", head)
    head = re.sub(r"\\let\\rseincludegraphics\\includegraphics\s*", "", head)
    k = head.find("\\renewcommand{\\includegraphics}")
    if k >= 0:                                  # the placeholder wrapper around \includegraphics
        _, e1 = tc.read_group(head, head.index("{", k + len("\\renewcommand{\\includegraphics}")))
        head = head[:k] + head[e1:]
    head = re.sub(r"\\newcommand\{\\todo\}\[1\]\{[^\n]*\}\s*", "", head)
    # a removed end-of-line comment joins its line to the next, as TeX does; in the preamble
    # each command goes back on a line of its own, for the reader
    head = re.sub(r"(?<=\S)[ \t]{2,}(?=\\)", "\n", head)
    body = strip_comments(body)
    body = inline_inputs(body, MS)
    doc = head.rstrip() + "\n\n\\begin{document}" + body
    doc = unwrap(doc, "rev")
    doc = re.sub(r"\\(?:begin|end)\{revblock\}[ \t]*\n?", "", doc)
    doc = write_out_metrics(doc, metrics)
    doc = drop_unused_packages(doc)
    doc, figs = place_figures(doc, out)
    doc = re.sub(r"\\bibliography\s*\{[^{}]*\}", "\\\\bibliography{references}", doc)
    doc = re.sub(r"[ \t]+\n", "\n", doc)
    doc = re.sub(r"\n{3,}", "\n\n", doc).lstrip("\n")
    n_refs = prune_bib(tc.cited_keys(doc), out)
    doc = HEADER + doc
    (out / "manuscript.tex").write_text(doc, encoding="utf-8", newline="\n")
    probs = []
    if read(out / "manuscript.tex").splitlines()[0] != ENGINE_DIRECTIVE:
        probs.append("line 1 of manuscript.tex is not the engine directive")
    # the submission system refuses a file name with more than one period or a special character,
    # and two files with one base name (the .bbl of the .tex excepted, which LaTeX requires)
    names = [p.name for p in out.iterdir()]
    bad = [n for n in names if not re.fullmatch(r"[A-Za-z0-9_]+\.[A-Za-z0-9]+", n)]
    if bad:
        probs.append(f"file names the submission system may refuse: {bad}")
    stems = [n.rsplit(".", 1)[0].lower() for n in names if not n.endswith(".bbl")]
    if len(stems) != len(set(stems)):
        probs.append("two files of the flat folder share a base name")
    for pat, what in ((r"\\input\b", "\\input"), (r"\\metric", "\\metric"), (r"\\rev\b", "\\rev"),
                      (r"revblock", "revblock"), (r"\.\./", "a parent path"), (r"sections/|rse/fig_", "a subfolder path"),
                      (r"\\todo\b", "\\todo")):
        if re.search(pat, doc):
            probs.append(f"manuscript.tex still holds {what}")
    odd = sorted({c for c in doc if ord(c) > 127 and c not in ALLOWED})
    if odd:
        probs.append("characters outside the allowed set: " + " ".join(f"U+{ord(c):04X}" for c in odd))
    odd = sorted({c for c in read(out / "references.bib") if ord(c) > 127 and c not in BIB_ALLOWED})
    if odd:
        probs.append("references.bib holds characters that pdfTeX may not read: "
                     + " ".join(f"U+{ord(c):04X}" for c in odd))
    return {"figures": figs, "n_references": n_refs, "problems": probs, "chars": len(doc),
            "non_ascii": sorted({f"U+{ord(c):04X}" for c in doc if ord(c) > 127})}


def pdf_tokens(path: Path) -> tuple[int, list[str], list[str]]:
    """(pages, every whitespace-separated token in reading order, every text line) of a PDF."""
    import fitz

    tokens, lines = [], []
    with fitz.open(path) as d:
        n = d.page_count
        for p in d:
            t = p.get_text()
            tokens += t.split()
            lines += [re.sub(r"\s+", " ", x).strip() for x in t.splitlines() if x.strip()]
    return n, tokens, lines


def build_and_compare(out: Path, reference: Path) -> tuple[list[str], str]:
    """Build the flat folder with tectonic and compare what it prints with the reference PDF.

    The test is on content, not on page breaks: the two builds may break a page one line
    apart (the working copy sets its rounded numbers through siunitx, the flat source as
    plain digits, and their boxes differ by a hair), and the publisher re-typesets the
    article in any case. Required: the same number of pages and the same multiset of
    tokens, so every word and every number is printed the same number of times. Reported
    beside it: whether every text line is the same too (it is when no line break moved)."""
    from collections import Counter

    r = subprocess.run(["tectonic", "--keep-intermediates", "manuscript.tex"], cwd=out, capture_output=True, text=True)
    if r.returncode != 0:
        return ["tectonic failed on the flat source:\n" + (r.stdout + r.stderr)[-1500:]], ""
    for ext in (".aux", ".log", ".out", ".blg", ".spl"):
        p = out / f"manuscript{ext}"
        if p.exists():
            p.unlink()
    if not (out / "manuscript.bbl").exists():
        return ["the build left no manuscript.bbl"], ""
    (na, ta, la), (nb, tb, lb) = pdf_tokens(out / "manuscript.pdf"), pdf_tokens(reference)
    probs = []
    if na != nb:
        probs.append(f"{na} pages against {nb} in {reference.name}")
    ca, cb = Counter(ta), Counter(tb)
    if ca != cb:
        only_a, only_b = list((ca - cb).elements())[:12], list((cb - ca).elements())[:12]
        probs.append(f"the printed tokens differ: only in the flat build {[ascii(x) for x in only_a]}, "
                     f"only in {reference.name} {[ascii(x) for x in only_b]}")
    note = (f"{na} pages, {len(ta)} tokens; " + ("every text line is identical too"
            if Counter(la) == Counter(lb) else
            f"{sum((Counter(la) - Counter(lb)).values())} lines break differently (same tokens)"))
    return probs, note


def main(argv) -> int:
    if not argv or argv[0].startswith("--"):
        raise SystemExit(__doc__)
    out = Path(argv[0]).resolve()
    rep = flatten(out)
    for name, src in rep["figures"]:
        print(f"  {name}  <-  {src}")
    print(f"manuscript.tex: {rep['chars']} characters, {len(rep['figures'])} figures, {rep['n_references']} references; "
          f"non-ASCII characters: {' '.join(rep['non_ascii']) or 'none'}")
    probs = list(rep["problems"])
    if "--build" in argv:
        ref = Path(argv[argv.index("--reference") + 1]) if "--reference" in argv else MS / "rse_main.pdf"
        more, note = build_and_compare(out, ref.resolve())
        probs += more
        if not more:
            print(f"built with tectonic; it prints the same words and numbers as {ref.name}: {note}")
    for p in probs:
        print("FAIL:", p)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
