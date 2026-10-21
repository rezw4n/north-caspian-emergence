"""Pure helpers for the verify-B manuscript checks (tools/verify.py, brief section 15).

LaTeX reading (comments, \\input expansion, environments, sections, citations, metric
keys, graphics), the number trace with its whitelist, the ERL word count, the colour-map
scan of figure scripts, numeric comparison of rebuilt tables, and the parsing of the
internal reviews and the submission-package list. Only ``expand_inputs`` touches the file
system. The checks that use these helpers may not be weakened after their first commit
without an entry in docs/DECISIONS.md (CLAUDE.md, integrity rule 5).
"""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

# ============================================================================ LaTeX basics


def strip_comments(tex: str) -> str:
    """Remove LaTeX comments (an unescaped % to the end of the line; \\% is kept)."""
    return re.sub(r"(?<!\\)%[^\n]*", "", tex)


_INPUT = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")


def expand_inputs(path: Path, skip: tuple[str, ...] = ("numbers",), _seen: frozenset = frozenset()
                  ) -> tuple[str, list[str]]:
    """The file's text, comments removed, with every \\input{x} / \\include{x} replaced by
    the (recursively expanded) text of x. Paths resolve against the directory of the top
    file, as LaTeX resolves them against the directory it runs in. Files whose stem is in
    ``skip`` (the generated macro file) are dropped. Returns (text, missing files)."""
    path = Path(path)
    base = path.parent

    def expand(p: Path, seen: frozenset) -> tuple[str, list[str]]:
        text = resolve_iffileexists(strip_comments(p.read_text(encoding="utf-8")), base)
        missing: list[str] = []

        def sub(m: re.Match) -> str:
            name = m.group(1).strip()
            q = base / name
            if q.suffix != ".tex":
                q = q.with_name(q.name + ".tex")
            if q.stem in skip:
                return " "
            if not q.exists():
                missing.append(name)
                return " "
            key = str(q.resolve())
            if key in seen:
                missing.append(f"{name} (recursive input)")
                return " "
            t, miss = expand(q, seen | {key})
            missing.extend(miss)
            return "\n" + t + "\n"

        return _INPUT.sub(sub, text), missing

    return expand(path, _seen | {str(path.resolve())})


def resolve_iffileexists(text: str, base: Path) -> str:
    """Replace each \\IfFileExists{file}{yes}{no} by the branch LaTeX would take, the file
    resolved against ``base`` (the directory LaTeX runs in). One inside a macro definition
    (its file argument holds a #) is left alone."""
    pat = re.compile(r"\\IfFileExists\s*\{")
    out, i = [], 0
    while True:
        m = pat.search(text, i)
        if not m:
            break
        try:
            name, j = read_group(text, m.end() - 1)
            yes, j = read_group(text, _skip_ws(text, j))
            no, j = read_group(text, _skip_ws(text, j))
        except ValueError:
            out.append(text[i:m.end()])
            i = m.end()
            continue
        out.append(text[i:m.start()])
        if "#" in name:
            out.append(text[m.start():j])
        else:
            out.append(resolve_iffileexists(yes if (Path(base) / name.strip()).exists() else no, base))
        i = j
    out.append(text[i:])
    return "".join(out)


def input_files(path: Path, skip: tuple[str, ...] = ("numbers",)) -> list[Path]:
    """Every file that ``path`` reads through \\input / \\include, recursively (existing
    ones only, resolved against the top file's directory)."""
    path = Path(path)
    base, out, todo = path.parent, [], [path]
    while todo:
        p = todo.pop()
        for name in _INPUT.findall(resolve_iffileexists(strip_comments(p.read_text(encoding="utf-8")), base)):
            q = base / name.strip()
            if q.suffix != ".tex":
                q = q.with_name(q.name + ".tex")
            if q.stem in skip or not q.exists() or q.resolve() in out:
                continue
            out.append(q.resolve())
            todo.append(q)
    return out


def read_group(tex: str, i: int, open_: str = "{", close: str = "}") -> tuple[str, int]:
    """The content of the balanced group that opens at tex[i] and the index after it.
    Escaped delimiters (\\{ \\}) do not count. Raises ValueError if unbalanced."""
    if i >= len(tex) or tex[i] != open_:
        raise ValueError(f"no {open_!r} at {i}")
    depth, j = 0, i
    while j < len(tex):
        c = tex[j]
        if c == "\\":
            j += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return tex[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced {open_!r} at {i}")


def _skip_ws(tex: str, j: int) -> int:
    while j < len(tex) and tex[j] in " \t\n\r":
        j += 1
    return j


def _skip_optional(tex: str, j: int) -> int:
    """Index after any [..] optional arguments (and the white space before them) at j."""
    while True:
        k = _skip_ws(tex, j)
        if k < len(tex) and tex[k] == "[":
            depth, m = 0, k
            while m < len(tex):
                c = tex[m]
                if c == "\\":
                    m += 2
                    continue
                if c in "[{":
                    depth += 1
                elif c in "]}":
                    depth -= 1
                    if depth == 0:
                        break
                m += 1
            j = m + 1
        else:
            return j


def transform_commands(tex: str, spec: dict[str, tuple[int, tuple[int, ...], str]],
                       optional: bool = True) -> str:
    """Replace each \\name in ``spec`` (name -> (n_args, kept arg indices, replacement)),
    with its star, its [..] optional arguments (if ``optional``) and its n_args brace
    groups, by ``replacement`` followed by the kept arguments (themselves transformed).
    A missing brace group ends the argument list."""
    pat = re.compile(r"\\([A-Za-z]+)\*?")
    out, i = [], 0
    while True:
        m = pat.search(tex, i)
        while m and m.group(1) not in spec:
            m = pat.search(tex, m.end())
        if not m:
            break
        n, keep, repl = spec[m.group(1)]
        out.append(tex[i:m.start()])
        j = _skip_optional(tex, m.end()) if optional else m.end()
        args = []
        for _ in range(n):
            k = _skip_ws(tex, j)
            if k < len(tex) and tex[k] == "{":
                try:
                    content, j = read_group(tex, k)
                except ValueError:
                    break
                args.append(content)
                if optional:
                    j = _skip_optional(tex, j) if len(args) < n else j
            else:
                break
        kept = " ".join(transform_commands(args[x], spec, optional) for x in keep if x < len(args))
        out.append(f" {repl} {kept} " if (repl or kept) else " ")
        i = j
    out.append(tex[i:])
    return "".join(out)


def environments(tex: str, names: tuple[str, ...]) -> list[tuple[int, int, str]]:
    """(start, end, body) of every \\begin{name}...\\end{name} for the given names
    (outermost only; same-name nesting is balanced)."""
    out = []
    pat = re.compile(r"\\(begin|end)\s*\{(" + "|".join(re.escape(n) for n in names) + r")\}")
    stack: list[tuple[str, int, int]] = []
    for m in pat.finditer(tex):
        if m.group(1) == "begin":
            stack.append((m.group(2), m.start(), m.end()))
        elif stack and stack[-1][0] == m.group(2):
            name, s, b = stack.pop()
            if not stack:
                out.append((s, m.end(), tex[b:m.start()]))
    return out


def remove_environments(tex: str, names: tuple[str, ...], repl=" ") -> str:
    """Replace each environment by ``repl`` (a string, or a function of its body)."""
    out, i = [], 0
    for s, e, body in environments(tex, names):
        out.append(tex[i:s])
        out.append(repl(body) if callable(repl) else repl)
        i = e
    out.append(tex[i:])
    return "".join(out)


def document_body(tex: str) -> str:
    """Text between \\begin{document} and \\end{document} (the whole text if absent)."""
    m = re.search(r"\\begin\s*\{document\}", tex)
    if not m:
        return tex
    e = re.search(r"\\end\s*\{document\}", tex[m.end():])
    return tex[m.end(): m.end() + e.start()] if e else tex[m.end():]


def abstract(tex: str) -> str | None:
    envs = environments(tex, ("abstract",))
    return envs[0][2] if envs else None


_SECTION_END = re.compile(r"\\end\s*\{document\}|\\begin\s*\{thebibliography\}|\\bibliography\s*\{"
                          r"|\\printbibliography|\\appendix\b")
_NEWCOMMAND = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\s*\\([A-Za-z]+)\s*\}?\s*(?:\[(\d)\])?\s*(?:\[[^\]]*\])?\s*\{")


def section_macros(tex: str) -> dict[str, int]:
    """User macros that issue a \\section (e.g. \\newcommand{\\SIsection}[2]{...\\section*{#1 #2}}),
    with their number of arguments."""
    out = {}
    for m in _NEWCOMMAND.finditer(tex):
        try:
            body, _ = read_group(tex, m.end() - 1)
        except ValueError:
            continue
        if re.search(r"\\section\b", body):
            out[m.group(1)] = int(m.group(2) or 0)
    return out


def sections(tex: str, macros: dict[str, int] | None = None) -> list[tuple[str, str]]:
    """(title, body) of every \\section / \\section*, and of every call of a sectioning
    macro in ``macros`` (name -> number of arguments; the title is their joined text).
    Subsections stay in the body. A body ends at the next section, the bibliography,
    \\appendix or \\end{document}."""
    macros = dict(macros or {})
    names = ["section"] + sorted(macros, key=len, reverse=True)
    pat = re.compile(r"\\(" + "|".join(names) + r")(?![A-Za-z])\*?\s*(?:\[[^\]]*\])?\s*(?=\{)")
    heads = []
    for m in pat.finditer(tex):
        n = macros.get(m.group(1), 1) or 1
        args, j = [], m.end()
        try:
            for _ in range(n):
                a, j = read_group(tex, _skip_ws(tex, j))
                args.append(a)
        except ValueError:
            continue
        heads.append((m.start(), j, " ".join(args)))
    out = []
    for k, (s, end, title) in enumerate(heads):
        stop = heads[k + 1][0] if k + 1 < len(heads) else len(tex)
        hard = _SECTION_END.search(tex, end, stop)
        out.append((plain_title(title), tex[end: hard.start() if hard else stop]))
    return out


def plain_title(title: str) -> str:
    t = re.sub(r"\\[A-Za-z]+\*?", " ", title)
    return re.sub(r"\s+", " ", t.replace("{", "").replace("}", "")).strip()


FLOATS = ("figure", "figure*", "table", "table*", "sidewaystable", "sidewaysfigure", "wrapfigure", "wraptable")

_CITE = re.compile(r"\\(?:no)?cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")


def cited_keys(tex: str) -> list[str]:
    """Citation keys in order of first use (comments must already be removed)."""
    out: list[str] = []
    for group in _CITE.findall(tex):
        for k in (s.strip() for s in group.split(",")):
            if k and k != "*" and k not in out:
                out.append(k)
    return out


def metric_keys(tex: str) -> list[str]:
    return re.findall(r"\\metric\s*\{([^{}]*)\}", tex)


def graphics(tex: str) -> list[str]:
    """Files named by \\includegraphics, in order."""
    return [g.strip() for g in re.findall(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}", tex)]


def graphics_paths(tex: str) -> list[str]:
    m = re.search(r"\\graphicspath\s*\{((?:\s*\{[^{}]*\})+)\s*\}", tex)
    return re.findall(r"\{([^{}]*)\}", m.group(1)) if m else []


def bib_resources(tex: str) -> list[str]:
    """.bib files named by \\bibliography{a,b} or \\addbibresource{x}."""
    out = []
    for g in re.findall(r"\\bibliography\s*\{([^{}]+)\}", tex):
        out += [s.strip() for s in g.split(",") if s.strip()]
    out += [s.strip() for s in re.findall(r"\\addbibresource\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}", tex)]
    return [s if s.endswith(".bib") else s + ".bib" for s in out]


def bib_keys(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text))


# ============================================================================ plain text

# Commands whose arguments are not running text: (n mandatory args, kept args, replacement).
# Citations print as reference numbers, which the ERL count excludes; cross-references
# and metrics print as one number each.
_NONTEXT = {
    **{c: (1, (), "") for c in ("label", "includegraphics", "input", "include", "url", "color", "pagecolor",
                                "hspace", "vspace", "cellcolor", "rowcolor", "columncolor", "graphicspath",
                                "bibliographystyle", "bibliography", "addbibresource", "printbibliography",
                                "usepackage", "documentclass", "linespread", "captionsetup", "hypersetup",
                                "raisebox", "scalebox", "arraycolsep", "pagestyle", "thispagestyle",
                                "nocite", "addcontentsline", "begin", "end")},
    **{c: (2, (), "") for c in ("setlength", "addtolength", "setcounter", "addtocounter", "renewcommand",
                                "newcommand", "providecommand", "fontsize", "resizebox", "rule",
                                "newenvironment", "renewenvironment")},
    "definecolor": (3, (), ""),
    "href": (2, (1,), ""),
    "textcolor": (2, (1,), ""),
    "multicolumn": (3, (2,), ""),
    "multirow": (3, (2,), ""),
    "parbox": (2, (1,), ""),
    "makebox": (1, (0,), ""),
    "textsuperscript": (1, (), ""),     # unit powers (km\textsuperscript{2})
    "textsubscript": (1, (), ""),
}
_CITE_CMDS = ("cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear", "citeyearpar",
              "Cite", "Citep", "Citet", "parencite", "textcite", "autocite", "footcite")
_REF_CMDS = ("ref", "eqref", "autoref", "cref", "Cref", "pageref", "nameref", "subref")

# Environments whose argument after \begin{..} is a column or width specification.
_SPEC_ENVS = {"tabular": 1, "tabular*": 2, "tabularx": 2, "longtable": 1, "array": 1, "minipage": 1,
              "tabulary": 2, "supertabular": 1}

_MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]", re.S)
_MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*",
              "eqnarray", "eqnarray*", "displaymath", "math")


def _drop_env_specs(tex: str) -> str:
    """Remove the column/width specification after \\begin{tabular} and similar."""
    pat = re.compile(r"\\begin\s*\{(" + "|".join(re.escape(e) for e in _SPEC_ENVS) + r")\}")
    out, i = [], 0
    for m in pat.finditer(tex):
        if m.start() < i:
            continue
        out.append(tex[i:m.start()])
        j = _skip_optional(tex, m.end())
        for _ in range(_SPEC_ENVS[m.group(1)]):
            k = _skip_ws(tex, j)
            if k < len(tex) and tex[k] == "{":
                _, j = read_group(tex, k)
                j = _skip_optional(tex, j)
        out.append(" ")
        i = j
    out.append(tex[i:])
    return "".join(out)


def to_plain(tex: str, mode: str = "numbers") -> str:
    """Rendered-text approximation of a LaTeX fragment (comments already removed).

    mode "numbers": \\metric{..} becomes the token METRICVAL; citations, cross-references,
      labels, file names, colours, lengths and table specifications are removed; in math,
      sub- and superscripts are removed (unit powers, symbol indices) and the rest kept;
      a thin space between digits joins them ("38\\,264" -> "38,264").
    mode "words": \\metric and \\ref-like commands become one word, citations nothing,
      each math group one word.
    """
    t = re.sub(r"\\metric\s*\{[^{}]*\}", " METRICVAL ", tex)
    t = _drop_env_specs(t)
    # math first, while \begin{equation} and its kin are still there
    if mode == "words":
        t = remove_environments(t, _MATH_ENVS, " MATH ")
        t = _MATH.sub(" MATH ", t)
    else:
        def strip_scripts(s: str) -> str:
            s = re.sub(r"[\^_]\s*\{[^{}]*\}", " ", s)
            s = re.sub(r"[\^_]\s*\\?[A-Za-z0-9+\-]", " ", s)
            for d in ("$", "\\(", "\\)", "\\[", "\\]"):
                s = s.replace(d, " ")
            return " " + s + " "
        t = remove_environments(t, _MATH_ENVS, strip_scripts)
        t = _MATH.sub(lambda m: strip_scripts(m.group(0)), t)
        t = re.sub(r"(?<=\d)\\[,;:!]\s*(?=\d)", ",", t)       # 38\,264 -> 38,264
    spec = dict(_NONTEXT)
    spec.update({c: (1, (), "") for c in _CITE_CMDS})
    spec.update({c: (1, (), "REFNUM" if mode == "words" else "") for c in _REF_CMDS})
    if mode == "words":
        spec["url"] = (1, (), "URL")
    t = transform_commands(t, spec)
    t = re.sub(r"\\\\\*?\s*(?:\[[^\]]*\])?", " ", t)          # line breaks with optional spacing
    t = re.sub(r"\\textminus\s*(\{\})?", "\u2212", t)
    t = t.replace("\\_", "_").replace("\\%", "%").replace("\\&", "&").replace("\\#", "#").replace("\\$", "$")
    t = re.sub(r"\\[,;:! ]", " ", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)                      # remaining command names
    t = t.replace("{", " ").replace("}", " ").replace("~", " ")
    if mode == "words":
        t = re.sub(r"-{2,}|\u2013|\u2014|&", " ", t)
    return re.sub(r"[ \t]+", " ", t)


# ============================================================================ word counts

_FRONT = {c: (1, (), "") for c in ("title", "author", "address", "ead", "date", "affiliation", "thanks",
                                   "caption", "submitto", "affil", "email", "orcid", "shorttitle")}
_FRONT["captionof"] = (2, (), "")


def word_count(tex: str) -> int:
    """Words in a LaTeX fragment as rendered (a math group, a metric or a cross-reference
    counts as one word; citations count as none)."""
    return sum(1 for w in to_plain(tex, "words").split() if re.search(r"[A-Za-z0-9]", w))


def main_text_for_count(tex: str) -> str:
    """The part of a manuscript ERL counts toward the Letter limit: the document body
    without title and author block, abstract, figures, tables (floats and bare tabulars),
    captions and the reference list (docs/journal_specs.md, ERL). Everything else counts,
    section headings, acknowledgements and statements included."""
    body = document_body(tex)
    body = remove_environments(body, ("abstract", "thebibliography", *FLOATS, "tabular", "tabular*", "tabularx",
                                      "longtable"))
    body = re.sub(r"\\maketitle\b", " ", body)
    return transform_commands(body, _FRONT)


# ============================================================================ number trace
#
# A numeral in the traced text passes only if it sits inside \metric{...} (it is then not
# a numeral of the source at all) or matches one of the explicit patterns below. Each
# pattern is a class the brief allows (section 15: "years, figure numbers, sample sizes
# defined in Methods"; the lead's list adds table, section and equation numbers and the
# D-/Amendment record numbers) or a name that merely contains digits. Nothing else is
# whitelisted: a design constant (a scenario depth, a nominal coverage, a percentile)
# is a number in the manuscript and goes through \metric like any result.

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan\\.|Feb\\.|Mar\\.|Apr\\.|Jun\\.|Jul\\.|Aug\\.|Sep\\.|Sept\\.|Oct\\.|Nov\\.|Dec\\.")
# not followed by a unit or a counted noun: such a numeral is a quantity, not a label
_UNIT_AFTER = r"(?!\s*(?:%|m\b|km\b|cm\b|mm\b|km2\b|m2\b|per\s*cent|percent|EECU|pixels?\b|scenes?\b))"
# a figure, table, section or equation number: 3, 3b, S2, (1), 2.3
_REF_NUM = r"\(?S?\d{1,3}(?:\.\d{1,2})*[a-z]?\)?(?![\d.])" + _UNIT_AFTER
_RANGE = r"(?:\s*(?:,|and|or|to|--|\u2013|-)\s*" + _REF_NUM + r")*"

WHITELIST: list[tuple[str, re.Pattern]] = [
    # cross-references written out: Figure 2b, Figs 2--4, Supplementary Table S1,
    # Section 2.3, Supplementary Note 4, Eq. (1), section 10 of the pre-registration, and
    # the numbered robustness checks of brief section 10 ("checks 1, 2 and 9")
    ("figure/table/section/equation/check number", re.compile(
        r"(?i)\b(?:Supplementary\s+|Suppl\.\s*)?(?:Figures?|Figs?\.?|Tables?|Sections?|Sects?\.?|Notes?|"
        r"Equations?|Eqs?\.?|(?:robustness\s+)?checks?)\s*" + _REF_NUM + _RANGE + r"|\u00a7\s*" + _REF_NUM + _RANGE)),
    # project record numbers: decisions D-105, PREREG Amendments 12 and 17
    ("D-/Amendment number", re.compile(r"\bD-\d+\b|(?i:\bAmendments?)\s+\d+(?:\s*(?:,|and|--|\u2013)\s*\d+)*")),
    # calendar: 1 January, January 1st, the 21st century, 1985--2026, the 1990s (a
    # four-digit 1800-2100 number that is not followed by a unit). Month names are
    # case-sensitive ("7 may differ" is not a date).
    ("date", re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:" + MONTHS + r")(?![A-Za-z])|(?:" + MONTHS
                        + r")\s+\d{1,2}(?:st|nd|rd|th)?\b(?!\d)")),
    ("century", re.compile(r"(?i)\b\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)\b")),
    ("year", re.compile(r"(?<![\d.,])(?:18\d\d|19\d\d|20\d\d|2100)s?(?![\d]|[.,]\d)" + _UNIT_AFTER)),
    # names that contain a numeral (products, sensors, datums, scenarios, versions)
    ("product, sensor or datum name", re.compile(
        r"\bLandsat\s+[4-9](?:\s*(?:,|and|or|/|--|\u2013|-)\s*[4-9])*\b|\bSentinel-[1-6][A-D]?\b|\bICESat-2\b"
        r"|\bGLO-(?:30|90)\b|\bWorldDEM-30\b|\bBHS-77\b|\bNAVD\s*88\b|\bOLI-2\b|\bTIRS-2\b|\bEIGEN-6C4\b"
        r"|\bCollection\s+2\b|\bLevel-[123]\b|\bTier\s+[12]\b|\b[Vv]ersion\s+\d+(?:\.\d+)*\b|\b3D\s+Elevation\s+Program\b"
        r"|\b3DEP\b"
        r"|\bSSP[1-5]-\d\.\d\b|\bRCP\s*\d\.\d\b")),
]

# Sample sizes defined in Methods: (value, noun pattern, word form, what it is). A numeral
# equal to the value, followed within two words by the noun ("7 coastal sectors") or
# closing a parenthesis after it ("Sectors mispredicted (of 7)"), passes only if the
# Methods text defines that sample size (``defined_sample_sizes``).
SAMPLE_SIZES: list[tuple[int, str, str, str]] = [
    (7, r"sectors?", "seven", "coastal sectors of the North Caspian study area"),
    (5, r"(?:products?|grids?)", "five",
     "global bathymetry products (grids) audited: GEBCO_2026, GEBCO_2022, ETOPO 2022, SRTM15+, ETOPO1"),
    (2, r"arms?", "two", "arms of Great Salt Lake (generality test)"),
]

NUMERAL = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:,\d{3})*(?:\.\d+)?")


def _sample_patterns(value: int, noun: str) -> re.Pattern:
    return re.compile(rf"(?i)(?<![\d.,])\b{value}\s+(?:[A-Za-z\-]+\s+){{0,2}}?{noun}\b"
                      rf"|\b{noun}\b[^.;:()]{{0,60}}\(\s*of\s+{value}\s*\)")


def defined_sample_sizes(methods_plain: str, sizes=None) -> list[tuple[int, str, str, str]]:
    """The sample sizes whose definition appears in the Methods text (plain, numbers
    mode): the value as digits or as a word, or a metric, within two words of the noun."""
    out = []
    for value, noun, word, what in (SAMPLE_SIZES if sizes is None else sizes):
        pat = re.compile(rf"(?i)\b(?:{value}|{word}|METRICVAL)\s+(?:[A-Za-z\-]+\s+){{0,2}}?{noun}\b")
        if pat.search(methods_plain):
            out.append((value, noun, word, what))
    return out


def untraced_numerals(tex: str, sample_sizes=()) -> list[tuple[str, str]]:
    """Numerals in a LaTeX fragment that are neither inside \\metric{} nor whitelisted:
    a list of (numeral, context). ``sample_sizes`` are the SAMPLE_SIZES entries that
    Methods defines (``defined_sample_sizes``)."""
    t = to_plain(tex, "numbers")

    def blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    for _, pat in WHITELIST:
        t = pat.sub(blank, t)
    for value, noun, _, _ in sample_sizes:
        t = _sample_patterns(value, noun).sub(lambda m: re.sub(rf"\b{value}\b", " " * len(str(value)), m.group(0)), t)
    out = []
    for m in NUMERAL.finditer(t):
        ctx = re.sub(r"\s+", " ", t[max(0, m.start() - 40): m.end() + 30]).strip()
        out.append((m.group(0), ctx))
    return out


# ============================================================================ colour maps

# Rainbow-type maps (CLAUDE.md: never jet or rainbow) and every other matplotlib built-in:
# CLAUDE.md asks for cmocean or Crameri maps. A ListedColormap of explicit colours is not a
# named map and passes; cmocean ("cmo.*") and Crameri ("cmc.*") registered names pass.
RAINBOW = {"jet", "rainbow", "gist_rainbow", "hsv", "nipy_spectral", "gist_ncar", "turbo", "prism", "flag",
           "Spectral", "spectral", "gist_stern", "CMRmap", "gist_earth", "terrain", "brg"}


def cmap_uses(source: str) -> list[tuple[int, str, str]]:
    """(line, name, how) of every matplotlib colour map named in Python source: string
    literals given as cmap=..., to get_cmap(...) or colormaps[...]; attributes of
    matplotlib's cm module (plt.cm.X, matplotlib.cm.X, mpl.cm.X, or cm.X when cm is
    matplotlib's). cmocean.cm.X and cmcrameri.cm.X are not matplotlib names."""
    cm_is_mpl = bool(re.search(r"^\s*(?:from\s+matplotlib\s+import\s+(?:[\w\s,]*\b)?cm\b|import\s+matplotlib\.cm\s+as\s+cm\b)",
                               source, re.M))
    found = []
    pats = [
        (re.compile(r"""\bcmap\s*=\s*['"]([A-Za-z0-9_.]+)['"]"""), "cmap="),
        (re.compile(r"""\b(?:get_cmap|colormaps\.get_cmap|colormaps\.get)\s*\(\s*['"]([A-Za-z0-9_.]+)['"]"""), "get_cmap"),
        (re.compile(r"""\bcolormaps\s*\[\s*['"]([A-Za-z0-9_.]+)['"]\s*\]"""), "colormaps[]"),
        (re.compile(r"(?<![\w.])(?:plt|pyplot|matplotlib|mpl)\.cm\.([A-Za-z_][A-Za-z0-9_]*)"), "cm attribute"),
        (re.compile(r"(?<![\w.])plt\.(jet|rainbow|hsv|nipy_spectral|gist_rainbow|turbo)\s*\("), "pyplot function"),
    ]
    if cm_is_mpl:
        pats.append((re.compile(r"(?<![\w.])cm\.([A-Za-z_][A-Za-z0-9_]*)"), "cm attribute"))
    for ln, line in enumerate(source.splitlines(), 1):
        code = line.split("#", 1)[0]
        for pat, how in pats:
            for m in pat.finditer(code):
                found.append((ln, m.group(1), how))
    return found


def cmap_problems(source: str, mpl_names: set[str]) -> list[str]:
    """Problems with the colour maps named in ``source``; ``mpl_names`` is the set of
    names registered with matplotlib (only names in it, or rainbow names, are maps)."""
    out = []
    for ln, name, how in cmap_uses(source):
        base = name[:-2] if name.endswith("_r") else name
        if base in RAINBOW:
            out.append(f"line {ln}: rainbow-type colour map {name!r} ({how})")
        elif name.startswith(("cmo.", "cmc.")):
            continue
        elif name in mpl_names:
            out.append(f"line {ln}: matplotlib colour map {name!r} is not cmocean or Crameri ({how})")
    return out


# ============================================================================ rebuilt tables

REPRO_REL_TOL = 1e-9    # relative tolerance for a rebuilt number (floating-point summation order)
REPRO_ABS_TOL = 1e-12   # absolute floor, for values at or near zero


def _num_eq(a: float, b: float, rel: float, abs_: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def compare_json(a, b, rel: float = REPRO_REL_TOL, abs_: float = REPRO_ABS_TOL, path: str = "") -> list[str]:
    """Paths at which two JSON values differ: numbers beyond the tolerance (an int and a
    float compare as numbers), anything else exactly (booleans are not numbers)."""
    num = (int, float)
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if a == b and type(a) is type(b) else [path or "/"]
    if isinstance(a, num) and isinstance(b, num):
        return [] if _num_eq(float(a), float(b), rel, abs_) else [f"{path or '/'}: {a!r} != {b!r}"]
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one" for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            out += compare_json(a[k], b[k], rel, abs_, f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path or '/'}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += compare_json(x, y, rel, abs_, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path or '/'}: {a!r} != {b!r}"]


# Run records (run.json, summary.json and the like) carry metadata of the run itself that a
# rebuild cannot reproduce: when it ran, how long it took and where the files were (D-133).
RUN_META_KEYS = frozenset({"created", "seconds", "elapsed_s", "wall_s", "started", "finished"})
_ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|/(?:tmp|home|Users|var|mnt|d|c)/)")


def strip_run_metadata(x):
    """x without run metadata: keys in RUN_META_KEYS are dropped and absolute paths
    become "<abs path>". Every other key and value is kept as it is."""
    if isinstance(x, dict):
        return {k: strip_run_metadata(v) for k, v in x.items() if k not in RUN_META_KEYS}
    if isinstance(x, list):
        return [strip_run_metadata(v) for v in x]
    if isinstance(x, str) and _ABS_PATH.match(x):
        return "<abs path>"
    return x


def compare_rebuilt_json(committed, rebuilt, rel: float = REPRO_REL_TOL,
                         abs_: float = REPRO_ABS_TOL) -> tuple[list[str], list[str]]:
    """(differences, added) between a committed JSON result and its rebuild, after
    strip_run_metadata. A key present only in the rebuild is "added" (a field the code
    has written since the committed run); every other difference, including a key
    missing from the rebuild, is a difference."""
    diffs: list[str] = []
    added: list[str] = []

    def walk(a, b, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            added.extend(f"{path}/{k}" for k in sorted(set(b) - set(a)))
            diffs.extend(f"{path}/{k}: missing from the rebuild" for k in sorted(set(a) - set(b)))
            for k in sorted(set(a) & set(b)):
                walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        else:
            diffs.extend(compare_json(a, b, rel, abs_, path))

    walk(strip_run_metadata(committed), strip_run_metadata(rebuilt), "")
    return diffs, added


def _as_number(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def compare_csv(a: str, b: str, rel: float = REPRO_REL_TOL, abs_: float = REPRO_ABS_TOL) -> list[str]:
    """Cells at which two CSV texts differ: numeric cells beyond the tolerance, others
    exactly; a different shape is one difference."""
    ra = list(csv.reader(io.StringIO(a)))
    rb = list(csv.reader(io.StringIO(b)))
    if len(ra) != len(rb):
        return [f"{len(ra)} rows != {len(rb)}"]
    out = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        if len(x) != len(y):
            out.append(f"row {i}: {len(x)} cells != {len(y)}")
            continue
        for j, (u, v) in enumerate(zip(x, y)):
            if u == v:
                continue
            fu, fv = _as_number(u), _as_number(v)
            if fu is None or fv is None or not _num_eq(fu, fv, rel, abs_):
                out.append(f"row {i} col {j}: {u!r} != {v!r}")
    return out


# ============================================================================ internal review

def md_sections(md: str) -> list[tuple[int, str, str]]:
    """(level, heading, body) of every markdown heading; a body runs to the next heading
    of the same or a higher level."""
    heads = [(m.start(), m.end(), len(m.group(1)), m.group(2).strip())
             for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*#*\s*$", md, re.M)]
    out = []
    for k, (s, e, lev, title) in enumerate(heads):
        stop = len(md)
        for s2, _, lev2, _ in heads[k + 1:]:
            if lev2 <= lev:
                stop = s2
                break
        out.append((lev, title, md[e:stop]))
    return out


def _items(body: str, level: int) -> list[str]:
    """Comments in a section body: its sub-headings if it has any, else its top-level list
    items (indent under two spaces), each with the text up to the next item."""
    subs = [(t, b) for lev, t, b in md_sections(body) if lev > level]
    if subs:
        top = min(lev for lev, _, _ in md_sections(body))
        return [f"{t}\n{b}".strip() for lev, t, b in md_sections(body) if lev == top]
    starts = [m.start() for m in re.finditer(r"^ ?(?:\d+[.)]|[-*+])\s+\S", body, re.M)]
    return [body[s: (starts[i + 1] if i + 1 < len(starts) else len(body))].strip() for i, s in enumerate(starts)]


def review_comments(md: str) -> dict[str, list[str]]:
    """The major and minor comments of a review report: the items under the first heading
    containing "major" and the first containing "minor"."""
    out: dict[str, list[str]] = {"major": [], "minor": []}
    for lev, title, body in md_sections(md):
        for kind in ("major", "minor"):
            if kind in title.lower() and not out[kind]:
                out[kind] = _items(body, lev)
    return out


def review_expertise(md: str) -> str:
    """The text of the review's "Expertise:" line (within its first 30 lines)."""
    for line in md.splitlines()[:30]:
        m = re.match(r"^\W*Expertise\W*:\s*(.+?)\s*$", line, re.I)
        if m:
            return m.group(1).strip("* ")
    return ""


def response_entries(md: str) -> dict[str, str]:
    """Response entries by comment ID (R<n>.M<k>): a line or heading that starts with the
    ID, and the text up to the next ID or heading."""
    pat = re.compile(r"^(?:#{1,6}\s*|\W{0,4})(R\d+\.M\d+)\b(.*)$", re.M)
    ms = list(pat.finditer(md))
    out = {}
    for i, m in enumerate(ms):
        stop = ms[i + 1].start() if i + 1 < len(ms) else len(md)
        nxt = re.search(r"^#{1,6}\s", md[m.end():stop], re.M)
        end = m.end() + nxt.start() if nxt else stop
        out[m.group(1)] = (m.group(2) + md[m.end():end]).strip()
    return out


# ============================================================================ package list

def listed_paths(md: str) -> list[str]:
    """Repository paths listed in a markdown file: back-ticked tokens with a slash or a
    file extension (no spaces, no URLs)."""
    out = []
    for tok in re.findall(r"`([^`\s]+)`", md):
        if "://" in tok or tok.startswith(("-", "$")):
            continue
        if "/" in tok or re.search(r"\.[A-Za-z0-9]{1,5}$", tok):
            if tok not in out:
                out.append(tok)
    return out
