"""Markdown to PDF for the supplementary files that are kept as Markdown (the pre-registration
record, manuscript/supplementary_prereg.md), with PyMuPDF only (no pandoc, no browser).

It reads the subset of Markdown those files use: headings, paragraphs, bold, italic, inline
code, links, bullet and numbered lists (nested by indentation), pipe tables, block quotes,
fenced code and rules. The text is not changed: every character of the source is carried into
the PDF (checked: the words of the PDF are the words of the source, in order).

Usage: python tools/md_to_pdf.py <in.md> <out.pdf> [--title "..."]
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.35; color: #111; }
h1 { font-size: 17pt; margin-top: 14pt; margin-bottom: 6pt; }
h2 { font-size: 13.5pt; margin-top: 13pt; margin-bottom: 5pt; }
h3 { font-size: 11.5pt; margin-top: 11pt; margin-bottom: 4pt; }
h4 { font-size: 10pt; margin-top: 9pt; margin-bottom: 3pt; }
p { margin-top: 0pt; margin-bottom: 6pt; }
li { margin-bottom: 2pt; }
code, pre { font-family: monospace; font-size: 8.5pt; }
pre { margin-bottom: 6pt; }
blockquote { margin-left: 14pt; color: #333; }
table { border-collapse: collapse; margin-bottom: 8pt; }
th, td { border: 0.5pt solid #999; padding: 2pt 5pt; font-size: 8.5pt; text-align: left; }
th { font-weight: bold; }
hr { margin-top: 8pt; margin-bottom: 8pt; }
a { color: #1a4f8b; }
"""


def inline(s: str) -> str:
    """Inline Markdown of one run of text to HTML. Code spans are taken out first, so that
    nothing inside them is read as emphasis."""
    codes: list[str] = []

    def stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", stash, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: "<code>" + html.escape(codes[int(m.group(1))], quote=False) + "</code>", s)


LIST = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")


def to_html(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    stack: list[tuple[int, str]] = []          # open lists: (indent, tag)
    para: list[str] = []
    i = 0

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
            para.clear()

    def close_lists(to_indent: int = -1):
        while stack and stack[-1][0] > to_indent:
            out.append(f"</li></{stack.pop()[1]}>")

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush_para(); close_lists()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j]), quote=False) + "</pre>")
            i = j + 1
            continue
        if not line.strip():
            flush_para()
            if stack and i + 1 < len(lines) and not LIST.match(lines[i + 1]) and not lines[i + 1].startswith("  "):
                close_lists()
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush_para(); close_lists()
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2).strip())}</h{n}>")
            i += 1
            continue
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", line):
            flush_para(); close_lists()
            out.append("<hr/>")
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush_para(); close_lists()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for k, r in enumerate(rows) if not (k == 1 and all(re.fullmatch(r":?-+:?", c) for c in r))]
            head = len(rows) > 1 and len(body) < len(rows)
            t = ["<table>"]
            for k, r in enumerate(body):
                tag = "th" if head and k == 0 else "td"
                t.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in r) + "</tr>")
            out.append("".join(t) + "</table>")
            continue
        m = LIST.match(line)
        if m:
            flush_para()
            indent, marker, text = len(m.group(1)), m.group(2), m.group(3)
            tag = "ol" if marker[0].isdigit() else "ul"
            if stack and indent < stack[-1][0]:
                close_lists(indent)
            if stack and indent == stack[-1][0]:
                out.append("</li>")
                if stack[-1][1] != tag:
                    out.append(f"</{stack.pop()[1]}>")
                    stack.append((indent, tag)); out.append(f"<{tag}>")
            else:
                stack.append((indent, tag)); out.append(f"<{tag}>")
            out.append("<li>" + inline(text))
            i += 1
            continue
        if line.startswith(">"):
            flush_para(); close_lists()
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i][1:].strip())
                i += 1
            out.append("<blockquote>" + "".join("<p>" + inline(x) + "</p>" for x in " ".join(q).split("  ") if x) + "</blockquote>")
            continue
        if stack and line.startswith("  "):          # continuation line of a list item
            out.append(" " + inline(line.strip()))
            i += 1
            continue
        close_lists()
        para.append(line)
        i += 1
    flush_para(); close_lists()
    return "\n".join(out)


def words(text: str) -> list[str]:
    """Letters and digits only, lower case: what survives both Markdown and PDF extraction.
    Ligatures and accented letters are first reduced to their base letters (NFKD), and all the
    letters and digits are then joined, because a PDF breaks words where a line breaks."""
    import unicodedata

    t = unicodedata.normalize("NFKD", text)
    return re.findall(r"[0-9a-z]", t.lower())


def convert(src: Path, dst: Path, title: str | None = None) -> dict:
    import fitz

    md = src.read_text(encoding="utf-8")
    story = fitz.Story(html=f"<html><body>{to_html(md)}</body></html>", user_css=CSS)
    page = fitz.paper_rect("a4")
    where = page + (48, 48, -48, -54)
    writer = fitz.DocumentWriter(str(dst))
    n, more = 0, True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        n += 1
    writer.close()
    with fitz.open(dst) as d:
        got = words(" ".join(p.get_text() for p in d))
    if title:
        with fitz.open(dst) as d:
            d.set_metadata({"title": title})
            d.saveIncr()
    # every word of the source, in order; Markdown marks and URLs inside links aside
    plain = re.sub(r"\]\([^)\s]+\)", "]", md)
    want = words(plain)
    k = next((k for k, (a, b) in enumerate(zip(want, got)) if a != b), None)
    return {"pages": n, "source_words": len(want), "pdf_words": len(got), "same_words": want == got,
            "first_difference": None if k is None else (k, "".join(want[max(0, k - 25):k + 25]), "".join(got[max(0, k - 25):k + 25]))}


def main(argv) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    title = argv[argv.index("--title") + 1] if "--title" in argv else None
    rep = convert(Path(argv[0]), Path(argv[1]), title)
    print(f"{argv[1]}: {rep['pages']} pages, {rep['pdf_words']} letters and digits"
          + ("; they are those of the source, in order" if rep["same_words"] else ""))
    if not rep["same_words"]:
        print(f"FAIL: the PDF's text differs from the source ({rep['source_words']} letters and digits); first "
              f"difference (index, source, pdf): {rep['first_difference']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
