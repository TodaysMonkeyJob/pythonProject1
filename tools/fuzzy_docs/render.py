"""
Document model → HTML (printed to PDF by headless Chrome) and DOCX (Office Open XML written by hand, stdlib only).

A document is a list of blocks (tuples):

* ``("h1" | "h2" | "h3", text)``, ``("p", text)``, ``("note", text)``, ``("formula", text)``,
  ``("ul", [text, …])``, ``("ol", [text, …])``, ``("table", header, rows, widths_or_None)``,
  ``("img", path, caption, width_cm)``, ``("pagebreak",)``, ``("toc",)``.

Inline markup in text: ``**bold**`` and ``` `code` ```.
"""
import base64
import html
import os
import re
import shutil
import subprocess
import zipfile
from typing import Any, List, Sequence, Tuple

Block = Tuple[Any, ...]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Arial", "Helvetica", sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1d1d1f; }
h1 { font-size: 19pt; color: #1f3b73; border-bottom: 2px solid #1f3b73; padding-bottom: 3px; margin-top: 0;
     page-break-before: always; }
h1.first { page-break-before: avoid; }
h2 { font-size: 14pt; color: #1f3b73; margin-top: 18px; }
h3 { font-size: 11.5pt; color: #333; margin-top: 14px; }
p { margin: 5px 0; text-align: justify; }
.note { background: #eef4ff; border-left: 4px solid #3a6fd8; padding: 6px 10px; margin: 8px 0; }
.formula { font-family: "Cambria Math", "STIX Two Math", "Times New Roman", serif; font-size: 11.5pt;
           background: #f6f6f6; border: 1px solid #ddd; padding: 6px 10px; margin: 8px 0; white-space: pre-wrap; }
code { font-family: Menlo, Consolas, monospace; font-size: 9pt; background: #f2f2f2; padding: 0 3px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; page-break-inside: auto; }
th { background: #1f3b73; color: #fff; text-align: left; padding: 4px 5px; }
td { border-bottom: 1px solid #ddd; padding: 3px 5px; vertical-align: top; }
tr { page-break-inside: avoid; }
figure { margin: 10px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9pt; color: #555; margin-top: 3px; }
.title { text-align: center; margin-top: 70mm; }
.title h1 { border: none; font-size: 26pt; page-break-before: avoid; }
.title p { text-align: center; font-size: 12pt; color: #444; }
.toc li { margin: 2px 0; }
.pb { page-break-after: always; }
"""


def inline_html(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)


def to_html(blocks: Sequence[Block], title: str, embed: bool = True) -> str:
    parts = [f"<!doctype html><html lang='uk'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body>"]
    headings = [(b[0], b[1]) for b in blocks if b[0] in ("h1", "h2")]
    first = True
    n = {"h1": 0, "h2": 0}
    figure = 0
    for block in blocks:
        kind = block[0]
        if kind == "title":
            parts.append("<div class='title'>" + f"<h1>{inline_html(block[1])}</h1>"
                         + "".join(f"<p>{inline_html(line)}</p>" for line in block[2]) + "</div><div class='pb'></div>")
        elif kind == "toc":
            items, c1, c2 = [], 0, 0
            for level, text in headings:
                if level == "h1":
                    c1, c2 = c1 + 1, 0
                    items.append(f"<li><b>{c1}. {inline_html(text)}</b></li>")
                else:
                    c2 += 1
                    items.append(f"<li style='margin-left:18px'>{c1}.{c2}. {inline_html(text)}</li>")
            parts.append("<h1 class='first'>Зміст</h1><ul class='toc' style='list-style:none;padding-left:0'>"
                         + "".join(items) + "</ul>")
            first = False
        elif kind == "h1":
            n["h1"] += 1
            n["h2"] = 0
            parts.append(f"<h1{' class=\"first\"' if first else ''}>{n['h1']}. {inline_html(block[1])}</h1>")
            first = False
        elif kind == "h2":
            n["h2"] += 1
            parts.append(f"<h2>{n['h1']}.{n['h2']}. {inline_html(block[1])}</h2>")
        elif kind == "h3":
            parts.append(f"<h3>{inline_html(block[1])}</h3>")
        elif kind == "p":
            parts.append(f"<p>{inline_html(block[1])}</p>")
        elif kind == "note":
            parts.append(f"<div class='note'>{inline_html(block[1])}</div>")
        elif kind == "formula":
            parts.append(f"<div class='formula'>{html.escape(block[1])}</div>")
        elif kind in ("ul", "ol"):
            parts.append(f"<{kind}>" + "".join(f"<li>{inline_html(i)}</li>" for i in block[1]) + f"</{kind}>")
        elif kind == "table":
            header, rows = block[1], block[2]
            parts.append("<table><thead><tr>" + "".join(f"<th>{inline_html(str(h))}</th>" for h in header)
                         + "</tr></thead><tbody>"
                         + "".join("<tr>" + "".join(f"<td>{inline_html(str(c))}</td>" for c in row) + "</tr>"
                                   for row in rows) + "</tbody></table>")
        elif kind == "img":
            figure += 1
            path, caption, width = block[1], block[2], block[3]
            if embed:
                with open(path, "rb") as f:
                    src = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            else:
                src = os.path.relpath(path)
            parts.append(f"<figure><img src='{src}' style='width:{width}cm'><figcaption>Рис. {figure}. "
                         f"{inline_html(caption)}</figcaption></figure>")
        elif kind == "pagebreak":
            parts.append("<div class='pb'></div>")
    parts.append("</body></html>")
    return "".join(parts)


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """:return: False when Chrome is not available"""
    if not os.path.isfile(CHROME):
        return False
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={os.path.abspath(pdf_path)}", "file://" + os.path.abspath(html_path)],
                   capture_output=True, timeout=300, check=False)
    return os.path.isfile(pdf_path)


# --- DOCX (Office Open XML) ---------------------------------------------------------------------------------------

def _x(text: str) -> str:
    return html.escape(text, quote=False)


def _runs(text: str, mono: bool = False, size: int = 0) -> str:
    """``**bold**`` / ``` `code` ``` → runs"""
    out = []
    for token in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not token:
            continue
        props = []
        if token.startswith("**"):
            token, props = token[2:-2], ["<w:b/>"]
        elif token.startswith("`"):
            token, props = token[1:-1], ['<w:rFonts w:ascii="Menlo" w:hAnsi="Menlo" w:cs="Menlo"/>',
                                         '<w:sz w:val="17"/>']
        if mono:
            props.append('<w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Cambria Math"/>')
        if size:
            props.append(f'<w:sz w:val="{size}"/>')
        rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
        lines = token.split("\n")
        for i, line in enumerate(lines):
            out.append(f'<w:r>{rpr}{"<w:br/>" if i else ""}<w:t xml:space="preserve">{_x(line)}</w:t></w:r>')
    return "".join(out)


def _p(text: str, style: str = "", mono: bool = False, extra_ppr: str = "", size: int = 0) -> str:
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + extra_ppr
    return f"<w:p><w:pPr>{ppr}</w:pPr>{_runs(text, mono, size)}</w:p>"


def _image(rid: str, idx: int, width_cm: float, size_px: Tuple[int, int], caption: str) -> str:
    cx = int(width_cm * 360000)
    cy = int(cx * size_px[1] / max(1, size_px[0]))
    drawing = (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
               f'<wp:docPr id="{idx}" name="Picture {idx}"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/'
               f'drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
               f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
               f'<pic:cNvPr id="{idx}" name="fig{idx}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill>'
               f'<a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr>'
               f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect">'
               f'<a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>'
               f'</w:r>')
    return (f'<w:p><w:pPr><w:jc w:val="center"/><w:keepNext/></w:pPr>{drawing}</w:p>'
            + _p(f"Рис. {idx}. {caption}", "Caption"))


def _png_size(path: str) -> Tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def _table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> str:
    def cell(text: str, head: bool) -> str:
        shade = '<w:shd w:val="clear" w:color="auto" w:fill="1F3B73"/>' if head else ""
        run_text = f"**{text}**" if head else text
        body = _runs(run_text, size=17)
        if head:
            body = body.replace("<w:rPr>", '<w:rPr><w:color w:val="FFFFFF"/>')
        return f"<w:tc><w:tcPr>{shade}</w:tcPr><w:p><w:pPr><w:spacing w:after=\"0\"/></w:pPr>{body}</w:p></w:tc>"
    borders = ('<w:tblBorders>' + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="BBBBBB"/>'
                                         for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
               + '</w:tblBorders>')
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>{borders}</w:tblPr>',
           "<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(cell(str(h), True) for h in header) + "</w:tr>"]
    for row in rows:
        out.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(cell(str(c), False) for c in row) + "</w:tr>")
    out.append("</w:tbl>")
    return "".join(out) + _p("")


STYLES_PATH = os.path.join(os.path.dirname(__file__), "styles.xml")    # Word paragraph styles (headings, notes …)


def to_docx(blocks: Sequence[Block], path: str, title: str) -> str:
    body: List[str] = []
    media: List[Tuple[str, str]] = []
    n1 = n2 = figure = 0
    headings = [(b[0], b[1]) for b in blocks if b[0] in ("h1", "h2")]
    for block in blocks:
        kind = block[0]
        if kind == "title":
            body.append(_p(block[1], "Title"))
            body += [_p(line, "Subtitle") for line in block[2]]
        elif kind == "toc":
            body.append(_p("Зміст", "Heading1"))
            c1 = c2 = 0
            for level, text in headings:
                if level == "h1":
                    c1, c2 = c1 + 1, 0
                    body.append(_p(f"**{c1}. {text}**", extra_ppr='<w:spacing w:after="20"/>'))
                else:
                    c2 += 1
                    body.append(_p(f"{c1}.{c2}. {text}", extra_ppr='<w:ind w:left="400"/><w:spacing w:after="20"/>'))
        elif kind == "h1":
            n1, n2 = n1 + 1, 0
            body.append(_p(f"{n1}. {block[1]}", "Heading1"))
        elif kind == "h2":
            n2 += 1
            body.append(_p(f"{n1}.{n2}. {block[1]}", "Heading2"))
        elif kind == "h3":
            body.append(_p(block[1], "Heading3"))
        elif kind == "p":
            body.append(_p(block[1]))
        elif kind == "note":
            body.append(_p(block[1], "Note"))
        elif kind == "formula":
            body.append(_p(block[1], "Formula", mono=True))
        elif kind in ("ul", "ol"):
            for i, item in enumerate(block[1], 1):
                body.append(_p(f"{'•' if kind == 'ul' else f'{i}.'}\t{item}", "ListParagraph"))
        elif kind == "table":
            body.append(_table(block[1], block[2]))
        elif kind == "img":
            figure += 1
            rid = f"rIdImg{figure}"
            media.append((rid, block[1]))
            body.append(_image(rid, figure, min(16.0, float(block[3])), _png_size(block[1]), block[2]))
        elif kind == "pagebreak":
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
    ns = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
          'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
          'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
          'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')
    sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1020" w:right="907" w:bottom="1020" '
            'w:left="907" w:header="500" w:footer="500" w:gutter="0"/></w:sectPr>')
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {ns}><w:body>'
                + "".join(body) + sect + "</w:body></w:document>")
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats'
            '.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            + "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                      f'relationships/image" Target="media/{os.path.basename(p)}"/>' for rid, p in media)
            + "</Relationships>")
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.'
                     'openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application'
                     '/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application'
                     '/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" '
                     'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.'
                     'wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/'
                     'vnd.openxmlformats-package.core-properties+xml"/></Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.'
                 'openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
                 'openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/'
                 'core-properties" Target="docProps/core.xml"/></Relationships>')
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.'
            'openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            f'<dc:title>{_x(title)}</dc:title><dc:language>uk-UA</dc:language></cp:coreProperties>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("word/document.xml", document)
        z.write(STYLES_PATH, "word/styles.xml")
        z.writestr("word/_rels/document.xml.rels", rels)
        for _, p in media:
            z.write(p, f"word/media/{os.path.basename(p)}")
    return path


def copy_figures(paths: Sequence[str], target: str) -> None:
    os.makedirs(target, exist_ok=True)
    for p in paths:
        shutil.copy(p, target)
