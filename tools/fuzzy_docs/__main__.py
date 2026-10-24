"""python -m tools.fuzzy_docs — build the fuzzy-logic documentation (PDF + DOCX + HTML)."""
import argparse
import os
import sys
from datetime import date

from tools.fuzzy_docs.content import build
from tools.fuzzy_docs.render import html_to_pdf, to_docx, to_html

TITLE = "Нечітка логіка системи безпеки IoT"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the fuzzy-logic documentation")
    parser.add_argument("--out", default=os.path.join("reports", f"fuzzy_docs_{date.today():%Y-%m-%d}"))
    parser.add_argument("--fuzzy", default="config/fuzzy.yaml")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    blocks = build(args.out, args.fuzzy)
    html_path = os.path.join(args.out, "fuzzy_logic.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(to_html(blocks, TITLE))
    pdf_path = os.path.join(args.out, "fuzzy_logic.pdf")
    pdf = html_to_pdf(html_path, pdf_path)
    docx_path = to_docx(blocks, os.path.join(args.out, "fuzzy_logic.docx"), TITLE)
    figures = sum(1 for b in blocks if b[0] == "img")
    print(f"{len([b for b in blocks if b[0] == 'h1'])} sections, {figures} figures, "
          f"{sum(len(b[2]) for b in blocks if b[0] == 'table')} table rows")
    print(f"→ {pdf_path if pdf else '(no Chrome: PDF skipped)'}\n→ {docx_path}\n→ {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
