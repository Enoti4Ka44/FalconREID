# -*- coding: utf-8 -*-
"""Сборка PDF сопроводительной документации из docs/SOLUTION.md.

    python docs/build_doc_pdf.py            # -> docs/ФальконТех_документация.pdf

Markdown → HTML (python-markdown) → PDF (Chromium через Playwright).
"""
import base64
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "SOLUTION.md"
OUT = DOCS / "ФальконТех_документация.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #1C1D22; }
h1 { color: #310F53; font-size: 20pt; border-bottom: 3px solid #FF0053; padding-bottom: 6px; }
h2 { color: #520978; font-size: 14pt; margin-top: 22px; page-break-after: avoid; }
h3 { color: #310F53; font-size: 12pt; page-break-after: avoid; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 9pt;
       background: #F4F0F8; padding: 1px 4px; border-radius: 3px; }
pre { background: #F4F0F8; padding: 10px; border-radius: 6px; font-size: 8.3pt;
      line-height: 1.25; overflow: hidden; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt;
        page-break-inside: avoid; }
th { background: #310F53; color: white; text-align: left; padding: 5px 7px; }
img { max-width: 100%; }
td { border-bottom: 1px solid #E4E0EC; padding: 4px 7px; vertical-align: top; }
.meta { color: #6A6280; font-size: 9pt; margin-bottom: 14px; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    # картинки встраиваются data-URI: страница собирается через set_content,
    # и относительные пути / file:// из неё недоступны
    body = re.sub(r'src="([^":]+)"',
                  lambda m: 'src="data:image/png;base64,'
                            + base64.b64encode((DOCS / m.group(1)).read_bytes()).decode() + '"',
                  body)
    meta = ('<div class="meta">Репозиторий: https://github.com/Enoti4Ka44/FalconREID · '
            'запуск и полный список внешних ресурсов — README.md</div>')
    body = body.replace("</h1>", "</h1>" + meta, 1)
    html = f"<!doctype html><html lang='ru'><meta charset='utf-8'><style>{CSS}</style><body>{body}</body></html>"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        pg = b.new_page()
        pg.set_content(html, wait_until="load")
        pg.pdf(path=str(OUT), format="A4", print_background=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>"
                               "ФАЛЬКОН·ReID — сопроводительная документация · "
                               "<span class='pageNumber'></span>/<span class='totalPages'></span></div>",
               margin={"top": "16mm", "bottom": "18mm", "left": "14mm", "right": "14mm"})
        b.close()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
