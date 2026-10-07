"""Generate the four demo PDFs into demo_docs/.

D1 annual_report.pdf     text + financial table + quarterly revenue bar chart (no bar labels)
D2 operations.pdf        plant operations table (Indian digit grouping)
D3 incident_memo.pdf     image-only scanned memo (noise + skew)
D4 press_release.pdf     text with a planted conflict: revenue ₹1,480 crore vs ₹1,450 crore in D1
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo_docs"

A4 = fitz.paper_rect("a4")
MARGIN = 56


def _new_page(doc: fitz.Document) -> fitz.Page:
    return doc.new_page(width=A4.width, height=A4.height)


def _write(page: fitz.Page, y: float, text: str, size: float = 11, bold: bool = False) -> float:
    """Write a wrapped paragraph, return next y."""
    font = "hebo" if bold else "helv"
    rect = fitz.Rect(MARGIN, y, A4.width - MARGIN, y + 400)
    rc = page.insert_textbox(rect, text, fontsize=size, fontname=font)
    used = 400 - rc if rc >= 0 else 400
    return y + used + size * 0.8


def _table(page: fitz.Page, y: float, rows: list[list[str]], col_w: list[float], row_h: float = 22) -> float:
    x0 = MARGIN
    for i, row in enumerate(rows):
        x = x0
        for j, cell in enumerate(row):
            r = fitz.Rect(x, y, x + col_w[j], y + row_h)
            page.draw_rect(r, color=(0, 0, 0), width=0.7)
            page.insert_textbox(r + (4, 5, -4, 0), cell, fontsize=10, fontname="hebo" if i == 0 else "helv",
                                align=fitz.TEXT_ALIGN_LEFT if j == 0 else fitz.TEXT_ALIGN_RIGHT)
            x += col_w[j]
        y += row_h
    return y + 14


def _bar_chart_png() -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    ax.bar(["Q1 FY24", "Q2 FY24", "Q3 FY24", "Q4 FY24"], [330, 350, 370, 400], color="#2f6db5")
    ax.set_ylabel("Revenue (₹ crore)")
    ax.set_ylim(0, 450)
    ax.set_yticks(range(0, 451, 50))
    ax.grid(axis="y", alpha=0.4)
    ax.set_title("Quarterly Revenue FY24 (₹ crore)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def annual_report(path: Path) -> None:
    doc = fitz.open()
    p = _new_page(doc)
    y = _write(p, MARGIN, "Asterix Mobility Ltd - Annual Report FY24", 18, bold=True)
    y = _write(p, y + 6, "1. Chairman's Message", 14, bold=True)
    y = _write(p, y, "FY24 was a year of disciplined growth. Revenue from operations grew to Rs 1,450 crore in FY24 "
                     "from Rs 1,210 crore in FY23, driven by higher volumes at our Pune and Chennai plants. "
                     "Net profit rose to Rs 112 crore. We employed 4,820 people across three plants at year end.")
    y = _write(p, y + 6, "2. Outlook", 14, bold=True)
    _write(p, y, "We target capacity utilisation above 85% across all plants in FY25 and will invest Rs 60 crore "
                 "in the Sanand facility to reduce unplanned downtime.")

    p = _new_page(doc)
    y = _write(p, MARGIN, "3. Financial Highlights", 14, bold=True)
    y = _write(p, y, "Table 1: Key financials (Rs crore)", 10)
    rows = [["Metric", "FY22", "FY23", "FY24"],
            ["Revenue", "980", "1,210", "1,450"],
            ["EBITDA", "142", "188", "236"],
            ["Net Profit", "61", "84", "112"]]
    _table(p, y, rows, [160, 90, 90, 90])

    p = _new_page(doc)
    y = _write(p, MARGIN, "4. Quarterly Performance", 14, bold=True)
    img_rect = fitz.Rect(MARGIN, y, MARGIN + 420, y + 238)
    p.insert_image(img_rect, stream=_bar_chart_png())
    _write(p, img_rect.y1 + 8, "Figure 1: Quarterly revenue, FY24 (Rs crore). Q4 was the strongest quarter.", 10)
    doc.save(path)


def operations(path: Path) -> None:
    doc = fitz.open()
    p = _new_page(doc)
    y = _write(p, MARGIN, "Plant Operations Data - FY24", 16, bold=True)
    y = _write(p, y, "1. Plant-wise performance", 13, bold=True)
    y = _write(p, y, "Table 1: Capacity, output and downtime by plant, FY24", 10)
    rows = [["Plant", "Capacity (units)", "Output (units)", "Utilisation (%)", "Downtime (hours)"],
            ["Pune", "1,20,000", "96,000", "80", "410"],
            ["Chennai", "80,000", "70,400", "88", "260"],
            ["Sanand", "50,000", "31,000", "62", "720"]]
    y = _table(p, y, rows, [90, 105, 100, 105, 110])
    _write(p, y, "Note: Sanand downtime includes the March 2024 coolant line incident (see memo IR-2024-07).", 10)
    doc.save(path)


def incident_memo(path: Path) -> None:
    """Render a memo, degrade it like a phone scan, and save as an image-only PDF."""
    import cv2

    src = fitz.open()
    p = _new_page(src)
    y = _write(p, MARGIN, "INTERNAL MEMO", 16, bold=True)
    y = _write(p, y, "Incident Report IR-2024-07", 13, bold=True)
    y = _write(p, y, "Date: 14 March 2024\nPlant: Sanand\nTo: Head of Operations", 12)
    y = _write(p, y + 6, "Summary", 13, bold=True)
    _write(p, y, "A coolant line failure caused an unplanned shutdown of 96 hours. Estimated production loss "
                 "was 4,800 units. Repair cost was Rs 38 lakh. The line was restored on 18 March 2024. "
                 "Corrective action: replace all coolant couplings by June 2024.", 12)
    pix = p.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    rng = np.random.default_rng(7)
    img = np.clip(img.astype(np.int16) - 18 + rng.normal(0, 14, img.shape), 0, 255).astype(np.uint8)
    h, w = img.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 1.2, 1.0)
    img = cv2.warpAffine(img, rot, (w, h), borderValue=235)
    ok, png = cv2.imencode(".png", img)
    out = fitz.open()
    page = _new_page(out)
    page.insert_image(page.rect, stream=png.tobytes())
    out.save(path)


def press_release(path: Path) -> None:
    doc = fitz.open()
    p = _new_page(doc)
    y = _write(p, MARGIN, "PRESS RELEASE", 16, bold=True)
    y = _write(p, y, "Asterix Mobility reports strong FY24 results", 14, bold=True)
    y = _write(p, y, "Pune, 20 May 2024 - Asterix Mobility Ltd today announced results for FY24. "
                     "Revenue rose 22% to Rs 1,480 crore, while net profit reached Rs 112 crore. "
                     "EBITDA margin expanded on better utilisation at the Chennai plant, which ran at 88% of capacity.")
    _write(p, y + 4, "\"Our teams delivered record output,\" said the CEO. The company will invest Rs 60 crore in Sanand in FY25.")
    doc.save(path)


def main(out_dir: Path = OUT) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    makers = [("annual_report.pdf", annual_report), ("operations.pdf", operations),
              ("incident_memo.pdf", incident_memo), ("press_release.pdf", press_release)]
    paths = []
    for name, fn in makers:
        fn(out_dir / name)
        paths.append(out_dir / name)
        print("wrote", out_dir / name)
    return paths


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
