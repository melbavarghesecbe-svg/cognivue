"""PDF -> Elements (text, headings, tables, figures, scans) + page PNGs + ledger facts."""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Optional

import fitz
import numpy as np

from ..ledger import Ledger, facts_from_table
from ..schema import Element, make_element_id
from . import scans, visual
from .textfacts import extract_text_facts
from .triage import classify_page, scan_quality

RENDER_DPI = 144
SCAN_DPI = 200


class PageCtx:
    """Per-page id counters and running section heading."""

    def __init__(self, doc_id: str, doc_name: str, page_no: int, section: str):
        self.doc_id, self.doc_name, self.page_no, self.section = doc_id, doc_name, page_no, section
        self.counts: dict[str, int] = defaultdict(int)

    def new_id(self, kind: str) -> str:
        self.counts[kind] += 1
        return make_element_id(self.doc_id, self.page_no, kind, self.counts[kind])

    def element(self, kind: str, text: str, bbox, **kw) -> Element:
        return Element(id=self.new_id(kind), doc_id=self.doc_id, doc_name=self.doc_name, page=self.page_no,
                       kind=kind, section=kw.pop("section", self.section), text=text, bbox=tuple(bbox), **kw)


def ingest_pdf(path: Path | str, store, llm, settings, trace: Optional[list] = None) -> tuple[str, list[str]]:
    """Ingest one PDF. Returns (doc_id, soft errors). Re-ingesting a file name replaces it."""
    path = Path(path)
    errors: list[str] = []
    old = store.find_doc(path.name)
    if old:
        store.delete_doc(old)
    doc_id = old or store.next_doc_id()
    pdf = fitz.open(path)
    store.add_doc(doc_id, path.name, str(path), pdf.page_count)
    ledger = Ledger(store)
    section = ""
    all_elements: list[Element] = []
    for i, page in enumerate(pdf):
        ctx = PageCtx(doc_id, path.name, i + 1, section)
        try:
            els, kind, quality, notes = _ingest_page(page, path, ctx, llm, settings, ledger, trace)
        except Exception as e:  # never let one page kill the document
            errors.append(f"{doc_id} p{i + 1}: {type(e).__name__}: {e}")
            els, kind, quality, notes = [], "error", 0.0, str(e)
        png = _render_page(page, settings.pages_dir / f"{doc_id}-p{i + 1}.png")
        store.add_page(doc_id, i + 1, kind, quality, str(png), notes)
        store.add_elements(els)
        all_elements += els
        section = ctx.section
    ledger.add(extract_text_facts(llm, doc_id, all_elements, trace))
    return doc_id, errors


def _render_page(page: fitz.Page, out: Path) -> Path:
    page.get_pixmap(dpi=RENDER_DPI).save(out)
    return out


def _crop_png(page: fitz.Page, bbox, out: Path, dpi: int = RENDER_DPI) -> Path:
    page.get_pixmap(dpi=dpi, clip=fitz.Rect(bbox)).save(out)
    return out


def _ingest_page(page, path, ctx: PageCtx, llm, settings, ledger, trace):
    kind = classify_page(page)
    if kind == "scan":
        els, quality, notes = _scan_page(page, ctx, llm, settings, trace)
        return els, kind, quality, notes
    tables = _tables(page, path, ctx, llm, settings, trace)
    figures = _figures(page, ctx, llm, settings, ledger, trace)
    texts = _text_blocks(page, ctx, exclude=[t.bbox for t in tables + figures])
    _attach_captions(tables + figures, texts)
    for t in tables:  # after captions, so unit hints like '(Rs crore)' are known
        ledger.add(facts_from_table(t, t.meta.get("rows") or []))
    return texts + tables + figures, kind, 1.0, ""


# -- text --------------------------------------------------------------------

def _text_blocks(page: fitz.Page, ctx: PageCtx, exclude: list) -> list[Element]:
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
    sizes = [s["size"] for b in blocks for l in b["lines"] for s in l["spans"] if s["text"].strip()]
    body = statistics.median(sizes) if sizes else 11
    out = []
    for b in blocks:
        spans = [s for l in b["lines"] for s in l["spans"]]
        text = " ".join(" ".join(s["text"] for s in l["spans"]) for l in b["lines"]).strip()
        if not text or _inside_any(b["bbox"], exclude):
            continue
        size = max(s["size"] for s in spans)
        bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans if s["text"].strip())
        if (size >= body * 1.15 or bold) and len(text) < 90:
            ctx.section = text
            out.append(ctx.element("heading", text, b["bbox"], section=text))
        else:
            out.append(ctx.element("text", text, b["bbox"]))
    return out


def _inside_any(bbox, rects) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects)


def _attach_captions(visuals: list[Element], texts: list[Element]) -> None:
    """'Table 1: ...' just above or 'Figure 1: ...' just below a visual becomes its caption."""
    for v in visuals:
        best = None
        for t in texts:
            if not t.text.lower().startswith(("table", "figure", "fig.", "chart")):
                continue
            gap = min(abs(t.bbox[1] - v.bbox[3]), abs(v.bbox[1] - t.bbox[3]))
            if gap < 40 and (best is None or gap < best[0]):
                best = (gap, t)
        if best:
            v.meta["caption"] = best[1].text
            v.text = f"{best[1].text}\n{v.text}"


# -- tables ------------------------------------------------------------------

def _tables(page, path, ctx: PageCtx, llm, settings, trace) -> list[Element]:
    import pdfplumber

    out = []
    with pdfplumber.open(path) as pdf:
        for t in pdf.pages[page.number].find_tables():
            rows = [[(c or "").strip() for c in r] for r in t.extract()]
            el = ctx.element("table", "", t.bbox)
            el.image_path = str(_crop_png(page, t.bbox, settings.pages_dir / f"{el.id}.png"))
            el.meta["extractor"] = "pdfplumber"
            if visual.is_ragged(rows):
                vt = visual.table_via_vlm(llm, Path(el.image_path).read_bytes(), el.id, trace)
                if vt and vt.header:
                    rows, el.meta["extractor"] = [vt.header] + vt.rows, "vlm"
            el.meta["rows"] = rows
            el.text = visual.table_markdown(rows[0], rows[1:]) if rows else ""
            out.append(el)
    return out


# -- figures / ChartLens -----------------------------------------------------

def _figures(page, ctx: PageCtx, llm, settings, ledger, trace) -> list[Element]:
    area = page.rect.width * page.rect.height
    out = []
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if r.width * r.height < 0.03 * area:
            continue
        el = ctx.element("figure", "", r)
        el.image_path = str(_crop_png(page, r, settings.pages_dir / f"{el.id}.png", dpi=170))
        vt = visual.chart_to_table(llm, Path(el.image_path).read_bytes(), el.id, trace)
        if vt and vt.header:
            el.meta.update(chart={"title": vt.title, "unit": vt.unit, "header": vt.header, "rows": vt.rows}, estimated=True)
            el.text = f"Chart: {vt.title} ({vt.unit}) [values estimated by ChartLens]\n" + visual.table_markdown(vt.header, vt.rows)
            el.meta["caption"] = vt.title + f" ({vt.unit})"
            ledger.add(facts_from_table(el, [vt.header] + vt.rows, estimated=True, modality="chart"))
        else:
            el.text = "Chart (data not extracted)"
            el.quality = 0.5
        out.append(el)
    return out


# -- scans -------------------------------------------------------------------

def _scan_page(page, ctx: PageCtx, llm, settings, trace) -> tuple[list[Element], float, str]:
    import cv2

    pix = page.get_pixmap(dpi=SCAN_DPI, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    clean = scans.cleanup(gray)
    scale = 72.0 / SCAN_DPI
    blocks = scans.group_lines(scans.run_ocr(clean), scale)
    conf = float(np.mean([b["conf"] for b in blocks])) if blocks else 0.0
    notes = f"ocr_conf={conf:.2f}, blocks={len(blocks)}"
    ctx.section = ctx.section or "Scanned page"
    if conf < settings.ocr_min_conf:
        ok, png = cv2.imencode(".png", clean)
        text = scans.vlm_transcribe(llm, png.tobytes(), f"{ctx.doc_id}-p{ctx.page_no}", trace)
        if text:
            el = ctx.element("scan", text, page.rect, meta={"source": "vlm", "ocr_conf": conf})
            return [el], scan_quality(max(conf, 0.75), gray), notes + ", vlm transcription"
    els = []
    for b in blocks:
        if len(b["text"]) < 40 and b["text"].isupper() or b["text"].lower().startswith(("incident report", "summary")):
            ctx.section = b["text"]
        els.append(ctx.element("scan", b["text"], b["bbox"], quality=b["conf"], meta={"source": "ocr"}))
    return els, scan_quality(conf, gray), notes
