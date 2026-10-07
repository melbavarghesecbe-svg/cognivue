"""Streamlit rendering helpers (no pipeline logic here)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from .ingest.pdf import RENDER_DPI
from .proofgraph import build_dot
from .schema import Answer, Element, Result

VERDICT_ICON = {"supported": "🟢", "partial": "🟡", "unsupported": "🔴", "unchecked": "⚪"}
MODALITY_BADGE = {"text": ":blue-badge[📝 text]", "table": ":green-badge[📊 table]",
                  "chart": ":violet-badge[📈 chart · estimated]", "scan": ":orange-badge[🖨️ scan]"}


def highlight(page_png: str, bbox, color=(255, 0, 0)) -> Image.Image:
    img = Image.open(page_png).convert("RGB")
    s = RENDER_DPI / 72.0
    x0, y0, x1, y1 = (v * s for v in bbox)
    ImageDraw.Draw(img).rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], outline=color, width=4)
    return img


def answer_card(res: Result) -> None:
    a: Answer = res.answer
    if a.refused:
        st.error(f"**Refused.** {a.refusal_reason}")
        return
    with st.container(border=True):
        st.markdown(" ".join(MODALITY_BADGE.get(m, m) for m in a.modalities))
        st.markdown(f"### {a.text}")
        if a.estimated:
            st.caption("⚠️ Includes values read from a chart by the vision model. These are estimates.")
        st.progress(a.confidence, text=f"Truth Meter confidence: {a.confidence:.2f}")
        st.caption(" · ".join(f"{k} {v:.2f}" for k, v in a.confidence_parts.items()))
    for c in a.conflicts:
        with st.container(border=True):
            st.warning(f"**Conflict: {c.metric} {c.period}.** {c.note}")
            for f in c.facts:
                st.markdown(f"- `{f.source_id}` ({f.modality}{', estimated' if f.estimated else ''}): **{f.raw}**")


def claim_chips(res: Result) -> None:
    st.markdown("#### Claims")
    for c in res.answer.claims:
        cites = " ".join(f"`{x}`" for x in c.cites)
        st.markdown(f"{VERDICT_ICON[c.verdict]} {c.text} {cites}")
        if c.reasons:
            st.caption("; ".join(c.reasons))
    removed = [v for t in res.trace if t.step == "verify" for v in t.data.get("verdicts", []) if v["verdict"] == "unsupported"]
    if removed:
        with st.expander(f"🔴 {len(removed)} claim(s) removed by the verifier"):
            for v in removed:
                st.markdown(f"- ~~{v['text']}~~: {'; '.join(v['reasons'])}")


def receipts(res: Result) -> None:
    if not res.answer.calcs:
        return
    st.markdown("#### Receipts (computed by Python, not the model)")
    for c in res.answer.calcs:
        with st.container(border=True):
            st.code(f"{c.name} = {c.expr}", language="python")
            st.write({"inputs": c.inputs, "result": c.result, "sources": c.sources, "error": c.error or None})


def evidence_viewer(res: Result, store) -> None:
    ids = sorted({x for c in res.answer.claims for x in c.cites}) or [e.id for e in res.evidence]
    if not ids:
        return
    st.markdown("#### Evidence viewer")
    pick = st.selectbox("Source element", ids, key=f"ev-{res.question}")
    el: Element | None = store.get(pick)
    if el is None:
        return
    st.caption(el.label())
    page = next((p for p in store.pages(el.doc_id) if p["page"] == el.page), None)
    cols = st.columns([3, 2])
    if page and Path(page["png"]).exists():
        cols[0].image(highlight(page["png"], el.bbox), use_container_width=True)
    cols[1].text_area("Extracted text", el.text, height=260, key=f"txt-{pick}")
    if el.kind == "figure" and el.meta.get("chart"):
        cols[1].caption("ChartLens values are estimated")


def proof_graph(res: Result) -> None:
    st.markdown("#### ProofGraph (from the run trace)")
    st.graphviz_chart(build_dot(res), use_container_width=True)
    with st.expander("Raw trace"):
        st.json([t.model_dump() for t in res.trace])


def triage_strip(store) -> None:
    for d in store.docs():
        st.markdown(f"**{d['id']} · {d['name']}**")
        pages = store.pages(d["id"])
        cols = st.columns(max(len(pages), 1))
        for col, p in zip(cols, pages):
            icon = {"native": "🟢", "mixed": "🟡", "scan": "🟠", "error": "🔴"}.get(p["kind"], "⚪")
            if Path(p["png"]).exists():
                col.image(p["png"], use_container_width=True)
            col.caption(f"{icon} p{p['page']} {p['kind']} · q={p['quality']:.2f}\n{p['notes'] or ''}")


def chartlens(store) -> None:
    figs = [e for e in store.all_elements() if e.kind == "figure"]
    for f in figs:
        st.markdown(f"**{f.id}** · {f.doc_name} p.{f.page}")
        c1, c2 = st.columns(2)
        if f.image_path and Path(f.image_path).exists():
            c1.image(f.image_path, use_container_width=True)
        chart = f.meta.get("chart")
        if chart:
            c2.caption(f"{chart['title']} ({chart['unit']}). Values are estimated.")
            c2.table([dict(zip(chart["header"], r)) for r in chart["rows"]])
        else:
            c2.info("Chart data not extracted (no cache or API key).")
