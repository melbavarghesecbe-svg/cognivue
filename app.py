"""COGNIVUE: Evidence-locked multimodal document intelligence.
Run: streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from cognivue import ui
from cognivue.bench import run_bench
from cognivue.config import get_settings
from cognivue.ingest import ingest_pdf
from cognivue.llm import get_llm
from cognivue.pipeline import Mode, Pipeline
from cognivue.store import Store

st.set_page_config(
    page_title="COGNIVUE — See. Understand. Verify.",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom B2B SaaS styles & suppress default Streamlit chrome
ui.inject_styles()

EXAMPLES = [
    "By what percentage did revenue grow from FY23 to FY24 according to the annual report?",
    "What revenue did Asterix Mobility report for FY24?",
    "According to the quarterly revenue chart, what was revenue in Q4 FY24?",
    "What share of Sanand's FY24 downtime hours was caused by the coolant line incident?",
    "Why did FY24 revenue fall to Rs 900 crore?",
]


@st.cache_resource
def get_pipeline() -> Pipeline:
    s = get_settings()
    return Pipeline(Store(s.db_path), get_llm(s), s)


def ingest_files(pipe: Pipeline, paths: list[Path]) -> None:
    with st.status("Ingesting documents…", expanded=True) as status:
        for p in paths:
            _, errs = ingest_pdf(p, pipe.store, pipe.llm, pipe.settings)
            st.write(f"{p.name}: {'ok' if not errs else '; '.join(errs)}")
        pipe.build_index()
        status.update(label=f"Ingested {len(paths)} file(s) · Embedder: {pipe.embed_name}", state="complete")


def render_sidebar(pipe: Pipeline) -> tuple[str, Mode]:
    s = pipe.settings

    # Brand Wordmark
    st.sidebar.markdown(
        """
        <div class="brand-container">
            <div class="brand-name">COGNIVUE</div>
            <div class="brand-tagline">See. Understand. Verify.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean SaaS Navigation matching reference design
    active_nav = st.sidebar.radio(
        "Navigation",
        options=["Ask", "Documents", "Bench"],
        index=0,
        key="app_navigation",
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="sidebar-section-label">DOCUMENTS</div>', unsafe_allow_html=True)

    # Document Upload Area
    st.sidebar.markdown(
        """
        <div style="background:#FFFFFF; border:1px solid #EBE4DC; border-radius:8px; padding:7px 10px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:8px;">
                <div>
                    <div style="font-size:12px; font-weight:600; color:#1E1815;">Upload PDFs</div>
                    <div style="font-size:10px; color:#8C7E75;">PDF up to 200 MB</div>
                </div>
            </div>
            <span style="color:#8C7E75; font-size:13px; font-weight:600;">›</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ups = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, help="PDF up to 200 MB", label_visibility="collapsed")
    if ups and st.sidebar.button("Ingest uploads", use_container_width=True):
        paths = []
        for u in ups:
            dest = s.data_dir / "uploads" / u.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(u.getvalue())
            paths.append(dest)
        ingest_files(pipe, paths)

    # Load Demo Documents Button
    if st.sidebar.button("Load demo documents", use_container_width=True):
        from scripts.make_demo_docs import OUT, main as make_docs

        paths = sorted(OUT.glob("*.pdf")) if len(list(OUT.glob("*.pdf"))) == 4 else make_docs()
        order = ["annual_report.pdf", "operations.pdf", "incident_memo.pdf", "press_release.pdf"]
        ingest_files(pipe, sorted(paths, key=lambda p: order.index(p.name) if p.name in order else 99))

    # Ingested Documents List
    docs = pipe.store.docs()
    if docs:
        st.sidebar.markdown('<div class="sidebar-section-label" style="margin-top: 0.9rem;">INDEXED CORPUS</div>', unsafe_allow_html=True)
        for d in docs:
            st.sidebar.markdown(
                f"""
                <div style="font-size: 12px; color: #332B27; padding: 0.22rem 0; display: flex; justify-content: space-between; align-items: center;">
                    <span><b>{d['id']}</b> · {d['name']}</span>
                    <span style="color: #8C7E75; font-size: 11px;">{d['pages']}p</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.caption("No documents loaded yet.")

    # Execution Mode Toggle
    st.sidebar.markdown('<div class="sidebar-section-label" style="margin-top: 1rem;">CONFIG</div>', unsafe_allow_html=True)
    baseline = st.sidebar.toggle(
        "Baseline mode (plain RAG) ⓘ",
        help="Disables FactLedger, visual re-look, and Truth Meter verifier",
    )
    mode = Mode.baseline() if baseline else Mode.full()

    # System Status Indicator
    st.sidebar.markdown(
        """
        <div style="font-size: 11.5px; color: #5C524F; margin-top: 1.5rem; padding-top: 0.75rem; border-top: 1px solid #EBE4DC; display: flex; align-items: center; gap: 7px;">
            <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: #10B981; box-shadow: 0 0 4px #10B981;"></span>
            <span style="font-weight: 500;">AI engine · Connected</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return active_nav, mode


def ask_view(pipe: Pipeline, mode: Mode) -> None:
    # Top Right Header Nav
    st.markdown(
        """
        <div class="top-nav-bar">
            <span class="top-nav-text">Document Intelligence</span>
            <span class="top-nav-divider">|</span>
            <span class="user-avatar">M</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hero Center Header
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-eyebrow-wrapper">
                <span class="hero-eyebrow-line"></span>
                <span class="hero-eyebrow-text">DOCUMENT INTELLIGENCE</span>
                <span class="hero-eyebrow-line"></span>
            </div>
            <h1 class="hero-title">COGNIVUE</h1>
            <div class="hero-tagline">See. Understand. Verify.</div>
            <p class="hero-description">Get answers grounded in text, tables, charts, and scanned evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Question Input Shell - Floating Query Card
    st.markdown('<div class="floating-query-card">', unsafe_allow_html=True)
    ex_choice = st.selectbox(
        "Example queries",
        [""] + EXAMPLES,
        index=0,
        placeholder="Or select a verified example question...",
        label_visibility="collapsed",
    )
    
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        q = st.text_input(
            "Ask a question",
            value=ex_choice,
            placeholder="Ask a question about your documents...",
            label_visibility="collapsed",
        )
    with col_btn:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not pipe.store.docs():
        st.markdown(
            """
            <div class="upload-dropzone-card">
                <div class="dropzone-title">Upload your documents</div>
                <div class="dropzone-sub">Drag and drop PDFs here, or click to browse</div>
                <div class="dropzone-caption">PDF up to 200 MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if ask_btn and q:
        if pipe.index is None:
            with st.spinner("Building hybrid vector index…"):
                pipe.build_index()
        with st.spinner("Planning → retrieving → reasoning → verifying…"):
            res = pipe.ask(q, mode)
            st.session_state["last_result"] = res
            st.session_state["last_question"] = q

    # Render previous or current result
    res: Result | None = st.session_state.get("last_result")
    if res is not None:
        for e in res.errors:
            st.caption(f"Notice: {e}")

        ui.answer_card(res)

        if not res.answer.refused:
            col_evidence, col_claims = st.columns([1, 1], gap="medium")
            with col_claims:
                ui.claim_chips(res)
                ui.receipts(res)
            with col_evidence:
                ui.evidence_viewer(res, pipe.store)

        ui.proof_graph(res)


def docs_view(pipe: Pipeline) -> None:
    st.markdown(
        """
        <div class="view-header">
            <div class="view-eyebrow">DOCUMENT REPOSITORY</div>
            <h1 class="view-title">Document intelligence & evidence.</h1>
            <p class="view-subtitle">Inspect page triage quality, extracted ChartLens figures, and the FactLedger.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pipe.store.docs():
        ui.empty_state("No documents yet", "Upload a PDF or click 'Load demo documents' in the sidebar to inspect page quality, figures, and facts.")
        return

    st.markdown('<div class="sidebar-section-label">PAGE TRIAGE & QUALITY</div>', unsafe_allow_html=True)
    ui.triage_strip(pipe.store)

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">CHARTLENS VISUAL EXTRACTIONS</div>', unsafe_allow_html=True)
    ui.chartlens(pipe.store)

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">FACTLEDGER (NORMALIZED FACTS)</div>', unsafe_allow_html=True)
    ui.factledger_view(pipe.store)


def bench_view(pipe: Pipeline) -> None:
    st.markdown(
        """
        <div class="view-header">
            <div class="view-eyebrow">EVALUATION BENCHMARK</div>
            <h1 class="view-title">Evaluation Bench</h1>
            <p class="view-subtitle">Measure answer, citation, numeric, and refusal accuracy across 25 labelled test cases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not pipe.store.docs():
        ui.empty_state("No documents available", "Please load the demo documents or upload test PDFs before executing the evaluation bench.")
        return

    col_btn, _ = st.columns([1.5, 4])
    with col_btn:
        run_clicked = st.button("Run evaluation bench (full + baseline)", type="primary", use_container_width=True)

    if run_clicked:
        if pipe.index is None:
            pipe.build_index()
        bar = st.progress(0.0)
        full_rows, full = run_bench(pipe, progress=lambda p, i: bar.progress(p / 2, text=f"Evaluating full pipeline: {i}"))
        base_rows, base = run_bench(pipe, baseline=True, progress=lambda p, i: bar.progress(0.5 + p / 2, text=f"Evaluating baseline: {i}"))
        bar.empty()
        st.session_state["bench_full"] = full
        st.session_state["bench_base"] = base
        st.session_state["bench_rows"] = full_rows

    full_summary = st.session_state.get("bench_full")
    base_summary = st.session_state.get("bench_base")
    full_rows = st.session_state.get("bench_rows")

    if full_summary:
        ui.bench_dashboard(full_summary, base_summary, full_rows)
    else:
        ui.empty_state(
            "No benchmark results yet",
            "Click 'Run evaluation bench' above to run the 25 labelled evaluation questions and evaluate accuracy.",
        )


def main() -> None:
    pipe = get_pipeline()
    active_nav, mode = render_sidebar(pipe)

    if "Ask" in active_nav:
        ask_view(pipe, mode)
    elif "Documents" in active_nav:
        docs_view(pipe)
    elif "Bench" in active_nav:
        bench_view(pipe)


main()
