"""COGNIVUE: evidence-locked multimodal document QA. Run: streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from cognivue import ui
from cognivue.bench import run_bench
from cognivue.config import get_settings
from cognivue.ingest import ingest_pdf
from cognivue.llm import get_llm
from cognivue.pipeline import Mode, Pipeline
from cognivue.store import Store

st.set_page_config(page_title="COGNIVUE", page_icon="🔎", layout="wide")

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
    with st.status("Ingesting…", expanded=True) as status:
        for p in paths:
            _, errs = ingest_pdf(p, pipe.store, pipe.llm, pipe.settings)
            st.write(f"{p.name}: {'ok' if not errs else '; '.join(errs)}")
        pipe.build_index()
        status.update(label=f"Ingested {len(paths)} file(s) · embedder: {pipe.embed_name}", state="complete")


def sidebar(pipe: Pipeline) -> Mode:
    s = pipe.settings
    st.sidebar.title("🔎 COGNIVUE")
    st.sidebar.caption("Every number traced to a source and checked by code.")
    st.sidebar.markdown(f"**LLM:** {s.gemini_model if s.gemini_api_key else 'no API key'}"
                        f"{' · CACHE_ONLY' if s.cache_only else ''}")
    ups = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if ups and st.sidebar.button("Ingest uploads"):
        paths = []
        for u in ups:
            dest = s.data_dir / "uploads" / u.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(u.getvalue())
            paths.append(dest)
        ingest_files(pipe, paths)
    if st.sidebar.button("Load demo documents"):
        from scripts.make_demo_docs import OUT, main as make_docs

        paths = sorted(OUT.glob("*.pdf")) if len(list(OUT.glob("*.pdf"))) == 4 else make_docs()
        order = ["annual_report.pdf", "operations.pdf", "incident_memo.pdf", "press_release.pdf"]
        ingest_files(pipe, sorted(paths, key=lambda p: order.index(p.name) if p.name in order else 99))
    st.sidebar.markdown("**Documents**")
    for d in pipe.store.docs():
        st.sidebar.caption(f"{d['id']} · {d['name']} · {d['pages']} p")
    baseline = st.sidebar.toggle("Baseline mode (no ledger / re-look / verifier)")
    return Mode.baseline() if baseline else Mode.full()


def ask_tab(pipe: Pipeline, mode: Mode) -> None:
    if not pipe.store.docs():
        st.info("Load the demo documents or upload PDFs from the sidebar.")
        return
    ex = st.selectbox("Example questions", [""] + EXAMPLES)
    q = st.text_input("Ask a question", value=ex)
    if not q or not st.button("Ask", type="primary"):
        return
    if pipe.index is None:
        with st.spinner("Building index…"):
            pipe.build_index()
    with st.spinner("Planning → retrieving → reasoning → verifying…"):
        res = pipe.ask(q, mode)
    for e in res.errors:
        st.caption(f"⚠️ {e}")
    ui.answer_card(res)
    if res.answer.refused:
        ui.proof_graph(res)
        return
    c1, c2 = st.columns([1, 1])
    with c1:
        ui.claim_chips(res)
        ui.receipts(res)
    with c2:
        ui.evidence_viewer(res, pipe.store)
    ui.proof_graph(res)


def docs_tab(pipe: Pipeline) -> None:
    st.subheader("Page triage")
    ui.triage_strip(pipe.store)
    st.subheader("ChartLens")
    ui.chartlens(pipe.store)
    st.subheader("FactLedger")
    st.dataframe([f.model_dump() for f in pipe.ledger.all()], use_container_width=True)


def bench_tab(pipe: Pipeline) -> None:
    st.caption("25 labelled questions: text, table, chart, scan, math, cross-doc, false premise, unanswerable.")
    if not st.button("Run bench (full + baseline)"):
        return
    if pipe.index is None:
        pipe.build_index()
    bar = st.progress(0.0)
    full_rows, full = run_bench(pipe, progress=lambda p, i: bar.progress(p / 2, text=f"full {i}"))
    base_rows, base = run_bench(pipe, baseline=True, progress=lambda p, i: bar.progress(0.5 + p / 2, text=f"baseline {i}"))
    st.table({"metric": list(full), "COGNIVUE": list(full.values()), "baseline": list(base.values())})
    st.dataframe(full_rows, use_container_width=True)


def main() -> None:
    pipe = get_pipeline()
    mode = sidebar(pipe)
    t1, t2, t3 = st.tabs(["Ask", "Documents", "Bench"])
    with t1:
        ask_tab(pipe, mode)
    with t2:
        docs_tab(pipe)
    with t3:
        bench_tab(pipe)


main()
