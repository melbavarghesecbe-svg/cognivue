# COGNIVUE: evidence-locked multimodal document intelligence

Hackathon problem **HNX26PSI01 (Multimodal Document Intelligence)**.

You upload mixed PDFs (text, tables, charts, scans) and ask complex questions. Every claim and number in an
answer is traced to a stored element (`D1-p7-F3` = document 1, page 7, figure 3) and **checked by code
before it is shown**. If there is no evidence, the app refuses.

## How it works

```
PDF ─► ingest ─► Element store (SQLite: doc, page, bbox, section)   page PNGs + triage (native/mixed/scan, quality)
         │         ├─ text blocks + headings (PyMuPDF)
         │         ├─ tables (pdfplumber; VLM fallback if ragged)
         │         ├─ charts ─► ChartLens (VLM → data table, marked "estimated")
         │         └─ scans ─► OpenCV cleanup ─► rapidocr ─► VLM transcription if OCR is weak
         └─► FactLedger (SQLite: metric, period, value, unit, source; lakh/crore aware)

question ─► planner (sub-questions + premises) ─► hybrid retrieval (bge-small/Chroma + BM25, RRF, ≥1 hit per relevant doc)
         ─► Premise Guard (ledger) ─► Reasoner: one multimodal call, strict JSON, chart/table crops re-attached
         ─► calc.py (ast-whitelisted, no eval) ─► Verifier / Truth Meter (ID, number-in-source, quote, batched entailment)
         ─► triangulation and conflicts (ledger) ─► confidence ─► answer card + receipts + ProofGraph
```

Hard rules, enforced in code:

| Rule | Where |
|---|---|
| Cite only element IDs from the evidence pack. Doc, page and bbox always come from the store | `verify.code_checks`, `store.py` |
| The LLM never does arithmetic. It returns variables and expressions, and Python evaluates them | `reasoner.py`, `calc.py` |
| Inputs must appear in the cited source | `pipeline._calculate` → `verify.number_supported` |
| Chart and table crops go to the VLM at query time, and the trace records it | `reasoner.visual_images`, trace `reason.relook` |
| Chart values are always "estimated" (−0.10 confidence) | `Fact.estimated`, `Claim.estimated`, `confidence.py` |
| Every model call goes through `llm.py`: disk cache, temperature 0, retry, fallback model, JSON validation, `CACHE_ONLY` | `llm.py` |
| Conflicts show both sources and never pick one silently. Confidence is capped at 0.6 | `conflicts.py`, `confidence.py` |
| Refusal: low retrieval score, insufficient evidence, false premise, nothing verified | `confidence.py` gates, `conflicts.premise_violations` |

Confidence = 0.30·retrieval + 0.30·verified ratio + 0.20·agreement + 0.20·page quality,
minus 0.10 if a chart estimate is used, and capped at 0.6 when there is a conflict.

## Install (Windows, Python 3.10+)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env      # then put your GEMINI_API_KEY in .env
```

The first run downloads the bge-small-en-v1.5 ONNX model (about 70 MB) into `data/models/`. If the download fails,
the index falls back to a hashed bag-of-words embedder and says so in the ingest status. On slow networks set
`EMBED_MODEL=hash` and `MIN_RETRIEVAL_SCORE=0.12` in `.env` (BM25 does most of the work then). The committed
bench results were produced in this mode.

## Config (`.env`)

| Key | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (empty) | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | primary model |
| `GEMINI_FALLBACK_MODEL` | `gemini-2.0-flash` | used after retries fail |
| `CACHE_ONLY` | `0` | `1` = never call the network; cache misses become soft refusals |

## Run

```powershell
.venv\Scripts\streamlit run app.py
```
In the sidebar, click **Load demo documents** (or upload PDFs), then ask a question in the **Ask** tab.

## Reproduce (offline)

```powershell
python scripts\make_demo_docs.py          # 4 demo PDFs → demo_docs/ (also committed)
python scripts\warm_cache.py              # with API key: ingest + run the bench, fills cache/llm/
$env:CACHE_ONLY=1; python scripts\warm_cache.py   # offline replay from the committed cache
.venv\Scripts\python -m pytest -q         # all tests run offline
```

Demo set:

* **D1** `annual_report.pdf`: text, financial table, quarterly revenue bar chart without labels
* **D2** `operations.pdf`: plant table with Indian digit grouping (`1,20,000`)
* **D3** `incident_memo.pdf`: image-only, noisy and skewed scanned memo
* **D4** `press_release.pdf`: **planted conflict** (revenue Rs 1,480 crore vs Rs 1,450 crore in D1)

## Sample input / output

**Q:** *By what percentage did revenue grow from FY23 to FY24 according to the annual report?*

```
Answer: Revenue grew 19.8% from Rs 1,210 crore in FY23 to Rs 1,450 crore in FY24.   [table]
  🟢 claim  cites D1-p2-T1  (quote "1,450")
  Receipt:  growth = pct_change(a, b)   a=1.21e10 (D1-p2-T1)  b=1.45e10 (D1-p2-T1)  → 19.83
  Confidence 0.8x · ProofGraph: Q → sub-questions → D1-p2-T1 → reasoner → calc → claim → answer
```

**Q:** *Why did FY24 revenue fall to Rs 900 crore?* → **Refused. False premise.** The documents show
1,450 (D1-p2-T1) and Rs 1,480 crore (D4-p1-X…).

## Bench

The **Bench** tab (or `scripts/warm_cache.py`) runs 25 labelled questions: text, table, chart, scan, math,
cross-doc, false premise and unanswerable. It reports answer, citation, numeric and refusal accuracy for COGNIVUE versus a
**baseline** with the ledger, visual re-look and verifier switched off. The results are written to `bench/results.json`.

## Scope

**MVP (built):** everything above.

**Stretch / not built:** a general chart detector (charts are found as embedded images), ChartLens for vector-drawn
charts, table-of-tables layouts, chat history, auth, summariser, reranker model, handwriting, non-English OCR,
documents over 100 pages.

## Layout

```
app.py                Streamlit UI (Ask / Documents / Bench)
cognivue/schema.py    Element, Fact, Claim, Calc, Conflict, Answer, Result
cognivue/llm.py       the only model gateway
cognivue/ingest/      pdf.py, triage.py, scans.py, visual.py (ChartLens), textfacts.py
cognivue/store.py     element store      cognivue/ledger.py  FactLedger
cognivue/index.py     hybrid index       cognivue/retrieve.py planner + evidence pack
cognivue/reasoner.py  cognivue/calc.py   cognivue/verify.py  cognivue/conflicts.py  cognivue/confidence.py
cognivue/pipeline.py  cognivue/proofgraph.py  cognivue/ui.py  cognivue/bench.py
bench/questions.json  scripts/  tests/
```
