# COGNIVUE
### Evidence-Locked Multimodal Document Intelligence
**See. Understand. Verify.**

Real-world documents combine narrative text, tables, charts, images, and scanned pages. COGNIVUE lets users ask questions across mixed PDFs and returns answers grounded in stored document evidence, page-level citations, code-checked calculations, and claim verification.

## Key Features

- Multimodal PDF processing for native text, tables, embedded figures, and scanned pages.
- Page triage with native, mixed, and scan classification plus quality scores.
- ChartLens visual extraction for chart images, with estimated values clearly marked.
- OCR processing for scanned pages using OpenCV cleanup and RapidOCR.
- Hybrid retrieval using dense embeddings, ChromaDB, BM25, and reciprocal rank fusion.
- Cross-document evidence packs that preserve document, page, element, and bounding-box references.
- FactLedger storage for normalized numeric facts from tables and text.
- Python-side calculations with validated inputs and calculation receipts.
- Claim verification for evidence IDs, quoted text, source numbers, and entailment.
- Conflict detection, confidence scoring, premise checks, and grounded refusal when evidence is insufficient.
- Streamlit views for asking questions, inspecting documents, and running the benchmark.

## How It Works

```text
Documents
    -> PDF processing
    -> Text, table, chart, and scan extraction
    -> Element Store and FactLedger
    -> Dense + BM25 retrieval
    -> Question planning and multimodal reasoning
    -> Python calculations and claim verification
    -> Evidence-backed answer with citations
```

1. **Process** — PyMuPDF and pdfplumber extract page content, tables, images, and metadata. Scanned pages are cleaned and OCR-processed.
2. **Store** — Elements, page quality information, document metadata, and normalized facts are persisted in SQLite. Page images and visual crops are retained for inspection.
3. **Retrieve** — A dense ChromaDB index and BM25 index are combined with reciprocal rank fusion.
4. **Reason** — Gemini receives the question and retrieved evidence, including relevant table or chart crops when visual re-checking is enabled.
5. **Verify** — Python evaluates permitted calculations. Claims are checked against evidence IDs, source numbers, quotes, and model-based entailment.
6. **Answer** — The UI shows the answer, citations, confidence information, receipts, conflicts, and the evidence trace, or refuses when support is inadequate.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.10+ |
| Framework and UI | Streamlit |
| PDF processing | PyMuPDF (`fitz`), pdfplumber |
| Scans and OCR | OpenCV, RapidOCR |
| Retrieval | ChromaDB, rank-bm25, reciprocal rank fusion |
| Embedding model | FastEmbed with `BAAI/bge-small-en-v1.5` by default; deterministic hash embeddings are available as an offline fallback |
| LLM/VLM | Google Gemini through the `google-genai` SDK |
| Database | SQLite |
| Validation and configuration | Pydantic, pydantic-settings, python-dotenv |
| Supporting libraries | NumPy, Pillow, Matplotlib, Graphviz |
| Testing | pytest |

## Installation

Windows PowerShell:

```powershell
git clone https://github.com/melbavarghesecbe-svg/cognivue.git
cd cognivue
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Configuration

Create a local environment file from the provided template:

```powershell
copy .env.example .env
```

Set `GEMINI_API_KEY` in `.env` for Gemini planning, reasoning, chart analysis, table fallback extraction, and scan transcription. Do not commit `.env`.

Important settings include:

| Setting | Purpose |
|---|---|
| `GEMINI_MODEL` | Primary Gemini model |
| `GEMINI_FALLBACK_MODEL` | Fallback Gemini model |
| `CACHE_ONLY` | Prevents network calls and uses cached model responses only |
| `EMBED_MODEL` | FastEmbed model name, or `hash` for deterministic offline embeddings |
| `MIN_RETRIEVAL_SCORE` | Retrieval refusal threshold |

The application creates its local database, page-image directory, model cache, and ChromaDB directory under `data/` as needed.

## Run

Start the Streamlit application:

```powershell
.venv\Scripts\streamlit run app.py
```

In the Ask view:

1. Upload one or more PDFs.
2. Click **Ingest uploads**.
3. Choose an example question or enter your own.
4. Click **Ask**.

The **Documents** view exposes page quality, visual extractions, and FactLedger data. The **Bench** view runs the labelled evaluation set in full and baseline modes.

## Reproduce the Demonstration

The repository includes four generated demo PDFs covering text, tables, charts, a scanned memo, and a planted cross-document revenue conflict.

To regenerate the demo documents:

```powershell
.venv\Scripts\python scripts\make_demo_docs.py
```

To ingest the demos and run the full and baseline benchmark:

```powershell
.venv\Scripts\python scripts\warm_cache.py
```

This command uses the configured Gemini key when cache entries are missing and writes benchmark output to `bench/results.json`.

For an offline replay, use existing cached responses:

```powershell
$env:CACHE_ONLY = "1"
.venv\Scripts\python scripts\warm_cache.py
```

Run the automated tests with:

```powershell
.venv\Scripts\python -m pytest -q
```

## Repository Layout

```text
app.py                 Streamlit application and view routing
cognivue/ingest/       PDF, table, chart, scan, OCR, and triage processing
cognivue/store.py      SQLite document, page, element, and fact store
cognivue/index.py      Dense and BM25 hybrid index
cognivue/retrieve.py   Question planning and evidence-pack assembly
cognivue/reasoner.py   Structured multimodal reasoning
cognivue/calc.py       Safe calculation evaluation
cognivue/verify.py     Claim and number verification
cognivue/pipeline.py   End-to-end question pipeline
cognivue/ui.py         Streamlit design system and result views
bench/                 Labelled benchmark questions and generated results
scripts/               Demo document generation and cache warming
tests/                 Ingestion and pipeline tests
```
