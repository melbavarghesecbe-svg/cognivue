"""Ingest the demo docs and run the bench (full + baseline), filling cache/llm/ for offline demos.

Usage:  python scripts/warm_cache.py            (needs GEMINI_API_KEY unless the cache is already warm)
        CACHE_ONLY=1 python scripts/warm_cache.py  (reproduce the committed results offline)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cognivue.bench import run_bench  # noqa: E402
from cognivue.config import get_settings  # noqa: E402
from cognivue.ingest import ingest_pdf  # noqa: E402
from cognivue.llm import get_llm  # noqa: E402
from cognivue.pipeline import Pipeline  # noqa: E402
from cognivue.store import Store  # noqa: E402
from scripts.make_demo_docs import OUT, main as make_docs  # noqa: E402

ORDER = ["annual_report.pdf", "operations.pdf", "incident_memo.pdf", "press_release.pdf"]


def main() -> None:
    s = get_settings()
    if s.db_path.exists():
        s.db_path.unlink()
    paths = [OUT / n for n in ORDER]
    if not all(p.exists() for p in paths):
        make_docs()
    store, llm = Store(s.db_path), get_llm(s)
    for p in paths:
        doc_id, errs = ingest_pdf(p, store, llm, s)
        print(doc_id, p.name, errs or "ok")
    pipe = Pipeline(store, llm, s)
    pipe.build_index()
    print("embedder:", pipe.embed_name)
    results = {}
    for name, baseline in (("full", False), ("baseline", True)):
        rows, summary = run_bench(pipe, baseline=baseline, progress=lambda p, i: print(f"  {name} {i}", flush=True))
        results[name] = {"summary": summary, "rows": rows}
        print(name, summary)
    out = ROOT / "bench" / "results.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
