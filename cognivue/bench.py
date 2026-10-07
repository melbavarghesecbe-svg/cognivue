"""Bench: 25 labelled questions, full pipeline vs baseline (no ledger, no visual re-look, no verifier)."""
from __future__ import annotations

import json
from pathlib import Path

from .numparse import close, find_numbers
from .pipeline import Mode, Pipeline
from .schema import Result

QUESTIONS = Path(__file__).resolve().parent.parent / "bench" / "questions.json"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def answer_numbers(res: Result) -> list[float]:
    text = res.answer.text + " " + " ".join(c.text for c in res.answer.claims)
    vals = [v for _, v in find_numbers(text)]
    vals += [c.result for c in res.answer.calcs if c.result is not None]
    return vals


def score(item: dict, res: Result) -> dict:
    refused = res.answer.refused
    expect_refuse = item.get("refuse", False)
    row = {"id": item["id"], "type": item["type"], "refused": refused, "confidence": res.answer.confidence}
    row["refusal_ok"] = refused == expect_refuse
    if expect_refuse:
        row.update(answer_ok=refused, citation_ok=None, numeric_ok=None)
        return row
    tol = item.get("tol", 0.01)
    numeric_ok = None
    if "value" in item:
        numeric_ok = (not refused) and any(close(v, item["value"], tol) for v in answer_numbers(res))
    text = (res.answer.text + " " + " ".join(c.text for c in res.answer.claims)).lower()
    kw_ok = all(k in text for k in item.get("keywords", []))
    cited_docs = {x.split("-")[0] for c in res.answer.claims for x in c.cites}
    row["citation_ok"] = (not refused) and set(item.get("docs", [])) <= cited_docs
    row["numeric_ok"] = numeric_ok
    row["answer_ok"] = (not refused) and kw_ok and (numeric_ok is not False)
    if item.get("conflict"):
        row["conflict_flagged"] = bool(res.answer.conflicts)
    return row


def summarise(rows: list[dict]) -> dict[str, float]:
    def pct(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(100 * sum(vals) / len(vals), 1) if vals else 0.0

    return {"answer_acc": pct("answer_ok"), "citation_acc": pct("citation_ok"),
            "numeric_acc": pct("numeric_ok"), "refusal_acc": pct("refusal_ok"), "n": len(rows)}


def run_bench(pipe: Pipeline, baseline: bool = False, progress=None) -> tuple[list[dict], dict[str, float]]:
    items = load_questions()
    mode = Mode.baseline() if baseline else Mode.full()
    rows = []
    for i, item in enumerate(items):
        res = pipe.ask(item["q"], mode)
        row = score(item, res)
        row["errors"] = "; ".join(res.errors)[:200]
        rows.append(row)
        if progress:
            progress((i + 1) / len(items), item["id"])
    return rows, summarise(rows)
