"""Numeric facts stated in running text (one cached LLM call per document, then code-checked)."""
from __future__ import annotations

import difflib
from typing import Optional

from pydantic import BaseModel, Field

from ..ledger import norm_metric, norm_period, unit_of
from ..numparse import find_numbers
from ..schema import Element, Fact


class TextFact(BaseModel):
    element_id: str
    metric: str
    period: str = ""
    value_text: str  # exactly as printed, e.g. "Rs 1,480 crore"
    quote: str


class TextFacts(BaseModel):
    facts: list[TextFact] = Field(default_factory=list)


PROMPT = (
    "Extract every numeric business fact (money, %, counts, hours, units) stated in these passages. "
    "For each give element_id (from the list), metric (short name, e.g. 'revenue', 'net profit', "
    "'Sanand downtime'), period (e.g. FY24, or empty), value_text exactly as printed, and a verbatim quote. "
    "Skip dates and growth rates.\n\nPASSAGES:\n"
)


def quote_in(quote: str, text: str, threshold: float = 0.85) -> bool:
    q, t = " ".join(quote.lower().split()), " ".join(text.lower().split())
    if q in t:
        return True
    m = difflib.SequenceMatcher(None, t, q).find_longest_match(0, len(t), 0, len(q))
    return m.size / max(len(q), 1) >= threshold


def extract_text_facts(llm, doc_id: str, elements: list[Element], trace: Optional[list]) -> list[Fact]:
    passages = [e for e in elements if e.kind in ("text", "scan") and find_numbers(e.text)]
    if not passages:
        return []
    body = "\n".join(f"[{e.id}] {e.text}" for e in passages)
    try:
        out = llm.call_json("text_facts", PROMPT + body, TextFacts, trace=trace)
    except Exception:
        return []
    by_id = {e.id: e for e in passages}
    facts = []
    for f in out.facts:
        el = by_id.get(f.element_id)
        nums = find_numbers(f.value_text)
        if not el or not nums or not quote_in(f.quote, el.text) or not quote_in(f.value_text, el.text, 0.7):
            continue  # code check: the fact must really be in the cited element
        unit = unit_of(f.value_text, f.value_text)
        facts.append(Fact(
            metric=f.metric, metric_norm=norm_metric(f.metric), period=norm_period(f.period) if f.period else "",
            value=nums[0][1], unit=unit, raw=f.value_text, source_id=el.id, doc_id=doc_id,
            modality="scan" if el.kind == "scan" else "text",
        ))
    return facts
