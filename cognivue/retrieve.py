"""Planner + evidence-pack assembly (multi sub-question, >=1 hit per relevant document)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .index import Hit, Index


class Premise(BaseModel):
    metric: str
    period: str = ""
    value_text: str


class Plan(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    premises: list[Premise] = Field(default_factory=list)


PLAN_PROMPT = (
    "Split the user question into 1-4 short self-contained retrieval sub-questions (one per fact needed). "
    "Also list any numeric facts the question ASSUMES to be true (premises), e.g. 'Why did revenue fall to "
    "Rs 900 crore in FY24?' assumes {metric: 'revenue', period: 'FY24', value_text: 'Rs 900 crore'}. "
    "Leave premises empty if none.\n\nQUESTION: "
)


def plan(llm, question: str, trace: Optional[list] = None) -> Plan:
    try:
        p = llm.call_json("planner", PLAN_PROMPT + question, Plan, trace=trace)
    except Exception:
        return Plan(sub_questions=[question])
    subs = [q for q in p.sub_questions if q.strip()][:4]
    return p.model_copy(update={"sub_questions": subs or [question]})


def build_pack(index: Index, question: str, sub_questions: list[str], k: int, min_score: float) -> list[Hit]:
    """Top hits per sub-question, plus the best hit of every document that clears min_score."""
    best: dict[str, Hit] = {}
    for q in [question, *sub_questions]:
        for h in index.search(q, k=k * 2):
            if h.id not in best or h.rrf > best[h.id].rrf:
                best[h.id] = h
    ranked = sorted(best.values(), key=lambda h: (-h.rrf, -h.dense))
    pack = ranked[:k]
    covered = {index.doc_of[h.id] for h in pack}
    for h in ranked[k:]:
        doc = index.doc_of[h.id]
        if doc not in covered and max(h.dense, h.bm25) >= min_score:
            pack.append(h)
            covered.add(doc)
    return pack
