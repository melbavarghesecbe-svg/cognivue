"""One multimodal call -> strict JSON (claims, variables, calcs, sufficient). No arithmetic by the model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .numparse import find_numbers, unit_scale_from_header
from .schema import Element, Fact


class Variable(BaseModel):
    name: str
    cite: str
    value_text: str  # exactly as printed in the cited element


class CalcSpec(BaseModel):
    name: str
    expr: str


class ClaimSpec(BaseModel):
    text: str
    cites: list[str] = Field(default_factory=list)
    quote: str = ""


class ReasonerOut(BaseModel):
    sufficient: bool
    missing: str = ""
    answer: str = ""
    claims: list[ClaimSpec] = Field(default_factory=list)
    variables: list[Variable] = Field(default_factory=list)
    calcs: list[CalcSpec] = Field(default_factory=list)


RULES = """You answer questions ONLY from the evidence below. Rules:
1. Every claim cites one or more evidence IDs exactly as given (e.g. D1-p2-T1). Never invent IDs.
2. 'quote' is a short verbatim span from the first cited element that supports the claim.
3. NEVER do arithmetic. If a number must be computed, declare 'variables' (name, cite, value_text copied
   exactly from the cited element) and 'calcs' (name, expr using variable names, + - * / and
   pct_change(old,new), cagr(start,end,years), round, abs, min, max; note: pct_change and cagr already return percentages, do NOT multiply by 100 again).
   Write the result in the answer and claims as a {calc_name} placeholder; Python fills it in.
4. If evidence is missing or the question cannot be answered, set sufficient=false and explain in 'missing'.
5. If sources disagree, state both values with their citations; do not pick one.
6. Values read from charts are estimates; say "approximately".
7. Images attached are the original table/chart crops for the IDs shown; use them to re-check values.
8. Resolve document references: D1 corresponds to Document A / Document 1; D2 corresponds to Document B / Document 2, etc. If the question asks 'According to Document B', cite and answer from D2 evidence."""


def evidence_block(pack: list[Element], facts: list[Fact]) -> str:
    from .index import doc_alias

    doc_summary = []
    seen_docs = set()
    for e in pack:
        if e.doc_id not in seen_docs:
            seen_docs.add(e.doc_id)
            alias = doc_alias(e.doc_id)
            doc_summary.append(f"- {e.doc_id} / {alias}: {e.doc_name}")
    prefix = ("INDEXED DOCUMENTS:\n" + "\n".join(doc_summary) + "\n\n") if doc_summary else ""

    lines = []
    for e in pack:
        tag = {"figure": "CHART (estimated values)", "table": "TABLE", "scan": "SCANNED TEXT"}.get(e.kind, "TEXT")
        alias_short = doc_alias(e.doc_id).split()[0:2]
        alias_lbl = f"{e.doc_id} ({' '.join(alias_short)})" if alias_short else e.doc_id
        lines.append(f"[{e.id}] {tag} | {alias_lbl} {e.doc_name} p.{e.page} | section: {e.section}\n{e.text}")
    if facts:
        lines.append("LEDGER FACTS (normalised, base units):")
        lines += [f"- {f.metric} {f.period}: {f.raw} (source {f.source_id}{', estimated' if f.estimated else ''})" for f in facts]
    return prefix + "\n\n".join(lines)


def visual_images(pack: list[Element], limit: int = 3) -> list[tuple[str, bytes]]:
    """Table/chart crops for the visual re-look (logged in the trace by llm.py)."""
    out = []
    for e in pack:
        if e.kind in ("figure", "table") and e.image_path and Path(e.image_path).exists():
            out.append((e.id, Path(e.image_path).read_bytes()))
        if len(out) >= limit:
            break
    return out


def reason(llm, question: str, sub_questions: list[str], pack: list[Element], facts: list[Fact],
           relook: bool = True, trace: Optional[list] = None) -> ReasonerOut:
    prompt = (f"{RULES}\n\nQUESTION: {question}\nSUB-QUESTIONS: {'; '.join(sub_questions)}\n\n"
              f"EVIDENCE:\n{evidence_block(pack, facts)}")
    images = visual_images(pack) if relook else []
    return llm.call_json("reasoner", prompt, ReasonerOut, images=images, trace=trace)


def resolve_variable(var: Variable, el: Optional[Element]) -> Optional[float]:
    """Value in base units. Bare numbers in a '(Rs crore)' table/chart are scaled by that context."""
    nums = find_numbers(var.value_text)
    if not nums:
        return None
    raw, value = nums[0]
    explicit_unit = any(c.isalpha() for c in raw) or "%" in raw
    if el is not None and el.kind in ("table", "figure") and not explicit_unit:
        value *= unit_scale_from_header(f"{el.meta.get('caption', '')} {el.section} {el.text[:120]}")
    return value
