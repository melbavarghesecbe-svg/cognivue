"""Truth Meter: ID check, number-in-source check, quote check, one batched entailment call."""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from .ingest.textfacts import quote_in
from .numparse import NUM_RE, close, number_variants, scale_of, unit_scale_from_header
from .schema import Calc, Claim, Element

PLACEHOLDER = re.compile(r"\{(\w+)\}")


class Entailment(BaseModel):
    idx: int
    label: str  # entailed | partial | contradicted | unrelated


class EntailmentOut(BaseModel):
    results: list[Entailment] = Field(default_factory=list)


def format_value(value: float, expr: str = "") -> str:
    if any(s in expr for s in ("* 100", "*100", "pct_change", "cagr")):
        return f"{value:.1f}%"
    if abs(value) >= 1e7:
        return f"Rs {value / 1e7:,.2f} crore"
    if abs(value) >= 1e5:
        return f"{value / 1e5:,.2f} lakh"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def fill_placeholders(text: str, calcs: list[Calc]) -> str:
    by_name = {c.name: c for c in calcs}

    def sub(m):
        c = by_name.get(m.group(1))
        if c is None:
            return m.group(0)
        return format_value(c.result, c.expr) if c.result is not None else "[calculation failed]"

    return PLACEHOLDER.sub(sub, text)


def _context_scale(el: Element) -> float:
    if el.kind in ("table", "figure"):
        return unit_scale_from_header(f"{el.meta.get('caption', '')} {el.section} {el.text[:120]}")
    return 1.0


def claim_numbers(text: str) -> list[float]:
    """Numbers asserted in a claim, ignoring years and list ordinals."""
    out = []
    for m in NUM_RE.finditer(text):
        v = float(m.group("num").replace(",", ""))
        unit = (m.group("unit") or "").strip()
        if not unit and (1990 <= v <= 2100 and "," not in m.group("num")):
            continue
        if not unit and v < 10 and re.match(r"\d\b", m.group("num")) and "." not in m.group("num"):
            continue  # Q1, 3 plants ... too ambiguous to check
        out.append(v * scale_of(unit) if unit != "%" else v)
    return out


def number_supported(value: float, sources: list[Element], calc_results: list[float]) -> bool:
    for el in sources:
        if any(close(value, v) for v in number_variants(el.text, _context_scale(el))):
            return True
    return any(close(value, r, 0.01) or close(value, round(r, 1), 0.01) for r in calc_results)


def code_checks(claim: Claim, pack: dict[str, Element], calcs: list[Calc]) -> Claim:
    reasons = []
    valid = [c for c in claim.cites if c in pack]
    if len(valid) < len(claim.cites):
        reasons.append(f"dropped unknown IDs: {sorted(set(claim.cites) - set(valid))}")
    if not valid:
        return claim.model_copy(update={"cites": [], "verdict": "unsupported", "reasons": reasons + ["no valid citation"]})
    sources = [pack[c] for c in valid]
    results = [c.result for c in calcs if c.result is not None]
    results += [r * 1e7 for r in results] + [r / 1e7 for r in results]
    bad = [v for v in claim_numbers(claim.text) if not number_supported(v, sources, results)]
    verdict = "supported"
    if bad:
        verdict = "unsupported"
        reasons.append(f"numbers not found in cited sources: {bad}")
    if claim.quote and not any(quote_in(claim.quote, s.text, 0.8) for s in sources):
        verdict = "unsupported" if verdict == "unsupported" else "partial"
        reasons.append("quote not found verbatim")
    estimated = any(s.kind == "figure" for s in sources)
    return claim.model_copy(update={"cites": valid, "verdict": verdict, "reasons": reasons, "estimated": estimated})


ENTAIL_PROMPT = (
    "For each numbered claim, judge whether its evidence entails it. Labels: entailed, partial, "
    "contradicted, unrelated. Computed numbers marked [calc] are already verified by code; judge only the wording.\n\n"
)


def entailment(llm, claims: list[Claim], pack: dict[str, Element], trace: Optional[list]) -> dict[int, str]:
    items = []
    for i, c in enumerate(claims):
        ev = " || ".join(pack[x].text[:600] for x in c.cites)
        items.append(f"#{i} CLAIM: {c.text}\nEVIDENCE: {ev}")
    if not items:
        return {}
    try:
        out = llm.call_json("entailment", ENTAIL_PROMPT + "\n\n".join(items), EntailmentOut, trace=trace)
    except Exception:
        return {}
    return {r.idx: r.label.lower() for r in out.results}


def apply_entailment(claim: Claim, label: Optional[str]) -> Claim:
    if claim.verdict == "unsupported":
        return claim
    if label is None:
        return claim.model_copy(update={"reasons": claim.reasons + ["entailment unavailable (code checks only)"]})
    if label in ("contradicted", "unrelated"):
        return claim.model_copy(update={"verdict": "unsupported", "reasons": claim.reasons + [f"entailment: {label}"]})
    if label == "partial":
        return claim.model_copy(update={"verdict": "partial", "reasons": claim.reasons + ["entailment: partial"]})
    return claim


def verify_claims(llm, claims: list[Claim], pack: dict[str, Element], calcs: list[Calc],
                  trace: Optional[list] = None) -> list[Claim]:
    """Returns every claim with a verdict; callers drop the unsupported ones."""
    checked = [code_checks(c, pack, calcs) for c in claims]
    candidates = [i for i, c in enumerate(checked) if c.verdict != "unsupported"]
    labels = entailment(llm, [checked[i] for i in candidates], pack, trace)
    for j, i in enumerate(candidates):
        checked[i] = apply_entailment(checked[i], labels.get(j) if labels else None)
    return checked
