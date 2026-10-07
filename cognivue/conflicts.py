"""Triangulation, conflicts and Premise Guard, all driven by the FactLedger."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .ledger import Ledger, norm_metric, norm_period
from .numparse import close, find_numbers
from .schema import Conflict, Fact


def tolerance(a: Fact, b: Fact) -> float:
    return 0.05 if (a.estimated or b.estimated) else 0.005


def group_facts(facts: list[Fact]) -> dict[tuple[str, str], list[Fact]]:
    groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
    for f in facts:
        if f.period:
            groups[(f.metric_norm, f.period)].append(f)
    return groups


def find_conflicts(ledger: Ledger, evidence_ids: list[str]) -> tuple[list[Conflict], float]:
    """Conflicts for metrics touched by the evidence; returns (conflicts, agreement 0..1)."""
    touched = {(f.metric_norm, f.period) for f in ledger.for_sources(evidence_ids) if f.period}
    conflicts, agree, compared = [], 0, 0
    for (metric, period) in sorted(touched):
        facts = ledger.find(metric, period)
        if len({f.source_id for f in facts}) < 2:
            continue
        compared += 1
        base = facts[0]
        if all(close(f.value, base.value, tolerance(f, base)) for f in facts[1:]):
            agree += 1
        else:
            conflicts.append(Conflict(metric=metric, period=period, facts=facts,
                                      note="Sources disagree; both values shown, none chosen."))
    agreement = 0.5 if compared == 0 else agree / compared
    return conflicts, agreement


def premise_violations(ledger: Ledger, premises: list, trace: Optional[list] = None) -> list[str]:
    """Each premise is {metric, period, value_text}. A violation = ledger has the metric/period but no source agrees."""
    out = []
    for p in premises:
        nums = find_numbers(p.value_text)
        if not nums:
            continue
        value = nums[0][1]
        facts = ledger.find(p.metric, p.period or None)
        if not facts:
            continue
        if any(close(value, f.value, max(tolerance(f, f), 0.02)) for f in facts):
            continue
        shown = "; ".join(f"{f.raw} ({f.source_id})" for f in facts[:3])
        out.append(f"The question assumes {norm_metric(p.metric)} {norm_period(p.period) if p.period else ''} = "
                   f"{p.value_text}, but the documents show {shown}.")
    if trace is not None:
        trace.append({"step": "premise_guard", "premises": [p.model_dump() for p in premises], "violations": out})
    return out
