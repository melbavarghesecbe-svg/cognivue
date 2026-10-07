"""Confidence score and refusal gates."""
from __future__ import annotations

from typing import Optional

from .schema import Claim

WEIGHTS = {"retrieval": 0.30, "verified": 0.30, "agreement": 0.20, "quality": 0.20}
CHART_PENALTY = 0.10
CONFLICT_CAP = 0.6


def verified_ratio(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    score = {"supported": 1.0, "partial": 0.5}
    return sum(score.get(c.verdict, 0.0) for c in claims) / len(claims)


def confidence(retrieval: float, verified: float, agreement: float, quality: float,
               chart_estimated: bool, has_conflict: bool) -> tuple[float, dict[str, float]]:
    parts = {
        "retrieval": max(0.0, min(retrieval, 1.0)),
        "verified": verified,
        "agreement": agreement,
        "quality": quality,
    }
    score = sum(WEIGHTS[k] * v for k, v in parts.items())
    if chart_estimated:
        score -= CHART_PENALTY
    if has_conflict:
        score = min(score, CONFLICT_CAP)
    return round(max(0.0, min(score, 1.0)), 3), parts


def retrieval_gate(top_score: float, min_score: float) -> Optional[str]:
    if top_score < min_score:
        return f"No sufficiently relevant evidence found (best match {top_score:.2f} < {min_score:.2f})."
    return None


def evidence_gate(sufficient: bool, missing: str) -> Optional[str]:
    return None if sufficient else f"Insufficient evidence: {missing or 'the documents do not answer this.'}"


def verification_gate(kept: list[Claim]) -> Optional[str]:
    return None if kept else "No claim survived verification, so no answer is shown."
