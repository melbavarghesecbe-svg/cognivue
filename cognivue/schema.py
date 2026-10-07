"""Core data contracts. Change only with team sign-off."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points, top-left origin

ElementKind = Literal["text", "heading", "table", "figure", "scan"]
Modality = Literal["text", "table", "chart", "scan"]
Verdict = Literal["supported", "partial", "unsupported", "unchecked"]

KIND_LETTER = {"text": "X", "heading": "H", "table": "T", "figure": "F", "scan": "S"}


def make_element_id(doc_id: str, page: int, kind: str, n: int) -> str:
    """D1-p7-F3: doc, 1-based page, kind letter, per-page counter."""
    return f"{doc_id}-p{page}-{KIND_LETTER[kind]}{n}"


class Element(BaseModel):
    id: str
    doc_id: str
    doc_name: str = ""
    page: int
    kind: ElementKind
    section: str = ""
    text: str = ""
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    image_path: Optional[str] = None
    quality: float = 1.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def modality(self) -> Modality:
        return {"figure": "chart", "table": "table", "scan": "scan"}.get(self.kind, "text")  # type: ignore[return-value]

    def label(self) -> str:
        what = {"figure": "Figure", "table": "Table", "scan": "Scan"}.get(self.kind, "Section")
        return f"{self.doc_name or self.doc_id} · p.{self.page} · {what}: {self.section or self.id}"


class Fact(BaseModel):
    id: Optional[int] = None
    metric: str
    metric_norm: str
    period: str = ""
    value: float  # in base units (crore/lakh expanded)
    unit: str = ""  # INR, %, units, hours, count ...
    raw: str = ""
    source_id: str  # element id
    doc_id: str
    modality: Modality
    estimated: bool = False


class Calc(BaseModel):
    name: str
    expr: str
    inputs: dict[str, float] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    result: Optional[float] = None
    error: str = ""


class Claim(BaseModel):
    text: str
    cites: list[str] = Field(default_factory=list)
    quote: str = ""
    verdict: Verdict = "unchecked"
    reasons: list[str] = Field(default_factory=list)
    estimated: bool = False


class Conflict(BaseModel):
    metric: str
    period: str
    facts: list[Fact]
    note: str = ""


class Answer(BaseModel):
    text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    calcs: list[Calc] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_parts: dict[str, float] = Field(default_factory=dict)
    refused: bool = False
    refusal_reason: str = ""
    modalities: list[str] = Field(default_factory=list)
    estimated: bool = False


class TraceStep(BaseModel):
    step: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Result(BaseModel):
    question: str
    answer: Answer = Field(default_factory=Answer)
    evidence: list[Element] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
