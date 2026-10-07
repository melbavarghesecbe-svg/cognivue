"""VLM helpers used at ingest: ChartLens (chart -> table) and table fallback."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VisualTable(BaseModel):
    title: str = ""
    unit: str = ""
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


CHART_PROMPT = (
    "You are ChartLens. Convert this chart into a data table. First column = category/period label "
    "exactly as on the axis, other columns = series. Read values from the axis scale; do not invent "
    "series. Put the unit (e.g. 'Rs crore', '%') in 'unit'. Return header and rows as strings."
)

TABLE_PROMPT = (
    "Extract this table exactly. 'header' is the first row; 'rows' are the remaining rows, "
    "one list of cell strings per row, keeping numbers as printed."
)


def chart_to_table(llm, png: bytes, label: str, trace: Optional[list]) -> Optional[VisualTable]:
    try:
        return llm.call_json("chartlens", CHART_PROMPT, VisualTable, images=[(label, png)], trace=trace)
    except Exception:
        return None


def table_via_vlm(llm, png: bytes, label: str, trace: Optional[list]) -> Optional[VisualTable]:
    try:
        return llm.call_json("table_fallback", TABLE_PROMPT, VisualTable, images=[(label, png)], trace=trace)
    except Exception:
        return None


def table_markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    lines += [" | ".join(c or "" for c in r) for r in rows]
    return "\n".join(lines)


def is_ragged(rows: list[list]) -> bool:
    """Extraction looks broken: inconsistent widths or many empty cells."""
    if len(rows) < 2:
        return True
    widths = {len(r) for r in rows}
    cells = [c for r in rows for c in r]
    empty = sum(1 for c in cells if c is None or not str(c).strip())
    return len(widths) > 1 or empty / max(len(cells), 1) > 0.3
