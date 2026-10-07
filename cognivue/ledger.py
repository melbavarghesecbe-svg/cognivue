"""FactLedger: numeric facts (metric, period, value, unit, source) in SQLite."""
from __future__ import annotations

import re
from typing import Optional

from .numparse import find_numbers, unit_scale_from_header
from .schema import Element, Fact
from .store import Store

SYNONYMS = {
    "revenue from operations": "revenue", "total revenue": "revenue", "net sales": "revenue", "sales": "revenue",
    "net profit": "net profit", "profit after tax": "net profit", "pat": "net profit",
    "utilization": "utilisation", "capacity utilisation": "utilisation",
}

PERIOD_RE = re.compile(r"\b(FY\s?'?\d{2,4}(?:-\d{2})?|Q[1-4]\s?(?:FY\s?\d{2,4})?|20\d{2}(?:-\d{2})?)\b", re.I)


def norm_metric(text: str) -> str:
    t = PERIOD_RE.sub(" ", re.sub(r"\(.*?\)", "", text or "")).lower()
    t = re.sub(r"[^a-z% ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for k, v in SYNONYMS.items():
        if t == k or t.startswith(k + " "):
            return v
    return t


def norm_period(text: str) -> str:
    """'FY 2023-24' / 'FY24' / '2023-24' -> 'FY24'; 'Q3 FY24' -> 'Q3 FY24'."""
    t = (text or "").upper().replace("'", "").strip()
    q = re.search(r"Q([1-4])", t)
    fy = re.search(r"(?:FY\s?)?(20)?(\d{2})-(\d{2})\b", t)
    if fy:
        year = fy.group(3)
    else:
        m = re.search(r"FY\s?(?:20)?(\d{2})\b", t) or re.search(r"\b20(\d{2})\b", t)
        year = m.group(1) if m else ""
    base = f"FY{year}" if year else t
    return f"Q{q.group(1)} {base}".strip() if q else base


def looks_like_period(text: str) -> bool:
    return bool(PERIOD_RE.search(text or ""))


def unit_of(header: str, cell: str) -> str:
    h = f"{header} {cell}".lower()
    if "%" in h:
        return "%"
    if "hour" in h:
        return "hours"
    if "₹" in h or re.search(r"\b(rs|inr|crores?|lakhs?)\b", h):
        return "INR"
    if "unit" in h:
        return "units"
    return ""


class Ledger:
    def __init__(self, store: Store):
        self.conn = store.conn

    def add(self, facts: list[Fact]) -> None:
        self.conn.executemany(
            "INSERT INTO facts (metric, metric_norm, period, value, unit, raw, source_id, doc_id, modality, estimated)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(f.metric, f.metric_norm, f.period, f.value, f.unit, f.raw, f.source_id, f.doc_id, f.modality, int(f.estimated)) for f in facts],
        )
        self.conn.commit()

    def all(self) -> list[Fact]:
        return [_row(r) for r in self.conn.execute("SELECT * FROM facts ORDER BY metric_norm, period")]

    def find(self, metric: str, period: Optional[str] = None) -> list[Fact]:
        q, args = "SELECT * FROM facts WHERE metric_norm=?", [norm_metric(metric)]
        if period:
            q += " AND period=?"
            args.append(norm_period(period))
        return [_row(r) for r in self.conn.execute(q, args)]

    def for_sources(self, element_ids: list[str]) -> list[Fact]:
        if not element_ids:
            return []
        marks = ",".join("?" * len(element_ids))
        return [_row(r) for r in self.conn.execute(f"SELECT * FROM facts WHERE source_id IN ({marks})", element_ids)]


def _row(r) -> Fact:
    d = dict(r)
    d["estimated"] = bool(d["estimated"])
    return Fact(**d)


# -- fact extraction from structured elements ---------------------------------

def facts_from_table(el: Element, rows: list[list[str]], estimated: bool = False, modality: str = "table") -> list[Fact]:
    """Rows[0] is the header. Supports periods-as-columns or periods-as-rows."""
    if len(rows) < 2:
        return []
    header = [c or "" for c in rows[0]]
    scale = unit_scale_from_header(" ".join(header) + " " + el.section + " " + el.meta.get("caption", ""))
    caption_unit = unit_of(el.section + " " + el.meta.get("caption", ""), "")
    facts: list[Fact] = []
    periods_in_header = sum(looks_like_period(h) for h in header[1:]) >= max(1, len(header[1:]) // 2)
    for row in rows[1:]:
        row = [c or "" for c in row]
        if not row or not row[0].strip():
            continue
        for j, cell in enumerate(row[1:], start=1):
            if j >= len(header):
                break
            nums = find_numbers(cell)
            if not nums:
                continue
            raw_val = nums[0][1]
            col_scale = unit_scale_from_header(header[j]) if unit_scale_from_header(header[j]) != 1.0 else scale
            unit = unit_of(header[j], cell) or caption_unit
            value = raw_val if unit == "%" else raw_val * col_scale
            if periods_in_header:
                metric, period = row[0], header[j]
            elif looks_like_period(row[0]):
                metric, period = header[j], row[0]
            else:  # entity rows (e.g. plants): metric = "<entity> <column>"
                metric = f"{row[0]} {header[j]}"
                period = header[j] if looks_like_period(header[j]) else _context_period(el)
            facts.append(Fact(
                metric=metric.strip(), metric_norm=norm_metric(metric), period=norm_period(period) if period else "",
                value=value, unit=unit, raw=cell.strip(), source_id=el.id, doc_id=el.doc_id,
                modality=modality, estimated=estimated,
            ))
    return facts


def _context_period(el: Element) -> str:
    m = PERIOD_RE.search(f"{el.meta.get('caption', '')} {el.section} {el.text[:200]}")
    return m.group(0) if m else ""
