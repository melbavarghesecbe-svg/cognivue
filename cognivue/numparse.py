"""Number parsing with Indian units (lakh/crore) and digit grouping."""
from __future__ import annotations

import re
from typing import Optional

SCALE = {
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9,
    "thousand": 1e3, "k": 1e3,
}

NUM_RE = re.compile(
    r"(?<![\w.])(?P<neg>[-−(])?\s?(?P<cur>₹|rs\.?\s?|inr\s?|\$)?\s?"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?"
    r"(?:\s?(?P<unit>%|crores?|cr\b\.?|lakhs?|lacs?|million|mn\b|billion|bn\b|thousand))?",
    re.I,
)


def scale_of(unit: str) -> float:
    return SCALE.get(unit.lower().rstrip("."), 1.0) if unit else 1.0


def parse_number(text: str) -> Optional[float]:
    """First number in text, scaled to base units. '₹1,23,45,678' -> 12345678; '3.2 lakh' -> 320000."""
    nums = find_numbers(text)
    return nums[0][1] if nums else None


def find_numbers(text: str) -> list[tuple[str, float]]:
    """All numbers as (raw, base_value). Percentages are kept as their face value."""
    out = []
    for m in NUM_RE.finditer(text or ""):
        val = float(m.group("num").replace(",", ""))
        unit = (m.group("unit") or "").strip()
        val *= scale_of(unit)
        if m.group("neg"):
            val = -val
        out.append((m.group(0).strip(), val))
    return out


def number_variants(text: str, context_scale: float = 1.0) -> set[float]:
    """Values a reader could take from text: raw, scaled by explicit unit, and scaled by context unit (e.g. table in ₹ crore)."""
    vals: set[float] = set()
    for m in NUM_RE.finditer(text or ""):
        base = float(m.group("num").replace(",", ""))
        unit = (m.group("unit") or "").strip()
        vals.add(base)
        vals.add(base * scale_of(unit))
        if not unit:
            vals.add(base * context_scale)
    return vals


def close(a: float, b: float, rel: float = 0.005) -> bool:
    if a == b:
        return True
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


def unit_scale_from_header(text: str) -> float:
    """'(₹ crore)' in a caption/header -> 1e7."""
    m = re.search(r"\b(crores?|lakhs?|lacs?|million|mn|billion|bn|thousand)\b", text or "", re.I)
    return scale_of(m.group(1)) if m else 1.0


def format_indian(value: float) -> str:
    """12345678 -> '₹1.23 crore' style human display for INR base values."""
    if abs(value) >= 1e7:
        return f"{value / 1e7:,.2f} crore"
    if abs(value) >= 1e5:
        return f"{value / 1e5:,.2f} lakh"
    return f"{value:,.2f}".rstrip("0").rstrip(".")
