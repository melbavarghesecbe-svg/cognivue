"""Scanned pages: OpenCV cleanup -> rapidocr -> VLM transcription if OCR is weak."""
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel

_OCR = None


class Transcription(BaseModel):
    text: str


def cleanup(gray: np.ndarray) -> np.ndarray:
    """Denoise, deskew, binarise."""
    import cv2

    den = cv2.fastNlMeansDenoising(gray, h=15)
    angle = estimate_skew(den)
    if abs(angle) > 0.2:
        h, w = den.shape
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        den = cv2.warpAffine(den, rot, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def estimate_skew(gray: np.ndarray) -> float:
    import cv2

    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle > 45:
        angle -= 90
    return float(angle) if abs(angle) < 10 else 0.0


def run_ocr(img: np.ndarray) -> list[tuple[list, str, float]]:
    """[(box_pts, text, conf)] or [] if OCR is unavailable."""
    global _OCR
    try:
        if _OCR is None:
            from rapidocr_onnxruntime import RapidOCR

            _OCR = RapidOCR()
        result, _ = _OCR(img)
    except Exception:
        return []
    return [(r[0], r[1], float(r[2])) for r in (result or [])]


def group_lines(lines: list[tuple[list, str, float]], scale: float) -> list[dict]:
    """Merge OCR lines into paragraph blocks by vertical gap; bbox in PDF points."""
    items = sorted(
        ({"y0": min(p[1] for p in b), "y1": max(p[1] for p in b), "x0": min(p[0] for p in b),
          "x1": max(p[0] for p in b), "text": t, "conf": c} for b, t, c in lines),
        key=lambda d: (d["y0"], d["x0"]),
    )
    blocks: list[dict] = []
    for it in items:
        h = it["y1"] - it["y0"]
        if blocks and it["y0"] - blocks[-1]["y1"] < 1.2 * h:
            b = blocks[-1]
            b["text"] += " " + it["text"]
            b.update(x0=min(b["x0"], it["x0"]), x1=max(b["x1"], it["x1"]), y1=max(b["y1"], it["y1"]), last_y0=it["y0"])
            b["confs"].append(it["conf"])
        else:
            blocks.append({**it, "last_y0": it["y0"], "confs": [it["conf"]]})
    return [
        {"text": b["text"], "conf": float(np.mean(b["confs"])),
         "bbox": (b["x0"] * scale, b["y0"] * scale, b["x1"] * scale, b["y1"] * scale)}
        for b in blocks
    ]


def vlm_transcribe(llm, png: bytes, label: str, trace: Optional[list]) -> Optional[str]:
    prompt = ("Transcribe this scanned page exactly, preserving numbers, dates and units. "
              "Do not summarise. Return {\"text\": \"...\"}.")
    try:
        return llm.call_json("scan_transcribe", prompt, Transcription, images=[(label, png)], trace=trace).text
    except Exception:
        return None
