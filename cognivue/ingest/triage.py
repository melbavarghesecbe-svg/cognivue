"""Per-page quality triage: native / mixed / scan, plus a 0..1 quality score."""
from __future__ import annotations

import fitz
import numpy as np


def image_coverage(page: fitz.Page) -> float:
    area = page.rect.width * page.rect.height
    covered = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        covered += r.width * r.height
    return min(covered / area, 1.0) if area else 0.0


def classify_page(page: fitz.Page) -> str:
    chars = len(page.get_text().strip())
    cov = image_coverage(page)
    if chars < 40 and cov > 0.5:
        return "scan"
    return "mixed" if cov > 0.08 else "native"


def sharpness(gray: np.ndarray) -> float:
    """Variance of Laplacian mapped to 0..1 (higher is sharper)."""
    import cv2

    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(v / 500.0, 1.0))


def scan_quality(ocr_conf: float, gray: np.ndarray | None) -> float:
    sharp = sharpness(gray) if gray is not None else 0.5
    return round(0.7 * ocr_conf + 0.3 * sharp, 3)
