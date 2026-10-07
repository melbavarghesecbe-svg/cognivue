"""Streamlit UI components and design system for COGNIVUE.
Minimal, production-grade B2B SaaS aesthetics: Linear / Notion / Vercel style.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from .ingest.pdf import RENDER_DPI
from .proofgraph import build_dot
from .schema import Answer, Element, Result


def inject_styles() -> None:
    """Inject clean luxury editorial design tokens matching reference design."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Playfair+Display:ital,wght@1,500;1,600;1,700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* 1. Global Streamlit Chrome Suppression */
        #MainMenu,
        header[data-testid="stHeader"],
        footer,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        .stDeployButton {
            visibility: hidden !important;
            display: none !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* 2. Base Typography & Warm Cream Palette */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #FAF7F2 !important;
            background-image: 
                radial-gradient(circle at -4% 102%, #4E0F1B 0%, #4E0F1B 15%, transparent 15.5%),
                radial-gradient(circle at -2% 102%, rgba(138, 58, 72, 0.22) 0%, rgba(138, 58, 72, 0.22) 25%, transparent 25.5%),
                radial-gradient(circle at 1% 1%, rgba(228, 214, 200, 0.45) 0%, rgba(228, 214, 200, 0.45) 28%, transparent 28.5%),
                radial-gradient(circle at 105% 102%, rgba(228, 214, 200, 0.45) 0%, rgba(228, 214, 200, 0.45) 24%, transparent 24.5%) !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
            color: #1E1815 !important;
            -webkit-font-smoothing: antialiased;
            letter-spacing: -0.01em;
        }

        /* 3. Main Container Spacing */
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
            max-width: 1160px !important;
        }

        /* 4. Sidebar Shell - Warm Linen */
        [data-testid="stSidebar"] {
            min-width: 260px !important;
            max-width: 275px !important;
            background-color: #F7F2EB !important;
            border-right: 1px solid #EBE4DC !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.25rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        [data-testid="stSidebarCollapseButton"] {
            color: #6E625E !important;
        }

        /* 5. Brand Header in Sidebar */
        .brand-container {
            padding: 0.25rem 0.25rem 1.1rem 0.25rem;
            border-bottom: 1px solid #EBE4DC;
            margin-bottom: 1rem;
        }
        .brand-name {
            font-size: 20px;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #50101C;
            line-height: 1.2;
        }
        .brand-tagline {
            font-family: 'Playfair Display', Georgia, serif;
            font-style: italic;
            font-size: 13px;
            font-weight: 600;
            color: #50101C;
            margin-top: 3px;
            letter-spacing: 0.02em;
        }

        /* 6. Sidebar Navigation Items */
        div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
            gap: 0.25rem !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            border-radius: 8px !important;
            padding: 0.5rem 0.85rem !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            color: #5A4F4A !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            border: 1px solid transparent !important;
            transition: all 0.15s ease !important;
            margin: 0 !important;
            background: transparent !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background-color: #EBE4DC !important;
            color: #1E1815 !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
            background-color: #50101C !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 2px 8px rgba(80, 16, 28, 0.25) !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) * {
            color: #FFFFFF !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
            display: none !important;
        }

        /* 7. Sidebar Section Labels */
        .sidebar-section-label {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8C7E75;
            margin: 1.25rem 0 0.5rem 0.25rem;
        }

        /* 8. Modern Buttons */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 13.5px !important;
            letter-spacing: -0.01em !important;
            transition: all 0.15s ease-in-out !important;
            border: 1px solid #D6CBC1 !important;
            background-color: #FFFFFF !important;
            color: #1E1815 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
            padding: 0.45rem 0.95rem !important;
        }
        .stButton > button:hover {
            background-color: #F7F2EB !important;
            border-color: #50101C !important;
            color: #50101C !important;
        }
        .stButton > button[kind="primary"] {
            background-color: #50101C !important;
            color: #FFFFFF !important;
            border: 1px solid #50101C !important;
            box-shadow: 0 2px 8px rgba(80, 16, 28, 0.25) !important;
            font-weight: 600 !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #3D0B14 !important;
            border-color: #3D0B14 !important;
            color: #FFFFFF !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 14px rgba(80, 16, 28, 0.35) !important;
        }

        /* 9. Inputs & Form Controls */
        .stTextInput input {
            border-radius: 8px !important;
            border: 1px solid #D6CBC1 !important;
            font-size: 13.5px !important;
            background-color: #FFFFFF !important;
            color: #1E1815 !important;
            padding: 0.6rem 0.9rem !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
        }
        .stTextInput input:focus {
            border-color: #50101C !important;
            box-shadow: 0 0 0 2px rgba(80, 16, 28, 0.15) !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: 8px !important;
            border: 1px solid #D6CBC1 !important;
            background-color: #FFFFFF !important;
            font-size: 13px !important;
        }

        /* 10. File Uploader */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF;
            border: 1px dashed #D6CBC1;
            border-radius: 8px;
            padding: 0.4rem 0.5rem;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #50101C;
        }
        [data-testid="stFileUploader"] section {
            padding: 0.4rem !important;
        }

        /* Toggle switch */
        div[data-testid="stToggle"] [aria-checked="true"] {
            background-color: #50101C !important;
        }

        /* 11. Reference Design Hero & Layout */
        .top-nav-bar {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 10px;
            margin-bottom: 0.75rem;
        }
        .top-nav-text {
            font-size: 12px;
            font-weight: 500;
            color: #6E625E;
        }
        .top-nav-divider {
            color: #D6CBC1;
        }
        .user-avatar {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #50101C;
            color: #FFFFFF;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 700;
        }

        .hero-container {
            text-align: center;
            margin: 0.5rem auto 1.5rem auto;
            max-width: 860px;
        }
        .hero-eyebrow-wrapper {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 0.4rem;
        }
        .hero-eyebrow-line {
            height: 1px;
            width: 70px;
            background: #D6CBC1;
        }
        .hero-eyebrow-text {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.18em;
            color: #7D716A;
            text-transform: uppercase;
        }
        .hero-brand {
            font-size: clamp(42px, 5vw, 64px);
            font-weight: 800;
            letter-spacing: 0.12em;
            color: #50101C;
            line-height: 1;
            margin: 0.9rem 0 0.7rem;
        }
        .hero-title {
            font-size: clamp(42px, 5vw, 64px);
            font-weight: 800;
            letter-spacing: 0.12em;
            color: #50101C;
            line-height: 1;
            margin: 0.9rem 0 0.7rem;
        }
        .hero-tagline {
            font-family: 'Playfair Display', Georgia, serif;
            font-style: italic;
            font-size: clamp(22px, 2.2vw, 30px);
            font-weight: 600;
            color: #50101C;
            margin: 0 0 0.8rem 0;
        }
        .hero-description {
            font-size: 14px;
            color: #6E625E;
            margin: 0;
            line-height: 1.5;
        }

        .sidebar-workspace-note {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            margin: 0.2rem 0.25rem 0.9rem;
            padding: 0.7rem 0.75rem;
            border: 1px solid #E4D9CF;
            border-radius: 8px;
            background: rgba(255, 253, 250, 0.68);
            color: #6E625E;
            font-size: 10.5px;
            line-height: 1.45;
        }
        .sidebar-workspace-note strong {
            color: #50101C;
            font-size: 11px;
            font-weight: 700;
        }

        .hero-upload-card {
            max-width: 720px;
            margin: 0 auto 1.25rem;
            padding: 1.1rem 1.25rem 1rem;
            text-align: center;
            background: #FFFDFC;
            border: 1px solid #DCCFC4;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(80, 16, 28, 0.05);
        }
        .hero-upload-title {
            color: #50101C;
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .hero-upload-help {
            color: #8C7E75;
            font-size: 11px;
            margin-bottom: 0.7rem;
        }
        .hero-upload-card [data-testid="stFileUploader"] {
            max-width: 560px;
            margin: 0 auto 0.7rem;
            text-align: left;
        }
        .hero-upload-card [data-testid="stFileUploaderDropzone"] {
            background: #FAF7F2 !important;
            border-color: #D6CBC1 !important;
        }

        .floating-query-card {
            background: #FFFFFF;
            border: 1px solid #EBE4DC;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 24px rgba(80, 16, 28, 0.04);
            margin-bottom: 1.5rem;
        }

        .upload-dropzone-card {
            border: 1.5px dashed #D6CBC1;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.5);
            padding: 2rem;
            text-align: center;
            margin: 1.5rem auto;
            max-width: 680px;
        }
        .dropzone-icon {
            font-size: 32px;
            margin-bottom: 0.5rem;
        }
        .dropzone-title {
            font-size: 16px;
            font-weight: 700;
            color: #1E1815;
            margin-bottom: 0.25rem;
        }
        .dropzone-sub {
            font-size: 13px;
            color: #6E625E;
            margin-bottom: 0.2rem;
        }
        .dropzone-caption {
            font-size: 11px;
            color: #8C7E75;
            margin-bottom: 1.25rem;
        }

        /* 12. View Header Elements */
        .view-header {
            margin-bottom: 1.5rem;
        }
        .view-eyebrow {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #8C7E75;
            margin-bottom: 0.35rem;
        }
        .view-title {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #50101C;
            margin: 0 0 0.35rem 0;
            line-height: 1.3;
        }
        .view-subtitle {
            font-size: 13px;
            color: #6E625E;
            margin: 0;
            line-height: 1.5;
        }

        /* 12. Answer Surfaces */
        .answer-surface {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        .answer-eyebrow {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.4rem;
        }
        .answer-text {
            font-size: 16px;
            font-weight: 600;
            color: #0f172a;
            line-height: 1.55;
            letter-spacing: -0.01em;
            margin-bottom: 1rem;
        }
        .answer-meta-grid {
            display: grid;
            grid-template-columns: minmax(180px, 1.2fr) minmax(200px, 1fr);
            gap: 1.5rem;
            padding-top: 0.85rem;
            border-top: 1px solid #f1f5f9;
        }
        .meta-col-title {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .confidence-row {
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
            margin-bottom: 0.25rem;
        }
        .confidence-val {
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
            font-variant-numeric: tabular-nums;
        }
        .confidence-parts {
            font-size: 11px;
            color: #64748b;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        .source-tags-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            align-items: center;
        }
        .source-tag {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.04em;
            color: #334155;
            background-color: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 0.15rem 0.45rem;
            display: inline-block;
        }
        .source-tag-est {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.04em;
            color: #6b21a8;
            background-color: #faf5ff;
            border: 1px solid #f3e8ff;
            border-radius: 4px;
            padding: 0.15rem 0.45rem;
            display: inline-block;
        }

        /* 13. Empty State Box */
        .empty-box {
            background-color: #ffffff;
            border: 1px dashed #d1d5db;
            border-radius: 8px;
            padding: 2.25rem 1.5rem;
            text-align: center;
            margin: 1rem 0;
        }
        .empty-title {
            font-size: 13.5px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.25rem;
        }
        .empty-desc {
            font-size: 12.5px;
            color: #64748b;
            line-height: 1.5;
            max-width: 440px;
            margin: 0 auto;
        }

        /* 14. Receipts SaaS Card */
        .receipt-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.5rem;
        }
        .receipt-header {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            margin-bottom: 0.5rem;
        }
        .receipt-value {
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.01em;
        }
        .receipt-badge {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #15803d;
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 4px;
            padding: 0.12rem 0.4rem;
        }
        .receipt-grid {
            display: grid;
            grid-template-columns: minmax(140px, 1.8fr) minmax(120px, 1.4fr) minmax(90px, 1fr);
            gap: 0.85rem;
            padding-top: 0.5rem;
            border-top: 1px solid #f1f5f9;
        }
        .receipt-label {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .receipt-content {
            font-size: 12px;
            color: #1e293b;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            line-height: 1.4;
        }

        /* 15. Claims Cards */
        .claim-item {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.4rem;
        }
        .claim-row {
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
        }
        .verdict-tag {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-radius: 4px;
            padding: 0.12rem 0.4rem;
            white-space: nowrap;
        }
        .verdict-supported { color: #15803d; background: #f0fdf4; border: 1px solid #bbf7d0; }
        .verdict-partial { color: #b45309; background: #fffbeb; border: 1px solid #fde68a; }
        .verdict-unsupported { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; }
        .verdict-unchecked { color: #4b5563; background: #f3f4f6; border: 1px solid #e5e7eb; }
        .claim-text {
            font-size: 13px;
            color: #1e293b;
            line-height: 1.45;
            flex: 1;
        }
        .cite-badge {
            font-size: 11px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: #2563eb;
            background: #eff6ff;
            border: 1px solid #dbeafe;
            border-radius: 3px;
            padding: 0.08rem 0.35rem;
            margin-left: 0.35rem;
        }
        .claim-quote {
            font-size: 11.5px;
            color: #64748b;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            margin-top: 0.25rem;
            margin-left: 5rem;
        }

        /* 16. Refusal & Conflict Banners */
        .refusal-card {
            background-color: #ffffff;
            border: 1px solid #fca5a5;
            border-left: 4px solid #dc2626;
            border-radius: 6px;
            padding: 1.1rem 1.35rem;
            margin-bottom: 1.25rem;
        }
        .refusal-badge {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #b91c1c;
            margin-bottom: 0.25rem;
        }
        .refusal-reason {
            font-size: 14.5px;
            font-weight: 600;
            color: #0f172a;
        }
        .conflict-card {
            background-color: #ffffff;
            border: 1px solid #fcd34d;
            border-left: 4px solid #d97706;
            border-radius: 6px;
            padding: 0.85rem 1.15rem;
            margin-bottom: 0.75rem;
        }
        .conflict-title {
            font-size: 12.5px;
            font-weight: 600;
            color: #92400e;
            margin-bottom: 0.2rem;
        }
        .conflict-note {
            font-size: 12px;
            color: #451a03;
            margin-bottom: 0.35rem;
        }
        .conflict-facts {
            font-size: 11.5px;
            color: #78350f;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }

        /* 17. Evidence Inspector Header */
        .inspector-meta {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            display: flex;
            gap: 1.25rem;
            margin-bottom: 0.75rem;
            font-size: 12px;
            color: #334155;
        }
        .inspector-meta span {
            font-weight: 600;
            color: #0f172a;
        }

        /* 18. Bench Metrics Cards */
        .bench-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        .bench-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 0.85rem 1rem;
        }
        .bench-metric-name {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        .bench-metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.02em;
        }
        .bench-delta {
            font-size: 11px;
            color: #15803d;
            margin-top: 0.15rem;
            font-weight: 500;
        }

        /* 19. Document Quality Indicators */
        .quality-badge {
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-radius: 4px;
            padding: 0.12rem 0.4rem;
            display: inline-block;
        }
        .quality-good { color: #15803d; background: #f0fdf4; border: 1px solid #bbf7d0; }
        .quality-acceptable { color: #b45309; background: #fffbeb; border: 1px solid #fde68a; }
        .quality-low { color: #b91c1c; background: #fef2f2; border: 1px solid #fecaca; }

        /* Streamlit Dataframe / Container fine-tuning */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #eaebed !important;
            border-radius: 6px !important;
            background-color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str) -> None:
    """Render a minimal, clean empty state container."""
    st.markdown(
        f"""
        <div class="empty-box">
            <div class="empty-title">{title}</div>
            <div class="empty-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def highlight(page_png: str, bbox, color=(220, 38, 38)) -> Image.Image:
    """Highlight bounding box on page PNG with a clean, sharp rectangle."""
    img = Image.open(page_png).convert("RGB")
    s = RENDER_DPI / 72.0
    x0, y0, x1, y1 = (v * s for v in bbox)
    ImageDraw.Draw(img).rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=color, width=3)
    return img


def _clean_error(msg: str) -> str:
    """Strip raw API JSON/stack traces from refusal reasons for clean UX."""
    if not msg:
        return "No supporting evidence found in the ingested documents."
    # If it's a giant JSON blob or API error, show a short friendly message
    if any(k in msg for k in ("RESOURCE_EXHAUSTED", "429", "ClientError", "quota", "API key", "NOT_FOUND", "no longer available")):
        if "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
            return "⚠️ The AI model is currently overloaded or your quota is exceeded. Please wait a moment and try again."
        if "no longer available" in msg or "NOT_FOUND" in msg:
            return "⚠️ The configured AI model is unavailable. Please check your GEMINI_MODEL setting in .env."
        return "⚠️ An AI API error occurred. Please retry your question."
    # Trim very long error blobs
    if len(msg) > 300:
        return msg[:300].rsplit(" ", 1)[0] + "…"
    return msg


def answer_card(res: Result) -> None:
    """Render the primary answer card with confidence and sources breakdown."""
    a: Answer = res.answer
    if a.refused:
        clean_reason = _clean_error(a.refusal_reason or "")
        st.markdown(
            f"""
            <div class="refusal-card">
                <div class="refusal-badge">Refused by Truth Meter</div>
                <div class="refusal-reason">{clean_reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Modality badges
    badges_html = []
    for m in a.modalities:
        if m == "chart":
            badges_html.append('<span class="source-tag-est">CHART (ESTIMATED)</span>')
        else:
            badges_html.append(f'<span class="source-tag">{m.upper()}</span>')
    sources_markup = " ".join(badges_html) if badges_html else '<span class="source-tag">TEXT</span>'

    # Confidence breakdown
    parts_str = " · ".join(f"{k} {v:.2f}" for k, v in a.confidence_parts.items())

    st.markdown(
        f"""
        <div class="answer-surface">
            <div class="answer-eyebrow">ANSWER</div>
            <div class="answer-text">{a.text}</div>
            <div class="answer-meta-grid">
                <div>
                    <div class="meta-col-title">Truth Meter Confidence</div>
                    <div class="confidence-row">
                        <span class="confidence-val">{int(a.confidence * 100)}%</span>
                        <span class="confidence-parts">{parts_str}</span>
                    </div>
                </div>
                <div>
                    <div class="meta-col-title">Source Modalities</div>
                    <div class="source-tags-row">
                        {sources_markup}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if a.estimated:
        st.caption("Includes values read from a chart by the vision model. These are marked as estimated.")

    # Conflicts
    for c in a.conflicts:
        facts_list = "".join(
            f"<div>• {f.source_id} ({f.modality}{', estimated' if f.estimated else ''}): <b>{f.raw}</b></div>"
            for f in c.facts
        )
        st.markdown(
            f"""
            <div class="conflict-card">
                <div class="conflict-title">Conflict: {c.metric} ({c.period})</div>
                <div class="conflict-note">{c.note}</div>
                <div class="conflict-facts">{facts_list}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def claim_chips(res: Result) -> None:
    """Render claims verified by code with status tags and citations."""
    claims = res.answer.claims
    if not claims:
        return

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1rem;">VERIFIED CLAIMS</div>', unsafe_allow_html=True)

    verdict_class = {
        "supported": "verdict-supported",
        "partial": "verdict-partial",
        "unsupported": "verdict-unsupported",
        "unchecked": "verdict-unchecked",
    }

    for c in claims:
        v_cls = verdict_class.get(c.verdict, "verdict-unchecked")
        cites_html = "".join(f'<span class="cite-badge">{x}</span>' for x in c.cites)
        quote_html = f'<div class="claim-quote">Quote in source: "{c.quote}"</div>' if c.quote else ""
        reason_html = f'<div class="claim-quote">{"; ".join(c.reasons)}</div>' if c.reasons else ""

        st.markdown(
            f"""
            <div class="claim-item">
                <div class="claim-row">
                    <span class="verdict-tag {v_cls}">{c.verdict}</span>
                    <div class="claim-text">{c.text} {cites_html}</div>
                </div>
                {quote_html}
                {reason_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Pruned claims
    removed = [
        v
        for t in res.trace
        if t.step == "verify"
        for v in t.data.get("verdicts", [])
        if v["verdict"] == "unsupported"
    ]
    if removed:
        with st.expander(f"Claims pruned by verifier ({len(removed)})"):
            for v in removed:
                st.markdown(f"- ~~{v['text']}~~ — {'; '.join(v.get('reasons', []))}")


def receipts(res: Result) -> None:
    """Render deterministic receipts evaluated by Python AST (no LLM arithmetic)."""
    calcs = res.answer.calcs
    if not calcs:
        return

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1rem;">COMPUTATIONAL RECEIPTS</div>', unsafe_allow_html=True)

    for c in calcs:
        res_display = f"{c.result:,.4g}" if c.result is not None else "Error"
        inputs_str = " · ".join(f"{k} = {v:,.4g}" for k, v in c.inputs.items()) if c.inputs else "None"
        sources_str = ", ".join(c.sources) if c.sources else "None"

        st.markdown(
            f"""
            <div class="receipt-card">
                <div class="receipt-header">
                    <span class="receipt-value">{res_display}</span>
                    <span class="receipt-badge">Verified calculation</span>
                </div>
                <div class="receipt-grid">
                    <div>
                        <div class="receipt-label">Formula</div>
                        <div class="receipt-content">{c.name} = {c.expr}</div>
                    </div>
                    <div>
                        <div class="receipt-label">Inputs</div>
                        <div class="receipt-content">{inputs_str}</div>
                    </div>
                    <div>
                        <div class="receipt-label">Sources</div>
                        <div class="receipt-content">{sources_str}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def evidence_viewer(res: Result, store) -> None:
    """Render inspector panel for selected cited element."""
    ids = sorted({x for c in res.answer.claims for x in c.cites}) or [e.id for e in res.evidence]
    if not ids:
        empty_state("No evidence selected", "Select a source from the answer to inspect the underlying document evidence.")
        return

    st.markdown('<div class="sidebar-section-label" style="margin-top: 1rem;">EVIDENCE INSPECTOR</div>', unsafe_allow_html=True)
    pick = st.selectbox("Inspect source element", ids, key=f"ev-{res.question}", label_visibility="collapsed")
    el: Element | None = store.get(pick)
    if el is None:
        return

    st.markdown(
        f"""
        <div class="inspector-meta">
            <div>Document: <span>{el.doc_name or el.doc_id}</span></div>
            <div>Page: <span>{el.page}</span></div>
            <div>Modality: <span>{el.modality.upper()}</span></div>
            <div>Element ID: <span>{el.id}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = next((p for p in store.pages(el.doc_id) if p["page"] == el.page), None)
    cols = st.columns([1, 1])

    with cols[0]:
        if page and Path(page["png"]).exists():
            st.image(highlight(page["png"], el.bbox), use_container_width=True)
        else:
            st.caption("No page image available.")

    with cols[1]:
        st.text_area("Extracted text", el.text, height=240, key=f"txt-{pick}")
        if el.kind == "figure" and el.meta.get("chart"):
            st.caption("ChartLens structured values are marked as estimated.")


def proof_graph(res: Result) -> None:
    """Render ProofGraph generated directly from the execution trace."""
    st.markdown('<div class="sidebar-section-label" style="margin-top: 1.5rem;">PROOFGRAPH TRACE</div>', unsafe_allow_html=True)
    st.graphviz_chart(build_dot(res), use_container_width=True)
    with st.expander("Inspect JSON trace"):
        st.json([t.model_dump() for t in res.trace])


def triage_strip(store) -> None:
    """Render compact Document Quality & Page Triage overview."""
    docs = store.docs()
    if not docs:
        empty_state("No documents available", "Upload PDFs or load the demo set to inspect page quality.")
        return

    for d in docs:
        st.markdown(f"**{d['name']}** (`{d['id']}` · {d['pages']} pages)")
        pages = store.pages(d["id"])
        if not pages:
            continue

        rows = []
        for p in pages:
            q = p["quality"]
            if q >= 0.85:
                status_html = '<span class="quality-badge quality-good">Good</span>'
            elif q >= 0.70:
                status_html = '<span class="quality-badge quality-acceptable">Acceptable</span>'
            else:
                status_html = '<span class="quality-badge quality-low">Low quality</span>'

            rows.append({
                "Page": f"Page {p['page']}",
                "Classification": p["kind"].capitalize(),
                "Quality Score": f"{q:.2f}",
                "Status": status_html,
                "Notes": p["notes"] or "—",
            })

        # Display quality table
        df = pd.DataFrame(rows)
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Thumbnail row
        with st.expander(f"View page thumbnails ({d['name']})"):
            cols = st.columns(max(len(pages), 1))
            for col, p in zip(cols, pages):
                if Path(p["png"]).exists():
                    col.image(p["png"], use_container_width=True)
                col.caption(f"p.{p['page']} · {p['kind']}")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)


def chartlens(store) -> None:
    """Render extracted figures and tabular estimates from ChartLens."""
    figs = [e for e in store.all_elements() if e.kind == "figure"]
    if not figs:
        empty_state("No charts extracted", "No figure elements were detected in the ingested documents.")
        return

    for f in figs:
        st.markdown(f"**{f.id}** · {f.doc_name} (Page {f.page})")
        c1, c2 = st.columns([1, 1])
        if f.image_path and Path(f.image_path).exists():
            c1.image(f.image_path, use_container_width=True)
        else:
            c1.caption("No chart image file on disk.")

        chart = f.meta.get("chart")
        if chart:
            with c2:
                st.markdown(
                    f"""
                    <div style="font-size: 13px; font-weight: 600; color: #0f172a; margin-bottom: 0.2rem;">
                        {chart.get('title', 'Chart Data')} <span class="quality-badge quality-acceptable">Estimated</span>
                    </div>
                    <div style="font-size: 11px; color: #64748b; margin-bottom: 0.5rem;">Unit: {chart.get('unit', 'value')}</div>
                    """,
                    unsafe_allow_html=True,
                )
                df_chart = pd.DataFrame(chart["rows"], columns=chart["header"])
                st.dataframe(df_chart, use_container_width=True, hide_index=True)
        else:
            c2.info("Chart data not extracted (no cached run or API key).")
        st.markdown("<div style='margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)


def factledger_view(store) -> None:
    """Render compact FactLedger data table or clean empty state."""
    facts = store.conn.execute("SELECT * FROM facts ORDER BY metric_norm, period").fetchall()
    if not facts:
        empty_state(
            "No extracted facts yet",
            "Upload or process a document to populate the FactLedger with normalized metrics and values.",
        )
        return

    rows = []
    for r in facts:
        d = dict(r)
        rows.append({
            "Metric": d["metric"],
            "Period": d["period"] or "—",
            "Normalized Value": f"{d['value']:,.4g}",
            "Unit": d["unit"] or "—",
            "Raw in Source": d["raw"] or "—",
            "Source ID": d["source_id"],
            "Modality": d["modality"].upper(),
            "Estimated": "Yes" if d["estimated"] else "No",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def bench_dashboard(full_summary: dict, base_summary: Optional[dict] = None, full_rows: Optional[list] = None) -> None:
    """Render clean Evaluation Bench metrics cards, comparison, and question records."""
    if not full_summary:
        empty_state(
            "No benchmark results",
            "Run the benchmark to evaluate accuracy across text, table, chart, and scanned test questions.",
        )
        return

    ans_acc = full_summary.get("answer_acc", 0.0)
    cite_acc = full_summary.get("citation_acc", 0.0)
    num_acc = full_summary.get("numeric_acc", 0.0)
    ref_acc = full_summary.get("refusal_acc", 0.0)

    delta_ans = f"+{ans_acc - base_summary.get('answer_acc', 0):.1f}% vs baseline" if base_summary else ""
    delta_cite = f"+{cite_acc - base_summary.get('citation_acc', 0):.1f}% vs baseline" if base_summary else ""
    delta_num = f"+{num_acc - base_summary.get('numeric_acc', 0):.1f}% vs baseline" if base_summary else ""
    delta_ref = f"+{ref_acc - base_summary.get('refusal_acc', 0):.1f}% vs baseline" if base_summary else ""

    st.markdown(
        f"""
        <div class="bench-grid">
            <div class="bench-card">
                <div class="bench-metric-name">Answer Accuracy</div>
                <div class="bench-metric-value">{ans_acc}%</div>
                <div class="bench-delta">{delta_ans}</div>
            </div>
            <div class="bench-card">
                <div class="bench-metric-name">Citation Accuracy</div>
                <div class="bench-metric-value">{cite_acc}%</div>
                <div class="bench-delta">{delta_cite}</div>
            </div>
            <div class="bench-card">
                <div class="bench-metric-name">Numeric Accuracy</div>
                <div class="bench-metric-value">{num_acc}%</div>
                <div class="bench-delta">{delta_num}</div>
            </div>
            <div class="bench-card">
                <div class="bench-metric-name">Refusal Accuracy</div>
                <div class="bench-metric-value">{ref_acc}%</div>
                <div class="bench-delta">{delta_ref}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if base_summary:
        st.markdown('<div class="sidebar-section-label">FULL PIPELINE VS BASELINE (PLAIN RAG)</div>', unsafe_allow_html=True)
        comparison_rows = [
            {"Metric": "Answer Accuracy", "COGNIVUE (Full)": f"{ans_acc}%", "Baseline (Plain RAG)": f"{base_summary.get('answer_acc', 0)}%"},
            {"Metric": "Citation Accuracy", "COGNIVUE (Full)": f"{cite_acc}%", "Baseline (Plain RAG)": f"{base_summary.get('citation_acc', 0)}%"},
            {"Metric": "Numeric Accuracy", "COGNIVUE (Full)": f"{num_acc}%", "Baseline (Plain RAG)": f"{base_summary.get('numeric_acc', 0)}%"},
            {"Metric": "Refusal Accuracy", "COGNIVUE (Full)": f"{ref_acc}%", "Baseline (Plain RAG)": f"{base_summary.get('refusal_acc', 0)}%"},
        ]
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    if full_rows:
        st.markdown('<div class="sidebar-section-label" style="margin-top: 1rem;">TEST CASE EVALUATION DETAILS</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(full_rows), use_container_width=True, hide_index=True)
