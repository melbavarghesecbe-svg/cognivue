"""SQLite element store: the single source of truth for doc, page and bbox."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from .schema import Element

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, name TEXT, path TEXT, pages INTEGER);
CREATE TABLE IF NOT EXISTS pages (doc_id TEXT, page INTEGER, kind TEXT, quality REAL, png TEXT,
    notes TEXT, PRIMARY KEY (doc_id, page));
CREATE TABLE IF NOT EXISTS elements (id TEXT PRIMARY KEY, doc_id TEXT, page INTEGER, kind TEXT,
    section TEXT, text TEXT, bbox TEXT, image_path TEXT, quality REAL, meta TEXT);
CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, metric TEXT, metric_norm TEXT,
    period TEXT, value REAL, unit TEXT, raw TEXT, source_id TEXT, doc_id TEXT, modality TEXT, estimated INTEGER);
"""


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # -- docs / pages --------------------------------------------------------
    def next_doc_id(self) -> str:
        n = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        return f"D{n + 1}"

    def find_doc(self, name: str) -> Optional[str]:
        row = self.conn.execute("SELECT id FROM docs WHERE name=?", (name,)).fetchone()
        return row["id"] if row else None

    def add_doc(self, doc_id: str, name: str, path: str, pages: int) -> None:
        self.conn.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)", (doc_id, name, path, pages))
        self.conn.commit()

    def docs(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM docs ORDER BY id")]

    def doc_name(self, doc_id: str) -> str:
        row = self.conn.execute("SELECT name FROM docs WHERE id=?", (doc_id,)).fetchone()
        return row["name"] if row else doc_id

    def add_page(self, doc_id: str, page: int, kind: str, quality: float, png: str, notes: str = "") -> None:
        self.conn.execute("INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?)", (doc_id, page, kind, quality, png, notes))
        self.conn.commit()

    def pages(self, doc_id: Optional[str] = None) -> list[dict]:
        q, args = "SELECT * FROM pages", ()
        if doc_id:
            q, args = q + " WHERE doc_id=?", (doc_id,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY doc_id, page", args)]

    def page_quality(self, doc_id: str, page: int) -> float:
        row = self.conn.execute("SELECT quality FROM pages WHERE doc_id=? AND page=?", (doc_id, page)).fetchone()
        return float(row["quality"]) if row else 1.0

    # -- elements ------------------------------------------------------------
    def add_elements(self, elements: Iterable[Element]) -> None:
        rows = [
            (e.id, e.doc_id, e.page, e.kind, e.section, e.text, json.dumps(e.bbox), e.image_path, e.quality, json.dumps(e.meta))
            for e in elements
        ]
        self.conn.executemany("INSERT OR REPLACE INTO elements VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def _row_to_element(self, r: sqlite3.Row) -> Element:
        return Element(
            id=r["id"], doc_id=r["doc_id"], doc_name=self.doc_name(r["doc_id"]), page=r["page"], kind=r["kind"],
            section=r["section"] or "", text=r["text"] or "", bbox=tuple(json.loads(r["bbox"])),
            image_path=r["image_path"], quality=r["quality"], meta=json.loads(r["meta"] or "{}"),
        )

    def get(self, element_id: str) -> Optional[Element]:
        r = self.conn.execute("SELECT * FROM elements WHERE id=?", (element_id,)).fetchone()
        return self._row_to_element(r) if r else None

    def get_many(self, ids: Iterable[str]) -> list[Element]:
        return [e for e in (self.get(i) for i in ids) if e is not None]

    def all_elements(self, doc_id: Optional[str] = None) -> list[Element]:
        q, args = "SELECT * FROM elements", ()
        if doc_id:
            q, args = q + " WHERE doc_id=?", (doc_id,)
        return [self._row_to_element(r) for r in self.conn.execute(q + " ORDER BY doc_id, page, id", args)]

    def delete_doc(self, doc_id: str) -> None:
        for t in ("docs WHERE id", "pages WHERE doc_id", "elements WHERE doc_id", "facts WHERE doc_id"):
            self.conn.execute(f"DELETE FROM {t}=?", (doc_id,))
        self.conn.commit()
