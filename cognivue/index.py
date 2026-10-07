"""Hybrid index: bge-small dense vectors in Chroma + BM25, fused with reciprocal rank fusion."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

from rank_bm25 import BM25Okapi

from .schema import Element

Embedder = Callable[[list[str]], list[list[float]]]
RRF_K = 60


MONTH_SYNONYMS: dict[str, str] = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr", "june": "jun",
    "july": "jul", "august": "aug", "september": "sep", "october": "oct", "november": "nov", "december": "dec",
    "jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june",
    "jul": "july", "aug": "august", "sep": "september", "oct": "october", "nov": "november", "dec": "december",
}


def doc_alias(doc_id: str) -> str:
    """Map D1 -> Document A / Doc 1, D2 -> Document B / Doc 2, etc."""
    try:
        num = int(doc_id.replace("D", ""))
        letter = chr(64 + num) if 1 <= num <= 26 else ""
        return f"Document {letter} Doc {num} Document {num}".strip()
    except Exception:
        return ""


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    res = []
    for t in toks:
        res.append(t)
        if t in MONTH_SYNONYMS:
            res.append(MONTH_SYNONYMS[t])
    return res


def hash_embedder(texts: list[str], dim: int = 256) -> list[list[float]]:
    """Deterministic offline fallback (bag of hashed tokens, L2-normalised). Used in tests."""
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in tokenize(t):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def fastembed_embedder(model: str, cache_dir: str) -> Embedder:
    from fastembed import TextEmbedding

    m = TextEmbedding(model_name=model, cache_dir=cache_dir)
    return lambda texts: [list(map(float, v)) for v in m.embed(texts)]


def load_embedder(settings, trace: Optional[list] = None) -> tuple[Embedder, str]:
    if settings.embed_model == "hash":  # explicit offline mode, no model download
        return hash_embedder, "hash"
    try:
        return fastembed_embedder(settings.embed_model, str(settings.data_dir / "models")), settings.embed_model
    except Exception as e:  # offline with no model downloaded
        if trace is not None:
            trace.append({"step": "embedder_fallback", "error": str(e)[:200]})
        return hash_embedder, "hash-fallback"


def index_text(e: Element) -> str:
    alias = doc_alias(e.doc_id)
    return f"{e.doc_id} {alias} {e.doc_name} {e.section}\n{e.text}".strip()


@dataclass
class Hit:
    id: str
    rrf: float
    dense: float  # cosine similarity 0..1
    bm25: float  # normalized BM25 relevance 0..1


class Index:
    def __init__(self, embedder: Embedder, chroma_dir: Optional[str] = None, name: str = "elements"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        cs = ChromaSettings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=chroma_dir, settings=cs) if chroma_dir else chromadb.EphemeralClient(settings=cs)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        self.col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        self.embed = embedder
        self.ids: list[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.doc_of: dict[str, str] = {}

    def build(self, elements: list[Element]) -> None:
        els = [e for e in elements if e.kind != "heading" and e.text.strip()]
        if not els:
            return
        self.ids = [e.id for e in els]
        self.doc_of = {e.id: e.doc_id for e in els}
        texts = [index_text(e) for e in els]
        self.col.add(ids=self.ids, embeddings=self.embed(texts), documents=texts,
                     metadatas=[{"doc_id": e.doc_id, "kind": e.kind} for e in els])
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if not self.ids or self.bm25 is None:
            return []
        n = min(len(self.ids), max(k * 3, 20))
        res = self.col.query(query_embeddings=self.embed([query]), n_results=n)
        dense_ids = res["ids"][0]
        dense_sim = {i: 1.0 - d for i, d in zip(dense_ids, res["distances"][0])}
        bm = self.bm25.get_scores(tokenize(query))
        bm_rank = sorted(range(len(self.ids)), key=lambda i: -bm[i])[:n]
        fused: dict[str, float] = {}
        for rank, i in enumerate(dense_ids):
            fused[i] = fused.get(i, 0) + 1 / (RRF_K + rank + 1)
        for rank, j in enumerate(bm_rank):
            if bm[j] > 0:
                fused[self.ids[j]] = fused.get(self.ids[j], 0) + 1 / (RRF_K + rank + 1)
        max_bm = max((float(score) for score in bm), default=0.0)
        bm_by_id = {
            self.ids[j]: (float(bm[j]) / max_bm if max_bm > 0 else 0.0)
            for j in range(len(self.ids))
        }
        hits = [Hit(i, s, max(dense_sim.get(i, 0.0), 0.0), bm_by_id.get(i, 0.0)) for i, s in fused.items()]
        return sorted(hits, key=lambda h: -h.rrf)[:k]
