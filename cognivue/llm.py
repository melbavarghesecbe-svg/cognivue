"""Single gateway for every model call.

Disk cache, temperature 0, retry with backoff, fallback provider,
Pydantic JSON validation (one repair attempt) and a CACHE_ONLY switch.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional, Protocol, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Image = tuple[str, bytes]  # (label, png bytes)


class LLMError(Exception):
    pass


class CacheMiss(LLMError):
    pass


class Provider(Protocol):
    name: str

    def generate(self, prompt: str, images: Sequence[Image]) -> str: ...


class GeminiProvider:
    def __init__(self, api_key: str, model: str):
        self.name = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, images: Sequence[Image]) -> str:
        from google.genai import types

        parts: list = []
        for label, data in images:
            parts.append(f"[image {label}]")
            parts.append(types.Part.from_bytes(data=data, mime_type="image/png"))
        parts.append(prompt)
        resp = self._get_client().models.generate_content(
            model=self.name,
            contents=parts,
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
        )
        return resp.text or ""


def cache_key(purpose: str, prompt: str, schema: type[BaseModel], images: Sequence[Image]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([purpose, prompt, schema.__name__]).encode())
    for label, data in images:
        h.update(label.encode())
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()[:32]


def extract_json(text: str) -> str:
    """Strip markdown fences / chatter around a JSON object."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return text[start:].strip()


class LLM:
    def __init__(
        self,
        providers: Sequence[Provider],
        cache_dir: Path,
        cache_only: bool = False,
        retries: int = 3,
        backoff_s: float = 2.0,
    ):
        self.providers = list(providers)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_only = cache_only
        self.retries = retries
        self.backoff_s = backoff_s

    # -- cache -------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, entry: dict) -> None:
        self._cache_path(key).write_text(json.dumps(entry, indent=1, ensure_ascii=False), encoding="utf-8")

    # -- calls -------------------------------------------------------------
    def call_json(
        self,
        purpose: str,
        prompt: str,
        schema: type[T],
        images: Sequence[Image] = (),
        trace: Optional[list] = None,
    ) -> T:
        key = cache_key(purpose, prompt, schema, images)
        log = {"purpose": purpose, "key": key, "images": [lbl for lbl, _ in images]}
        cached = self._read_cache(key)
        if cached is not None:
            try:
                out = schema.model_validate(cached["output"])
                _log(trace, log | {"cache": "hit", "model": cached.get("model", "")})
                return out
            except (ValidationError, KeyError):
                pass
        if self.cache_only:
            _log(trace, log | {"cache": "miss", "error": "CACHE_ONLY"})
            raise CacheMiss(f"{purpose}: not in cache and CACHE_ONLY is on")

        errors: list[str] = []
        for provider in self.providers:
            try:
                out = self._call_provider(provider, prompt, schema, images)
            except LLMError as e:
                errors.append(f"{provider.name}: {e}")
                continue
            self._write_cache(key, {"purpose": purpose, "model": provider.name, "output": out.model_dump(mode="json")})
            _log(trace, log | {"cache": "miss", "model": provider.name})
            return out
        _log(trace, log | {"cache": "miss", "error": "; ".join(errors)})
        raise LLMError("; ".join(errors) or "no providers configured")

    def _call_provider(self, provider: Provider, prompt: str, schema: type[T], images: Sequence[Image]) -> T:
        full = f"{prompt}\n\nReturn ONLY JSON matching this schema:\n{json.dumps(schema.model_json_schema())}"
        last = ""
        for attempt in range(self.retries):
            try:
                raw = provider.generate(full, images)
            except Exception as e:  # network/quota errors from the SDK
                last = f"{type(e).__name__}: {e}"
                time.sleep(self.backoff_s * (2**attempt))
                continue
            try:
                return schema.model_validate_json(extract_json(raw))
            except ValidationError as e:
                last = f"invalid JSON: {str(e)[:300]}"
                full = f"{prompt}\n\nYour previous reply was invalid ({last}). Return ONLY valid JSON for schema:\n{json.dumps(schema.model_json_schema())}"
        raise LLMError(last)


def _log(trace: Optional[list], entry: dict) -> None:
    if trace is not None:
        trace.append(entry)


def get_llm(settings=None) -> LLM:
    from .config import get_settings

    s = settings or get_settings()
    providers: list[Provider] = []
    if s.gemini_api_key:
        providers = [GeminiProvider(s.gemini_api_key, s.gemini_model), GeminiProvider(s.gemini_api_key, s.gemini_fallback_model)]
    return LLM(providers, s.cache_dir, cache_only=s.cache_only, retries=s.llm_retries, backoff_s=s.llm_backoff_s)
