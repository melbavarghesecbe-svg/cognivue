import json
from pathlib import Path

import pytest

from cognivue.config import Settings
from cognivue.llm import LLM


class FakeProvider:
    """Returns canned JSON by purpose keyword found in the prompt; counts calls."""

    def __init__(self, replies=None, name="fake", fail_times=0):
        self.name = name
        self.replies = replies or {}
        self.calls = 0
        self.fail_times = fail_times

    def generate(self, prompt, images):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("boom")
        for key, reply in self.replies.items():
            if key in prompt:
                return reply if isinstance(reply, str) else json.dumps(reply)
        return "{}"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache", gemini_api_key="", llm_backoff_s=0)
    for d in (s.data_dir, s.cache_dir, s.pages_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s


@pytest.fixture
def offline_llm(settings) -> LLM:
    """No providers + CACHE_ONLY: every VLM call soft-fails."""
    return LLM([], settings.cache_dir, cache_only=True, backoff_s=0)
