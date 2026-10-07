"""Settings loaded from .env / environment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_fallback_model: str = "gemini-2.0-flash"
    cache_only: bool = False
    llm_retries: int = 3
    llm_backoff_s: float = 2.0

    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / "cache" / "llm"  # committed so CACHE_ONLY demos work offline

    embed_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6
    min_retrieval_score: float = 0.35
    ocr_min_conf: float = 0.80

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cognivue.db"

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.data_dir, s.cache_dir, s.pages_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
