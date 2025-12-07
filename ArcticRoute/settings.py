"""Centralised environment-driven settings."""

from __future__ import annotations

try:
    from pydantic import BaseSettings
except ImportError:  # pragma: no cover - fallback for Pydantic v2
    try:
        from pydantic.v1 import BaseSettings  # type: ignore[attr-defined]
    except ImportError:
        from pydantic_settings import BaseSettings  # type: ignore[assignment]


class Settings(BaseSettings):
    """Application settings sourced from environment variables."""

    DATA_DIR: str = "./data"
    CACHE_DIR: str = "./.cache"
    LOG_DIR: str = "./logs"
    ENV: str = "dev"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]
