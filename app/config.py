"""Runtime configuration.

Everything defaults to the offline implementations, so the pipeline runs with no
API keys. Real providers are switched on purely through environment variables
(see `.env.example`). Invalid values fail fast at startup with a clear message.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class ConfigError(ValueError):
    pass


class Settings(BaseModel):
    llm_provider: Literal["fake", "openai", "anthropic", "groq"] = "fake"
    llm_model: str | None = None
    ocr_provider: Literal["mock", "tesseract"] = "mock"
    vector_store: Literal["memory", "pinecone"] = "memory"
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_dim: int = Field(default=256, ge=8, le=4096)

    # Pinecone - read from the environment only; SecretStr keeps the key out of logs/reprs.
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "doc-intel-reference"
    pinecone_namespace: str = "reference"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    pinecone_create_index: bool = False

    # Bounded loops: at most this many re-OCR attempts per document.
    max_retries: int = Field(default=2, ge=0, le=5)
    # Structured-output attempts before the structuring agent gives up.
    max_structuring_attempts: int = Field(default=2, ge=1, le=5)
    # Below this OCR confidence a missing field is treated as a likely OCR miss.
    ocr_confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    # Below this record confidence the supervisor will not auto-approve.
    approval_confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    rag_top_k: int = Field(default=3, ge=1, le=20)

    batch_max_concurrency: int = Field(default=4, ge=1, le=32)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    documents_dir: Path = DATA_DIR / "sample_documents"
    reference_dir: Path = DATA_DIR / "reference"

    @model_validator(mode="after")
    def _real_llm_needs_model(self) -> "Settings":
        if self.llm_provider != "fake" and not self.llm_model:
            raise ValueError(f"LLM_MODEL must be set when LLM_PROVIDER={self.llm_provider}")
        return self

    @property
    def pinecone_configured(self) -> bool:
        return self.pinecone_api_key is not None and bool(self.pinecone_api_key.get_secret_value())

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        if env is None:
            try:  # optional: pick up a local .env file
                from dotenv import load_dotenv

                load_dotenv(PROJECT_ROOT / ".env")
            except ImportError:
                pass
            env = dict(os.environ)
        # Each field maps to its upper-case env var; empty values mean "use the default".
        values = {name: env[name.upper()] for name in cls.model_fields
                  if env.get(name.upper(), "").strip()}
        try:
            return cls(**values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])).upper() or 'config'}: {e['msg']}"
                for e in exc.errors())
            raise ConfigError(f"invalid configuration - {problems}") from None
