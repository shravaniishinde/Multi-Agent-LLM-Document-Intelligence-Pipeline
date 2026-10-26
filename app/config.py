"""Runtime configuration.

Everything defaults to the offline implementations, so the pipeline runs with no
API keys. Real providers are switched on purely through environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


class Settings(BaseModel):
    llm_provider: Literal["fake", "openai", "anthropic", "groq"] = "fake"
    llm_model: str | None = None
    ocr_provider: Literal["mock", "tesseract"] = "mock"
    vector_store: Literal["memory", "pinecone"] = "memory"
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_dim: int = 256

    pinecone_api_key: str | None = None
    pinecone_index: str = "doc-intel-reference"
    pinecone_namespace: str = "reference"

    # Bounded loops: at most this many re-OCR attempts per document.
    max_retries: int = Field(default=2, ge=0, le=5)
    # Structured-output attempts before the structuring agent gives up.
    max_structuring_attempts: int = Field(default=2, ge=1, le=5)
    # Below this OCR confidence a missing field is treated as a likely OCR miss.
    ocr_confidence_threshold: float = Field(default=0.80, ge=0, le=1)
    # Below this record confidence the supervisor will not auto-approve.
    approval_confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    rag_top_k: int = 3

    fixtures_dir: Path = FIXTURES_DIR

    @classmethod
    def from_env(cls) -> "Settings":
        try:  # optional: pick up a local .env file
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
        except ImportError:
            pass
        env = os.environ
        values: dict = {
            "llm_provider": env.get("LLM_PROVIDER", "fake"),
            "llm_model": env.get("LLM_MODEL") or None,
            "ocr_provider": env.get("OCR_PROVIDER", "mock"),
            "vector_store": env.get("VECTOR_STORE", "memory"),
            "embedding_provider": env.get("EMBEDDING_PROVIDER", "hashing"),
            "pinecone_api_key": env.get("PINECONE_API_KEY") or None,
            "pinecone_index": env.get("PINECONE_INDEX", "doc-intel-reference"),
            "pinecone_namespace": env.get("PINECONE_NAMESPACE", "reference"),
        }
        for key, name in [
            ("max_retries", "MAX_RETRIES"),
            ("max_structuring_attempts", "MAX_STRUCTURING_ATTEMPTS"),
            ("embedding_dim", "EMBEDDING_DIM"),
            ("rag_top_k", "RAG_TOP_K"),
        ]:
            if env.get(name):
                values[key] = int(env[name])
        for key, name in [
            ("ocr_confidence_threshold", "OCR_CONFIDENCE_THRESHOLD"),
            ("approval_confidence_threshold", "APPROVAL_CONFIDENCE_THRESHOLD"),
        ]:
            if env.get(name):
                values[key] = float(env[name])
        return cls(**values)
