"""Vector store adapters: in-memory (default) and Pinecone."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.observability import log_event


class VectorRecord(BaseModel):
    id: str
    values: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryMatch(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None: ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int = 3,
              filter: dict[str, Any] | None = None) -> list[QueryMatch]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine search. Fine for reference data in the hundreds."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.id] = record

    def query(self, vector: list[float], top_k: int = 3,
              filter: dict[str, Any] | None = None) -> list[QueryMatch]:
        candidates = [
            r for r in self._records.values()
            if not filter or all(r.metadata.get(k) == v for k, v in filter.items())
        ]
        scored = [QueryMatch(id=r.id, score=_cosine(vector, r.values), metadata=r.metadata)
                  for r in candidates]
        # Sort by score, then id, so ties are deterministic.
        scored.sort(key=lambda m: (-m.score, m.id))
        return scored[:top_k]


class PineconeVectorStore(VectorStore):
    """Pinecone serverless index. Pass `index` to inject a client (used in tests).

    Plain equality filters are translated to Pinecone's `{"field": {"$eq": v}}`;
    values that are already operator dicts (`{"$in": [...]}`) pass through.
    """

    @classmethod
    def from_settings(cls, settings: Settings, client: Any = None,
                      spec_factory: Any = None) -> "PineconeVectorStore":
        """Connect using env-derived settings; optionally create the index if missing."""
        if not settings.pinecone_configured:
            raise ValueError("PINECONE_API_KEY is not set")
        if client is None:
            from pinecone import Pinecone

            client = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        name = settings.pinecone_index
        if settings.pinecone_create_index and not _index_exists(client, name):
            if spec_factory is None:
                from pinecone import ServerlessSpec as spec_factory
            client.create_index(name=name, dimension=settings.embedding_dim,
                                metric=settings.pinecone_metric,
                                spec=spec_factory(cloud=settings.pinecone_cloud,
                                                  region=settings.pinecone_region))
            log_event("pinecone_index_created", index=name, dimension=settings.embedding_dim)
        return cls(index=client.Index(name), namespace=settings.pinecone_namespace)

    def __init__(self, api_key: str | None = None, index_name: str | None = None,
                 namespace: str = "", index: Any = None):
        if index is None:
            if not api_key or not index_name:
                raise ValueError("PineconeVectorStore needs PINECONE_API_KEY and PINECONE_INDEX")
            from pinecone import Pinecone

            index = Pinecone(api_key=api_key).Index(index_name)
        self.index = index
        self.namespace = namespace

    def upsert(self, records: list[VectorRecord]) -> None:
        self.index.upsert(
            vectors=[{"id": r.id, "values": r.values,
                      # Pinecone rejects null metadata values.
                      "metadata": {k: v for k, v in r.metadata.items() if v is not None}}
                     for r in records],
            namespace=self.namespace,
        )

    def query(self, vector: list[float], top_k: int = 3,
              filter: dict[str, Any] | None = None) -> list[QueryMatch]:
        kwargs: dict[str, Any] = {"vector": vector, "top_k": top_k,
                                  "include_metadata": True, "namespace": self.namespace}
        if filter:
            kwargs["filter"] = {k: v if isinstance(v, dict) else {"$eq": v}
                                for k, v in filter.items()}
        response = self.index.query(**kwargs)
        matches = response["matches"] if isinstance(response, dict) else response.matches
        return [
            QueryMatch(id=_get(m, "id"), score=float(_get(m, "score")),
                       metadata=dict(_get(m, "metadata") or {}))
            for m in matches
        ]


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _index_exists(client: Any, name: str) -> bool:
    if hasattr(client, "has_index"):
        return bool(client.has_index(name))
    return name in client.list_indexes().names()


def build_vector_store(settings: Settings) -> VectorStore:
    """Pinecone when selected *and* usable; otherwise the offline in-memory store.

    Missing credentials or a missing `pinecone` package degrade to offline mode
    with a warning instead of stopping the service. Bad credentials still fail.
    """
    if settings.vector_store != "pinecone":
        return InMemoryVectorStore()
    if not settings.pinecone_configured:
        log_event("vector_store_fallback", level=logging.WARNING,
                  reason="PINECONE_API_KEY not set", using="memory")
        return InMemoryVectorStore()
    try:
        return PineconeVectorStore.from_settings(settings)
    except ImportError:
        log_event("vector_store_fallback", level=logging.WARNING,
                  reason="pinecone package not installed", using="memory")
        return InMemoryVectorStore()
