"""Vector store adapters: in-memory (default) and Pinecone."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings


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

    Metadata equality filters are translated to Pinecone's `{"field": {"$eq": v}}`.
    """

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
            vectors=[{"id": r.id, "values": r.values, "metadata": r.metadata} for r in records],
            namespace=self.namespace,
        )

    def query(self, vector: list[float], top_k: int = 3,
              filter: dict[str, Any] | None = None) -> list[QueryMatch]:
        kwargs: dict[str, Any] = {"vector": vector, "top_k": top_k,
                                  "include_metadata": True, "namespace": self.namespace}
        if filter:
            kwargs["filter"] = {k: {"$eq": v} for k, v in filter.items()}
        response = self.index.query(**kwargs)
        matches = response["matches"] if isinstance(response, dict) else response.matches
        return [
            QueryMatch(id=_get(m, "id"), score=float(_get(m, "score")),
                       metadata=dict(_get(m, "metadata") or {}))
            for m in matches
        ]


def _get(obj: Any, key: str) -> Any:
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "pinecone":
        return PineconeVectorStore(settings.pinecone_api_key, settings.pinecone_index,
                                   settings.pinecone_namespace)
    return InMemoryVectorStore()
