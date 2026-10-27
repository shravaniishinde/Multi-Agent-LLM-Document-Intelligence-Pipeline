"""Chunk -> embed -> store -> retrieve.

The retriever indexes *reference* knowledge (customer master data, document
policies), not the documents being processed. Agents use it to ask "is this
value consistent with what we already know?"
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.adapters.vector_store import VectorRecord, VectorStore
from app.rag.embeddings import EmbeddingProvider


class ReferenceDocument(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Split into word windows of `chunk_size` words overlapping by `overlap`."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    return [" ".join(words[i:i + chunk_size])
            for i in range(0, max(len(words) - overlap, 1), step)]


class Retriever:
    def __init__(self, embeddings: EmbeddingProvider, store: VectorStore,
                 chunk_size: int = 80, overlap: int = 15):
        self.embeddings = embeddings
        self.store = store
        self.chunk_size = chunk_size
        self.overlap = overlap

    def index(self, documents: list[ReferenceDocument]) -> int:
        records: list[VectorRecord] = []
        for doc in documents:
            chunks = chunk_text(doc.text, self.chunk_size, self.overlap)
            vectors = self.embeddings.embed_documents(chunks)
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                records.append(VectorRecord(
                    id=f"{doc.id}#{i}", values=vector,
                    metadata={**doc.metadata, "doc_id": doc.id, "text": chunk},
                ))
        self.store.upsert(records)
        return len(records)

    def retrieve(self, query: str, top_k: int = 3,
                 filter: dict[str, Any] | None = None) -> list[RetrievedChunk]:
        vector = self.embeddings.embed_query(query)
        matches = self.store.query(vector, top_k=top_k, filter=filter)
        return [
            RetrievedChunk(id=m.id, text=m.metadata.get("text", ""), score=round(m.score, 4),
                           metadata={k: v for k, v in m.metadata.items() if k != "text"})
            for m in matches
        ]
