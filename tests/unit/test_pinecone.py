"""Pinecone adapter against an in-process fake - no network, no credentials."""

import json
import math

from app.adapters.llm import FakeLLMAdapter
from app.adapters.vector_store import PineconeVectorStore, VectorRecord
from app.config import DATA_DIR, Settings
from app.rag.embeddings import HashingEmbeddings
from app.rag.knowledge_base import load_reference_documents
from app.rag.retriever import Retriever
from app.schemas.document import DocumentInput
from app.services import DocumentService


class FakeIndex:
    """Stores vectors and answers queries like a Pinecone Index (incl. $eq/$in filters)."""

    def __init__(self):
        self.namespaces: dict[str, dict[str, dict]] = {}
        self.queries: list[dict] = []

    def upsert(self, vectors, namespace):
        self.namespaces.setdefault(namespace, {}).update({v["id"]: v for v in vectors})

    @staticmethod
    def _matches(metadata, flt):
        for key, cond in (flt or {}).items():
            value = metadata.get(key)
            if "$eq" in cond and value != cond["$eq"]:
                return False
            if "$in" in cond and value not in cond["$in"]:
                return False
        return True

    def query(self, vector, top_k, include_metadata, namespace, filter=None):
        self.queries.append({"top_k": top_k, "namespace": namespace, "filter": filter})
        scored = []
        for v in self.namespaces.get(namespace, {}).values():
            if self._matches(v["metadata"], filter):
                dot = sum(a * b for a, b in zip(vector, v["values"]))
                norm = math.sqrt(sum(a * a for a in vector)) * math.sqrt(sum(b * b for b in v["values"]))
                scored.append({"id": v["id"], "score": dot / norm if norm else 0.0,
                               "metadata": v["metadata"]})
        scored.sort(key=lambda m: (-m["score"], m["id"]))
        return {"matches": scored[:top_k]}


class FakeClient:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.created: list[dict] = []
        self.index = FakeIndex()

    def has_index(self, name):
        return name in self.existing

    def create_index(self, **kwargs):
        self.created.append(kwargs)
        self.existing.add(kwargs["name"])

    def Index(self, name):  # noqa: N802 - mirrors the Pinecone client API
        self.index_name = name
        return self.index


def pinecone_settings(**env):
    return Settings.from_env({"VECTOR_STORE": "pinecone", "PINECONE_API_KEY": "pk-test",
                              "PINECONE_INDEX": "docs", "PINECONE_NAMESPACE": "ref", **env})


def test_from_settings_uses_env_index_and_namespace():
    client = FakeClient(existing={"docs"})
    store = PineconeVectorStore.from_settings(pinecone_settings(), client=client)
    assert client.index_name == "docs" and store.namespace == "ref" and not client.created


def test_from_settings_creates_missing_index_when_enabled():
    client = FakeClient()
    spec = lambda **kw: ("spec", kw)  # noqa: E731
    PineconeVectorStore.from_settings(
        pinecone_settings(PINECONE_CREATE_INDEX="true", PINECONE_REGION="eu-west-1"),
        client=client, spec_factory=spec)
    created = client.created[0]
    assert created["name"] == "docs" and created["dimension"] == 256
    assert created["metric"] == "cosine"
    assert created["spec"] == ("spec", {"cloud": "aws", "region": "eu-west-1"})


def test_null_metadata_dropped_and_operator_filters_pass_through():
    index = FakeIndex()
    store = PineconeVectorStore(index=index, namespace="ns")
    store.upsert([VectorRecord(id="a", values=[1.0, 0.0], metadata={"kind": "x", "note": None}),
                  VectorRecord(id="b", values=[0.0, 1.0], metadata={"kind": "y"})])
    assert index.namespaces["ns"]["a"]["metadata"] == {"kind": "x"}
    hits = store.query([1.0, 0.0], top_k=5, filter={"kind": {"$in": ["x", "y"]}})
    assert [h.id for h in hits] == ["a", "b"]
    assert index.queries[-1]["filter"] == {"kind": {"$in": ["x", "y"]}}


def test_full_pipeline_on_pinecone_adapter_matches_offline_results(mock_ocr):
    """Swap the in-memory store for the Pinecone adapter: every routing decision is unchanged."""
    settings = Settings()
    store = PineconeVectorStore(index=FakeIndex(), namespace="ref")
    retriever = Retriever(HashingEmbeddings(settings.embedding_dim), store)
    retriever.index(load_reference_documents(DATA_DIR / "reference"))
    service = DocumentService(mock_ocr, FakeLLMAdapter(), retriever, settings)

    manifest = json.loads((DATA_DIR / "sample_documents" / "manifest.json").read_text())
    for doc_id, expected in manifest.items():
        result = service.process(DocumentInput(document_id=doc_id))
        assert result.decision.value == expected["expected_decision"], doc_id
    assert service.components["vector_store"] == "PineconeVectorStore"
    assert all(q["namespace"] == "ref" for q in store.index.queries)
