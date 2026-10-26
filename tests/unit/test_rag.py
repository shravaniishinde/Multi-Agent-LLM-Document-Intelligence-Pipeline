import pytest

from app.adapters.vector_store import InMemoryVectorStore, PineconeVectorStore, VectorRecord
from app.config import FIXTURES_DIR
from app.rag.embeddings import HashingEmbeddings
from app.rag.knowledge_base import load_reference_documents
from app.rag.retriever import ReferenceDocument, Retriever, chunk_text


def test_chunk_text_windows_overlap():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), chunk_size=40, overlap=10)
    assert [c.split()[0] for c in chunks] == ["w0", "w30", "w60"]
    assert chunks[-1].split()[-1] == "w99"
    assert chunk_text("") == []
    with pytest.raises(ValueError):
        chunk_text("a b", chunk_size=5, overlap=5)


def test_hashing_embeddings_are_deterministic_and_normalized():
    emb = HashingEmbeddings(dim=64)
    a, b = emb.embed_query("approval limit policy"), emb.embed_query("approval limit policy")
    assert a == b and len(a) == 64
    assert sum(x * x for x in a) == pytest.approx(1.0)


def test_in_memory_store_ranks_and_filters():
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", values=[1, 0], metadata={"kind": "x"}),
                  VectorRecord(id="b", values=[0.7, 0.7], metadata={"kind": "y"}),
                  VectorRecord(id="c", values=[0, 1], metadata={"kind": "x"})])
    assert [m.id for m in store.query([1, 0], top_k=2)] == ["a", "b"]
    assert [m.id for m in store.query([1, 0], top_k=5, filter={"kind": "x"})] == ["a", "c"]


def test_reference_documents_load():
    docs = load_reference_documents(FIXTURES_DIR / "reference")
    kinds = {d.metadata["kind"] for d in docs}
    assert kinds == {"customer", "policy"}
    acme = next(d for d in docs if d.id == "customer-CUST-1001")
    assert acme.metadata["billing_currency"] == "USD"
    assert acme.metadata["approval_limit"] == 25000.0


def test_retriever_finds_relevant_policy(retriever):
    hits = retriever.retrieve("which currency must a customer be invoiced in",
                              top_k=1, filter={"kind": "policy"})
    assert hits[0].id == "policy-currency-policy#0"
    assert "billing currency" in hits[0].text


def test_retriever_semantic_customer_lookup(retriever):
    hits = retriever.retrieve("Globex GmbH", top_k=1, filter={"kind": "customer"})
    assert hits[0].metadata["customer_id"] == "CUST-1002"


def test_retriever_indexes_long_documents_as_multiple_chunks():
    retriever = Retriever(HashingEmbeddings(32), InMemoryVectorStore(), chunk_size=10, overlap=2)
    count = retriever.index([ReferenceDocument(id="long", text=" ".join(["word"] * 30))])
    assert count == 4


class FakePineconeIndex:
    """Records calls and answers like the Pinecone client's Index object."""

    def __init__(self):
        self.upserts, self.queries = [], []

    def upsert(self, vectors, namespace):
        self.upserts.append((vectors, namespace))

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return {"matches": [{"id": "customer-CUST-1001#0", "score": 0.91,
                             "metadata": {"text": "Customer Acme", "kind": "customer"}}]}


def test_pinecone_adapter_maps_requests_and_responses():
    index = FakePineconeIndex()
    store = PineconeVectorStore(index=index, namespace="ref")
    retriever = Retriever(HashingEmbeddings(16), store)
    retriever.index([ReferenceDocument(id="customer-CUST-1001", text="Customer Acme",
                                       metadata={"kind": "customer"})])
    hits = retriever.retrieve("acme", top_k=2, filter={"kind": "customer"})

    vectors, namespace = index.upserts[0]
    assert namespace == "ref" and vectors[0]["id"] == "customer-CUST-1001#0"
    assert vectors[0]["metadata"]["text"] == "Customer Acme"
    assert index.queries[0]["filter"] == {"kind": {"$eq": "customer"}}
    assert index.queries[0]["top_k"] == 2
    assert hits[0].score == 0.91 and hits[0].text == "Customer Acme"


def test_pinecone_requires_credentials_without_injected_index():
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        PineconeVectorStore()
