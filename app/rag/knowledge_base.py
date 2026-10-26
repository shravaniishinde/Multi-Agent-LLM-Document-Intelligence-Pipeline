"""Load reference data (customer master records + policy text) for RAG."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.adapters.vector_store import build_vector_store
from app.rag.embeddings import build_embeddings
from app.rag.retriever import ReferenceDocument, Retriever


def load_reference_documents(reference_dir: Path) -> list[ReferenceDocument]:
    reference_dir = Path(reference_dir)
    docs: list[ReferenceDocument] = []

    for c in json.loads((reference_dir / "customers.json").read_text(encoding="utf-8")):
        text = (
            f"Customer {c['customer_name']} has customer ID {c['customer_id']}. "
            f"Billing currency is {c['billing_currency']}. Region: {c['region']}. "
            f"Documents above {c['approval_limit']} {c['billing_currency']} require "
            f"manager approval. Registered address: {c['address']}."
        )
        docs.append(ReferenceDocument(
            id=f"customer-{c['customer_id']}", text=text,
            metadata={"kind": "customer", "customer_id": c["customer_id"],
                      "customer_name": c["customer_name"],
                      "billing_currency": c["billing_currency"],
                      "approval_limit": float(c["approval_limit"])},
        ))

    # policies.md: one "## Title" section per policy.
    policies = (reference_dir / "policies.md").read_text(encoding="utf-8")
    for section in policies.split("\n## ")[1:]:
        title, _, body = section.partition("\n")
        slug = title.strip().lower().replace(" ", "-")
        docs.append(ReferenceDocument(id=f"policy-{slug}", text=f"{title.strip()}. {body.strip()}",
                                      metadata={"kind": "policy", "title": title.strip()}))
    return docs


def build_retriever(settings: Settings) -> Retriever:
    retriever = Retriever(build_embeddings(settings), build_vector_store(settings))
    retriever.index(load_reference_documents(settings.fixtures_dir / "reference"))
    return retriever
