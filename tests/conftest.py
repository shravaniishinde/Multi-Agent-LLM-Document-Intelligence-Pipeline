import pytest

from app.adapters.llm import FakeLLMAdapter
from app.adapters.ocr import MockOCRAdapter
from app.config import FIXTURES_DIR, Settings
from app.rag.knowledge_base import build_retriever
from app.schemas.extraction import ExtractedFields


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def retriever(settings):
    return build_retriever(settings)


@pytest.fixture
def mock_ocr():
    return MockOCRAdapter(FIXTURES_DIR / "documents")


@pytest.fixture
def fake_llm():
    return FakeLLMAdapter()


@pytest.fixture
def good_fields() -> ExtractedFields:
    return ExtractedFields(
        document_type="invoice", reference_number="INV-20431", customer_name="Acme Corporation",
        customer_id="CUST-1001", document_date="2026-03-14", subtotal="1,150.00", tax="92.00",
        total_amount="1,242.00", currency="USD",
    )
