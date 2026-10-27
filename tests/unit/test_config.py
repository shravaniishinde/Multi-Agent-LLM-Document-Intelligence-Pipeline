import json
import logging
import sys

import pytest

from app.adapters.vector_store import InMemoryVectorStore, build_vector_store
from app.config import ConfigError, Settings
from app.observability import JsonFormatter


def test_empty_environment_is_fully_offline():
    s = Settings.from_env({})
    assert (s.llm_provider, s.ocr_provider, s.vector_store, s.embedding_provider) == \
        ("fake", "mock", "memory", "hashing")
    assert s.max_retries == 2 and not s.pinecone_configured


def test_env_values_are_parsed_and_blank_values_ignored():
    s = Settings.from_env({"MAX_RETRIES": "4", "RAG_TOP_K": "5", "PINECONE_CREATE_INDEX": "true",
                           "OCR_CONFIDENCE_THRESHOLD": "0.7", "LLM_MODEL": "  "})
    assert s.max_retries == 4 and s.rag_top_k == 5 and s.pinecone_create_index
    assert s.ocr_confidence_threshold == 0.7 and s.llm_model is None


@pytest.mark.parametrize("env,mentions", [
    ({"MAX_RETRIES": "9"}, "MAX_RETRIES"),
    ({"MAX_RETRIES": "two"}, "MAX_RETRIES"),
    ({"LLM_PROVIDER": "gpt"}, "LLM_PROVIDER"),
    ({"VECTOR_STORE": "faiss"}, "VECTOR_STORE"),
    ({"APPROVAL_CONFIDENCE_THRESHOLD": "1.5"}, "APPROVAL_CONFIDENCE_THRESHOLD"),
    ({"LLM_PROVIDER": "openai"}, "LLM_MODEL"),
])
def test_invalid_configuration_fails_fast_with_named_variable(env, mentions):
    with pytest.raises(ConfigError, match=mentions):
        Settings.from_env(env)


def test_pinecone_selected_without_credentials_falls_back_to_memory():
    store = build_vector_store(Settings.from_env({"VECTOR_STORE": "pinecone"}))
    assert isinstance(store, InMemoryVectorStore)


def test_pinecone_package_missing_falls_back_to_memory(monkeypatch):
    monkeypatch.setitem(sys.modules, "pinecone", None)  # makes `import pinecone` fail
    settings = Settings.from_env({"VECTOR_STORE": "pinecone", "PINECONE_API_KEY": "pk-test"})
    assert isinstance(build_vector_store(settings), InMemoryVectorStore)


def test_api_key_never_appears_in_repr_or_dump():
    s = Settings.from_env({"PINECONE_API_KEY": "pk-super-secret"})
    assert s.pinecone_configured
    assert "pk-super-secret" not in repr(s)
    assert "pk-super-secret" not in s.model_dump_json()


def test_json_log_redacts_sensitive_fields():
    record = logging.LogRecord("docintel", logging.INFO, __file__, 1, "something", None, None)
    record.fields = {"document_id": "d1", "api_key": "sk-123", "auth_token": "t"}
    line = json.loads(JsonFormatter().format(record))
    assert line["event"] == "something" and line["document_id"] == "d1"
    assert line["api_key"] == "***" and line["auth_token"] == "***"
