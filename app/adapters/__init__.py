"""Adapters isolate every external dependency (LLM, OCR engine, vector DB).

Each module exposes an abstract interface, an offline implementation used by
default and in tests, and a real implementation that is imported lazily so the
optional SDK is only needed when that provider is selected.
"""
