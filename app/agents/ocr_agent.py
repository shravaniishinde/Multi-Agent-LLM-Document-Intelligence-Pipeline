"""Agent 1 - OCR Extraction.

Runs OCR through the adapter, cleans the text, and pulls candidate key/value
fields out of it. Field extraction is a deterministic label parser: scanned
business documents are mostly "Label: value" lines, and keeping this step
rule-based means the LLM is only used where judgement is needed.
"""

from __future__ import annotations

import re
import unicodedata

from app.adapters.ocr import OCRAdapter, OCRError
from app.graph.state import PipelineState
from app.schemas.extraction import ExtractedFields

# Normalized label -> ExtractedFields attribute.
LABELS: dict[str, str] = {
    "invoice": "reference_number",
    "invoice no": "reference_number",
    "invoice number": "reference_number",
    "reference no": "reference_number",
    "reference number": "reference_number",
    "receipt no": "reference_number",
    "receipt number": "reference_number",
    "po number": "reference_number",
    "customer": "customer_name",
    "customer name": "customer_name",
    "bill to": "customer_name",
    "billed to": "customer_name",
    "customer id": "customer_id",
    "account id": "customer_id",
    "client id": "customer_id",
    "date": "document_date",
    "invoice date": "document_date",
    "issue date": "document_date",
    "document date": "document_date",
    "subtotal": "subtotal",
    "tax": "tax",
    "vat": "tax",
    "gst": "tax",
    "total": "total_amount",
    "total due": "total_amount",
    "amount due": "total_amount",
    "grand total": "total_amount",
    "currency": "currency",
    "address": "address",
    "billing address": "address",
}

DOCUMENT_TYPES = [("purchase order", "purchase_order"), ("receipt", "receipt"),
                  ("invoice", "invoice")]
CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "₹": "INR", "$": "USD"}

_CONTROL = re.compile(r"[^\S\n]+")


def normalize_text(text: str) -> str:
    """Unicode-normalize, drop control characters, collapse spaces, drop blank lines."""
    text = unicodedata.normalize("NFKC", text).replace("\t", " ")
    text = "".join(ch for ch in text if ch == "\n" or unicodedata.category(ch)[0] != "C")
    lines = (_CONTROL.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _normalize_label(label: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", label.lower()).split())


def extract_fields(text: str) -> ExtractedFields:
    values: dict[str, str] = {}
    for line in text.splitlines():
        label, sep, value = line.partition(":")
        if not sep or not value.strip():
            continue
        field = LABELS.get(_normalize_label(label))
        if field and field not in values:  # first occurrence wins
            values[field] = value.strip()

    first_line = text.splitlines()[0].lower() if text else ""
    for keyword, doc_type in DOCUMENT_TYPES:
        if keyword in first_line:
            values["document_type"] = doc_type
            break

    # No explicit currency line: infer from the total ("EUR 1,200.00" or "€1,200.00").
    if "currency" not in values and values.get("total_amount"):
        total = values["total_amount"]
        code = re.search(r"\b[A-Z]{3}\b", total)
        if code:
            values["currency"] = code.group(0)
        else:
            for symbol, iso in CURRENCY_SYMBOLS.items():
                if symbol in total:
                    values["currency"] = iso
                    break
    return ExtractedFields(**values)


class OCRAgent:
    def __init__(self, ocr: OCRAdapter):
        self.ocr = ocr

    def __call__(self, state: PipelineState) -> dict:
        attempt = state.get("retry_count", 0)
        try:
            result = self.ocr.extract(state["document"], attempt=attempt)
        except OCRError as exc:
            return {
                "ocr_result": None,
                "raw_text": "",
                "extracted_fields": ExtractedFields(),
                "errors": [f"ocr: {exc}"],
                "events": [f"OCR failed: {exc}"],
            }
        text = normalize_text(result.text)
        result = result.model_copy(update={"text": text})
        fields = extract_fields(text)
        return {
            "ocr_result": result,
            "raw_text": text,
            "extracted_fields": fields,
            "events": [
                f"OCR completed (engine={result.engine}, attempt={attempt}, "
                f"confidence={result.confidence:.2f}, fields={len(fields.present())})"
            ],
        }
