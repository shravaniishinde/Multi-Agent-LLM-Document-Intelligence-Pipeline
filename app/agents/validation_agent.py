"""Agent 2 - Field Validation.

Deterministic checks first (presence, formats, arithmetic), then contextual
checks against reference data retrieved through RAG.

Retry policy, encoded in each issue's `retryable` flag:
- format errors (garbled dates/amounts/IDs) are always retryable - classic OCR noise;
- content errors (missing field, totals don't add up) are retryable only when
  OCR confidence was low, i.e. the scan is a plausible cause;
- reference-data conflicts are never retryable - re-scanning won't change them.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from app.config import Settings
from app.graph.state import PipelineState
from app.rag.retriever import RetrievedChunk, Retriever
from app.schemas.document import (
    CUSTOMER_ID_PATTERN,
    REFERENCE_NUMBER_PATTERN,
    SUPPORTED_CURRENCIES,
)
from app.schemas.extraction import ExtractedFields
from app.schemas.validation import FieldResult, IssueCode, ValidationIssue, ValidationResult

REQUIRED_FIELDS = ["reference_number", "customer_name", "customer_id", "document_date",
                   "total_amount", "currency"]

# Day-first on purpose: the sample documents use DD/MM/YYYY.
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%B %d, %Y", "%b %d, %Y",
                "%d %B %Y", "%d %b %Y"]
_AMOUNT = re.compile(r"^(\d{1,3}(,\d{3})+|\d+)(\.\d{1,2})?$")
_AMOUNT_NOISE = re.compile(r"[A-Z]{3}|[€£₹$\s]")


def parse_date(value: str) -> date | None:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(value: str) -> Decimal | None:
    cleaned = _AMOUNT_NOISE.sub("", value)
    if not _AMOUNT.match(cleaned):
        return None
    try:
        return Decimal(cleaned.replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def _names_match(a: str, b: str) -> bool:
    norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())  # noqa: E731
    return norm(a) == norm(b)


class ValidationAgent:
    def __init__(self, retriever: Retriever | None, settings: Settings):
        self.retriever = retriever
        self.settings = settings

    def validate(self, fields: ExtractedFields, ocr_confidence: float
                 ) -> tuple[ValidationResult, list[RetrievedChunk]]:
        low_confidence = ocr_confidence < self.settings.ocr_confidence_threshold
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        results: dict[str, FieldResult] = {}

        def fail(field: str, code: IssueCode, message: str, retryable: bool) -> None:
            errors.append(ValidationIssue(field=field, code=code, message=message,
                                          retryable=retryable))
            results[field] = FieldResult(field=field, value=getattr(fields, field, None),
                                         valid=False, message=message)

        def ok(field: str, normalized: str) -> None:
            results[field] = FieldResult(field=field, value=getattr(fields, field),
                                         valid=True, normalized=normalized)

        for field in REQUIRED_FIELDS:
            if not getattr(fields, field):
                fail(field, IssueCode.MISSING_FIELD, "required field not found",
                     retryable=low_confidence)

        if fields.reference_number:
            ref = fields.reference_number.upper()
            if re.match(REFERENCE_NUMBER_PATTERN, ref):
                ok("reference_number", ref)
            else:
                fail("reference_number", IssueCode.INVALID_FORMAT,
                     f"{fields.reference_number!r} is not like INV-0001", retryable=True)

        if fields.customer_name:
            if len(fields.customer_name.strip()) >= 2:
                ok("customer_name", fields.customer_name.strip())
            else:
                fail("customer_name", IssueCode.INVALID_FORMAT, "name too short", retryable=True)

        if fields.customer_id:
            cid = fields.customer_id.upper()
            if re.match(CUSTOMER_ID_PATTERN, cid):
                ok("customer_id", cid)
            else:
                fail("customer_id", IssueCode.INVALID_FORMAT,
                     f"{fields.customer_id!r} is not like CUST-0001", retryable=True)

        if fields.document_date:
            parsed = parse_date(fields.document_date)
            if parsed is None:
                fail("document_date", IssueCode.INVALID_DATE,
                     f"cannot parse date {fields.document_date!r}", retryable=True)
            elif parsed > date.today():
                fail("document_date", IssueCode.FUTURE_DATE,
                     f"{parsed.isoformat()} is in the future", retryable=low_confidence)
            else:
                ok("document_date", parsed.isoformat())

        amounts: dict[str, Decimal] = {}
        for field in ("total_amount", "subtotal", "tax"):
            raw = getattr(fields, field)
            if not raw:
                continue
            amount = parse_amount(raw)
            if amount is None:
                fail(field, IssueCode.INVALID_AMOUNT, f"cannot parse amount {raw!r}",
                     retryable=True)
            elif field == "total_amount" and amount <= 0:
                fail(field, IssueCode.NON_POSITIVE_AMOUNT, "total must be positive",
                     retryable=low_confidence)
            else:
                amounts[field] = amount
                ok(field, str(amount))
        if {"subtotal", "tax", "total_amount"} <= amounts.keys():
            expected = amounts["subtotal"] + amounts["tax"]
            if abs(expected - amounts["total_amount"]) > Decimal("0.01"):
                fail("total_amount", IssueCode.AMOUNT_MISMATCH,
                     f"subtotal + tax = {expected} but total = {amounts['total_amount']}",
                     retryable=low_confidence)
                amounts.pop("total_amount")

        if fields.currency:
            currency = fields.currency.upper()
            if currency in SUPPORTED_CURRENCIES:
                ok("currency", currency)
            else:
                fail("currency", IssueCode.INVALID_CURRENCY,
                     f"{fields.currency!r} is not a supported ISO currency", retryable=True)

        if fields.document_type:
            ok("document_type", fields.document_type)
        else:
            warnings.append(ValidationIssue(field="document_type", code=IssueCode.INVALID_FORMAT,
                                            message="document type not recognised"))
        if fields.address:
            ok("address", fields.address)

        context = self._contextual_checks(results, amounts, errors, warnings)

        checked = [r for name, r in results.items() if name in REQUIRED_FIELDS]
        valid_ratio = sum(r.valid for r in checked) / len(REQUIRED_FIELDS)
        result = ValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            field_results=list(results.values()),
            confidence=round(valid_ratio * ocr_confidence, 3),
        )
        return result, context

    def _contextual_checks(self, results: dict[str, FieldResult], amounts: dict[str, Decimal],
                           errors: list[ValidationIssue], warnings: list[ValidationIssue]
                           ) -> list[RetrievedChunk]:
        """RAG: compare the document with customer master data and policies."""
        if self.retriever is None:
            return []
        good = lambda f: results.get(f) is not None and results[f].valid  # noqa: E731
        context: list[RetrievedChunk] = []

        if good("customer_id") and good("customer_name"):
            cid, name = results["customer_id"].normalized, results["customer_name"].normalized
            # Metadata-filtered lookup by ID, then a semantic lookup by name.
            by_id = self.retriever.retrieve(f"customer {name} {cid}", top_k=1,
                                            filter={"kind": "customer", "customer_id": cid})
            context += by_id
            if not by_id:
                by_name = self.retriever.retrieve(f"customer {name}", top_k=1,
                                                  filter={"kind": "customer"})
                match = by_name[0] if by_name else None
                if match and _names_match(match.metadata["customer_name"], name):
                    context.append(match)
                    errors.append(ValidationIssue(
                        field="customer_id", code=IssueCode.CUSTOMER_MISMATCH,
                        message=f"{name} is registered as {match.metadata['customer_id']}, "
                                f"not {cid}"))
                else:
                    warnings.append(ValidationIssue(
                        field="customer_id", code=IssueCode.UNKNOWN_CUSTOMER,
                        message=f"no reference record for {cid}"))
            else:
                ref = by_id[0].metadata
                if not _names_match(ref["customer_name"], name):
                    errors.append(ValidationIssue(
                        field="customer_name", code=IssueCode.CUSTOMER_MISMATCH,
                        message=f"{cid} belongs to {ref['customer_name']}, not {name}"))
                currency = results.get("currency")
                if currency and currency.valid and currency.normalized != ref["billing_currency"]:
                    errors.append(ValidationIssue(
                        field="currency", code=IssueCode.CURRENCY_POLICY,
                        message=f"{cid} is billed in {ref['billing_currency']}, "
                                f"document uses {currency.normalized}"))
                total = amounts.get("total_amount")
                if total is not None and total > Decimal(str(ref["approval_limit"])):
                    warnings.append(ValidationIssue(
                        field="total_amount", code=IssueCode.APPROVAL_LIMIT_EXCEEDED,
                        message=f"{total} exceeds approval limit {ref['approval_limit']:.2f} "
                                f"for {cid}"))

        doc_type = results["document_type"].normalized if good("document_type") else "document"
        context += self.retriever.retrieve(f"{doc_type} validation policy required fields",
                                           top_k=2, filter={"kind": "policy"})
        return context

    def __call__(self, state: PipelineState) -> dict:
        fields = state.get("extracted_fields") or ExtractedFields()
        ocr = state.get("ocr_result")
        result, context = self.validate(fields, ocr.confidence if ocr else 0.0)
        if result.valid:
            note = f" ({len(result.warnings)} warning(s))" if result.warnings else ""
            event = f"Validation passed{note}"
        else:
            event = "Validation failed: " + ", ".join(
                f"{e.field}={e.code.value}" for e in result.errors)
        return {"validation": result, "retrieved_context": context, "events": [event]}
