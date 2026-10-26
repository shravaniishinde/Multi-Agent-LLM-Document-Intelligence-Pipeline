# Pipeline walkthrough

What happens to one document, step by step, and what each step writes to state.

| Step | Node | Reads | Writes |
|---|---|---|---|
| 1 | `ocr` | `document`, `retry_count` (as OCR attempt) | `ocr_result`, `raw_text`, `extracted_fields` |
| 2 | `validate` | `extracted_fields`, `ocr_result.confidence` | `validation`, `retrieved_context` |
| 3 | `structure` (only if valid) | `validation.normalized_fields()`, `raw_text` | `record` or `structuring_error`, `structuring_attempts` |
| 4 | `supervisor` | everything above | `review`, `decision` |
| — | `retry` | `retry_count` | `retry_count + 1`, clears per-pass results |

Every node also appends to `events` (human-readable trace) and, on failure, to
`errors`. Both use LangGraph's additive reducer.

## Validation rules

| Code | Field(s) | Error / warning | Retryable |
|---|---|---|---|
| `MISSING_FIELD` | required fields | error | only if OCR confidence < 0.80 |
| `INVALID_FORMAT` | reference number, customer ID | error | yes |
| `INVALID_DATE` | date | error | yes |
| `FUTURE_DATE` | date | error | only if OCR confidence < 0.80 |
| `INVALID_AMOUNT` | total, subtotal, tax | error | yes |
| `NON_POSITIVE_AMOUNT` | total | error | only if OCR confidence < 0.80 |
| `AMOUNT_MISMATCH` | subtotal + tax ≠ total | error | only if OCR confidence < 0.80 |
| `INVALID_CURRENCY` | currency not ISO/supported | error | yes |
| `CUSTOMER_MISMATCH` | ID ↔ name disagree with reference data (RAG) | error | no |
| `CURRENCY_POLICY` | currency ≠ customer billing currency (RAG) | error | no |
| `UNKNOWN_CUSTOMER` | no reference record (RAG) | warning → supervisor escalates | — |
| `APPROVAL_LIMIT_EXCEEDED` | total > customer approval limit (RAG) | warning → supervisor escalates | — |

Validation confidence = (valid required fields / required fields) × OCR confidence.

## Supervisor policy (in order)

1. Validation failed → **ESCALATE** (lists the errors).
2. No schema-valid record after the allowed attempts → **ESCALATE**.
3. An escalating warning (approval limit, unknown customer) → **ESCALATE**.
4. Record confidence < 0.75 → **RETRY** if retries are left, otherwise **ESCALATE**.
5. Otherwise → **APPROVE**, citing the reference chunks it was checked against.

The LLM review is then merged in. The final decision is the more conservative of the
policy decision and the LLM decision. If the LLM output is invalid, the policy review
is used as-is.
