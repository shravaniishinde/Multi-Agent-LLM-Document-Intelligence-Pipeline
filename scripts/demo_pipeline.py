"""Run the sample documents through the pipeline and print each workflow.

    python scripts/demo_pipeline.py              # all sample documents
    python scripts/demo_pipeline.py inv_004      # selected documents
    python scripts/demo_pipeline.py --json       # full ProcessingResult JSON

Runs offline by default (mock OCR, fake LLM, in-memory vector store).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.schemas.document import DocumentInput  # noqa: E402
from app.schemas.review import ProcessingResult  # noqa: E402
from app.services import BatchService, build_pipeline  # noqa: E402

SHOWN_FIELDS = ["document_type", "reference_number", "customer_name", "customer_id",
                "document_date", "total_amount", "currency"]


def print_result(result: ProcessingResult, scenario: str | None) -> None:
    print(f"\nDocument: {result.document_id}" + (f"  ({scenario})" if scenario else ""))
    for event in result.events:
        print(f"  -> {event}")
    if result.extracted_fields:
        present = result.extracted_fields.present()
        print("  Extracted fields:")
        for name in SHOWN_FIELDS:
            print(f"     {name:<17} {present.get(name, '-')}")
    if result.validation and result.validation.errors:
        print("  Validation errors:")
        for issue in result.validation.errors:
            print(f"     - {issue}")
    if result.validation and result.validation.warnings:
        print("  Validation warnings:")
        for issue in result.validation.warnings:
            print(f"     - {issue}")
    print(f"  Retry count: {result.retry_count}")
    print(f"  Final decision: {result.decision.value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="*", help="document ids (default: all samples)")
    parser.add_argument("--json", action="store_true", help="print full JSON results")
    args = parser.parse_args()

    settings = Settings.from_env()
    manifest = json.loads((settings.fixtures_dir / "documents" / "manifest.json").read_text())
    ids = args.documents or sorted(manifest)

    service = build_pipeline(settings)
    print(f"Pipeline: OCR={settings.ocr_provider}, LLM={settings.llm_provider}, "
          f"vector store={settings.vector_store}, max retries={settings.max_retries}")
    summary = BatchService(service).process_many([DocumentInput(document_id=i) for i in ids])

    if args.json:
        print(summary.model_dump_json(indent=2))
        return

    for result in summary.results:
        print_result(result, manifest.get(result.document_id, {}).get("scenario"))

    print("\n" + "=" * 60)
    print(f"Processed {summary.total} documents: " +
          ", ".join(f"{k}={v}" for k, v in summary.by_decision.items()) +
          f", total retries={summary.total_retries}")
    labelled = [r for r in summary.results if "expected_decision" in manifest.get(r.document_id, {})]
    if labelled:
        agree = sum(r.decision.value == manifest[r.document_id]["expected_decision"]
                    for r in labelled)
        print(f"Routing matches the expected decision on {agree}/{len(labelled)} "
              f"hand-labelled sample documents (synthetic fixtures, not an accuracy benchmark).")


if __name__ == "__main__":
    main()
