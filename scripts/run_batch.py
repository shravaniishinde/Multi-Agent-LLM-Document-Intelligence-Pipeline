"""Process every document in a directory and print batch statistics.

    python scripts/run_batch.py --input data/sample_documents
    python scripts/run_batch.py --input scans/ --concurrency 8 --output results.json

A directory with a manifest.json is treated as a mock-OCR sample set; otherwise
each .txt/.png/.jpg/.tif/.pdf file is one document (needs OCR_PROVIDER=tesseract
for images/PDFs). Uses the same environment configuration as the API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.observability import configure_logging  # noqa: E402
from app.services import BatchService, build_pipeline, discover_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="directory of documents")
    parser.add_argument("--output", type=Path, help="write per-document results as JSON")
    parser.add_argument("--concurrency", type=int, help="max documents in flight")
    parser.add_argument("--log-format", choices=["json", "text"], default=None,
                        help="structured logs to stderr (default: LOG_FORMAT or json)")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.log_format:
        settings = settings.model_copy(update={"log_format": args.log_format})
    configure_logging(settings)

    documents = discover_documents(args.input)
    if not documents:
        sys.exit(f"no supported documents found in {args.input}")
    service = build_pipeline(settings)
    summary = BatchService(service, args.concurrency).process_many(documents)

    print(f"{'document':<20} {'status':<10} {'decision':<9} {'retries':>7} {'ms':>9}")
    for r in summary.results:
        print(f"{r.document_id:<20} {r.status:<10} {r.decision.value:<9} "
              f"{r.retry_count:>7} {r.processing_ms:>9.1f}")
    print()
    print(f"Total documents       {summary.total}")
    print(f"Successful (approved) {summary.successful}")
    print(f"Escalated             {summary.escalated}")
    print(f"Failed (errors)       {summary.failed}")
    print(f"Retries               {summary.total_retries}")
    print(f"Validation failures   {summary.validation_failures}")
    print(f"Avg processing time   {summary.average_processing_ms:.1f} ms")
    print(f"Wall-clock time       {summary.wall_clock_ms:.1f} ms")

    if args.output:
        report = {"summary": summary.model_dump(mode="json", exclude={"results"}),
                  "results": [r.model_dump(mode="json") for r in summary.results]}
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
