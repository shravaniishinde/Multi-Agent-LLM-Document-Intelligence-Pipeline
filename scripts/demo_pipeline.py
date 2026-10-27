"""Walk sample documents through the pipeline, then show the same thing over HTTP.

    python scripts/demo_pipeline.py                  # pipeline walkthrough + API demo
    python scripts/demo_pipeline.py inv_004 inv_005  # selected documents
    python scripts/demo_pipeline.py --no-api         # skip the API part
    python scripts/demo_pipeline.py --json           # full ProcessingResult JSON

Runs offline by default (mock OCR, fake LLM, in-memory vector store) and takes
a few seconds. The API part starts uvicorn on a free local port and stops it.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.schemas.document import DocumentInput  # noqa: E402
from app.schemas.review import ProcessingResult  # noqa: E402
from app.services import BatchService, build_pipeline  # noqa: E402

SHOWN_FIELDS = ["document_type", "reference_number", "customer_name", "customer_id",
                "document_date", "total_amount", "currency"]
STAGES = [("Document received", "RECEIVED"), ("OCR", "OCR"), ("RAG", "RAG"),
          ("Validation", "VALIDATE"), ("Retry", "RETRY"), ("Structur", "STRUCTURE"),
          ("Supervisor", "SUPERVISOR")]
DEFAULT_DOCS = ["inv_001", "inv_004", "inv_005", "inv_006", "inv_008"]


def stage(event: str) -> str:
    return next((label for prefix, label in STAGES if event.startswith(prefix)), "")


def print_result(result: ProcessingResult, scenario: str | None) -> None:
    print(f"\n{'-' * 72}\nDocument: {result.document_id}"
          + (f"  ({scenario})" if scenario else ""))
    for event in result.events:
        print(f"  {stage(event):<10} | {event}")
    if result.extracted_fields:
        present = result.extracted_fields.present()
        print("  Extracted fields: " + ", ".join(
            f"{name}={present[name]}" for name in SHOWN_FIELDS if name in present))
    if result.validation and result.validation.errors:
        print("  Validation errors:")
        for issue in result.validation.errors:
            print(f"     - {issue}")
    if result.validation and result.validation.warnings:
        print("  Validation warnings:")
        for issue in result.validation.warnings:
            print(f"     - {issue}")
    if result.record:
        rec = result.record
        print(f"  Structured record: {rec.document_type.value} {rec.reference_number} | "
              f"{rec.customer_id} | {rec.document_date} | {rec.total_amount} {rec.currency} | "
              f"confidence={rec.confidence}")
    print(f"  FINAL: {result.decision.value}  (retries={result.retry_count}, "
          f"{result.processing_ms:.1f} ms)")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def api_demo(settings: Settings) -> None:
    import httpx
    import uvicorn

    from app.main import create_app

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings=settings), host="127.0.0.1",
                                           port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        print(f"\n{'=' * 72}\nAPI demo - uvicorn running at {base}\n{'=' * 72}")
        with httpx.Client(base_url=base, timeout=30) as client:
            print("\nGET /health")
            print(json.dumps(client.get("/health").json(), indent=2))

            print('\nPOST /documents/process {"document_id": "inv_004"}')
            body = client.post("/documents/process", json={"document_id": "inv_004"}).json()
            keys = ["document_id", "status", "decision", "document_type", "extracted_fields",
                    "confidence", "retry_count", "review_reasons", "errors", "warnings",
                    "processing_ms"]
            print(json.dumps({k: body[k] for k in keys}, indent=2))

            print("\nPOST /documents/batch (inv_001, inv_003, inv_007)")
            batch = client.post("/documents/batch", json={"documents": [
                {"document_id": d} for d in ("inv_001", "inv_003", "inv_007")]}).json()
            print(json.dumps({k: v for k, v in batch.items() if k != "results"}, indent=2))

            print("\nGET /metrics")
            print(json.dumps(client.get("/metrics").json(), indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="*", help=f"document ids (default: {DEFAULT_DOCS})")
    parser.add_argument("--all", action="store_true", help="run every sample document")
    parser.add_argument("--json", action="store_true", help="print full JSON results")
    parser.add_argument("--no-api", action="store_true", help="skip the FastAPI demo")
    args = parser.parse_args()

    settings = Settings.from_env()
    manifest = json.loads((settings.documents_dir / "manifest.json").read_text())
    ids = sorted(manifest) if args.all else (args.documents or DEFAULT_DOCS)

    service = build_pipeline(settings)
    print(f"Pipeline components: {service.components} | max retries={settings.max_retries}")
    summary = BatchService(service).process_many([DocumentInput(document_id=i) for i in ids])

    if args.json:
        print(summary.model_dump_json(indent=2))
        return

    for result in summary.results:
        print_result(result, manifest.get(result.document_id, {}).get("scenario"))

    print(f"\n{'=' * 72}")
    print(f"Processed {summary.total}: successful={summary.successful}, "
          f"escalated={summary.escalated}, failed={summary.failed}, "
          f"retries={summary.total_retries}, validation failures={summary.validation_failures}")
    labelled = [r for r in summary.results if "expected_decision" in manifest.get(r.document_id, {})]
    if labelled:
        agree = sum(r.decision.value == manifest[r.document_id]["expected_decision"]
                    for r in labelled)
        print(f"Routing matches the hand-labelled expected decision on {agree}/{len(labelled)} "
              f"synthetic sample documents (a routing check, not an accuracy benchmark).")

    if not args.no_api:
        api_demo(settings)


if __name__ == "__main__":
    main()
