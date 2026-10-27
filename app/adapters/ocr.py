"""OCR adapters.

`attempt` lets the pipeline ask for a harder second pass on retry. The mock
replays pre-recorded scan text per attempt; Tesseract switches on heavier image
preprocessing (upscale + binarise), which is the usual real-world retry lever.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import Settings
from app.schemas.document import DocumentInput
from app.schemas.extraction import OCRResult


class OCRError(RuntimeError):
    pass


class OCRAdapter(ABC):
    name: str

    @abstractmethod
    def extract(self, document: DocumentInput, attempt: int = 0) -> OCRResult: ...


class MockOCRAdapter(OCRAdapter):
    """Deterministic OCR backed by fixture text files.

    `documents/manifest.json` maps a document id to one text file per OCR attempt
    plus the confidence that attempt reports. Attempts past the last file reuse
    the last one, so a "persistently bad scan" is just a single noisy file.
    Unknown ids fall back to reading `file_path` if it is a .txt file.
    """

    name = "mock"

    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        manifest = self.documents_dir / "manifest.json"
        self.manifest: dict = json.loads(manifest.read_text()) if manifest.exists() else {}

    def extract(self, document: DocumentInput, attempt: int = 0) -> OCRResult:
        entry = self.manifest.get(document.document_id)
        if entry:
            index = min(attempt, len(entry["scans"]) - 1)
            path = self.documents_dir / entry["scans"][index]
            confidence = entry["confidence"][index]
        elif document.file_path and document.file_path.endswith(".txt"):
            path, confidence = Path(document.file_path), 1.0
        else:
            raise OCRError(f"mock OCR has no scan for {document.document_id!r}")
        if not path.exists():
            raise OCRError(f"scan file not found: {path}")
        return OCRResult(
            text=path.read_text(encoding="utf-8"),
            confidence=confidence,
            engine=self.name,
            attempt=attempt,
            metadata={"source": path.name},
        )


class TesseractOCRAdapter(OCRAdapter):
    """Local OCR via pytesseract (needs the `tesseract` binary installed).

    Images are read with Pillow; PDFs need `pdf2image` + poppler. Confidence is
    the mean word confidence Tesseract reports, scaled to 0..1.
    """

    name = "tesseract"

    def __init__(self) -> None:
        try:
            import pytesseract  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise OCRError(
                "Tesseract OCR needs `pip install pytesseract pillow` and the tesseract binary"
            ) from exc

    def _load_pages(self, path: Path):
        from PIL import Image

        if path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path

            return convert_from_path(str(path), dpi=300)
        return [Image.open(path)]

    @staticmethod
    def _preprocess(image, attempt: int):
        from PIL import ImageOps

        image = ImageOps.grayscale(image)
        if attempt >= 1:
            image = image.resize((image.width * 2, image.height * 2))
            image = ImageOps.autocontrast(image)
            image = image.point(lambda px: 255 if px > 150 else 0)
        return image

    def extract(self, document: DocumentInput, attempt: int = 0) -> OCRResult:
        import pytesseract

        if not document.file_path:
            raise OCRError("Tesseract OCR needs a file_path")
        path = Path(document.file_path)
        if path.suffix.lower() == ".txt":
            return OCRResult(text=path.read_text(encoding="utf-8"), confidence=1.0,
                             engine="text-passthrough", attempt=attempt)

        texts, confidences = [], []
        pages = self._load_pages(path)
        for page in pages:
            image = self._preprocess(page, attempt)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences += [float(c) for c in data["conf"] if float(c) >= 0]
            texts.append(pytesseract.image_to_string(image))
        mean_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
        return OCRResult(
            text="\n".join(texts),
            confidence=round(min(max(mean_conf, 0.0), 1.0), 3),
            engine=self.name,
            page_count=len(pages),
            attempt=attempt,
            metadata={"source": path.name, "preprocessing": "enhanced" if attempt else "basic"},
        )


def build_ocr(settings: Settings) -> OCRAdapter:
    if settings.ocr_provider == "tesseract":
        return TesseractOCRAdapter()
    return MockOCRAdapter(settings.documents_dir)
