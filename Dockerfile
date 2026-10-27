# Document Intelligence Pipeline API - runs fully offline by default
# (mock OCR, fake LLM, in-memory vector store). Configure real providers via env vars.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Real-provider SDKs are opt-in: docker build --build-arg REQUIREMENTS=requirements-optional.txt .
# (Tesseract OCR would additionally need the tesseract-ocr apt package.)
ARG REQUIREMENTS=requirements.txt
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r ${REQUIREMENTS}

COPY app ./app
COPY data ./data
COPY scripts ./scripts

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
