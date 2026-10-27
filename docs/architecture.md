# Architecture

## Request flow

```mermaid
flowchart TD
    Client([Client]) --> API[FastAPI<br/>app/main.py, app/api]
    Batch([scripts/run_batch.py]) --> BS
    API --> DS[DocumentService<br/>timing, correlation id, metrics]
    API --> BS[BatchService<br/>bounded thread pool]
    BS --> DS
    DS --> ORCH{{LangGraph orchestrator<br/>StateGraph + recursion_limit}}

    ORCH --> OCR[OCR Agent]
    OCR --> VAL[Validation Agent]
    VAL --> RAGR[RAG retrieval<br/>customer + policy chunks]
    RAGR --> RR{Retry router}
    RR -->|retry: retryable errors,<br/>retry_count < MAX_RETRIES| RETRY[retry node<br/>retry_count += 1]
    RETRY --> OCR
    RR -->|continue: valid| STR[Structuring Agent]
    RR -->|continue: not retryable<br/>or retries exhausted| SUP
    STR -->|schema error,<br/>attempts left| STR
    STR --> SUP[Supervisor Agent]
    SUP -->|RETRY, budget left| RETRY
    SUP --> APPROVED([APPROVED])
    SUP --> ESCALATED([ESCALATED])
```

(RAG retrieval happens inside the Validation Agent; it is drawn as its own step for clarity.)

## Adapters: offline vs. real

```mermaid
flowchart LR
    subgraph Agents
        OCRA[OCR Agent]
        VA[Validation Agent]
        SA[Structuring Agent]
        SUPA[Supervisor Agent]
    end

    subgraph Interfaces
        OCRI[[OCRAdapter]]
        LLMI[[LLMAdapter<br/>invoke_structured]]
        RET[[Retriever]]
        EMBI[[EmbeddingProvider]]
        VSI[[VectorStore]]
    end

    subgraph Offline["Offline / mock (default, tests, CI, Docker)"]
        MOCKOCR[MockOCRAdapter]
        FAKELLM[FakeLLMAdapter]
        HASH[HashingEmbeddings]
        MEM[InMemoryVectorStore]
    end

    subgraph Real["Real integrations (env-selected, lazy imports)"]
        TESS[Tesseract]
        PROV[LLM provider<br/>OpenAI / Anthropic / Groq<br/>via LangChain]
        OAIE[OpenAI embeddings]
        PC[(Pinecone)]
    end

    OCRA --> OCRI
    SA --> LLMI
    SUPA --> LLMI
    VA --> RET
    RET --> EMBI
    RET --> VSI

    OCRI -.-> MOCKOCR
    OCRI -.-> TESS
    LLMI -.-> FAKELLM
    LLMI -.-> PROV
    EMBI -.-> HASH
    EMBI -.-> OAIE
    VSI -.-> MEM
    VSI -.-> PC
```

Selection happens once at startup in the `build_*` factories, from `Settings.from_env()`:

| Variable | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `fake` · `openai` · `anthropic` · `groq` | real providers need `LLM_MODEL` (validated) |
| `OCR_PROVIDER` | `mock` · `tesseract` | |
| `VECTOR_STORE` | `memory` · `pinecone` | falls back to `memory` (with a warning) when credentials or the SDK are missing |
| `EMBEDDING_PROVIDER` | `hashing` · `openai` | Pinecone index dimension must equal `EMBEDDING_DIM` |

The graph never imports a provider. It receives adapter instances, so the LLM is
never hard-coded into the workflow.

## Layers

* **`app/schemas`** are the contracts between agents. They are typed Pydantic objects in state.
* **`app/agents`** are plain callables `state -> partial update` with injected dependencies. They don't know about LangGraph or HTTP.
* **`app/graph`** holds wiring (`workflow.py`) and pure routing functions (`routing.py`).
* **`app/services`** contains `DocumentService`, which owns timing, correlation IDs, exception containment and metrics recording, and `BatchService`, which adds bounded concurrency and summary statistics.
* **`app/api`** is the HTTP layer only: request validation, path safety and response mapping.
* **`app/adapters`** are the only modules that touch third-party SDKs, and they import them lazily.
* **`app/observability.py`** holds the JSON log formatter (with redaction) and the in-memory metrics store.

## Design decisions

| Decision | Why |
|---|---|
| Rule-based field extraction in the OCR agent | Scanned business documents are mostly `Label: value`. This is cheap and testable, and keeps the LLM for the parts that need judgement. |
| Deterministic validation; RAG rules on chunk metadata | Validation must be reproducible and auditable. The chunk text is still there for an LLM reviewer. |
| LLM output validated twice | Once against the schema, once against values validation already verified (`LOCKED_FIELDS`). |
| Supervisor = policy + LLM, most conservative wins | The LLM can add caution, never remove it. |
| `retryable` flag per validation issue | Whether to retry depends on *why* validation failed, not just *whether* it failed. |
| Retry counters + `recursion_limit` | Two independent bounds. A test proves the backstop with a deliberately broken router. |
| `process()` never raises | One bad document must not crash an API request handler or a batch. Failures become `status="failed"` results. |
| Thread pool for batches (not `graph.batch`) | Gives per-document timing and error isolation around the whole service call, including code outside the graph. |
| In-memory metrics and result store | Enough for one instance. Production would use Prometheus/OTel and a database. |

## Extending

* **New document type:** add labels to `LABELS`, a prefix to `REFERENCE_NUMBER_PATTERN`, and a section to `data/reference/policies.md`.
* **Another LLM provider:** add a branch to `build_llm`; any LangChain chat model with `with_structured_output` works.
* **Cloud OCR:** subclass `OCRAdapter`.
* **Large knowledge base:** index once offline instead of in `build_retriever` at startup.
