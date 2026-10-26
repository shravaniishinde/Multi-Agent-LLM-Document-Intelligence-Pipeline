# Architecture

## Layers

```mermaid
flowchart LR
    subgraph Services
        DS[DocumentService]
        BS[BatchService<br/>graph.batch]
    end
    subgraph Graph["LangGraph StateGraph"]
        OCRA[OCR Extraction Agent]
        VA[Field Validation Agent]
        SA[Schema Structuring Agent]
        SUP[Supervisory Review Agent]
        RT[retry node]
    end
    subgraph RAG
        RET[Retriever]
        EMB[Embeddings<br/>LangChain interface]
        KB[(Reference data<br/>customers + policies)]
    end
    subgraph Adapters
        OCRAD[OCRAdapter<br/>Mock | Tesseract]
        LLM[LLMAdapter<br/>Fake | LangChain chat model]
        VS[VectorStore<br/>InMemory | Pinecone]
    end

    BS --> DS --> Graph
    OCRA --> OCRAD
    VA --> RET
    SA --> LLM
    SUP --> LLM
    RET --> EMB
    RET --> VS
    KB -. indexed at startup .-> RET
```

* **Schemas** (`app/schemas`) are the contracts between agents. Each agent reads
  typed objects from state and writes typed objects back.
* **Agents** (`app/agents`) are plain callables `state -> partial update`, built with
  their dependencies injected. They don't know about LangGraph. That keeps them
  unit-testable without running a graph.
* **Graph** (`app/graph`) holds only wiring and routing. Routing functions are pure
  and take `Settings`, so every branch can be tested directly.
* **Adapters** (`app/adapters`) are the only modules that touch third-party SDKs, and
  they import those SDKs lazily. The default `requirements.txt` therefore doesn't
  include any of them.

## Design decisions

| Decision | Why |
|---|---|
| Rule-based field extraction in the OCR agent | Scanned business documents are mostly `Label: value`. A deterministic parser is cheaper, testable, and leaves the LLM for the parts that need judgement. |
| Deterministic validation, RAG checks on metadata | Validation must be reproducible and auditable. Retrieved chunks have structured metadata (currency, limits) for the rules, and text for the LLM. |
| LLM output validated twice | First against the Pydantic schema, then against the values validation already verified (`LOCKED_FIELDS`). The LLM can format and enrich, but it can't rewrite facts. |
| Supervisor = policy + LLM, most conservative wins | The LLM can add caution, such as escalating something the rules missed. It can never remove caution. |
| `retryable` flag per issue | The retry decision depends on *why* validation failed, not just *whether* it failed. |
| Counters + `recursion_limit` | Two independent bounds, so no loop can run forever even if a routing bug slips in. |
| `HashingEmbeddings` implements LangChain `Embeddings` | Gives deterministic offline tests. Any real embedding model is a one-line swap. |

## Extending

* **Real LLM**: `LLM_PROVIDER=openai|anthropic|groq` plus `LLM_MODEL`. To add a provider, add
  a branch to `build_llm`.
* **Real OCR**: `OCR_PROVIDER=tesseract`. A cloud OCR would be another
  `OCRAdapter` subclass.
* **Pinecone**: `VECTOR_STORE=pinecone`. `build_retriever` indexes the reference data
  on startup. For a large knowledge base you would index once, offline, instead.
* **New document type**: add labels to `LABELS`, a reference-number prefix to
  `REFERENCE_NUMBER_PATTERN`, and a policy section to `policies.md`.
