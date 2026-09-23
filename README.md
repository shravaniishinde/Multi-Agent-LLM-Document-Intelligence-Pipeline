# Multi-Agent LLM Document Intelligence Pipeline

An end-to-end multi-agent document intelligence system for automated invoice reconciliation. The pipeline uses LangGraph to coordinate specialized agents that extract information from invoices, match invoices against purchase orders, detect discrepancies, recommend resolutions, and escalate high-risk cases for human review.

The system is designed to handle real-world document variations including scanned PDFs, different invoice layouts, missing purchase-order references, and pricing or quantity discrepancies.

## Architecture

```mermaid
flowchart TD
    A[Invoice PDF] --> B[Document Intelligence Agent]
    B --> C[Matching Agent]
    C --> D[Discrepancy Detection Agent]
    D --> E[Resolution Agent]

    D -->|High Risk / Low Confidence| F[Human Review Agent]
    E --> G{Decision}

    G -->|AUTO_APPROVE| H[Approved Invoice]
    G -->|REQUEST_CLARIFICATION| I[Clarification Required]
    G -->|ESCALATE_TO_HUMAN| F

    F --> J[Human Review / Final Decision]
```
## Workflow

Invoice PDF
    ↓
OCR / Document Extraction
    ↓
Structured Invoice Data
    ↓
Purchase Order Matching
    ↓
Discrepancy Detection
    ↓
Resolution Recommendation
    ↓
Decision
    ├── AUTO_APPROVE
    ├── REQUEST_CLARIFICATION
    └── ESCALATE_TO_HUMAN

