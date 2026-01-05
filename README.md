## Overview

This project implements an internal AI-powered Q&A chatbot for analyzing credit card fraud.
The system answers both:
- data-driven analytical questions from a transaction dataset, and
- conceptual questions from fraud research documents.

The chatbot is designed to minimize hallucination by explicitly separating
database analytics and document-based reasoning.

---

## 📌 Problem Statement

The goal is to build a chatbot capable of answering analytical and conceptual questions such as:

- How does the daily or monthly fraud rate fluctuate over a two-year period?
- Which merchants or merchant categories exhibit the highest incidence of fraud?
- What are the primary methods by which credit card fraud is committed?
- What are the core components of an effective fraud detection system?
- How much higher are fraud rates for transactions outside the EEA?

The chatbot must reason over:
- **Tabular fraud transaction data**
- **External documents / reports** (via Retrieval-Augmented Generation)

---

## Requirement Coverage

| Requirement | Implementation |
|-------------------|----------------|
| Python-based solution | Python 3.10 |
| Store tabular data in database | SQLite (analytical, read-only) |
| Integrate LLM (open/free allowed) | vLLM / OpenAI-compatible API |
| RAG allowed | FAISS + SentenceTransformers |
| Multi-layer prompt | 4-phase agent pipeline |
| UI | Streamlit |
| Quality scoring | Evidence-based score (Phase 4) |

## System Architecture

The system uses a deterministic multi-phase pipeline:

1. Phase 1 – Intent Routing  
   Classifies the user question into:
   - database analytics,
   - document-based reasoning,
   - both, or
   - general / out-of-scope.

2. Phase 2 – NL-to-SQL Analytics  
   Converts analytical questions into safe, read-only SQLite queries.

3. Phase 3 – Retrieval-Augmented Generation (RAG)  
   Retrieves relevant excerpts from fraud research documents.

4. Phase 4 – Final Answer Composition  
   Combines evidence from database and documents into a final answer
   with citations and a quality score.

## Data Setup (Required)

Raw datasets and documents are intentionally not included in this repository.

The `data/` directory is located at the **project root level**, alongside the
application code and Docker configuration files.

Before running the application, please prepare the following **data/** structure:

```text
app/
├── ...
data/
├── raw/
│   ├── fraudTrain.csv
│   ├── fraudTest.csv (optional)
│   ├── Bhatla.pdf
│   ├── EBA_ECB 2024 Report on Payment Fraud.pdf
└── processed/
```

### Execution method
The application is designed to be started using Docker Compose.

1. Clone the repository
```bash
git clone https://github.com/masgani/QA-chatbot.git
cd QA-chatbot
```

2. run on Docker Compose 
```bash
docker compose up --build
```

3. Web UI（Streamlit）
```bash
http://localhost:8501
```




## LLM Inference

The chatbot tested in an open-source large language model served locally on AWS EC2 using vLLM.

### Model
- **Model:** gpt-oss-20b
- **Source:** openai/gpt-oss-20b
- **Parameter size:** ~20B
- **Serving framework:** vLLM (OpenAI-compatible API)

### Deployment Environment
- **Compute:** AWS EC2 GPU instance (A10G, 24 GB VRAM)
- **AMI:** Deep Learning OSS NVIDIA Driver AMI
- **Serving mode:** Local inference (no external API dependency)

## Data Sources

### Transaction Dataset
- Simulated credit card transactions - Kaggle dataset (2019–2020)
- Stored in SQLite for analytical queries

### Documents
- Credit card fraud research paper (Bhatla et al.)
- EBA/ECB 2024 Report on Payment Fraud

## Stateless Design

The current system is intentionally designed as a **stateless chatbot**.

Each user query is processed independently without relying on previous conversation
history. This design simplifies reasoning, improves reproducibility of answers,
and avoids unintended information leakage across queries.

Chat history persistence and multi-turn contextual reasoning are not implemented
in this version but can be added as a future enhancement.

## RAG Limitations

The Retrieval-Augmented Generation (RAG) pipeline indexes text content extracted
from PDF documents.

Visual elements such as charts, figures, and complex tables embedded in the documents
are not fully parsed or semantically structured. As a result:
- insights presented only in charts may not be directly retrievable,
- numerical values embedded in figures may require manual interpretation.

This limitation is common in text-based RAG systems and can be addressed in future work
using layout-aware document parsing or multimodal models.
