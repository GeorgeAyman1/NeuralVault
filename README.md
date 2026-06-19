# NeuralVault

Scalable long-term semantic memory engine for LLM systems using custom vector indexing and approximate nearest-neighbor retrieval.

---

## Overview

NeuralVault is an AI infrastructure project focused on building a scalable semantic memory architecture for large language models (LLMs).

Instead of relying on short-term context windows, NeuralVault enables persistent long-term memory through embedding-based retrieval and custom vector indexing systems.

The project explores:

- semantic memory storage
- vector similarity retrieval
- approximate nearest-neighbor (ANN) search
- intelligent memory ranking
- scalable vector indexing
- long-term AI memory architectures

NeuralVault is designed as a systems engineering and AI infrastructure project rather than a simple chatbot application.

---

# Core Features

## Phase 1 — Core Semantic Memory System
- Embedding generation pipeline
- Semantic memory storage
- Cosine similarity retrieval
- Metadata persistence
- FastAPI retrieval APIs

## Phase 2 — Vector Indexing Engine
- Custom ANN retrieval engine
- Disk-based vector storage
- Memory-mapped indexing
- Vector partitioning and clustering
- Retrieval benchmarking

## Phase 3 — Intelligent Memory Architecture
- Memory ranking
- Recency-aware retrieval
- Importance scoring
- Memory decay
- Memory consolidation

## Phase 4 — Scalable Retrieval Infrastructure
- Async retrieval APIs
- Multi-threaded querying
- Distributed memory partitions
- Retrieval caching

## Phase 5 — LLM Integration
- Semantic recall for LLMs
- Long-term conversational memory
- Retrieval-augmented context injection

---

# Why NeuralVault?

Traditional databases search for exact matches.

NeuralVault searches for meaning.

Instead of storing only text, NeuralVault stores vector embeddings that represent semantic relationships between memories, conversations, and knowledge.

This enables:
- semantic recall
- contextual memory retrieval
- behavioral similarity search
- long-term AI memory persistence

---

# Planned Architecture

```text
[LLM/User Input]
        ↓
[Embedding Pipeline]
        ↓
[Vector Storage Engine]
        ↓
[ANN Retrieval Engine]
        ↓
[Memory Ranking Layer]
        ↓
[Semantic Recall]
        ↓
[LLM Context Injection]
```

---

# Running NeuralVault

## Local (development)

```bash
pip install -r requirements-dev.txt          # runtime + test tooling
uvicorn main:app --reload                     # serves on http://localhost:8000
```

Open `http://localhost:8000/docs` for the interactive API.

## Docker

```bash
cp .env.example .env                          # add your ANTHROPIC_API_KEY
docker compose up -d                          # builds + runs, persists ./data and ./logs
```

The embedding model is cached in a named volume, so restarts are fast.

## Configuration

All settings are environment variables (see `.env.example`). LLM chat needs an
API key for the selected provider; without one the service still runs
(ingestion, search, memory management) and `/memory/chat` returns 503.

NeuralVault supports two LLM providers via `NEURALVAULT_LLM_PROVIDER`:

- `anthropic` (default) — Claude Opus 4.8, set `ANTHROPIC_API_KEY`.
- `groq` — free OpenAI-compatible models (e.g. `llama-3.3-70b-versatile`),
  set `GROQ_API_KEY` (free at https://console.groq.com). The model and base
  URL default sensibly per provider.

A live end-to-end check against the free Groq tier:

```bash
GROQ_API_KEY=gsk-... python scripts/live_chat_test_groq.py
```

## Key endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /memory/add` · `/add-batch` | Store memories |
| `POST /memory/search` | Semantic retrieval |
| `POST /memory/chat` | LLM answer grounded in retrieved memories (with source attribution) |
| `POST /memory/ingest/{pdf,docx,directory,notebook}` | Document ingestion |
| `POST /memory/{prune,summarize,consolidation/merge}` | Memory intelligence |
| `POST /memory/build-index` | Build the IVF index for scale |
| `GET /health` · `/health/ready` · `/metrics` | Ops |

## Tests

```bash
pytest                                         # 103 tests, fully offline (LLM mocked)
ruff check core api main.py vec_db.py tests    # lint
```

CI runs both on every push and pull request (`.github/workflows/test.yml`).
