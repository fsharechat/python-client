# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) intelligent Q&A system for 飞享IM (FshareChat), built with LangGraph state machine orchestration and Claude API. Answers questions about FshareChat features, deployment, and usage via both CLI and HTTP interfaces.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then add ANTHROPIC_API_KEY
```

## Running

```bash
# Interactive CLI (builds vector store on first run, takes 1-2 min)
python cli.py

# HTTP service (http://localhost:8000, auto-generates /docs)
python service.py

# Rebuild vector store from scraped + static data
python ingest.py
```

## Architecture

The system follows a multi-stage RAG pipeline orchestrated by a **LangGraph state machine** in `graph.py`:

```
Question → [classify] → on_topic → [retrieve] → [grade_docs] → [generate] (streaming)
                      → off_topic → [reject]
                                            → no relevant docs → [fallback]
```

**State** (`QAState`): `question`, `retrieved_docs`, `relevant_docs`, `answer`, `route`

**Key design choices:**
- **Dual knowledge sources**: `ingest.py` combines live web scraping (4 FshareChat URLs) with the large static knowledge block in `config.py`. The static block is the primary fallback.
- **Per-document grading**: `grade_docs` node calls Claude individually for each retrieved chunk — quality filtering over raw retrieval.
- **Adaptive thinking**: All Claude calls use `claude-opus-4-7` with `thinking={"type": "adaptive"}`.
- **Local embeddings**: `BAAI/bge-small-zh-v1.5` runs locally for Chinese-optimized embeddings; ChromaDB persists to `./chroma_db/`.
- **Retriever binding via closure**: `build_graph(retriever)` closes over the retriever so node functions stay pure.

**Entry points:**
- `cli.py` — loads vectorstore, builds graph, calls `graph.invoke()`
- `service.py` — FastAPI with `POST /ask` (sync), `POST /stream` (SSE), `GET /health`; vectorstore initialized in `lifespan`

**Configuration** (`config.py`): API key, model name, ChromaDB path, embedding model name, scrape URLs, and the static knowledge base string.

## HTTP API

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "飞享IM支持哪些平台？"}'
# Response: {"question": "...", "answer": "...", "route": "on_topic|off_topic"}

curl -X POST http://localhost:8000/stream -H "Content-Type: application/json" -d '{"question": "..."}'
# SSE stream, terminated by "data: [DONE]"
```

## No Test Suite

There is no automated test suite. The project uses manual `curl` testing. No linting or CI/CD configuration exists.
