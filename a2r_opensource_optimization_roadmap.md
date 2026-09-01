# A2R Open-Source & Optimization Roadmap (v2 Architecture)

## Executive Strategy: $0 Cost, Local-First, Portfolio-Grade

This document outlines the architecture, optimization strategy, and implementation roadmap for transforming **A2R** into a 100% open-source, local-first retrieval router that costs $0/month to operate while delivering sub-second latency and an impressive portfolio piece for machine learning and AI systems engineering interviews.

---

## Architecture Overview

### 1. Zero-Cost LLM Stack: Ollama + Local Quantized Models
- **Inference Engine**: [Ollama](https://ollama.ai/) running locally on CPU/GPU.
- **Model**: `Qwen 2.5 3B / 7B` or `Llama 3.2 3B` (GGUF 4-bit / 5-bit quantized).
- **Fallback Logic**: If Ollama daemon is offline or under extreme load, fallback immediately to local extractive synthesis (`DeterministicProvider`) without dropping requests or throwing 500 errors.

### 2. Semantic Query Caching (Sub-5ms Hits)
- **Problem**: In enterprise search, 25-40% of queries are repeated or semantically identical ("refund window", "how long do I have to request a refund?", "what is the return policy?").
- **Solution**: Vector similarity query caching.
  - Query text is encoded into a dense vector using the local embedding model (`all-MiniLM-L6-v2`).
  - Cosine similarity is computed against stored cached embeddings.
  - If `cosine_similarity >= threshold (0.85)`: Return the cached response, citations, and source badges immediately without hitting vector search or LLM generation.
  - **Latency**: Sub-5ms response times on cache hits.

### 3. Multi-Turn Session Persistence
- **Storage**: SQLite with Write-Ahead Logging (`WAL` mode) for concurrent reads/writes.
- **Schema**:
  - `sessions`: `(id, user_id, title, created_at, updated_at, is_deleted)`
  - `messages`: `(id, session_id, role, content, query_id, metadata, timestamp)`
- **Conversation Context Compression**: Sliding window of the last 10 conversation turns formatted into markdown prompt injection to provide continuity without exceeding context window limits.

### 4. Real-Time Token Streaming (Server-Sent Events)
- **Protocol**: HTTP SSE (`text/event-stream`).
- **Events Emitted**:
  1. `{"event": "cache_hit", "similarity": 0.94}` (if cache hit)
  2. `{"event": "status", "message": "Analyzing query and ranking knowledge collections..."}`
  3. `{"event": "route", "domain": "billing", "pipeline": "Billing & Invoicing"}`
  4. `{"event": "token", "token": "Refunds "}`
  5. `{"event": "token", "token": "must "}`
  6. `{"event": "done", "result": {...}}` (with confidence, sources, latency_ms)

### 5. Web Search Fallback (Zero-Cost DuckDuckGo)
- When queries fall outside indexed documentation (`confidence < 0.35` across all internal collections), optionally route to DuckDuckGo Instant Answer / HTML search without needing paid API keys.
- If external search is disabled or yields no results, output the honesty guardrail: *"This question is outside the indexed knowledge base. I cannot verify an answer from the available internal documents."*

---

## Hardware Budget & Memory Footprint

| Component | RAM / VRAM | Disk Space | Purpose |
|---|---|---|---|
| **Ollama Daemon + Qwen 2.5 3B (Q4_K_M)** | ~2.5 GB RAM (or VRAM) | ~2.0 GB | Local generation & classification |
| **Sentence-Transformers (`all-MiniLM-L6-v2`)** | ~400 MB RAM | ~90 MB | Query & chunk embedding |
| **ChromaDB Vector Store** | ~150 MB RAM | ~10-50 MB | Persistent local HNSW vector index |
| **SQLite (Cache, Sessions, Bandit)** | ~50 MB RAM | ~5 MB | WAL transactional storage |
| **FastAPI + Uvicorn + Gradio** | ~120 MB RAM | N/A | Async API server & UI |
| **Total System Footprint** | **~3.2 - 4.0 GB** | **~2.5 GB** | Runs comfortably on any standard laptop or cloud VM |

---

## Production Interview Talking Points

1. **Learned Contextual Multi-Armed Bandit**:
   - Rather than static keyword rules or expensive LLM-only routing, A2R maintains a domain-to-pipeline weight matrix.
   - Learns online from explicit user feedback (`accept` / `reject`), adjusting weights with exponential moving average and exploration ($\epsilon$-greedy).
2. **Deterministic Fallbacks & Resilience**:
   - If the local LLM daemon crashes or is unavailable, the pipeline falls back gracefully to extractive retrieval and rule-based classification. Zero 500 errors.
3. **Semantic Query Caching**:
   - Prevents redundant inference costs and slashes latency from ~400ms down to <5ms for frequent customer queries.
4. **Honesty & Anti-Hallucination Guardrails**:
   - Multi-metric validation agent scores answers for relevance, grounding against retrieved chunks, and completeness.
   - Strictly refuses out-of-domain questions with transparent "Outside indexed knowledge" badges.
