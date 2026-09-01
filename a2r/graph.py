from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterator
from uuid import uuid4

from a2r.agents import analyze_query, validate_answer
from a2r.agents.web_search import search_duckduckgo
from a2r.learning import BanditRouter
from a2r.llm import LLMProvider, provider_from_config
from a2r.pipelines import RAGPipeline
from a2r.settings import load_config, project_path
from a2r.storage import ChatSessionManager, PredictionLogger, SemanticCache, VectorStoreManager


class A2REngine:
    """Explicit, auditable retrieval flow with semantic caching and multi-turn sessions."""

    def __init__(
        self,
        config: dict | None = None,
        provider: LLMProvider | None = None,
        vector_store: VectorStoreManager | None = None,
        cache: SemanticCache | None = None,
        session_manager: ChatSessionManager | None = None,
    ):
        self.config = config or load_config()
        self.provider = provider or provider_from_config(self.config)
        self.pipelines = self.config["pipelines"]
        ids = [pipeline["id"] for pipeline in self.pipelines]
        router = self.config["router"]
        self.router = BanditRouter(
            project_path(router["weights_db"]),
            ids,
            router["learning_rate"],
            router["exploration_rate"],
            router["min_weight"],
        )
        self.vector_store = vector_store or VectorStoreManager(self.config)
        self.rag = RAGPipeline(self.provider, self.config["vector_store"]["min_similarity"])
        self.logger = PredictionLogger(project_path(self.config["storage"]["prediction_db"]))

        # Chat session persistence
        session_db = project_path(
            self.config.get("storage", {}).get("session_db", "./chat_sessions.sqlite")
        )
        self.session_manager = session_manager or ChatSessionManager(session_db)

        # Semantic cache
        cache_cfg = self.config.get("cache", {})
        self.cache_enabled = cache_cfg.get("enabled", True)
        if cache:
            self.cache = cache
        elif self.cache_enabled and hasattr(self.vector_store, "encoder"):
            cache_path = project_path(cache_cfg.get("db_path", "./query_cache.sqlite"))
            threshold = float(cache_cfg.get("threshold", 0.85))
            self.cache = SemanticCache(cache_path, encoder=self.vector_store.encoder.encode, threshold=threshold)
        else:
            self.cache = None

    def _get_conversation_context(self, session_id: str) -> tuple[str, list[dict]]:
        if not session_id:
            return "", []
        max_turns = self.config.get("sessions", {}).get("max_history_turns", 10)
        history = self.session_manager.load_session_messages(session_id, limit=max_turns * 2)
        if not history:
            return "", []
        formatted = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted), history

    def query(self, query: str, session_id: str = "") -> dict[str, Any]:
        started_at = time.perf_counter()
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        if len(query) > self.config["runtime"]["max_query_chars"]:
            raise ValueError("query exceeds configured length limit")

        # 1. Semantic Cache check
        if self.cache_enabled and self.cache:
            hit, cached_rec, similarity = self.cache.lookup(query)
            if hit and cached_rec:
                query_id = str(uuid4())
                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                result = {
                    "query_id": query_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": cached_rec["answer"],
                    "answer_source": cached_rec["answer_source"],
                    "pipeline_used": cached_rec["pipeline_used"],
                    "domain": cached_rec["domain"],
                    "confidence": cached_rec["confidence"],
                    "retrieval_confidence": cached_rec["confidence"],
                    "sources": cached_rec["sources"],
                    "source_badge": f"Cached ({round(similarity * 100)}% match)",
                    "source_badge_color": "cyan",
                    "needs_human_review": False,
                    "used_fallback": False,
                    "cache_hit": True,
                    "latency_ms": latency_ms,
                }
                self.logger.log_prediction(result)
                if session_id:
                    self.session_manager.save_message(session_id, "user", query, query_id=query_id)
                    self.session_manager.save_message(
                        session_id, "assistant", result["answer"], query_id=query_id, metadata={"cache_hit": True}
                    )
                return result

        # 2. Multi-turn context
        conv_context, _ = self._get_conversation_context(session_id)

        # 3. Query analysis & routing
        analysis, classification_fallback = analyze_query(query, self.provider)
        pipeline_order = self.router.rank_pipelines(analysis.domain)

        best: tuple[str, list[dict], str, float, bool] | None = None
        for pipeline_id in pipeline_order:
            chunks = self.vector_store.retrieve(pipeline_id, query)
            answer, confidence, answer_fallback = self.rag.run(
                query, chunks, conversation_context=conv_context
            )
            if answer:
                best = pipeline_id, chunks, answer, confidence, answer_fallback
                break

        query_id = str(uuid4())
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

        if best is None:
            # Check Web Search Fallback if enabled
            web_cfg = self.config.get("web_fallback", {})
            web_results = search_duckduckgo(query, max_results=web_cfg.get("max_results", 3)) if web_cfg.get("enabled", True) else []

            if web_results:
                answer = web_results[0]["snippet"]
                sources = [f"{r['title']}: {r['url']}" for r in web_results]
                result = {
                    "query_id": query_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": answer,
                    "answer_source": "web_search",
                    "pipeline_used": None,
                    "domain": analysis.domain,
                    "confidence": 0.65,
                    "retrieval_confidence": 0.65,
                    "sources": sources,
                    "source_badge": "Web search (DuckDuckGo)",
                    "source_badge_color": "amber",
                    "needs_human_review": False,
                    "used_fallback": True,
                    "cache_hit": False,
                    "latency_ms": latency_ms,
                }
            else:
                result = {
                    "query_id": query_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": "This question is outside the indexed knowledge base. I cannot verify an answer from the available internal documents.",
                    "answer_source": "out_of_scope",
                    "pipeline_used": None,
                    "domain": analysis.domain,
                    "confidence": 0.0,
                    "retrieval_confidence": 0.0,
                    "sources": [],
                    "source_badge": "Outside indexed knowledge",
                    "source_badge_color": "gray",
                    "needs_human_review": False,
                    "used_fallback": classification_fallback,
                    "cache_hit": False,
                    "latency_ms": latency_ms,
                }
        else:
            pipeline_id, chunks, answer, retrieval_confidence, answer_fallback = best
            scores, validation_fallback = validate_answer(query, answer, chunks, self.provider)
            pipeline = next(item for item in self.pipelines if item["id"] == pipeline_id)
            sources = [f"{chunk['source']} (chunk {chunk['chunk_index']})" for chunk in chunks[:3]]
            result = {
                "query_id": query_id,
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "answer_source": "rag_pipeline",
                "pipeline_used": pipeline_id,
                "domain": analysis.domain,
                "confidence": round(scores.confidence, 3),
                "retrieval_confidence": round(retrieval_confidence, 3),
                "sources": sources,
                "source_badge": pipeline["name"],
                "source_badge_color": "green",
                "needs_human_review": scores.confidence < 0.45,
                "used_fallback": classification_fallback or answer_fallback or validation_fallback,
                "cache_hit": False,
                "latency_ms": latency_ms,
            }

            # Store in semantic cache if valid answer
            if self.cache_enabled and self.cache:
                self.cache.store(query, result)

        self.logger.log_prediction(result)

        # Save to chat history if session_id provided
        if session_id:
            self.session_manager.save_message(session_id, "user", query, query_id=query_id)
            self.session_manager.save_message(
                session_id,
                "assistant",
                result["answer"],
                query_id=query_id,
                metadata={
                    "source_badge": result["source_badge"],
                    "source_badge_color": result["source_badge_color"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                },
            )

        return result

    def stream_query(self, query: str, session_id: str = "") -> Iterator[dict[str, Any]]:
        """Streaming retrieval and answer generation."""
        started_at = time.perf_counter()
        query = query.strip()
        if not query:
            yield {"event": "error", "message": "query must not be blank"}
            return
        if len(query) > self.config["runtime"]["max_query_chars"]:
            yield {"event": "error", "message": "query exceeds configured length limit"}
            return

        # 1. Semantic Cache check
        if self.cache_enabled and self.cache:
            hit, cached_rec, similarity = self.cache.lookup(query)
            if hit and cached_rec:
                query_id = str(uuid4())
                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                yield {"event": "cache_hit", "similarity": round(similarity, 3)}
                words = cached_rec["answer"].split(" ")
                for i, word in enumerate(words):
                    yield {"event": "token", "token": word + (" " if i < len(words) - 1 else "")}
                result = {
                    "query_id": query_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": cached_rec["answer"],
                    "answer_source": cached_rec["answer_source"],
                    "pipeline_used": cached_rec["pipeline_used"],
                    "domain": cached_rec["domain"],
                    "confidence": cached_rec["confidence"],
                    "sources": cached_rec["sources"],
                    "source_badge": f"Cached ({round(similarity * 100)}% match)",
                    "source_badge_color": "cyan",
                    "needs_human_review": False,
                    "used_fallback": False,
                    "cache_hit": True,
                    "latency_ms": latency_ms,
                }
                self.logger.log_prediction(result)
                if session_id:
                    self.session_manager.save_message(session_id, "user", query, query_id=query_id)
                    self.session_manager.save_message(
                        session_id, "assistant", result["answer"], query_id=query_id, metadata={"cache_hit": True}
                    )
                yield {"event": "done", "result": result}
                return

        # 2. Multi-turn context
        conv_context, _ = self._get_conversation_context(session_id)

        # 3. Routing
        yield {"event": "status", "message": "Analyzing query and ranking knowledge collections..."}
        analysis, classification_fallback = analyze_query(query, self.provider)
        pipeline_order = self.router.rank_pipelines(analysis.domain)

        best_pipeline_id: str | None = None
        best_chunks: list[dict] = []

        for p_id in pipeline_order:
            chunks = self.vector_store.retrieve(p_id, query)
            if chunks and max(c["score"] for c in chunks) >= self.config["vector_store"]["min_similarity"]:
                best_pipeline_id = p_id
                best_chunks = chunks
                break

        query_id = str(uuid4())
        answer_tokens: list[str] = []

        if not best_pipeline_id:
            # Fallback web search
            web_cfg = self.config.get("web_fallback", {})
            web_results = search_duckduckgo(query, max_results=web_cfg.get("max_results", 3)) if web_cfg.get("enabled", True) else []

            if web_results:
                yield {"event": "status", "message": "Retrieving external knowledge via DuckDuckGo..."}
                answer = web_results[0]["snippet"]
                sources = [f"{r['title']}: {r['url']}" for r in web_results]
                words = answer.split(" ")
                for i, word in enumerate(words):
                    tok = word + (" " if i < len(words) - 1 else "")
                    answer_tokens.append(tok)
                    yield {"event": "token", "token": tok}

                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                result = {
                    "query_id": query_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": answer,
                    "answer_source": "web_search",
                    "pipeline_used": None,
                    "domain": analysis.domain,
                    "confidence": 0.65,
                    "retrieval_confidence": 0.65,
                    "sources": sources,
                    "source_badge": "Web search (DuckDuckGo)",
                    "source_badge_color": "amber",
                    "needs_human_review": False,
                    "used_fallback": True,
                    "cache_hit": False,
                    "latency_ms": latency_ms,
                }
            else:
                out_scope_msg = "This question is outside the indexed knowledge base. I cannot verify an answer from the available internal documents."
                words = out_scope_msg.split(" ")
                for i, word in enumerate(words):
                    tok = word + (" " if i < len(words) - 1 else "")
                    yield {"event": "token", "token": tok}

                latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
                result = {
                    "query_id": query_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": out_scope_msg,
                    "answer_source": "out_of_scope",
                    "pipeline_used": None,
                    "domain": analysis.domain,
                    "confidence": 0.0,
                    "retrieval_confidence": 0.0,
                    "sources": [],
                    "source_badge": "Outside indexed knowledge",
                    "source_badge_color": "gray",
                    "needs_human_review": False,
                    "used_fallback": classification_fallback,
                    "cache_hit": False,
                    "latency_ms": latency_ms,
                }
        else:
            pipeline = next(item for item in self.pipelines if item["id"] == best_pipeline_id)
            yield {"event": "route", "domain": analysis.domain, "pipeline": pipeline["name"]}

            # Stream answer generation
            for token in self.rag.stream_run(query, best_chunks, conversation_context=conv_context):
                answer_tokens.append(token)
                yield {"event": "token", "token": token}

            full_answer = "".join(answer_tokens).strip() or best_chunks[0]["text"]
            scores, validation_fallback = validate_answer(query, full_answer, best_chunks, self.provider)
            sources = [f"{chunk['source']} (chunk {chunk['chunk_index']})" for chunk in best_chunks[:3]]
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

            result = {
                "query_id": query_id,
                "session_id": session_id,
                "query": query,
                "answer": full_answer,
                "answer_source": "rag_pipeline",
                "pipeline_used": best_pipeline_id,
                "domain": analysis.domain,
                "confidence": round(scores.confidence, 3),
                "retrieval_confidence": round(scores.confidence, 3),
                "sources": sources,
                "source_badge": pipeline["name"],
                "source_badge_color": "green",
                "needs_human_review": scores.confidence < 0.45,
                "used_fallback": classification_fallback or validation_fallback,
                "cache_hit": False,
                "latency_ms": latency_ms,
            }

            if self.cache_enabled and self.cache:
                self.cache.store(query, result)

        self.logger.log_prediction(result)

        if session_id:
            self.session_manager.save_message(session_id, "user", query, query_id=query_id)
            self.session_manager.save_message(
                session_id,
                "assistant",
                result["answer"],
                query_id=query_id,
                metadata={
                    "source_badge": result["source_badge"],
                    "source_badge_color": result["source_badge_color"],
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                },
            )

        yield {"event": "done", "result": result}

    def feedback(self, query_id: str, signal: str) -> float | None:
        row = self.logger.record_feedback(query_id, 1.0 if signal == "accept" else -1.0)
        if row is None:
            return None
        if row["feedback_signal"] is not None:  # already recorded: do not learn twice
            return None
        if not row["selected_pipeline"]:
            return 0.0
        return self.router.update_weight(row["domain"], row["selected_pipeline"], 1.0 if signal == "accept" else -1.0)

    def ensure_ingested(self) -> dict[str, int]:
        """Populate an empty local/public-demo vector store from bundled data."""
        added: dict[str, int] = {}
        for pipeline in self.pipelines:
            collection = self.vector_store.get_collection(pipeline["id"])
            if collection.count() == 0:
                added[pipeline["id"]] = self.vector_store.ingest_directory(
                    pipeline["id"], project_path(pipeline["data_dir"])
                )
            else:
                added[pipeline["id"]] = 0
        return added

    def health(self) -> dict[str, Any]:
        cache_st = self.cache.stats() if self.cache else {"cache_size": 0, "hits": 0, "misses": 0}
        return {
            "status": "ok",
            "runtime_mode": self.config["runtime"]["mode"],
            "model_status": self.provider.status(),
            "cache": cache_st,
        }


def build_engine(
    config: dict | None = None,
    provider: LLMProvider | None = None,
    vector_store: VectorStoreManager | None = None,
    cache: SemanticCache | None = None,
    session_manager: ChatSessionManager | None = None,
) -> A2REngine:
    return A2REngine(
        config=config,
        provider=provider,
        vector_store=vector_store,
        cache=cache,
        session_manager=session_manager,
    )
