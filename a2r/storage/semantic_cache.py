from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

import numpy as np


class SemanticCache:
    """Fast semantic cache using NumPy cosine similarity and SQLite persistence.

    Achieves sub-5ms lookups for semantically similar queries without redundant LLM calls.
    """

    def __init__(
        self,
        db_path: str | Path,
        encoder: Callable[[list[str]], Any],
        threshold: float = 0.85,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        self.threshold = threshold
        self._hits = 0
        self._misses = 0
        self._vectors: np.ndarray | None = None
        self._records: list[dict[str, Any]] = []
        self._init_db()
        self._load_cache()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT UNIQUE NOT NULL,
                    query_embedding BLOB NOT NULL,
                    answer TEXT NOT NULL,
                    answer_source TEXT NOT NULL,
                    pipeline_used TEXT,
                    domain TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sources TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_query_cache_time
                ON query_cache(created_at DESC);
                """
            )

    def _load_cache(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query_text, query_embedding, answer, answer_source,
                       pipeline_used, domain, confidence, sources, created_at, hit_count
                FROM query_cache
                ORDER BY id ASC
                """
            ).fetchall()

        if not rows:
            self._vectors = None
            self._records = []
            return

        vector_list = []
        records = []
        for row in rows:
            vec = np.frombuffer(row["query_embedding"], dtype=np.float32)
            vector_list.append(vec)
            try:
                sources = json.loads(row["sources"])
            except (json.JSONDecodeError, TypeError):
                sources = []
            records.append({
                "id": row["id"],
                "query_text": row["query_text"],
                "answer": row["answer"],
                "answer_source": row["answer_source"],
                "pipeline_used": row["pipeline_used"],
                "domain": row["domain"],
                "confidence": row["confidence"],
                "sources": sources,
                "created_at": row["created_at"],
                "hit_count": row["hit_count"],
            })

        matrix = np.vstack(vector_list)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._vectors = matrix / np.maximum(norms, 1e-12)
        self._records = records

    def lookup(
        self, query_text: str, threshold: float | None = None
    ) -> tuple[bool, dict[str, Any] | None, float]:
        query_text = query_text.strip()
        cutoff = threshold if threshold is not None else self.threshold

        if self._vectors is None or len(self._records) == 0 or not query_text:
            self._misses += 1
            return False, None, 0.0

        # Encode input query
        raw_vec = self._encode(query_text)
        norm = np.linalg.norm(raw_vec)
        if norm > 0:
            raw_vec = raw_vec / norm

        # Cosine similarity via dot product with normalized vectors
        scores = np.dot(self._vectors, raw_vec)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= cutoff:
            self._hits += 1
            hit_record = dict(self._records[best_idx])
            hit_record["hit_count"] += 1

            # Asynchronously / immediately update hit count in SQLite
            try:
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE query_cache SET hit_count = hit_count + 1 WHERE id = ?",
                        (hit_record["id"],),
                    )
            except sqlite3.Error:
                pass

            return True, hit_record, best_score

        self._misses += 1
        return False, None, best_score

    def store(self, query_text: str, result: dict[str, Any]) -> bool:
        query_text = query_text.strip()
        if not query_text or not result.get("answer"):
            return False

        # Encode query
        raw_vec = self._encode(query_text)
        norm = np.linalg.norm(raw_vec)
        normalized_vec = raw_vec / np.maximum(norm, 1e-12)

        now = time.time()
        sources_json = json.dumps(result.get("sources", []))

        try:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO query_cache (
                        query_text, query_embedding, answer, answer_source,
                        pipeline_used, domain, confidence, sources, created_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(query_text) DO UPDATE SET
                        answer = excluded.answer,
                        sources = excluded.sources,
                        confidence = excluded.confidence
                    """,
                    (
                        query_text,
                        raw_vec.tobytes(),
                        result.get("answer", ""),
                        result.get("answer_source", "rag_pipeline"),
                        result.get("pipeline_used"),
                        result.get("domain", "general"),
                        float(result.get("confidence", 0.0)),
                        sources_json,
                        now,
                    ),
                )
                row_id = cur.lastrowid
        except sqlite3.Error:
            return False

        new_rec = {
            "id": row_id,
            "query_text": query_text,
            "answer": result.get("answer", ""),
            "answer_source": result.get("answer_source", "rag_pipeline"),
            "pipeline_used": result.get("pipeline_used"),
            "domain": result.get("domain", "general"),
            "confidence": float(result.get("confidence", 0.0)),
            "sources": result.get("sources", []),
            "created_at": now,
            "hit_count": 0,
        }

        # Update in-memory structures
        if self._vectors is None:
            self._vectors = normalized_vec.reshape(1, -1)
            self._records = [new_rec]
        else:
            self._vectors = np.vstack([self._vectors, normalized_vec.reshape(1, -1)])
            self._records.append(new_rec)

        return True

    def _encode(self, text: str) -> np.ndarray:
        output = self.encoder([text])
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[0]
        return arr

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "cache_size": len(self._records),
            "hits": self._hits,
            "misses": self._misses,
            "total_lookups": total,
            "hit_rate": round(self._hits / max(1, total), 3),
            "threshold": self.threshold,
        }

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM query_cache")
        self._vectors = None
        self._records = []
        self._hits = 0
        self._misses = 0
