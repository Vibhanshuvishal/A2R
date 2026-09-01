from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone


class PredictionLogger:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
                query_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, query TEXT NOT NULL,
                domain TEXT NOT NULL, selected_pipeline TEXT, retrieval_confidence REAL,
                final_confidence REAL, answer_source TEXT NOT NULL, final_answer TEXT NOT NULL,
                sources TEXT NOT NULL, feedback_signal REAL, feedback_timestamp TEXT)""")

    def log_prediction(self, result: dict) -> None:
        with self._connect() as conn:
            conn.execute("""INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""", (
                result["query_id"], self._now(), result["query"], result["domain"], result.get("pipeline_used"),
                result["retrieval_confidence"], result["confidence"], result["answer_source"], result["answer"],
                json.dumps(result["sources"]),
            ))

    def record_feedback(self, query_id: str, signal: float) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM predictions WHERE query_id = ?", (query_id,)).fetchone()
            if row is None:
                return None
            if row["feedback_signal"] is not None:
                return dict(row)
            conn.execute("UPDATE predictions SET feedback_signal = ?, feedback_timestamp = ? WHERE query_id = ?", (signal, self._now(), query_id))
            return dict(row)

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT count(*) FROM predictions").fetchone()[0]
            sources = dict(conn.execute("SELECT answer_source, count(*) FROM predictions GROUP BY answer_source").fetchall())
            average = conn.execute("SELECT coalesce(avg(final_confidence), 0) FROM predictions").fetchone()[0]
            return {"total_queries": total, "by_source": sources, "average_confidence": round(average, 3)}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
