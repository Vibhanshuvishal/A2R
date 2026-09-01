from __future__ import annotations

import random
import sqlite3
from pathlib import Path


class BanditRouter:
    def __init__(self, db_path: str | Path, pipeline_ids: list[str], learning_rate: float = 0.05, exploration_rate: float = 0.1, min_weight: float = 0.1):
        self.db_path, self.pipeline_ids = str(db_path), pipeline_ids
        self.learning_rate, self.exploration_rate, self.min_weight = learning_rate, exploration_rate, min_weight
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS weights (domain TEXT, pipeline_id TEXT, weight REAL, query_count INTEGER, PRIMARY KEY(domain, pipeline_id))")
            for domain in ("billing", "product", "hr", "general"):
                for pipeline in self.pipeline_ids:
                    conn.execute("INSERT OR IGNORE INTO weights VALUES (?, ?, .5, 0)", (domain, pipeline))

    def rank_pipelines(self, domain: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT pipeline_id, weight FROM weights WHERE domain = ?", (domain,)).fetchall()
        ranking = sorted(rows, key=lambda row: max(row["weight"], self.min_weight), reverse=True)
        ids = [row["pipeline_id"] for row in ranking]
        if random.random() < self.exploration_rate:
            random.shuffle(ids)
        return ids

    def update_weight(self, domain: str, pipeline_id: str, signal: float) -> float:
        normalized = (signal + 1) / 2
        with self._connect() as conn:
            row = conn.execute("SELECT weight FROM weights WHERE domain = ? AND pipeline_id = ?", (domain, pipeline_id)).fetchone()
            if row is None:
                raise KeyError(f"Unknown weight: {domain}/{pipeline_id}")
            weight = min(1.0, max(self.min_weight, row["weight"] + self.learning_rate * (normalized - row["weight"])))
            conn.execute("UPDATE weights SET weight = ?, query_count = query_count + 1 WHERE domain = ? AND pipeline_id = ?", (weight, domain, pipeline_id))
        return weight

    def matrix(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM weights ORDER BY domain, pipeline_id").fetchall()
        result: dict = {}
        for row in rows:
            result.setdefault(row["domain"], {})[row["pipeline_id"]] = {"weight": row["weight"], "query_count": row["query_count"]}
        return result
