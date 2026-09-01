from __future__ import annotations

import sqlite3
from pathlib import Path
import random


class BanditRouter:
    def __init__(self, db_path: Path, pipelines: list[str], learning_rate: float = 0.05, exploration_rate: float = 0.10, min_weight: float = 0.10):
        self.db_path = db_path
        self.pipelines = pipelines
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.min_weight = min_weight
        self._init_db()
        self.weights = self._load_weights()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bandit_weights (
                    pipeline_id TEXT PRIMARY KEY,
                    weight REAL NOT NULL
                );
            """)
            for pid in self.pipelines:
                conn.execute(
                    "INSERT OR IGNORE INTO bandit_weights (pipeline_id, weight) VALUES (?, 1.0);",
                    (pid,)
                )

    def _load_weights(self) -> dict[str, float]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pipeline_id, weight FROM bandit_weights;")
            rows = cursor.fetchall()
            return {row[0]: float(row[1]) for row in rows}

    def route(self, domain_hint: str, explore: bool = True) -> str:
        if explore and random.random() < self.exploration_rate:
            return random.choice(self.pipelines)
        if domain_hint in self.weights:
            return domain_hint
        return max(self.weights, key=self.weights.get)

    def update(self, pipeline_id: str, reward: float):
        if pipeline_id not in self.weights:
            return
        old_weight = self.weights[pipeline_id]
        new_weight = max(self.min_weight, old_weight + self.learning_rate * (reward - old_weight))
        self.weights[pipeline_id] = new_weight
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE bandit_weights SET weight = ? WHERE pipeline_id = ?;", (new_weight, pipeline_id))
