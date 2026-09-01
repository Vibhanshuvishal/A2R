from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any
from uuid import uuid4


class ChatSessionManager:
    """SQLite-backed chat session and multi-turn message history storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    is_deleted INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    query_id TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    tokens_consumed INTEGER DEFAULT 0,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions(user_id, is_deleted, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                ON messages(session_id, timestamp ASC);
                """
            )

    def create_session(self, user_id: str = "default", title: str = "New Conversation") -> str:
        session_id = str(uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, created_at, updated_at, is_deleted) VALUES (?, ?, ?, ?, ?, 0)",
                (session_id, user_id, title.strip() or "New Conversation", now, now),
            )
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id = ? AND is_deleted = 0",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def list_sessions(self, user_id: str = "default", limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.user_id = ? AND s.is_deleted = 0
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        query_id: str = "",
        metadata: dict[str, Any] | None = None,
        tokens_consumed: int = 0,
    ) -> str:
        now = time.time()
        msg_id = str(uuid4())
        meta_json = json.dumps(metadata or {})
        with self._connect() as conn:
            # Ensure session exists or create it
            existing = conn.execute(
                "SELECT id FROM sessions WHERE id = ? AND is_deleted = 0", (session_id,)
            ).fetchone()
            if not existing:
                title = content[:40].replace("\n", " ").strip() if role == "user" else "Conversation"
                conn.execute(
                    "INSERT INTO sessions (id, user_id, title, created_at, updated_at, is_deleted) VALUES (?, ?, ?, ?, ?, 0)",
                    (session_id, "default", title or "New Conversation", now, now),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
                )

            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, query_id, metadata, tokens_consumed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, session_id, role, content, query_id, meta_json, tokens_consumed, now),
            )
        return msg_id

    def load_session_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, query_id, metadata, tokens_consumed, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except (json.JSONDecodeError, TypeError):
                    item["metadata"] = {}
                results.append(item)
            return results

    def update_title(self, session_id: str, title: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND is_deleted = 0",
                (title.strip(), time.time(), session_id),
            )
            return cur.rowcount > 0

    def delete_session(self, session_id: str, soft: bool = True) -> bool:
        with self._connect() as conn:
            if soft:
                cur = conn.execute(
                    "UPDATE sessions SET is_deleted = 1, updated_at = ? WHERE id = ?",
                    (time.time(), session_id),
                )
            else:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0
