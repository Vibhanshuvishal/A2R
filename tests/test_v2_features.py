from __future__ import annotations

import numpy as np
import pytest

from a2r.graph import build_engine
from a2r.llm.provider import DeterministicProvider
from a2r.storage.chat_session import ChatSessionManager
from a2r.storage.semantic_cache import SemanticCache
from tests.conftest import FakeVectorStore


def test_chat_session_lifecycle(tmp_path):
    db_path = tmp_path / "test_chat.sqlite"
    mgr = ChatSessionManager(db_path)

    # 1. Create session
    sess_id = mgr.create_session(user_id="user1", title="First Chat")
    assert sess_id

    # 2. Add messages
    m1 = mgr.save_message(sess_id, "user", "What is the return policy?")
    m2 = mgr.save_message(sess_id, "assistant", "You have 30 days.", metadata={"confidence": 0.95})
    assert m1 and m2

    # 3. Load messages
    messages = mgr.load_session_messages(sess_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the return policy?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["metadata"]["confidence"] == 0.95

    # 4. List sessions
    sessions = mgr.list_sessions(user_id="user1")
    assert len(sessions) == 1
    assert sessions[0]["title"] == "First Chat"
    assert sessions[0]["message_count"] == 2

    # 5. Update title
    assert mgr.update_title(sess_id, "Updated Return Policy Chat")
    assert mgr.get_session(sess_id)["title"] == "Updated Return Policy Chat"

    # 6. Delete session
    assert mgr.delete_session(sess_id, soft=True)
    assert mgr.get_session(sess_id) is None
    assert len(mgr.list_sessions(user_id="user1")) == 0


def test_semantic_cache_hit_and_miss(tmp_path):
    db_path = tmp_path / "test_cache.sqlite"

    # Simple deterministic encoder for testing
    embeddings = {
        "how to refund": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "how to get a refund": np.array([0.98, 0.1, 0.0], dtype=np.float32),  # very close
        "what is quantum physics": np.array([0.0, 1.0, 0.0], dtype=np.float32),  # orthogonal
    }

    def fake_encoder(texts):
        return [embeddings.get(t.lower(), np.array([0.0, 0.0, 1.0], dtype=np.float32)) for t in texts]

    cache = SemanticCache(db_path, encoder=fake_encoder, threshold=0.85)

    # 1. Miss on empty cache
    hit, rec, sim = cache.lookup("how to refund")
    assert not hit
    assert rec is None

    # 2. Store item
    stored = cache.store("how to refund", {
        "answer": "Refunds are processed within 30 days.",
        "answer_source": "rag_pipeline",
        "pipeline_used": "billing",
        "domain": "billing",
        "confidence": 0.9,
        "sources": ["agreement.md#0"],
    })
    assert stored

    # 3. Hit on similar query (similarity ~0.99 > 0.85)
    hit, rec, sim = cache.lookup("how to get a refund")
    assert hit
    assert rec["answer"] == "Refunds are processed within 30 days."
    assert sim >= 0.85

    # 4. Miss on unrelated query
    hit, rec, sim = cache.lookup("what is quantum physics")
    assert not hit

    # 5. Stats
    stats = cache.stats()
    assert stats["cache_size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 2  # initial empty miss + quantum physics miss
    assert stats["hit_rate"] > 0

    # 6. Clear
    cache.clear()
    assert cache.stats()["cache_size"] == 0


def test_engine_with_session_and_cache(tmp_path, test_config):
    # Configure test with real session manager and cache
    sess_mgr = ChatSessionManager(tmp_path / "sessions.sqlite")
    
    # Simple embedding mock
    def fake_encode(texts):
        return np.array([[1.0, 0.0, 0.0]] * len(texts), dtype=np.float32)

    cache = SemanticCache(tmp_path / "cache.sqlite", encoder=fake_encode, threshold=0.85)

    chunks = [{"text": "Refunds take 30 days.", "source": "billing.md", "chunk_index": 0, "score": 0.95}]
    vector_store = FakeVectorStore({"billing": chunks})

    test_config["cache"] = {"enabled": True}
    engine = build_engine(
        test_config,
        provider=DeterministicProvider(),
        vector_store=vector_store,
        cache=cache,
        session_manager=sess_mgr,
    )

    sess_id = sess_mgr.create_session(title="Billing Inquiry")

    # 1. First query (cache miss, runs RAG, saves to session and cache)
    r1 = engine.query("How do I request a refund?", session_id=sess_id)
    assert r1["answer_source"] == "rag_pipeline"
    assert not r1["cache_hit"]
    assert len(sess_mgr.load_session_messages(sess_id)) == 2

    # 2. Second query (cache hit, instant response, saves to session)
    r2 = engine.query("How do I request a refund?", session_id=sess_id)
    assert r2["cache_hit"]
    assert "Cached" in r2["source_badge"]
    assert len(sess_mgr.load_session_messages(sess_id)) == 4


def test_engine_streaming_query(tmp_path, test_config):
    chunks = [{"text": "Refund requests are processed in 30 days.", "source": "billing.md", "chunk_index": 0, "score": 0.9}]
    engine = build_engine(
        test_config,
        provider=DeterministicProvider(),
        vector_store=FakeVectorStore({"billing": chunks}),
    )

    events = list(engine.stream_query("What is the refund deadline?"))
    event_types = [e["event"] for e in events]

    assert "token" in event_types
    assert "done" in event_types
    done_event = next(e for e in events if e["event"] == "done")
    assert done_event["result"]["answer"]
