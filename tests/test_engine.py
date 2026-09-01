from a2r.graph import build_engine
from a2r.llm.provider import DeterministicProvider
from tests.conftest import FakeVectorStore


def test_out_of_scope_is_honest_and_has_no_sources(test_config):
    engine = build_engine(test_config, provider=DeterministicProvider(), vector_store=FakeVectorStore())
    result = engine.query("Who invented the internet?")
    assert result["answer_source"] == "out_of_scope"
    assert result["sources"] == []
    assert "outside the indexed knowledge base" in result["answer"].lower()


def test_grounded_answer_lists_chunk_source_and_feedback_is_idempotent(test_config):
    chunks = [{"text": "Refund requests must be submitted within thirty days.", "source": "enterprise_agreement.md", "chunk_index": 0, "score": 0.9}]
    engine = build_engine(test_config, provider=DeterministicProvider(), vector_store=FakeVectorStore({"billing": chunks}))
    result = engine.query("What is the refund policy?")
    assert result["answer_source"] == "rag_pipeline"
    assert result["sources"] == ["enterprise_agreement.md (chunk 0)"]
    assert engine.feedback(result["query_id"], "accept") > 0.5
    assert engine.feedback(result["query_id"], "accept") is None
