import httpx
import pytest

from a2r.llm.provider import DeterministicProvider, ModelUnavailable, OllamaProvider


def test_deterministic_provider_classifies_without_network():
    provider = DeterministicProvider()
    response = provider.complete("Classify this query: What is the refund policy?", json_mode=True)
    assert '"domain": "billing"' in response


def test_ollama_errors_are_wrapped_as_model_unavailable(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OllamaProvider("qwen3:4b")

    with pytest.raises(ModelUnavailable):
        provider.complete("hello")


def test_ollama_uses_local_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"response": "hello from ollama"})

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = OllamaProvider("qwen3:4b", base_url="http://localhost:11434")
    assert provider.complete("hi") == "hello from ollama"
    assert captured["url"] == "http://localhost:11434/api/generate"
