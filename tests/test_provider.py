from unittest.mock import Mock, patch

import httpx
import pytest

from a2r.llm.provider import DeterministicProvider, ModelUnavailable, OllamaProvider


def test_deterministic_provider_classifies_without_network():
    answer = DeterministicProvider().complete("TASK: classify_query\nQUERY: how do I set up webhook authentication", json_mode=True)
    assert '"domain": "product"' in answer


def test_ollama_errors_are_wrapped_as_model_unavailable():
    with patch("a2r.llm.provider.httpx.post", side_effect=httpx.ConnectError("down")):
        with pytest.raises(ModelUnavailable):
            OllamaProvider("http://localhost:11434", "qwen3:4b").complete("hello")


def test_ollama_uses_local_endpoint():
    response = Mock()
    response.json.return_value = {"response": "local answer"}
    response.raise_for_status.return_value = None
    with patch("a2r.llm.provider.httpx.post", return_value=response) as post:
        assert OllamaProvider("http://localhost:11434", "qwen3:4b").complete("hello") == "local answer"
    assert post.call_args.args[0].endswith("/api/generate")
