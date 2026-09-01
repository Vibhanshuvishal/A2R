from a2r.agents.query_analyzer import analyze_query
from a2r.llm.provider import LLMProvider, ModelUnavailable


class BrokenProvider(LLMProvider):
    name = "broken"
    def complete(self, prompt, *, json_mode=False):
        raise ModelUnavailable("not running")
    def status(self):
        return {"ready": False}


def test_analyzer_uses_safe_fallback_when_model_is_unavailable():
    analysis, used_fallback = analyze_query("Where is my invoice?", BrokenProvider())
    assert used_fallback is True
    assert analysis.domain == "billing"
