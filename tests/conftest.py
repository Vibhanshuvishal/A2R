from __future__ import annotations

from copy import deepcopy

import pytest

from a2r.settings import load_config


@pytest.fixture
def test_config(tmp_path):
    config = deepcopy(load_config())
    config["llm"]["provider"] = "deterministic"
    config["router"]["weights_db"] = str(tmp_path / "weights.sqlite")
    config["storage"]["prediction_db"] = str(tmp_path / "predictions.sqlite")
    config["storage"]["session_db"] = str(tmp_path / "sessions.sqlite")
    config["cache"] = {"enabled": False}
    config["web_fallback"] = {"enabled": False}
    config["router"]["exploration_rate"] = 0
    return config


class FakeVectorStore:
    def __init__(self, responses=None):
        self.responses = responses or {}

    def retrieve(self, pipeline_id, query):
        return self.responses.get(pipeline_id, [])
