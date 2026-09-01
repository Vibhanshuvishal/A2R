from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Iterator

import httpx


class ModelUnavailable(RuntimeError):
    """The selected local model cannot currently serve a request."""


class LLMProvider(ABC):
    """Small provider seam; no implementation requires an API key."""

    name: str

    @abstractmethod
    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        raise NotImplementedError

    def stream_complete(self, prompt: str) -> Iterator[str]:
        """Stream token-by-token completion."""
        full_text = self.complete(prompt)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    @abstractmethod
    def status(self) -> dict[str, Any]:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            body = response.json()
            if not body.get("response"):
                raise ModelUnavailable("Ollama returned an empty response")
            return body["response"]
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailable(f"Ollama is unavailable: {exc}") from exc

    def stream_complete(self, prompt: str) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0},
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelUnavailable(f"Ollama stream is unavailable: {exc}") from exc

    def status(self) -> dict[str, Any]:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            models = {model.get("name") for model in response.json().get("models", [])}
            return {
                "provider": self.name,
                "model": self.model,
                "ready": self.model in models or f"{self.model}:latest" in models,
                "detail": "ready" if self.model in models else "run: ollama pull qwen3:4b",
            }
        except httpx.HTTPError:
            return {"provider": self.name, "model": self.model, "ready": False, "detail": "Ollama is not running"}


class DeterministicProvider(LLMProvider):
    """Offline, predictable fallback for availability and tests.

    It never fabricates facts: answer generation is handled extractively by the
    RAG pipeline and this provider only supplies classifier/validator JSON.
    """

    name = "deterministic"

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        lower = prompt.lower()
        if "classify_query" in lower:
            query = _after_marker(prompt, "QUERY:").lower()
            domain = _domain_for(query)
            intent = "how_to" if any(word in query for word in ("how", "setup", "configure")) else "policy_lookup"
            return json.dumps({"domain": domain, "intent": intent, "complexity": "simple", "entities": _entities(query)})
        if "validate_answer" in lower:
            return json.dumps({"relevance": 0.75, "grounding": 1.0, "completeness": 0.65})
        raise ModelUnavailable("Deterministic provider delegates generation to extractive fallback")

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "model": "rules", "ready": True, "detail": "offline fallback"}


class TransformersProvider(LLMProvider):
    """Optional in-process provider for a Gradio ZeroGPU public demo.

    Imports are deliberately lazy: local mode never needs torch/transformers.
    """

    name = "transformers"

    def __init__(self, model: str):
        self.model = model
        self._generator = None
        self._gpu_generate = None

    def _load(self):
        if self._generator is not None:
            return self._generator
        try:
            from transformers import pipeline
            self._generator = pipeline("text-generation", model=self.model, device_map="auto")
            return self._generator
        except Exception as exc:
            raise ModelUnavailable(f"Public demo model could not load: {exc}") from exc

    def _generate(self, prompt: str):
        return self._load()(prompt, max_new_tokens=512, do_sample=False, return_full_text=False)

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        # Hugging Face injects `spaces` in a ZeroGPU Space.  Locally this
        # remains a normal Transformers call and needs no Spaces dependency.
        if self._gpu_generate is None:
            try:
                import spaces
                self._gpu_generate = spaces.GPU(duration=60)(self._generate)
            except ImportError:
                self._gpu_generate = self._generate
        result = self._gpu_generate(prompt)
        return result[0]["generated_text"]

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "ready": True, "detail": "loads on first request"}


def provider_from_config(config: dict) -> LLMProvider:
    llm = config["llm"]
    if config["runtime"]["mode"] == "public_demo":
        return TransformersProvider(llm["public_demo_model"])
    if llm.get("provider") == "deterministic":
        return DeterministicProvider()
    return OllamaProvider(llm["ollama_base_url"], llm["model"], config["runtime"]["request_timeout_seconds"])


def _after_marker(text: str, marker: str) -> str:
    return text.split(marker, 1)[-1].strip()


def _domain_for(query: str) -> str:
    labels = {
        "billing": ("refund", "invoice", "price", "pricing", "payment", "plan", "tax", "credit"),
        "product": ("api", "webhook", "integration", "feature", "search", "export", "permission", "mobile"),
        "hr": ("leave", "pto", "benefit", "conduct", "onboarding", "performance", "expense", "remote work"),
    }
    return next((domain for domain, words in labels.items() if any(word in query for word in words)), "general")


def _entities(query: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", query)
    return words[:5] or ["query"]
