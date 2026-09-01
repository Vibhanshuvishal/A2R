from __future__ import annotations

from typing import Any
from a2r.llm.provider import LLMProvider


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_words])
        chunks.append(chunk)
        start += step
    return chunks


class RAGPipeline:
    def __init__(self, provider: LLMProvider, min_similarity: float):
        self.provider = provider
        self.min_similarity = min_similarity

    def generate(self, query: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        if not retrieved_chunks:
            return "Answer outside the indexed knowledge base.", []
        context = "\n---\n".join(c["text"] for c in retrieved_chunks)
        prompt = f"Answer using only context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.provider.complete(prompt), retrieved_chunks
