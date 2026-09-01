from __future__ import annotations

from typing import Iterator

from a2r.llm.provider import LLMProvider, ModelUnavailable


class RAGPipeline:
    def __init__(self, provider: LLMProvider, min_similarity: float):
        self.provider, self.min_similarity = provider, min_similarity

    def run(self, query: str, chunks: list[dict], conversation_context: str = "") -> tuple[str, float, bool]:
        if not chunks or max(chunk["score"] for chunk in chunks) < self.min_similarity:
            return "", 0.0, False
        confidence = sum(chunk["score"] for chunk in chunks[:3]) / min(3, len(chunks))
        context = "\n\n".join(f"[{chunk['source']}#{chunk['chunk_index']}] {chunk['text']}" for chunk in chunks[:3])
        history_block = f"\nCONVERSATION HISTORY:\n{conversation_context}\n" if conversation_context else ""
        prompt = f"""Answer using only CONTEXT.{history_block} If it is insufficient, output INSUFFICIENT_CONTEXT.
QUESTION: {query}\nCONTEXT:\n{context}"""
        try:
            answer = self.provider.complete(prompt)
            if not answer or "INSUFFICIENT_CONTEXT" in answer.upper():
                return "", 0.0, False
            return answer.strip(), confidence, False
        except ModelUnavailable:
            # Extractive fallback retains a source and cannot introduce new facts.
            return chunks[0]["text"], confidence, True

    def stream_run(self, query: str, chunks: list[dict], conversation_context: str = "") -> Iterator[str]:
        if not chunks or max(chunk["score"] for chunk in chunks) < self.min_similarity:
            return
        context = "\n\n".join(f"[{chunk['source']}#{chunk['chunk_index']}] {chunk['text']}" for chunk in chunks[:3])
        history_block = f"\nCONVERSATION HISTORY:\n{conversation_context}\n" if conversation_context else ""
        prompt = f"""Answer using only CONTEXT.{history_block} If it is insufficient, output INSUFFICIENT_CONTEXT.
QUESTION: {query}\nCONTEXT:\n{context}"""
        try:
            for token in self.provider.stream_complete(prompt):
                yield token
        except ModelUnavailable:
            yield chunks[0]["text"]
