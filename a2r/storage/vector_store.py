from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from a2r.settings import load_config, project_path


class VectorStoreManager:
    """Persistent, local Chroma collections with cached hash IDs."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        vector = self.config["vector_store"]
        self.encoder = SentenceTransformer(self.config["embedding"]["model"])
        self.client = chromadb.PersistentClient(path=str(project_path(vector["persist_dir"])))

    def get_collection(self, pipeline_id: str):
        vector = self.config["vector_store"]
        return self.client.get_or_create_collection(
            f"{vector['collection_prefix']}_{pipeline_id}", metadata={"hnsw:space": "cosine"}
        )

    def ingest_directory(self, pipeline_id: str, data_dir: str | Path) -> int:
        collection = self.get_collection(pipeline_id)
        chunks: list[tuple[str, str, int]] = []
        for source in sorted(Path(data_dir).glob("*")):
            if source.suffix.lower() not in {".md", ".txt"}:
                continue
            for index, text in enumerate(self._chunks(source.read_text(encoding="utf-8"))):
                chunks.append((source.name, text, index))
        if not chunks:
            return 0
        ids = [sha256(f"{pipeline_id}|{source}|{index}|{text}".encode()).hexdigest() for source, text, index in chunks]
        existing = set(collection.get(ids=ids, include=[])["ids"])
        pending = [(item, item_id) for item, item_id in zip(chunks, ids) if item_id not in existing]
        if not pending:
            return 0
        batch_size = self.config["embedding"]["batch_size"]
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            texts = [item[0][1] for item in batch]
            embeddings = self.encoder.encode(texts, normalize_embeddings=True).tolist()
            collection.upsert(
                ids=[item[1] for item in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"source": item[0][0], "chunk_index": item[0][2]} for item in batch],
            )
        return len(pending)

    def retrieve(self, pipeline_id: str, query: str, top_k: int | None = None) -> list[dict]:
        collection = self.get_collection(pipeline_id)
        if collection.count() == 0:
            return []
        top_k = min(top_k or self.config["vector_store"]["top_k"], collection.count())
        embedding = self.encoder.encode([query], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=embedding, n_results=top_k, include=["documents", "metadatas", "distances"])
        return [
            {
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "score": max(0.0, 1.0 - float(distance)),
            }
            for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def _chunks(self, text: str) -> Iterable[str]:
        words = text.split()
        size = self.config["vector_store"]["chunk_words"]
        overlap = self.config["vector_store"]["chunk_overlap_words"]
        step = max(1, size - overlap)
        for start in range(0, len(words), step):
            chunk = words[start : start + size]
            if chunk:
                yield " ".join(chunk)
            if start + size >= len(words):
                break
