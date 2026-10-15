"""Vector retrieval over the Qdrant storage boundary."""

from dataclasses import dataclass

from hybrid_rag.embeddings.base import EmbeddingService
from hybrid_rag.storage.qdrant_store import QdrantStore


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked piece of evidence returned by retrieval."""

    chunk_id: str
    text: str
    page: int
    source: str
    score: float


class VectorRetriever:
    """Embed a question and convert Qdrant points into application records."""

    def __init__(self, embedder: EmbeddingService, store: QdrantStore) -> None:
        if embedder.dimension != store.dimension:
            raise ValueError("Embedder and collection dimensions differ")
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        points = self.store.search(self.embedder.embed_query(question), limit=top_k)
        results: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            text = payload.get("text")
            page = payload.get("page")
            source = payload.get("source")
            if (not isinstance(chunk_id, str) or not isinstance(text, str)
                    or type(page) is not int or not isinstance(source, str)):
                raise RuntimeError("Qdrant point has an invalid chunk payload")
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                page=page,
                source=source,
                score=float(point.score),
            ))
        return results
