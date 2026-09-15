"""Explicit Qdrant storage operations, independent of embedding inference."""

from dataclasses import dataclass, field
from hashlib import sha256
import math
import os

from qdrant_client import QdrantClient, models

from hybrid_rag.embeddings.base import Vector
from hybrid_rag.ingestion.chunking import DocumentChunk


@dataclass(frozen=True)
class QdrantSettings:
    path: str = ".qdrant"
    collection: str = "hybrid_rag_bge_small_v1"
    url: str | None = None
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> "QdrantSettings":
        return cls(
            path=os.getenv("QDRANT_PATH", ".qdrant"),
            collection=os.getenv("QDRANT_COLLECTION", "hybrid_rag_bge_small_v1"),
            url=os.getenv("QDRANT_URL") or None,
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )

    def connect(self) -> QdrantClient:
        if self.url:
            return QdrantClient(url=self.url, api_key=self.api_key, timeout=30)
        return QdrantClient(path=self.path)


class QdrantStore:
    """The caller owns and closes the client. One named cosine vector per point."""

    def __init__(self, client: QdrantClient, collection: str,
                 dimension: int, embedding_space: str) -> None:
        if type(dimension) is not int or dimension <= 0 or not embedding_space.strip():
            raise ValueError("A positive dimension and nonempty embedding space are required")
        self.client = client
        self.collection = collection
        self.dimension = dimension
        self.embedding_space = embedding_space
        # A compact vector name binds the schema to a declared processing version.
        self.vector_name = "dense_" + sha256(embedding_space.encode()).hexdigest()[:24]

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={self.vector_name: models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE)},
            )
        vectors = self.client.get_collection(self.collection).config.params.vectors
        if not isinstance(vectors, dict) or set(vectors) != {self.vector_name}:
            raise ValueError("Collection embedding space differs; use a new collection")
        config = vectors[self.vector_name]
        if config.size != self.dimension or config.distance != models.Distance.COSINE:
            raise ValueError("Collection dimension or distance metric differs")

    def _validate_vector(self, vector: Vector) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"Expected {self.dimension} vector coordinates")
        if not all(math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError("Vector must contain finite coordinates and be nonzero")

    def upsert(self, chunks: list[DocumentChunk], vectors: list[Vector],
               document_id: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")
        for vector in vectors:
            self._validate_vector(vector)
        # Bounded network requests; embedding batches are handled by indexing.py.
        for start in range(0, len(chunks), 64):
            points = [models.PointStruct(
                id=chunk["chunk_id"],
                vector={self.vector_name: vector},
                payload={**chunk, "document_id": document_id,
                         "embedding_space": self.embedding_space},
            ) for chunk, vector in zip(chunks[start:start + 64], vectors[start:start + 64])]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: Vector, limit: int = 3) -> list[models.ScoredPoint]:
        self._validate_vector(vector)
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return self.client.query_points(
            collection_name=self.collection, query=vector, using=self.vector_name,
            limit=limit, with_payload=True, with_vectors=False,
        ).points

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count
