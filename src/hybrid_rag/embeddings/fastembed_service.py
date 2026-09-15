"""Local CPU embeddings through FastEmbed; no database client involved."""

import math
from pathlib import Path

from fastembed import TextEmbedding

from hybrid_rag.embeddings.base import Vector


class FastEmbedService:
    """Load one reusable model instance; initial construction may download it.

    FastEmbed handles tokenization, model inference, pooling and role-specific
    processing. Overlong input can be truncated by the model's tokenizer.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        cache_dir: str | Path = ".cache/fastembed",
    ) -> None:
        supported = {model["model"]: model for model in TextEmbedding.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        self._model_name = model_name
        self._dimension = supported[model_name]["dim"]
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir), threads=2,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string")
        if not text.strip():
            raise ValueError("Embedding input must not be empty or whitespace-only")

    def _convert(self, raw_vectors, expected_count: int) -> list[Vector]:
        # Consume the generator here: inference errors surface at this boundary.
        vectors = [[float(value) for value in row] for row in raw_vectors]
        if len(vectors) != expected_count:
            raise RuntimeError("Embedding provider returned the wrong number of vectors")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise RuntimeError("Embedding provider returned an unexpected dimension")
            if not all(math.isfinite(value) for value in vector) or not any(vector):
                raise RuntimeError("Embedding provider returned a nonfinite or zero vector")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        if isinstance(texts, str):
            raise TypeError("Pass a list of texts, not a single string")
        for text in texts:
            self._validate_text(text)
        if not texts:
            return []
        return self._convert(self._model.passage_embed(texts, batch_size=32), len(texts))

    def embed_query(self, text: str) -> Vector:
        self._validate_text(text)
        return self._convert(self._model.query_embed(text), 1)[0]
