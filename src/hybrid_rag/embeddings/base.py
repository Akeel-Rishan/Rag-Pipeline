"""The small contract an embedding provider must implement."""

from typing import Protocol

Vector = list[float]


class EmbeddingService(Protocol):
    """Documents and queries must use the same model and compatible roles."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Return one query vector in the matching embedding space."""
        ...
