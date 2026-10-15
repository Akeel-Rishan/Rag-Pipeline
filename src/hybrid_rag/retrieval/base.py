"""Shared retrieval result types; no ranking implementation lives here."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked piece of evidence returned by a retriever."""

    chunk_id: str
    text: str
    page: int
    source: str
    score: float


class Retriever(Protocol):
    """Common ranking boundary implemented by vector, BM25, and hybrid search."""

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]: ...
