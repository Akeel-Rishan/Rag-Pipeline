"""Shared retrieval result types; no ranking implementation lives here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked piece of evidence returned by a retriever."""

    chunk_id: str
    text: str
    page: int
    source: str
    score: float
