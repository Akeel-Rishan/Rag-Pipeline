"""Provider-independent reranking contracts and result records."""

from dataclasses import dataclass
from typing import Protocol

from hybrid_rag.retrieval.base import RetrievedChunk


class RelevanceScorer(Protocol):
    """Score query-document pairs; higher values must mean more relevant."""

    @property
    def model_name(self) -> str: ...

    def score(self, question: str, documents: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankedChunk(RetrievedChunk):
    """A cross-encoder result retaining its original retrieval position."""

    retrieval_rank: int
    retrieval_score: float
    candidate: RetrievedChunk


@dataclass(frozen=True)
class RerankingRun:
    """Both stages of one run, useful for debugging and evaluation."""

    question: str
    candidates: tuple[RetrievedChunk, ...]
    results: tuple[RerankedChunk, ...]
