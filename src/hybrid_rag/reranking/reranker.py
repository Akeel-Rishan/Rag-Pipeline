"""Retrieve a broad candidate set and rerank it to a smaller final set."""

from dataclasses import dataclass
import math

from hybrid_rag.reranking.base import RelevanceScorer, RerankedChunk, RerankingRun
from hybrid_rag.retrieval.base import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RerankingConfig:
    retrieve_k: int = 20
    final_k: int = 5

    def __post_init__(self) -> None:
        if type(self.retrieve_k) is not int or self.retrieve_k <= 0:
            raise ValueError("retrieve_k must be a positive integer")
        if type(self.final_k) is not int or self.final_k <= 0:
            raise ValueError("final_k must be a positive integer")
        if self.final_k > self.retrieve_k:
            raise ValueError("final_k must not exceed retrieve_k")


class RerankingRetriever:
    """Orchestrate high-recall retrieval followed by cross-encoder scoring."""

    def __init__(self, retriever: Retriever, scorer: RelevanceScorer,
                 config: RerankingConfig | None = None) -> None:
        self.retriever = retriever
        self.scorer = scorer
        self.config = config or RerankingConfig()

    def run(self, question: str, final_k: int | None = None) -> RerankingRun:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        result_count = self.config.final_k if final_k is None else final_k
        if type(result_count) is not int or result_count <= 0:
            raise ValueError("final_k must be a positive integer")
        if result_count > self.config.retrieve_k:
            raise ValueError("final_k must not exceed retrieve_k")

        candidates = self.retriever.retrieve(question, self.config.retrieve_k)
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.chunk_id in seen:
                raise RuntimeError(f"Retriever returned duplicate chunk_id {candidate.chunk_id}")
            if not math.isfinite(candidate.score):
                raise RuntimeError("Retriever returned a nonfinite score")
            seen.add(candidate.chunk_id)
        scores = self.scorer.score(question, [candidate.text for candidate in candidates])
        if len(scores) != len(candidates):
            raise RuntimeError("Reranker returned the wrong number of scores")
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Reranker returned a nonfinite score")

        ranked = [RerankedChunk(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            page=candidate.page,
            source=candidate.source,
            score=score,
            retrieval_rank=rank,
            retrieval_score=candidate.score,
            candidate=candidate,
        ) for rank, (candidate, score) in enumerate(
            zip(candidates, scores), start=1)]
        # Original retrieval rank makes equal cross-encoder scores deterministic.
        ranked.sort(key=lambda result: (-result.score, result.retrieval_rank))
        return RerankingRun(
            question=question,
            candidates=tuple(candidates),
            results=tuple(ranked[:result_count]),
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[RerankedChunk]:
        """Return final results; use run() when the candidate trace is also needed."""
        return list(self.run(question, final_k=top_k).results)
