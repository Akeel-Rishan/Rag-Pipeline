"""Weighted Reciprocal Rank Fusion over independent retrieval rankings."""

from dataclasses import dataclass
import math

from hybrid_rag.retrieval.base import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RRFConfig:
    """Fusion settings; raw component scores are never combined."""

    rank_constant: int = 60
    candidate_k: int = 20
    vector_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self) -> None:
        if type(self.rank_constant) is not int or self.rank_constant < 0:
            raise ValueError("rank_constant must be a nonnegative integer")
        if type(self.candidate_k) is not int or self.candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        for name, weight in (("vector_weight", self.vector_weight),
                             ("bm25_weight", self.bm25_weight)):
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.vector_weight == 0 and self.bm25_weight == 0:
            raise ValueError("At least one retriever weight must be positive")


@dataclass(frozen=True)
class HybridRetrievedChunk(RetrievedChunk):
    """A fused result with enough detail to reproduce its RRF score."""

    vector_rank: int | None
    vector_score: float | None
    vector_contribution: float
    bm25_rank: int | None
    bm25_score: float | None
    bm25_contribution: float


class HybridRetriever:
    """Retrieve independent candidate lists and fuse their ranks with RRF."""

    def __init__(self, vector_retriever: Retriever, bm25_retriever: Retriever,
                 config: RRFConfig | None = None) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.config = config or RRFConfig()

    @staticmethod
    def _validate_results(results: list[RetrievedChunk], name: str) -> None:
        seen: set[str] = set()
        for result in results:
            if result.chunk_id in seen:
                raise RuntimeError(f"{name} returned duplicate chunk_id {result.chunk_id}")
            if not math.isfinite(result.score):
                raise RuntimeError(f"{name} returned a nonfinite score")
            seen.add(result.chunk_id)

    def fuse(self, vector_results: list[RetrievedChunk],
             bm25_results: list[RetrievedChunk], top_k: int = 3) -> list[HybridRetrievedChunk]:
        """Fuse already-ranked lists, allowing inspection without repeat retrieval."""
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self._validate_results(vector_results, "Vector retriever")
        self._validate_results(bm25_results, "BM25 retriever")

        records: dict[str, dict[str, object]] = {}
        for name, results, weight in (
            ("vector", vector_results, self.config.vector_weight),
            ("bm25", bm25_results, self.config.bm25_weight),
        ):
            for rank, result in enumerate(results, start=1):
                existing = records.get(result.chunk_id)
                if existing is None:
                    existing = {
                        "chunk": result,
                        "vector_rank": None,
                        "vector_score": None,
                        "vector_contribution": 0.0,
                        "bm25_rank": None,
                        "bm25_score": None,
                        "bm25_contribution": 0.0,
                    }
                    records[result.chunk_id] = existing
                else:
                    chunk = existing["chunk"]
                    assert isinstance(chunk, RetrievedChunk)
                    if (chunk.text, chunk.page, chunk.source) != (
                            result.text, result.page, result.source):
                        raise RuntimeError(
                            f"Retrievers disagree on payload for chunk_id {result.chunk_id}")
                existing[f"{name}_rank"] = rank
                existing[f"{name}_score"] = result.score
                existing[f"{name}_contribution"] = (
                    weight / (self.config.rank_constant + rank)
                )

        fused: list[HybridRetrievedChunk] = []
        for record in records.values():
            chunk = record["chunk"]
            assert isinstance(chunk, RetrievedChunk)
            vector_contribution = float(record["vector_contribution"])
            bm25_contribution = float(record["bm25_contribution"])
            fused.append(HybridRetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page=chunk.page,
                source=chunk.source,
                score=vector_contribution + bm25_contribution,
                vector_rank=record["vector_rank"],
                vector_score=record["vector_score"],
                vector_contribution=vector_contribution,
                bm25_rank=record["bm25_rank"],
                bm25_score=record["bm25_score"],
                bm25_contribution=bm25_contribution,
            ))

        def ordering(result: HybridRetrievedChunk) -> tuple[float, int, str]:
            component_ranks = [rank for rank in (
                result.vector_rank, result.bm25_rank) if rank is not None]
            return (-result.score, min(component_ranks), result.chunk_id)

        return sorted(fused, key=ordering)[:top_k]

    def retrieve(self, question: str, top_k: int = 3) -> list[HybridRetrievedChunk]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        candidate_k = max(top_k, self.config.candidate_k)
        vector_results = self.vector_retriever.retrieve(question, candidate_k)
        bm25_results = self.bm25_retriever.retrieve(question, candidate_k)
        return self.fuse(vector_results, bm25_results, top_k)
