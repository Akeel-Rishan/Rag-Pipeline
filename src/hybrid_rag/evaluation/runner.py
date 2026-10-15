"""Evaluate one retrieval/generation configuration on labeled cases."""

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable

from hybrid_rag.evaluation.dataset import EvaluationCase
from hybrid_rag.evaluation.metrics import (
    GenerationMetrics,
    RetrievalMetrics,
    generation_metrics,
    retrieval_metrics,
)
from hybrid_rag.generation.prompting import build_context
from hybrid_rag.retrieval.base import RetrievedChunk, Retriever


AnswerFunction = Callable[[str, list[RetrievedChunk], str], str]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    answer: str
    retrieved_ids: tuple[str, ...]
    retrieval: RetrievalMetrics
    generation: GenerationMetrics


@dataclass(frozen=True)
class EvaluationSummary:
    system: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    context_relevance: float
    answer_relevance: float
    faithfulness: float
    groundedness: float
    hallucination_rate: float
    cases: tuple[CaseResult, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_system(system: str, retriever: Retriever, cases: list[EvaluationCase],
                    k: int, answer: AnswerFunction) -> EvaluationSummary:
    if not cases:
        raise ValueError("At least one evaluation case is required")
    results = []
    for case in cases:
        chunks = retriever.retrieve(case.question, k)
        context = build_context(chunks)
        generated = answer(case.question, chunks, context)
        retrieval = retrieval_metrics(
            [chunk.chunk_id for chunk in chunks], case.relevant_chunk_ids, k)
        generation = generation_metrics(generated, case.expected_answer, context)
        results.append(CaseResult(
            case_id=case.case_id,
            answer=generated,
            retrieved_ids=tuple(chunk.chunk_id for chunk in chunks),
            retrieval=retrieval,
            generation=generation,
        ))

    return EvaluationSummary(
        system=system,
        recall_at_k=mean(result.retrieval.recall_at_k for result in results),
        precision_at_k=mean(result.retrieval.precision_at_k for result in results),
        mrr=mean(result.retrieval.reciprocal_rank for result in results),
        # With binary manual relevance labels, this baseline defines context
        # relevance as the share of top-k chunks labeled relevant.
        context_relevance=mean(result.retrieval.precision_at_k for result in results),
        answer_relevance=mean(result.generation.answer_relevance for result in results),
        faithfulness=mean(result.generation.faithfulness for result in results),
        groundedness=mean(result.generation.groundedness for result in results),
        hallucination_rate=mean(result.generation.hallucination_rate for result in results),
        cases=tuple(results),
    )
