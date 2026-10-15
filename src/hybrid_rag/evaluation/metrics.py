"""Simple explicit retrieval and generation metrics; no model judge required."""

from dataclasses import dataclass
import re


_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalized_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class GenerationMetrics:
    answer_relevance: float
    faithfulness: float
    groundedness: float
    hallucination_rate: float


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: set[str] | frozenset[str],
                      k: int) -> RetrievalMetrics:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant_ids:
        raise ValueError("At least one relevant ID is required")
    top_ids = retrieved_ids[:k]
    relevant_retrieved = len(set(top_ids) & set(relevant_ids))
    first_rank = next((rank for rank, chunk_id in enumerate(top_ids, start=1)
                       if chunk_id in relevant_ids), None)
    return RetrievalMetrics(
        recall_at_k=relevant_retrieved / len(relevant_ids),
        precision_at_k=relevant_retrieved / k,
        reciprocal_rank=0.0 if first_rank is None else 1 / first_rank,
    )


def token_f1(actual: str, expected: str) -> float:
    actual_tokens = normalized_tokens(actual)
    expected_tokens = normalized_tokens(expected)
    if not actual_tokens or not expected_tokens:
        return 0.0
    from collections import Counter
    overlap = sum((Counter(actual_tokens) & Counter(expected_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(actual_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def generation_metrics(answer: str, expected_answer: str, context: str,
                       claim_support_threshold: float = 0.7) -> GenerationMetrics:
    if not 0 <= claim_support_threshold <= 1:
        raise ValueError("claim_support_threshold must be between 0 and 1")
    answer_tokens = normalized_tokens(answer)
    context_tokens = set(normalized_tokens(context))
    groundedness = (
        sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)
        if answer_tokens else 0.0
    )
    claims = [claim for claim in _CLAIM_SPLIT.split(answer.strip()) if claim.strip()]
    supported = 0
    for claim in claims:
        claim_tokens = normalized_tokens(claim)
        coverage = (sum(token in context_tokens for token in claim_tokens) / len(claim_tokens)
                    if claim_tokens else 0.0)
        supported += coverage >= claim_support_threshold
    faithfulness = supported / len(claims) if claims else 0.0
    return GenerationMetrics(
        answer_relevance=token_f1(answer, expected_answer),
        faithfulness=faithfulness,
        groundedness=groundedness,
        hallucination_rate=1 - faithfulness,
    )
