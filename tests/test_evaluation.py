"""Tests for manually labeled, framework-free RAG evaluation."""

import json

import pytest

from hybrid_rag.evaluation.answering import ExtractiveAnswerer
from hybrid_rag.evaluation.dataset import EvaluationCase, EvidenceLocation, load_evaluation_dataset
from hybrid_rag.evaluation.metrics import generation_metrics, retrieval_metrics, token_f1
from hybrid_rag.evaluation.runner import evaluate_system
from hybrid_rag.retrieval.base import RetrievedChunk


def hit(chunk_id, text=None):
    return RetrievedChunk(chunk_id, text or f"text {chunk_id}", 1, "source.txt", 1.0)


def test_recall_precision_and_reciprocal_rank():
    metrics = retrieval_metrics(["wrong", "relevant-a", "other", "relevant-b"],
                                {"relevant-a", "relevant-b"}, k=3)
    assert metrics.recall_at_k == .5
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank == .5


def test_no_relevant_retrieval_has_zero_metrics():
    metrics = retrieval_metrics(["wrong"], {"relevant"}, 2)
    assert metrics.recall_at_k == 0
    assert metrics.precision_at_k == 0
    assert metrics.reciprocal_rank == 0


def test_token_f1_known_values():
    assert token_f1("red blue", "red green") == .5
    assert token_f1("unrelated", "expected") == 0


def test_generation_support_metrics_detect_unsupported_claim():
    metrics = generation_metrics(
        "Project notes remain available. Mars is made of cheese.",
        "Project notes remain available.",
        "The policy says project notes remain available until closure.",
    )
    assert metrics.answer_relevance > 0
    assert metrics.faithfulness == .5
    assert metrics.hallucination_rate == .5
    assert 0 < metrics.groundedness < 1


def test_extractive_answerer_selects_query_relevant_sentence():
    chunks = [hit("a", "The kitchen closes at six. Notes remain during investigation.")]
    answer = ExtractiveAnswerer(max_sentences=1).answer(
        "What happens to investigation notes?", chunks)
    assert answer == "Notes remain during investigation."


def test_evaluate_system_aggregates_cases():
    case = EvaluationCase(
        "case", "question alpha", "alpha answer", frozenset({"right"}),
        (EvidenceLocation("source.txt", 1),),
    )

    class Retriever:
        def retrieve(self, question, top_k=3):
            return [hit("right", "alpha answer"), hit("wrong", "noise")]

    summary = evaluate_system(
        "system", Retriever(), [case], 2,
        lambda question, chunks, context: "alpha answer",
    )
    assert summary.recall_at_k == 1
    assert summary.precision_at_k == .5
    assert summary.context_relevance == .5
    assert summary.mrr == 1
    assert summary.answer_relevance == 1
    assert summary.faithfulness == 1
    assert summary.hallucination_rate == 0


def test_curated_dataset_loads_and_has_unique_ids():
    cases = load_evaluation_dataset("evals/rag_eval.json")
    assert len(cases) == 6
    assert len({case.case_id for case in cases}) == 6
    assert all(case.relevant_chunk_ids for case in cases)


def test_invalid_dataset_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"id": "missing-fields"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_evaluation_dataset(path)
