"""Reranking orchestration and local cross-encoder adapter tests."""

from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.reranking.fastembed_cross_encoder import FastEmbedCrossEncoder
from hybrid_rag.reranking.reranker import RerankingConfig, RerankingRetriever
from hybrid_rag.rag import RAGPipeline
from hybrid_rag.retrieval.base import RetrievedChunk


def hit(chunk_id, retrieval_score):
    return RetrievedChunk(chunk_id, f"text {chunk_id}", 1, "source.txt", retrieval_score)


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, top_k=3):
        self.calls.append((question, top_k))
        return self.results[:top_k]


class StubScorer:
    model_name = "controlled-cross-encoder"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, question, documents):
        self.calls.append((question, documents))
        return self.scores


def test_retrieves_broad_candidates_then_reorders_and_truncates():
    retriever = StubRetriever([hit("a", .9), hit("b", .8), hit("c", .7)])
    scorer = StubScorer([-.5, 4.0, 1.0])
    reranker = RerankingRetriever(
        retriever, scorer, RerankingConfig(retrieve_k=3, final_k=2))

    run = reranker.run("question")
    assert [item.chunk_id for item in run.candidates] == ["a", "b", "c"]
    assert [item.chunk_id for item in run.results] == ["b", "c"]
    assert run.results[0].score == 4.0
    assert run.results[0].retrieval_rank == 2
    assert run.results[0].retrieval_score == .8
    assert run.results[0].candidate is run.candidates[1]
    assert retriever.calls == [("question", 3)]
    assert scorer.calls == [("question", ["text a", "text b", "text c"])]


def test_equal_reranker_scores_keep_candidate_order_and_override_final_k():
    retriever = StubRetriever([hit("a", 2), hit("b", 1)])
    reranker = RerankingRetriever(
        retriever, StubScorer([.5, .5]), RerankingConfig(2, 1))
    assert [item.chunk_id for item in reranker.retrieve("q", top_k=2)] == ["a", "b"]


def test_reranked_results_can_feed_existing_rag_context():
    retriever = StubRetriever([hit("a", .9), hit("b", .8), hit("c", .7)])
    reranker = RerankingRetriever(
        retriever, StubScorer([0, 3, 2]), RerankingConfig(3, 2))

    class ContextCheckingLLM:
        def generate(self, system_prompt, user_prompt):
            assert user_prompt.index("text b") < user_prompt.index("text c")
            assert "text a" not in user_prompt
            return "Reranked evidence [1]."

    result = RAGPipeline(reranker, ContextCheckingLLM()).answer("question", top_k=2)
    assert [item.chunk_id for item in result.retrieved_chunks] == ["b", "c"]


@pytest.mark.parametrize("values", [
    {"retrieve_k": 0}, {"retrieve_k": True}, {"final_k": 0},
    {"final_k": True}, {"retrieve_k": 2, "final_k": 3},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        RerankingConfig(**values)


@pytest.mark.parametrize("final_k", [0, -1, True, 1.5, 4])
def test_invalid_final_k_override(final_k):
    reranker = RerankingRetriever(
        StubRetriever([]), StubScorer([]), RerankingConfig(3, 2))
    with pytest.raises(ValueError):
        reranker.run("question", final_k)


def test_duplicate_nonfinite_and_wrong_score_count_rejected():
    with pytest.raises(RuntimeError, match="duplicate"):
        RerankingRetriever(
            StubRetriever([hit("a", 1), hit("a", 2)]), StubScorer([1, 2]),
            RerankingConfig(2, 1)).run("q")
    with pytest.raises(RuntimeError, match="nonfinite"):
        RerankingRetriever(
            StubRetriever([hit("a", float("nan"))]), StubScorer([1]),
            RerankingConfig(1, 1)).run("q")
    with pytest.raises(RuntimeError, match="wrong number"):
        RerankingRetriever(
            StubRetriever([hit("a", 1)]), StubScorer([]),
            RerankingConfig(1, 1)).run("q")
    with pytest.raises(RuntimeError, match="nonfinite"):
        RerankingRetriever(
            StubRetriever([hit("a", 1)]), StubScorer([float("inf")]),
            RerankingConfig(1, 1)).run("q")


@pytest.fixture
def cross_encoder():
    with patch("hybrid_rag.reranking.fastembed_cross_encoder.TextCrossEncoder") as factory:
        factory.list_supported_models.return_value = [{"model": "test-reranker"}]
        backend = MagicMock()
        factory.return_value = backend
        yield FastEmbedCrossEncoder("test-reranker", batch_size=2), backend


def test_cross_encoder_adapter_preserves_score_order(cross_encoder):
    adapter, backend = cross_encoder
    backend.rerank.return_value = iter([1, -2])
    assert adapter.score("query", ["first", "second"]) == [1.0, -2.0]
    backend.rerank.assert_called_once_with(
        "query", ["first", "second"], batch_size=2)
    assert adapter.model_name == "test-reranker"


@pytest.mark.parametrize("scores", [[], [1], [float("nan"), 1]])
def test_cross_encoder_bad_output_rejected(cross_encoder, scores):
    adapter, backend = cross_encoder
    backend.rerank.return_value = iter(scores)
    with pytest.raises(RuntimeError):
        adapter.score("query", ["first", "second"])


def test_cross_encoder_empty_input_skips_model(cross_encoder):
    adapter, backend = cross_encoder
    assert adapter.score("query", []) == []
    backend.rerank.assert_not_called()


def test_cross_encoder_rejects_bad_input_and_model(cross_encoder):
    adapter, _ = cross_encoder
    with pytest.raises(ValueError):
        adapter.score(" ", ["document"])
    with pytest.raises(ValueError):
        adapter.score("query", [""])
    with pytest.raises(TypeError):
        adapter.score("query", "document")
    with pytest.raises(ValueError):
        FastEmbedCrossEncoder("missing-model")
