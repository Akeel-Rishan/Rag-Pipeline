"""Tests for weighted Reciprocal Rank Fusion without score mixing."""

import pytest

from hybrid_rag.retrieval.base import RetrievedChunk
from hybrid_rag.retrieval.hybrid import HybridRetriever, RRFConfig


def hit(chunk_id, score, text=None, page=1, source="source.txt"):
    return RetrievedChunk(chunk_id, text or f"text {chunk_id}", page, source, score)


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, top_k=3):
        self.calls.append((question, top_k))
        return self.results[:top_k]


def make_fusion(config=None):
    vector = StubRetriever([])
    bm25 = StubRetriever([])
    return HybridRetriever(vector, bm25, config), vector, bm25


def test_rrf_rewards_cross_retriever_rank_agreement():
    fusion, _, _ = make_fusion()
    results = fusion.fuse(
        [hit("a", .9), hit("b", .8), hit("c", .7)],
        [hit("b", 500), hit("c", 20), hit("a", 1)],
        top_k=3,
    )
    assert [result.chunk_id for result in results] == ["b", "a", "c"]
    first = results[0]
    assert first.vector_rank == 2
    assert first.bm25_rank == 1
    assert first.vector_score == .8
    assert first.bm25_score == 500
    assert first.vector_contribution == pytest.approx(1 / 62)
    assert first.bm25_contribution == pytest.approx(1 / 61)
    assert first.score == pytest.approx(1 / 62 + 1 / 61)


def test_raw_score_magnitude_does_not_change_fusion():
    fusion, _, _ = make_fusion()
    ordinary = fusion.fuse([hit("a", .9), hit("b", .8)],
                           [hit("b", 3), hit("a", 2)], 2)
    extreme = fusion.fuse([hit("a", .000001), hit("b", .0000001)],
                          [hit("b", 10**100), hit("a", -10**100)], 2)
    assert [(item.chunk_id, item.score) for item in ordinary] == [
        (item.chunk_id, item.score) for item in extreme]


def test_weights_are_adjustable_and_a_single_list_result_survives():
    config = RRFConfig(rank_constant=10, vector_weight=2, bm25_weight=.5)
    fusion, _, _ = make_fusion(config)
    results = fusion.fuse([hit("vector-only", .5)], [hit("bm25-only", 8)], 2)
    assert [item.chunk_id for item in results] == ["vector-only", "bm25-only"]
    assert results[0].score == pytest.approx(2 / 11)
    assert results[0].bm25_rank is None
    assert results[1].score == pytest.approx(.5 / 11)
    assert results[1].vector_rank is None


def test_retrieve_requests_candidate_depth_then_limits_output():
    vector = StubRetriever([hit("a", .9), hit("b", .8)])
    bm25 = StubRetriever([hit("b", 4), hit("a", 3)])
    fusion = HybridRetriever(vector, bm25, RRFConfig(candidate_k=7))
    results = fusion.retrieve("question", top_k=1)
    assert len(results) == 1
    assert vector.calls == [("question", 7)]
    assert bm25.calls == [("question", 7)]


@pytest.mark.parametrize("config", [
    {"rank_constant": -1}, {"rank_constant": True},
    {"candidate_k": 0}, {"candidate_k": True},
    {"vector_weight": -1}, {"bm25_weight": float("nan")},
    {"vector_weight": True}, {"vector_weight": 0, "bm25_weight": 0},
])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        RRFConfig(**config)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    fusion, _, _ = make_fusion()
    with pytest.raises(ValueError):
        fusion.retrieve("question", top_k)


def test_duplicate_ids_nonfinite_scores_and_payload_disagreement_rejected():
    fusion, _, _ = make_fusion()
    with pytest.raises(RuntimeError, match="duplicate"):
        fusion.fuse([hit("a", 2), hit("a", 1)], [], 2)
    with pytest.raises(RuntimeError, match="nonfinite"):
        fusion.fuse([hit("a", float("inf"))], [], 1)
    with pytest.raises(RuntimeError, match="disagree"):
        fusion.fuse([hit("a", 1, text="first")],
                    [hit("a", 1, text="different")], 1)
