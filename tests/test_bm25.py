"""Deterministic tests for the independent BM25 implementation."""

import pytest

from hybrid_rag.retrieval.bm25 import BM25Retriever, tokenize


def chunk(chunk_id, text, page=1, source="corpus.txt"):
    return {"chunk_id": chunk_id, "text": text, "page": page, "source": source}


def test_rare_exact_term_ranks_matching_chunk_first():
    retriever = BM25Retriever([
        chunk("a", "The service reported error ERR-8492 during startup.", 4, "errors.pdf"),
        chunk("b", "The service started normally after maintenance.", 2, "guide.pdf"),
        chunk("c", "General service troubleshooting steps.", 9, "guide.pdf"),
    ])
    results = retriever.retrieve("How do I investigate ERR-8492?", top_k=3)
    assert [result.chunk_id for result in results] == ["a"]
    assert results[0].page == 4
    assert results[0].source == "errors.pdf"
    assert results[0].score > 0


def test_term_frequency_saturates():
    retriever = BM25Retriever([
        chunk("once", "quasar"),
        chunk("twice", "quasar quasar"),
    ], b=0)
    results = retriever.retrieve("quasar", top_k=2)
    scores = {result.chunk_id: result.score for result in results}
    assert scores["twice"] > scores["once"]
    assert scores["twice"] < scores["once"] * 2


def test_length_normalization_favors_focused_chunk():
    retriever = BM25Retriever([
        chunk("short", "satellite handbook"),
        chunk("long", "satellite " + "filler " * 30),
    ])
    assert [result.chunk_id for result in retriever.retrieve("satellite", 2)] == [
        "short", "long"]


def test_repeated_query_term_does_not_change_score():
    retriever = BM25Retriever([chunk("a", "blue database")])
    assert retriever.retrieve("blue")[0].score == pytest.approx(
        retriever.retrieve("blue blue blue")[0].score)


def test_no_lexical_match_returns_no_results():
    retriever = BM25Retriever([chunk("a", "alpha beta")])
    assert retriever.retrieve("unrelated") == []


def test_tokenizer_casefolds_and_preserves_common_code_separators():
    assert tokenize("Fix ERR-8492 in API/v1 with Foo_Bar.") == [
        "fix", "err-8492", "in", "api/v1", "with", "foo_bar"]


@pytest.mark.parametrize(("k1", "b"), [
    (0, .75), (-1, .75), (float("nan"), .75), (1.5, -0.1),
    (1.5, 1.1), (1.5, float("inf")),
])
def test_invalid_parameters(k1, b):
    with pytest.raises(ValueError):
        BM25Retriever([], k1=k1, b=b)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    retriever = BM25Retriever([chunk("a", "text")])
    with pytest.raises(ValueError):
        retriever.retrieve("text", top_k)


def test_empty_and_all_tokenless_corpora_return_no_results():
    assert BM25Retriever([]).retrieve("query") == []
    assert BM25Retriever([chunk("a", "---")]).retrieve("query") == []


def test_duplicate_ids_and_malformed_chunks_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        BM25Retriever([chunk("same", "one"), chunk("same", "two")])
    with pytest.raises(ValueError, match="valid"):
        BM25Retriever([{"chunk_id": "a", "text": "missing metadata"}])
