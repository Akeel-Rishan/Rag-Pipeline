"""Offline contract and arithmetic tests; model quality is a separate experiment."""

from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.embeddings.similarity import cosine_similarity


@pytest.mark.parametrize(("a", "b", "expected"), [
    ([1, 2], [2, 1], 0.8), ([1, 0], [0, 1], 0),
    ([1, 2], [2, 4], 1), ([1, 0], [-1, 0], -1),
])
def test_cosine_known_answers(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)


@pytest.mark.parametrize(("a", "b"), [
    ([], []), ([1], [1, 2]), ([0, 0], [1, 2]),
    ([float("nan")], [1]), ([float("inf")], [1]),
])
def test_cosine_invalid_vectors(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


@pytest.fixture
def adapter():
    with patch("hybrid_rag.embeddings.fastembed_service.TextEmbedding") as factory:
        factory.list_supported_models.return_value = [{"model": "test-model", "dim": 2}]
        backend = MagicMock()
        factory.return_value = backend
        yield FastEmbedService("test-model"), backend


def test_document_order_and_query_role(adapter):
    service, backend = adapter
    backend.passage_embed.return_value = iter([[1, 2], [3, 4]])
    backend.query_embed.return_value = iter([[5, 6]])
    assert service.embed_documents(["first", "second"]) == [[1.0, 2.0], [3.0, 4.0]]
    backend.passage_embed.assert_called_once_with(["first", "second"], batch_size=32)
    assert service.embed_query("question") == [5.0, 6.0]
    backend.query_embed.assert_called_once_with("question")
    assert service.model_name == "test-model"
    assert service.dimension == 2


def test_empty_batch_skips_inference(adapter):
    service, backend = adapter
    assert service.embed_documents([]) == []
    backend.passage_embed.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n"])
def test_blank_text_rejected(adapter, text):
    service, _ = adapter
    with pytest.raises(ValueError):
        service.embed_documents([text])
    with pytest.raises(ValueError):
        service.embed_query(text)


def test_wrong_input_type(adapter):
    service, _ = adapter
    with pytest.raises(TypeError):
        service.embed_documents("not a list")
    with pytest.raises(TypeError):
        service.embed_query(123)


@pytest.mark.parametrize("vectors", [[], [[1]], [[0, 0]], [[float("nan"), 1]]])
def test_malformed_provider_output(adapter, vectors):
    service, backend = adapter
    backend.passage_embed.return_value = iter(vectors)
    with pytest.raises(RuntimeError):
        service.embed_documents(["text"])
