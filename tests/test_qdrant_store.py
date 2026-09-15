"""Real local-Qdrant tests with controlled vectors; no network or model needed."""

import pytest
from qdrant_client import QdrantClient

from hybrid_rag.ingestion.indexing import index_document, stable_chunks
from hybrid_rag.storage.qdrant_store import QdrantSettings, QdrantStore


class FakeEmbedder:
    dimension = 2
    model_name = "test"

    def embed_documents(self, texts):
        return [[1.0, 0.0] if text == "dog" else [0.0, 1.0] for text in texts]


def chunks():
    return [{"chunk_id": "random", "text": text, "page": i + 1, "source": "sample.pdf"}
            for i, text in enumerate(["dog", "database"])]


def test_repeat_index_search_and_persistence(tmp_path):
    path = str(tmp_path / "db")
    client = QdrantClient(path=path)
    try:
        store = QdrantStore(client, "test", 2, "test-space")
        original = chunks()
        index_document(original, "corpus/sample.pdf", FakeEmbedder(), store)
        again = chunks()
        again[0]["chunk_id"] = "different-random-id"
        index_document(again, "corpus/sample.pdf", FakeEmbedder(), store)
        assert store.count() == 2
        hits = store.search([1.0, 0.0], limit=2)
        assert hits[0].score == pytest.approx(1)
        assert hits[1].score == pytest.approx(0)
        assert hits[0].payload["text"] == "dog"
        assert hits[0].payload["source"] == "sample.pdf"
        assert hits[0].payload["page"] == 1
        assert str(hits[0].id) == hits[0].payload["chunk_id"]
        assert original[0]["chunk_id"] == "random"
    finally:
        client.close()
    reopened = QdrantClient(path=path)
    try:
        assert QdrantStore(reopened, "test", 2, "test-space").count() == 2
    finally:
        reopened.close()


def test_document_identity_disambiguates_same_filename():
    assert stable_chunks(chunks(), "a/sample.pdf")[0]["chunk_id"] != stable_chunks(chunks(), "b/sample.pdf")[0]["chunk_id"]
    assert stable_chunks(chunks(), "a") == stable_chunks(chunks(), "a")


def test_incompatible_collection_and_invalid_vectors():
    client = QdrantClient(":memory:")
    try:
        store = QdrantStore(client, "test", 2, "space-a")
        store.ensure_collection()
        for dimension, space in [(3, "space-a"), (2, "space-b")]:
            with pytest.raises(ValueError):
                QdrantStore(client, "test", dimension, space).ensure_collection()
        for vector in ([1.0], [0.0, 0.0], [float("nan"), 1.0]):
            with pytest.raises(ValueError):
                store.search(vector)
        with pytest.raises(ValueError):
            store.upsert(stable_chunks(chunks(), "doc"), [[1.0, 0.0]], "doc")
        with pytest.raises(ValueError):
            store.search([1.0, 0.0], limit=0)
    finally:
        client.close()


def test_environment_settings(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "test-secret")
    monkeypatch.setenv("QDRANT_COLLECTION", "custom")
    settings = QdrantSettings.from_env()
    assert settings.url == "http://localhost:6333"
    assert settings.collection == "custom"
    assert "test-secret" not in repr(settings)
