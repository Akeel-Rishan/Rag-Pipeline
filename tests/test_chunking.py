"""Check boundaries, coverage, page provenance, and invalid configuration."""

from copy import deepcopy
from pathlib import Path
from uuid import UUID

import pytest

from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.pdf_loader import load_pdf


def page(text, number=3, source="document.pdf"):
    return {"text": text, "page": number, "source": source}


@pytest.mark.parametrize(("size", "overlap", "expected"), [
    (4, 0, ["abcd", "efgh", "ij"]),
    (4, 1, ["abcd", "defg", "ghij"]),
    (4, 2, ["abcd", "cdef", "efgh", "ghij"]),
    (10, 2, ["abcdefghij"]),
    (20, 2, ["abcdefghij"]),
    (1, 0, list("abcdefghij")),
])
def test_known_windows(size, overlap, expected):
    chunks = chunk_documents([page("abcdefghij")], chunk_size=size, overlap=overlap)
    assert [chunk["text"] for chunk in chunks] == expected


@pytest.mark.parametrize(("size", "expected_count"), [(300, 8), (1000, 3)])
def test_experiment_sizes_preserve_all_text(size, expected_count):
    text = "0123456789" * 200  # Exactly 2,000 characters.
    chunks = chunk_documents([page(text)], chunk_size=size, overlap=50)
    assert len(chunks) == expected_count
    assert all(0 < len(chunk["text"]) <= size for chunk in chunks)
    reconstructed = chunks[0]["text"] + "".join(c["text"][50:] for c in chunks[1:])
    assert reconstructed == text


def test_metadata_empty_pages_and_input_preservation():
    documents = [page("abc", 1), page("", 2), page("def", 3, "other.pdf")]
    original = deepcopy(documents)
    chunks = chunk_documents(documents, chunk_size=2, overlap=0)
    assert [(c["source"], c["page"], c["text"]) for c in chunks] == [
        ("document.pdf", 1, "ab"), ("document.pdf", 1, "c"),
        ("other.pdf", 3, "de"), ("other.pdf", 3, "f"),
    ]
    assert documents == original


def test_identical_chunks_have_distinct_uuid_ids_including_across_calls():
    first = chunk_documents([page("same"), page("same")])
    second = chunk_documents([page("same")])
    ids = [c["chunk_id"] for c in first + second]
    assert len(set(ids)) == 3
    assert all(UUID(identifier).version == 4 for identifier in ids)


def test_empty_input():
    assert chunk_documents([]) == []
    assert chunk_documents([page("")]) == []


def test_unicode_and_whitespace_are_not_cleaned():
    text = "  café\n猫🙂  "
    chunks = chunk_documents([page(text)], chunk_size=3, overlap=0)
    assert "".join(c["text"] for c in chunks) == text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (3, -1), (3, 3), (3, 4)])
def test_invalid_ranges(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=size, overlap=overlap)


@pytest.mark.parametrize(("size", "overlap"), [(3.5, 0), (3, 0.5), (True, 0), (3, False)])
def test_invalid_types(size, overlap):
    with pytest.raises(TypeError):
        chunk_documents([], chunk_size=size, overlap=overlap)


def test_pdf_loader_to_chunker():
    sample = Path(__file__).resolve().parents[1] / "data/sample/example.pdf"
    documents = load_pdf(sample)
    chunks = chunk_documents(documents, chunk_size=15, overlap=3)
    assert {c["page"] for c in chunks} == {1, 3}
    assert all(c["source"] == "example.pdf" for c in chunks)
    for document in documents:
        parts = [c["text"] for c in chunks if c["page"] == document["page"]]
        restored = parts[0] + "".join(part[3:] for part in parts[1:]) if parts else ""
        assert restored == document["text"]
