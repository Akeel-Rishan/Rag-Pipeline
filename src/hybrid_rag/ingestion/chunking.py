"""A transparent character-window chunker; no tokenizer or framework needed."""

from collections.abc import Iterable
from uuid import uuid4

from hybrid_rag.ingestion.pdf_loader import DocumentPage


class DocumentChunk(DocumentPage):
    """A text slice with its page provenance and a new identifier."""

    chunk_id: str


def chunk_documents(
    documents: Iterable[DocumentPage],
    *,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Split each page independently into overlapping character windows.

    Sizes count Python Unicode code points, not bytes or model tokens.
    Empty text produces no chunks; all other text, including whitespace,
    is preserved exactly. Input records are never modified. UUIDs are new
    on each call, so they are not stable identifiers for repeat ingestion.
    """
    # bool is an int subclass, but True/False are not meaningful sizes here.
    if type(chunk_size) is not int or type(overlap) is not int:
        raise TypeError("chunk_size and overlap must be integers")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[DocumentChunk] = []
    step = chunk_size - overlap

    for document in documents:
        text = document["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({
                "chunk_id": str(uuid4()),
                "text": text[start:end],
                "page": document["page"],
                "source": document["source"],
            })
            # Stop before emitting a redundant overlap-only tail.
            if end == len(text):
                break
            start += step

    return chunks
