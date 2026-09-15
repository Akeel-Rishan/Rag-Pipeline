"""Join chunks, embedding inference and storage with repeatable point IDs."""

import json
from uuid import NAMESPACE_URL, uuid5

from hybrid_rag.embeddings.base import EmbeddingService
from hybrid_rag.ingestion.chunking import DocumentChunk
from hybrid_rag.storage.qdrant_store import QdrantStore


def stable_chunks(chunks: list[DocumentChunk], document_id: str) -> list[DocumentChunk]:
    """Use the complete ordered chunk list of ONE document; do not mutate it.

    document_id must be a stable unique identity, e.g. a corpus-relative path.
    Changed content creates new IDs; this does not delete obsolete points.
    """
    if not document_id.strip():
        raise ValueError("document_id must not be blank")
    return [{**chunk, "chunk_id": str(uuid5(NAMESPACE_URL, json.dumps(
        ["hybrid-rag-chunk-v1", document_id, chunk["page"], position, chunk["text"]],
        ensure_ascii=True, separators=(",", ":"),
    )))} for position, chunk in enumerate(chunks)]


def index_document(chunks: list[DocumentChunk], document_id: str,
                   embedder: EmbeddingService, store: QdrantStore) -> int:
    """Upsert one document in batches; retries converge for unchanged input.

    A multi-batch indexing operation is not atomic. On failure earlier batches
    can remain stored. Pass the same complete chunk list when retrying.
    """
    if embedder.dimension != store.dimension:
        raise ValueError("Embedder and collection dimensions differ")
    prepared = stable_chunks(chunks, document_id)
    if any(not chunk["text"].strip() for chunk in prepared):
        raise ValueError("Cannot index blank chunks; inspect extracted text first")
    store.ensure_collection()
    for start in range(0, len(prepared), 64):
        batch = prepared[start:start + 64]
        vectors = embedder.embed_documents([chunk["text"] for chunk in batch])
        store.upsert(batch, vectors, document_id)
    return len(prepared)
