"""Index teaching text twice, then search persistent Qdrant storage."""

import argparse
from importlib.metadata import version
from pathlib import Path

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.indexing import index_document
from hybrid_rag.ingestion.pdf_loader import load_pdf
from hybrid_rag.storage.qdrant_store import QdrantSettings, QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="When can project notes be deleted during an active investigation?")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--document-id", help="Stable corpus identity; defaults to the input's resolved path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.pdf or root / "data/sample/chunking_example.txt"
    pages = load_pdf(path) if args.pdf else [
        {"text": path.read_text(encoding="utf-8"), "page": 1, "source": path.name}]
    chunks = chunk_documents(pages, chunk_size=1000, overlap=100)
    embedder = FastEmbedService(cache_dir=root / ".cache/fastembed")
    settings = QdrantSettings.from_env()
    client = settings.connect()
    try:
        # Change this version if artifacts, tokenizer or role processing change.
        space = f"{embedder.model_name}|fastembed-{version('fastembed')}-onnx-q|roles-default|v1"
        store = QdrantStore(client, settings.collection, embedder.dimension, space)
        identity = args.document_id or path.resolve().as_posix()
        index_document(chunks, identity, embedder, store)
        first = store.count()
        index_document(chunks, identity, embedder, store)
        second = store.count()
        if first != second:
            raise RuntimeError("Repeat indexing unexpectedly changed point count")
        print(f"Mode: {'server' if settings.url else 'persistent local (exact search)'}")
        print(f"Collection: {settings.collection}; points after first/second indexing: {first}/{second}")
        print(f"Query: {args.query}")
        for rank, hit in enumerate(store.search(embedder.embed_query(args.query)), start=1):
            payload = hit.payload or {}
            print(f"\n{rank}. score={hit.score:.4f} source={payload.get('source')} page={payload.get('page')}")
            print(f"chunk_id={payload.get('chunk_id')}\n{payload.get('text')}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
