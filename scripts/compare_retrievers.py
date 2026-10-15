"""Compare vector and BM25 rankings without combining their scores."""

import argparse
from importlib.metadata import version
from pathlib import Path

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.indexing import index_document, stable_chunks
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.vector import VectorRetriever
from hybrid_rag.storage.qdrant_store import QdrantSettings, QdrantStore


def preview(text: str, width: int = 76) -> str:
    single_line = " ".join(text.split())
    return single_line if len(single_line) <= width else single_line[:width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question", nargs="?",
        default="Why keep source and page metadata?",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = root / "data/sample/chunking_example.txt"
    pages = [{"text": path.read_text(encoding="utf-8"), "page": 1, "source": path.name}]
    chunks = chunk_documents(pages, chunk_size=1000, overlap=100)
    identity = path.resolve().as_posix()
    lexical_chunks = stable_chunks(chunks, identity)

    embedder = FastEmbedService(cache_dir=root / ".cache/fastembed")
    settings = QdrantSettings.from_env()
    client = settings.connect()
    try:
        space = f"{embedder.model_name}|fastembed-{version('fastembed')}-onnx-q|roles-default|v1"
        store = QdrantStore(client, settings.collection, embedder.dimension, space)
        index_document(chunks, identity, embedder, store)
        vector_results = VectorRetriever(embedder, store).retrieve(args.question, args.top_k)
        bm25_results = BM25Retriever(lexical_chunks).retrieve(args.question, args.top_k)
    finally:
        client.close()

    print(f"Query: {args.question}\n")
    print(f"{'VECTOR RETRIEVAL':<90} | BM25 RETRIEVAL")
    print(f"{'-' * 90}-+-{'-' * 90}")
    for rank in range(max(len(vector_results), len(bm25_results))):
        vector = vector_results[rank] if rank < len(vector_results) else None
        bm25 = bm25_results[rank] if rank < len(bm25_results) else None
        vector_header = (
            f"{rank + 1}. score={vector.score:.4f} id={vector.chunk_id[:8]} "
            f"{vector.source} p.{vector.page}"
            if vector else ""
        )
        bm25_header = (
            f"{rank + 1}. score={bm25.score:.4f} id={bm25.chunk_id[:8]} "
            f"{bm25.source} p.{bm25.page}"
            if bm25 else ""
        )
        print(f"{vector_header:<90} | {bm25_header}")
        vector_text = preview(vector.text) if vector else ""
        bm25_text = preview(bm25.text) if bm25 else ""
        print(f"   {vector_text:<87} |    {bm25_text}")
        print()
    print("Scores are internal to each retriever and must not be compared across columns.")
    print("No fusion, reranking, or LLM generation was performed.")


if __name__ == "__main__":
    main()
