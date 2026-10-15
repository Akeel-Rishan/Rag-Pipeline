"""Show vector, BM25, and weighted-RRF rankings without answer generation."""

import argparse
from importlib.metadata import version
from pathlib import Path

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.indexing import index_document, stable_chunks
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.hybrid import HybridRetriever, RRFConfig
from hybrid_rag.retrieval.vector import VectorRetriever
from hybrid_rag.storage.qdrant_store import QdrantSettings, QdrantStore


DEFAULT_QUERIES = [
    "Why keep source and page metadata?",
    "When can project notes be deleted during an active investigation?",
    "How can scanned pages or tables cause extraction problems?",
]


def ranking_line(rank, result):
    return (
        f"{rank}. id={result.chunk_id[:8]} score={result.score:.6f} "
        f"source={result.source} page={result.page}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", dest="queries",
                        help="Query to compare; repeat for multiple queries")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF rank constant, not the result count")
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    args = parser.parse_args()

    config = RRFConfig(
        rank_constant=args.rrf_k,
        candidate_k=args.candidate_k,
        vector_weight=args.vector_weight,
        bm25_weight=args.bm25_weight,
    )
    root = Path(__file__).resolve().parents[1]
    path = root / "data/sample/chunking_example.txt"
    pages = [{"text": path.read_text(encoding="utf-8"), "page": 1, "source": path.name}]
    chunks = chunk_documents(pages, chunk_size=1000, overlap=100)
    identity = path.resolve().as_posix()

    embedder = FastEmbedService(cache_dir=root / ".cache/fastembed")
    settings = QdrantSettings.from_env()
    client = settings.connect()
    try:
        space = f"{embedder.model_name}|fastembed-{version('fastembed')}-onnx-q|roles-default|v1"
        store = QdrantStore(client, settings.collection, embedder.dimension, space)
        index_document(chunks, identity, embedder, store)
        vector = VectorRetriever(embedder, store)
        bm25 = BM25Retriever(stable_chunks(chunks, identity))
        hybrid = HybridRetriever(vector, bm25, config)
        candidate_k = max(args.top_k, config.candidate_k)

        print(
            f"RRF config: rank_constant={config.rank_constant}, "
            f"vector_weight={config.vector_weight}, bm25_weight={config.bm25_weight}, "
            f"candidate_k={config.candidate_k}"
        )
        for question in args.queries or DEFAULT_QUERIES:
            vector_results = vector.retrieve(question, candidate_k)
            bm25_results = bm25.retrieve(question, candidate_k)
            fused = hybrid.fuse(vector_results, bm25_results, args.top_k)
            print(f"\nQUERY: {question}")
            print("\nVector ranking:")
            for rank, result in enumerate(vector_results[:args.top_k], start=1):
                print(ranking_line(rank, result))
            print("\nBM25 ranking:")
            for rank, result in enumerate(bm25_results[:args.top_k], start=1):
                print(ranking_line(rank, result))
            print("\nHybrid RRF ranking and calculation:")
            for rank, result in enumerate(fused, start=1):
                vector_term = (
                    f"{config.vector_weight}/({config.rank_constant}+{result.vector_rank})"
                    if result.vector_rank is not None else "0"
                )
                bm25_term = (
                    f"{config.bm25_weight}/({config.rank_constant}+{result.bm25_rank})"
                    if result.bm25_rank is not None else "0"
                )
                print(
                    f"{ranking_line(rank, result)} = {vector_term} + {bm25_term} "
                    f"= {result.vector_contribution:.6f} + "
                    f"{result.bm25_contribution:.6f}"
                )
    finally:
        client.close()

    print("\nRaw cosine and BM25 scores were displayed but never added or normalized.")
    print("No LLM generation was performed.")


if __name__ == "__main__":
    main()
