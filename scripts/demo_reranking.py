"""Retrieve hybrid candidates, cross-encode them, and show ranking changes."""

import argparse
from importlib.metadata import version
from pathlib import Path

from qdrant_client import QdrantClient

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.indexing import index_document, stable_chunks
from hybrid_rag.reranking.fastembed_cross_encoder import FastEmbedCrossEncoder
from hybrid_rag.reranking.reranker import RerankingConfig, RerankingRetriever
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.hybrid import HybridRetriever, RRFConfig
from hybrid_rag.retrieval.vector import VectorRetriever
from hybrid_rag.storage.qdrant_store import QdrantStore


def preview(text: str, width: int = 100) -> str:
    single_line = " ".join(text.split())
    return single_line if len(single_line) <= width else single_line[:width - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question", nargs="?",
        default="When can project notes be deleted during an active investigation?",
    )
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--model", default="Xenova/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()
    config = RerankingConfig(args.retrieve_k, args.final_k)

    root = Path(__file__).resolve().parents[1]
    path = root / "data/sample/chunking_example.txt"
    pages = [{"text": path.read_text(encoding="utf-8"), "page": 1, "source": path.name}]
    # Smaller windows create enough candidates to demonstrate retrieve-20/rerank-to-5.
    chunks = chunk_documents(pages, chunk_size=400, overlap=50)
    identity = "phase-9-demo/chunking_example-400-50"
    prepared = stable_chunks(chunks, identity)

    embedder = FastEmbedService(cache_dir=root / ".cache/fastembed")
    client = QdrantClient(":memory:")
    try:
        space = f"{embedder.model_name}|fastembed-{version('fastembed')}-onnx-q|roles-default|v1"
        store = QdrantStore(client, "phase_9_demo", embedder.dimension, space)
        index_document(chunks, identity, embedder, store)
        hybrid = HybridRetriever(
            VectorRetriever(embedder, store),
            BM25Retriever(prepared),
            RRFConfig(candidate_k=args.retrieve_k),
        )
        scorer = FastEmbedCrossEncoder(
            args.model, cache_dir=root / ".cache/fastembed", batch_size=8)
        run = RerankingRetriever(hybrid, scorer, config).run(args.question)
    finally:
        client.close()

    print(f"Query: {run.question}")
    print(f"Model: {scorer.model_name}")
    print(f"Requested retrieve_k={config.retrieve_k}; candidates found={len(run.candidates)}")
    print(f"Requested final_k={config.final_k}; final results={len(run.results)}")
    print("\nHybrid candidate ranking before reranking:")
    for rank, candidate in enumerate(run.candidates, start=1):
        print(
            f"{rank}. id={candidate.chunk_id[:8]} hybrid={candidate.score:.6f} "
            f"source={candidate.source} page={candidate.page}\n"
            f"   {preview(candidate.text)}"
        )

    print("\nRanking after cross-encoder reranking:")
    for rank, result in enumerate(run.results, start=1):
        movement = result.retrieval_rank - rank
        movement_text = "same" if movement == 0 else (
            f"up {movement}" if movement > 0 else f"down {-movement}")
        print(
            f"{rank}. id={result.chunk_id[:8]} cross_encoder={result.score:.6f} "
            f"old_rank={result.retrieval_rank} movement={movement_text} "
            f"source={result.source} page={result.page}\n"
            f"   {preview(result.text)}"
        )
    print("\nThe cross-encoder score replaces the hybrid score only for ordering these candidates.")
    print("No LLM generation was performed.")


if __name__ == "__main__":
    main()
