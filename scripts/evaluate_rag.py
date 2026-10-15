"""Compare vector, hybrid, and reranked RAG with transparent metrics."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from qdrant_client import QdrantClient

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.evaluation.answering import ExtractiveAnswerer
from hybrid_rag.evaluation.dataset import load_evaluation_dataset
from hybrid_rag.evaluation.runner import evaluate_system
from hybrid_rag.generation.openai_compatible import OpenAICompatibleLLM, OpenAICompatibleSettings
from hybrid_rag.generation.prompting import SYSTEM_PROMPT, build_user_prompt
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.indexing import index_document, stable_chunks
from hybrid_rag.reranking.fastembed_cross_encoder import FastEmbedCrossEncoder
from hybrid_rag.reranking.reranker import RerankingConfig, RerankingRetriever
from hybrid_rag.retrieval.bm25 import BM25Retriever
from hybrid_rag.retrieval.hybrid import HybridRetriever, RRFConfig
from hybrid_rag.retrieval.vector import VectorRetriever
from hybrid_rag.storage.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--live-llm", action="store_true",
                        help="Use configured LLM instead of reproducible extractive baseline")
    args = parser.parse_args()
    if args.k <= 0 or args.retrieve_k < args.k:
        parser.error("Require 0 < k <= retrieve-k")

    root = Path(__file__).resolve().parents[1]
    source_path = root / "data/sample/chunking_example.txt"
    dataset_path = root / "evals/rag_eval.json"
    report_path = root / "evals/reports/phase_10_results.json"
    cases = load_evaluation_dataset(dataset_path)
    pages = [{"text": source_path.read_text(encoding="utf-8"),
              "page": 1, "source": source_path.name}]
    chunks = chunk_documents(pages, chunk_size=400, overlap=50)
    identity = "eval/chunking_example.txt"
    prepared = stable_chunks(chunks, identity)
    corpus_ids = {chunk["chunk_id"] for chunk in prepared}
    labeled_ids = set().union(*(case.relevant_chunk_ids for case in cases))
    if not labeled_ids <= corpus_ids:
        raise RuntimeError("Evaluation labels do not match the configured corpus/chunking")

    if args.live_llm:
        llm = OpenAICompatibleLLM(OpenAICompatibleSettings.from_env())
        def answer(question, chunks, context):
            return llm.generate(SYSTEM_PROMPT, build_user_prompt(question, context))
        answer_mode = f"live LLM: {llm.settings.model}"
    else:
        answerer = ExtractiveAnswerer(max_sentences=2)
        def answer(question, chunks, context):
            return answerer.answer(question, chunks)
        answer_mode = "deterministic extractive proxy (not an LLM)"

    embedder = FastEmbedService(cache_dir=root / ".cache/fastembed")
    client = QdrantClient(":memory:")
    try:
        space = f"{embedder.model_name}|fastembed-{version('fastembed')}-onnx-q|roles-default|v1"
        store = QdrantStore(client, "phase_10_eval", embedder.dimension, space)
        index_document(chunks, identity, embedder, store)
        vector = VectorRetriever(embedder, store)
        hybrid = HybridRetriever(
            vector,
            BM25Retriever(prepared),
            RRFConfig(candidate_k=args.retrieve_k),
        )
        reranked = RerankingRetriever(
            hybrid,
            FastEmbedCrossEncoder(cache_dir=root / ".cache/fastembed"),
            RerankingConfig(retrieve_k=args.retrieve_k, final_k=args.k),
        )
        systems = [
            ("vector-only", vector),
            ("hybrid-RRF", hybrid),
            ("hybrid+reranking", reranked),
        ]
        summaries = [evaluate_system(name, retriever, cases, args.k, answer)
                     for name, retriever in systems]
    finally:
        client.close()

    print(f"Cases: {len(cases)}; K={args.k}; generation={answer_mode}\n")
    headers = ("System", "Recall@K", "Precision@K", "MRR", "ContextRel",
               "AnswerRel", "Faithful", "Grounded", "Halluc.")
    print(f"{headers[0]:<20} " + " ".join(f"{header:>11}" for header in headers[1:]))
    print("-" * 116)
    for summary in summaries:
        values = (summary.recall_at_k, summary.precision_at_k, summary.mrr,
                  summary.context_relevance, summary.answer_relevance,
                  summary.faithfulness, summary.groundedness,
                  summary.hallucination_rate)
        print(f"{summary.system:<20} " + " ".join(f"{value:>11.3f}" for value in values))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "k": args.k,
        "retrieve_k": args.retrieve_k,
        "generation_mode": answer_mode,
        "systems": [summary.to_dict() for summary in summaries],
    }, indent=2), encoding="utf-8")
    print(f"\nDetailed per-case report: {report_path}")
    print("ContextRel equals manually labeled Precision@K in this baseline.")
    print("Token/claim metrics are transparent proxies, not semantic model-judge scores.")


if __name__ == "__main__":
    main()
