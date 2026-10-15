"""Run basic vector RAG and print the exact context sent to the LLM."""

import argparse
from importlib.metadata import version
from pathlib import Path

from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.generation.openai_compatible import (
    LLMGenerationError,
    OpenAICompatibleLLM,
    OpenAICompatibleSettings,
)
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.indexing import index_document
from hybrid_rag.rag import RAGPipeline
from hybrid_rag.retrieval.vector import VectorRetriever
from hybrid_rag.storage.qdrant_store import QdrantSettings, QdrantStore


class OfflineDemoLLM:
    """Deterministic test double, not a language model."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "Project notes for an active investigation must remain available until "
            "the investigation ends, even when seven years have passed [1]."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question", nargs="?",
        default="When can project notes be deleted during an active investigation?",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--offline-demo", action="store_true",
        help="Use a transparent deterministic test double instead of calling an LLM",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = root / "data/sample/chunking_example.txt"
    pages = [{"text": path.read_text(encoding="utf-8"), "page": 1, "source": path.name}]
    chunks = chunk_documents(pages, chunk_size=1000, overlap=100)
    embedder = FastEmbedService(cache_dir=root / ".cache/fastembed")
    settings = QdrantSettings.from_env()
    client = settings.connect()
    try:
        space = f"{embedder.model_name}|fastembed-{version('fastembed')}-onnx-q|roles-default|v1"
        store = QdrantStore(client, settings.collection, embedder.dimension, space)
        index_document(chunks, path.resolve().as_posix(), embedder, store)
        llm = OfflineDemoLLM() if args.offline_demo else OpenAICompatibleLLM(
            OpenAICompatibleSettings.from_env())
        try:
            result = RAGPipeline(VectorRetriever(embedder, store), llm).answer(
                args.question, top_k=args.top_k)
        except LLMGenerationError as error:
            parser.exit(1, f"error: {error}\n")

        print(f"Question:\n{result.question}")
        print("\nExact context passed to the LLM:\n---")
        print(result.context)
        print("---")
        print(f"\nAnswer:\n{result.answer}")
        print("\nCited sources:")
        for citation in result.citations:
            print(
                f"[{citation.label}] source={citation.source} page={citation.page} "
                f"chunk_id={citation.chunk_id}"
            )
        if args.offline_demo:
            print("\nGeneration mode: deterministic offline test double (no live LLM call)")
        else:
            print(f"\nGeneration model: {OpenAICompatibleSettings.from_env().model}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
