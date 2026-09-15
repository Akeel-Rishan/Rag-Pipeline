"""Generate real vectors and inspect cosine similarity; no retrieval."""

import argparse
import json
from pathlib import Path

from hybrid_rag.embeddings.base import EmbeddingService
from hybrid_rag.embeddings.fastembed_service import FastEmbedService
from hybrid_rag.embeddings.similarity import cosine_similarity
from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.pdf_loader import load_pdf


def demonstrate(service: EmbeddingService, full: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    chunk = chunk_documents(load_pdf(root / "data/sample/example.pdf"))[0]
    texts = [chunk["text"], "A dog is playing in the park.",
             "A puppy is having fun outdoors in a park.",
             "The database backup completed at midnight."]
    vectors = service.embed_documents(texts)
    query = "Where is the dog playing?"
    query_vector = service.embed_query(query)
    report = {
        "model": service.model_name, "dimension": service.dimension,
        "chunk": chunk,
        "examples": [{"text": text, "embedding": vector}
                     for text, vector in zip(texts, vectors)],
        "query": {"text": query, "embedding": query_vector},
        "similarity": {
            "paraphrases": cosine_similarity(vectors[1], vectors[2]),
            "unrelated": cosine_similarity(vectors[1], vectors[3]),
            "query_and_sentence": cosine_similarity(query_vector, vectors[1]),
        },
    }
    print(f"Model: {service.model_name}; dimension: {service.dimension}")
    for text, vector in zip(texts, vectors):
        print(f"\nText: {text}\nLength: {len(vector)}")
        print(f"{'Full vector' if full else 'First 8 coordinates'}: {vector if full else vector[:8]}")
    print(f"\nCosine scores: {report['similarity']}")
    print(f"Hand-check: cosine([1, 2], [2, 1]) = {cosine_similarity([1, 2], [2, 1]):.6f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Print all coordinates")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    service = FastEmbedService(cache_dir=root / ".cache/fastembed")
    report = demonstrate(service, args.full)
    output = root / "data/processed/embedding_demo.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull vectors saved to {output}")


if __name__ == "__main__":
    main()
