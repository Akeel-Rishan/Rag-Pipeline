"""Compare 300- and 1,000-character windows on identical input."""

import argparse
import json
from pathlib import Path

from hybrid_rag.ingestion.chunking import chunk_documents
from hybrid_rag.ingestion.pdf_loader import PDFLoadError, load_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Use a PDF instead of the teaching text")
    parser.add_argument("--overlap", type=int, default=50, help="Shared characters (default: 50)")
    parser.add_argument("--show-text", action="store_true", help="Print full chunk records as JSON")
    args = parser.parse_args()
    try:
        if args.pdf:
            documents = load_pdf(args.pdf)
        else:
            fixture = Path(__file__).resolve().parents[1] / "data/sample/chunking_example.txt"
            documents = [{"text": fixture.read_text(encoding="utf-8"),
                          "page": 1, "source": fixture.name}]

        # Validate both configurations before printing either result.
        comparisons = [(size, chunk_documents(documents, chunk_size=size, overlap=args.overlap))
                       for size in (300, 1000)]
        original_length = sum(len(document["text"]) for document in documents)
        print(f"Input: {len(documents)} page records, {original_length} characters")
        for size, chunks in comparisons:
            lengths = [len(chunk["text"]) for chunk in chunks]
            total = sum(lengths)
            print(f"\nchunk_size={size}, overlap={args.overlap}: {len(chunks)} chunks")
            print(f"Lengths: {lengths}")
            print(f"Total chunk characters: {total}; repeated characters: {total - original_length}")
            if args.show_text:
                print(json.dumps(chunks, indent=2, ensure_ascii=True))
    except (OSError, ValueError, PDFLoadError) as exc:
        parser.exit(1, f"Could not compare chunks: {exc}\n")


if __name__ == "__main__":
    main()
