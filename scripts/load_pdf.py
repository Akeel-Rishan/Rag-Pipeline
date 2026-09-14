"""Print extracted PDF pages as JSON. Run from an installed environment."""

import argparse
import json
from pathlib import Path

from hybrid_rag.ingestion.pdf_loader import PDFLoadError, load_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a PDF file")
    args = parser.parse_args()
    try:
        pages = load_pdf(args.path)
    except (OSError, ValueError, PDFLoadError) as exc:
        parser.exit(1, f"Could not load PDF: {exc}\n")
    # ASCII escapes keep Unicode text safe on Windows consoles and reversible.
    print(json.dumps(pages, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
