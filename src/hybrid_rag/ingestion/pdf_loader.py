"""Extract PDF pages without cleaning, chunking, or performing OCR."""

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader
from pypdf.errors import PyPdfError


class DocumentPage(TypedDict):
    """A page's extracted text and its original source location."""

    text: str
    page: int
    source: str


class PDFLoadError(Exception):
    """A PDF could not be read or its text could not be extracted."""


def load_pdf(path: str | Path) -> list[DocumentPage]:
    """Load all physical pages, including empty ones, in original order.

    Page numbers start at one; source is the filename, not a unique ID.
    Encrypted PDFs are unsupported. Parsing failures raise PDFLoadError;
    no partial result is returned. Missing/inaccessible files retain their
    native OSError subclasses so callers can distinguish filesystem errors.
    """
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {pdf_path}")

    pages: list[DocumentPage] = []
    location = "document structure"
    try:
        # Binary mode reads PDF bytes; the context manager closes the file.
        with pdf_path.open("rb") as stream:
            reader = PdfReader(stream, strict=False)
            if reader.is_encrypted:
                raise PDFLoadError(f"Encrypted PDFs are unsupported: {pdf_path.name}")

            for page_number, pdf_page in enumerate(reader.pages, start=1):
                location = f"page {page_number}"
                text = pdf_page.extract_text() or ""
                pages.append({
                    "text": text,
                    "page": page_number,
                    "source": pdf_path.name,
                })
    except PyPdfError as exc:
        raise PDFLoadError(
            f"Could not parse {pdf_path.name} ({location}): {exc}"
        ) from exc

    return pages
