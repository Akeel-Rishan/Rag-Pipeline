"""Behavior checks using real PDFs plus controlled extraction failures."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.errors import PdfReadError

from hybrid_rag.ingestion.pdf_loader import PDFLoadError, load_pdf


SAMPLE = Path(__file__).resolve().parents[1] / "data/sample/example.pdf"


def test_sample_preserves_text_blank_page_and_metadata():
    pages = load_pdf(SAMPLE)
    assert len(pages) == 3
    assert [page["page"] for page in pages] == [1, 2, 3]
    assert all(page["source"] == "example.pdf" for page in pages)
    assert pages[0]["text"].strip() == "Document loading preserves source metadata."
    assert pages[1]["text"] == ""
    assert pages[2]["text"].strip() == "This is physical page three."


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_wrong_extension(tmp_path):
    with pytest.raises(ValueError, match="Expected a .pdf"):
        load_pdf(tmp_path / "document.txt")


@pytest.mark.parametrize("content", [b"", b"This is not a PDF."])
def test_invalid_pdf(tmp_path, content):
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)
    with pytest.raises(PDFLoadError) as error:
        load_pdf(path)
    assert isinstance(error.value.__cause__, PdfReadError)


def test_encrypted_pdf(tmp_path):
    path = tmp_path / "locked.pdf"
    with PdfWriter() as writer:
        writer.add_blank_page(width=612, height=792)
        writer.encrypt("test-password")
        writer.write(path)
    with pytest.raises(PDFLoadError, match="Encrypted"):
        load_pdf(path)


def test_none_extraction_becomes_empty_text():
    with patch("pypdf._page.PageObject.extract_text", return_value=None):
        assert all(page["text"] == "" for page in load_pdf(SAMPLE))


def test_extraction_failure_does_not_return_partial_pages():
    with patch("pypdf._page.PageObject.extract_text",
               side_effect=["First page", PdfReadError("Damaged content")]):
        with pytest.raises(PDFLoadError, match="page 2"):
            load_pdf(SAMPLE)
