# Hybrid RAG Knowledge Assistant

A portfolio project built one phase at a time to understand hybrid retrieval
and evidence-based answer generation.

## Current scope

Phase 2 adds PDF loading into dictionaries with text, physical page number,
and source filename. No chunking, embeddings, retrieval, or generation yet.

Requires Python 3.12 or newer. Development currently uses Python 3.12.

## Setup (Windows PowerShell)

Run these commands from the project root. Create the environment only if it
does not already exist:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The install needs package-index access. The optional `dev` extra installs
pytest. Dependencies are not locked yet; a reproducibility strategy will be
added as the actual application dependencies take shape.

Using the environment's Python directly avoids PowerShell activation-policy
issues. Optional activation: `.\.venv\Scripts\Activate.ps1`.

## Run and verify

```powershell
.\.venv\Scripts\python.exe -m hybrid_rag
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import hybrid_rag; print(hybrid_rag.__file__)"
```

The first command prints:

```text
Hybrid RAG: PDF loading is ready. Run scripts/load_pdf.py to load a PDF.
```

The second should report no broken requirements. The third should point to
`src/hybrid_rag/__init__.py` in this project, confirming the editable install.

## Load a PDF (Phase 2)

After installing dependencies, run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/load_pdf.py data/sample/example.pdf
.\.venv\Scripts\python.exe -m pytest -q
```

The included synthetic PDF has three pages: text, a blank page, then text.
Output is a JSON list of records such as
`{"text": "This is physical page three.", "page": 3, "source": "example.pdf"}`.
To load your own document, replace the sample path; quote paths with spaces.

The Python interface is `load_pdf(path)` in
`hybrid_rag.ingestion.pdf_loader`. It returns ordinary dictionaries, annotated
with `TypedDict` for static tooling (not runtime validation).

Empty pages remain in the list with empty text. Page numbers are one-based
physical positions, not printed page labels. Text is not cleaned or chunked.
Missing files raise `FileNotFoundError`; other filesystem errors remain
`OSError` subclasses. Wrong extensions raise `ValueError`. Encrypted files
and pypdf parsing errors raise `PDFLoadError`, with the original parsing
exception chained for debugging. Unexpected programming errors propagate.
No partial list is returned on failure. The CLI reports handled errors on
stderr with exit code 1.

`strict=False` permits pypdf's recoverable repairs and warnings; successful
parsing does not guarantee correct text. Scans need OCR, and columns, tables,
fonts, and existing OCR layers can yield missing or out-of-order text. All
pages are held in memory; this is not a hardened public-upload service.
Source filenames can collide across folders and are not unique document IDs.

The fixture can be recreated with `scripts/create_sample_pdf.py` when it is
absent. The generator refuses to overwrite an existing file. Its low-level
PDF drawing commands are only for the fixed ASCII teaching fixture.

Reference: [pypdf text extraction documentation](https://pypdf.readthedocs.io/en/stable/user/extract-text.html).

## Structure

- `src/hybrid_rag/`: application package and future component subpackages.
- `tests/`: PDF loader tests.
- `scripts/`: PDF loading example and synthetic fixture generator.
- `data/sample/`: small, shareable example documents.
- `data/raw/`: local input documents, excluded from Git.
- `data/processed/`: generated data, excluded from Git.
- `evals/`: future evaluation datasets; generated reports are excluded from Git.
- `docs/`: future architecture decisions and operating instructions.
- `.venv/`: local environment, excluded from Git; recreate on each machine.

Empty folders contain `.gitkeep` files because Git tracks files, not folders.

## Configuration

`.env.example` is a public template. Phase 1 requires no configuration and
does not load `.env` files. Future phases will introduce settings when needed.
Keep real credentials out of source control. Git ignore rules do not remove
files that have already been tracked.

## Direction

Documents will be parsed, cleaned, chunked, and indexed in Qdrant and BM25.
Queries will use both retrievers, result fusion, reranking, bounded context,
and LLM generation with citations. Evaluation and an HTTP service follow.
