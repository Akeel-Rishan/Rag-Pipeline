# Hybrid RAG Knowledge Assistant

A portfolio project built one phase at a time to understand hybrid retrieval
and evidence-based answer generation.

## Current scope

Phase 4 adds local embeddings and a manual cosine similarity demonstration.
PDF loading and character chunking preserve source and page metadata.
There is no vector database, retrieval pipeline, or answer generation yet.

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
Hybrid RAG: local embeddings are ready. Run scripts/demo_embeddings.py.
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

## Chunking (Phase 3)

```python
from hybrid_rag.ingestion.pdf_loader import load_pdf
from hybrid_rag.ingestion.chunking import chunk_documents

pages = load_pdf("data/sample/example.pdf")
chunks = chunk_documents(pages, chunk_size=300, overlap=50)
```

Output records have `chunk_id`, `text`, `page`, and `source`. Lengths count
Python Unicode code points, not tokens or bytes. The window advances by
`chunk_size - overlap`. Both parameters must be integers, with
`chunk_size > 0` and `0 <= overlap < chunk_size`. The default overlap is 100;
set it explicitly when experimenting with smaller sizes.

The chunker processes pages independently, never mutates input records, skips
empty text, and preserves whitespace in nonempty text. It stops when a window
reaches the page end, avoiding a redundant overlap-only tail. It can split
words, sentences, and even multi-code-point visible characters. It provides
no semantic boundary detection. UUID4 IDs are practically unique but change
on reruns; they do not provide idempotent database ingestion.

### Compare 300 vs 1000 characters

```powershell
.\.venv\Scripts\python.exe scripts/compare_chunks.py
.\.venv\Scripts\python.exe scripts/compare_chunks.py --show-text
.\.venv\Scripts\python.exe scripts/compare_chunks.py --overlap 0
.\.venv\Scripts\python.exe scripts/compare_chunks.py --pdf "data/raw/your-document.pdf" --show-text
.\.venv\Scripts\python.exe -m pytest -q
```

The default input is the original teaching text in
`data/sample/chunking_example.txt`, represented as a single synthetic page.
The tiny Phase 2 PDF is too short to demonstrate these two sizes well.
Both configurations use the same input and 50 characters of overlap by
default. Compare chunk counts, lengths, repeated characters, and boundaries.
The overlap has different percentages at the two sizes: about 17% vs 5%.
This is a fixed absolute-overlap comparison, not a fixed-ratio comparison.

Find the evidence answering: "When can project notes be deleted if an
investigation is still active?" Is the rule and its exception in one chunk?
Inspect the context before and after each split. Repeat with zero overlap.
These observations do not measure retrieval accuracy; that needs retrieval
and labeled questions in a later phase.

### Optional library comparison (not installed or used)

LangChain's separate `langchain-text-splitters` package can preserve paragraph
and word boundaries where possible with a recursive character splitter:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, length_function=len,
)
library_chunks = splitter.create_documents(
    [page["text"] for page in pages],
    metadatas=[{"source": page["source"], "page": page["page"]} for page in pages],
)
```

The result contains LangChain Document objects with `page_content` and
`metadata`, not our dictionaries. It does not assign our `chunk_id` field.
Overlap is a target and depends on separator boundaries. This is not
embedding-based semantic splitting or identical fixed-window output.
Reference: [recursive splitting documentation](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter).

## Embeddings (Phase 4)

Install updated dependencies, then run the real-model experiment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/demo_embeddings.py
.\.venv\Scripts\python.exe scripts/demo_embeddings.py --full
.\.venv\Scripts\python.exe -m pytest -q
```

The initial run downloads `BAAI/bge-small-en-v1.5` model artifacts into the
ignored `.cache/fastembed/` directory. Subsequent runs reuse the cache.
Inference runs locally on CPU; no API key or database is needed. FastEmbed
uses ONNX Runtime rather than requiring the PyTorch training stack.
The model is English, has 384 dimensions and a 512-token input window.
Overlong text can be truncated: our character chunker does not guarantee
token limits. Token-aware input validation remains future work.

`EmbeddingService` in `embeddings/base.py` is a typing Protocol with model
identity, dimension, `embed_documents(texts)` and `embed_query(text)`. A new
provider implements that contract without changing callers. This is not
runtime enforcement or automatic configuration; select the adapter when
constructing the application. `FastEmbedService` loads one model instance,
uses passage/query methods, and converts vectors into ordinary float lists.
For this BGE model in FastEmbed 0.7.4, both role methods delegate to the same
embedding operation without adding an instruction. The separate interface
allows future adapters to apply model-specific role handling. FastEmbed's
selected artifact is `qdrant/bge-small-en-v1.5-onnx-q`, a quantized ONNX model;
its coordinates may differ from other implementations of the original model.
Blank text is rejected; an empty batch returns an empty list. Provider errors
propagate for debugging. Do not construct a model for every chunk.

The demo embeds an actual PDF chunk, two related sentences, an unrelated
sentence, and a query. It prints dimensions, coordinate previews and cosine
scores. Complete vectors and model identity are written to
`data/processed/embedding_demo.json` (overwritten on each demo run, ignored by
Git). Document metadata stays outside the embedding model and is kept in the
report. Only text goes into embedding inference.

Manual cosine: for `[1, 2]` and `[2, 1]`, the dot product is 4, both lengths
are sqrt(5), and the cosine is 4/5 = 0.8. Our implementation validates equal
dimensions and finite, nonzero vectors, then explicitly multiplies and sums.
Zero vectors have no direction. Cosine is not a probability or a universal
relevance threshold. Matching dimensions alone cannot establish matching
embedding spaces. The simple arithmetic also rejects numeric overflow.

Offline tests use a fake backend to check adapter behavior and known vectors
to test the math. They do not validate learned semantic quality. The demo is
a separate real-model smoke experiment, not a retrieval evaluation suite.

Observed demo scores: related sentences 0.9233, unrelated sentences 0.3251,
and query versus the dog sentence 0.8072. Every vector had 384 coordinates.
These are illustrative measurements, not general accuracy guarantees.

Changing models, revisions, tokenization or role instructions may require
re-embedding all documents. Never mix old document vectors with incompatible
query vectors. Model weights and dependencies are not yet revision-locked;
pinning artifacts is necessary for stricter reproducibility. The present
model is a small English baseline, not a multilingual or domain-quality claim.

References: [FastEmbed supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
and [role-specific embedding methods](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/).

## Structure

- `src/hybrid_rag/`: application package and future component subpackages.
- `tests/`: PDF loader and chunker tests.
- `scripts/`: PDF loading, sample generation, and chunk-size comparison.
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
