# Hybrid RAG Knowledge Assistant

A portfolio project built one phase at a time to understand hybrid retrieval
and evidence-based answer generation.

## Current scope

Phase 1 provides project packaging, a local virtual environment, module
boundaries, and a startup check. No RAG components are implemented yet.

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
Hybrid RAG project setup is ready. No RAG components are implemented yet.
```

The second should report no broken requirements. The third should point to
`src/hybrid_rag/__init__.py` in this project, confirming the editable install.
No automated test cases exist yet; pytest will report no tests collected.

## Structure

- `src/hybrid_rag/`: application package and future component subpackages.
- `tests/`: future automated tests.
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
