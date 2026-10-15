"""A small independent BM25 index for teaching lexical retrieval."""

from collections import Counter
from dataclasses import dataclass
import math
import re

from hybrid_rag.ingestion.chunking import DocumentChunk
from hybrid_rag.retrieval.base import RetrievedChunk


# Keep common code/ID separators inside tokens: ERR-8492, v1.2 and foo_bar.
_TOKEN_PATTERN = re.compile(r"(?u)(?<!\w)\w+(?:[./:+#-]\w+)*(?!\w)")


def tokenize(text: str) -> list[str]:
    """Case-fold text into words and code-like lexical terms."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [match.group().casefold() for match in _TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True)
class _IndexedChunk:
    chunk_id: str
    text: str
    page: int
    source: str
    term_frequencies: Counter[str]
    length: int


class BM25Retriever:
    """Rank an in-memory, fixed chunk corpus using Okapi BM25."""

    def __init__(self, chunks: list[DocumentChunk], *, k1: float = 1.5,
                 b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")
        self.k1 = k1
        self.b = b
        self._documents: list[_IndexedChunk] = []
        self._document_frequencies: Counter[str] = Counter()
        seen_ids: set[str] = set()

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            page = chunk.get("page")
            source = chunk.get("source")
            if (not isinstance(chunk_id, str) or not chunk_id
                    or not isinstance(text, str) or type(page) is not int
                    or not isinstance(source, str)):
                raise ValueError("Each chunk needs valid chunk_id, text, page, and source fields")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in BM25 corpus: {chunk_id}")
            seen_ids.add(chunk_id)
            tokens = tokenize(text)
            frequencies = Counter(tokens)
            self._document_frequencies.update(frequencies.keys())
            self._documents.append(_IndexedChunk(
                chunk_id=chunk_id,
                text=text,
                page=page,
                source=source,
                term_frequencies=frequencies,
                length=len(tokens),
            ))

        self._average_length = (
            sum(document.length for document in self._documents) / len(self._documents)
            if self._documents else 0.0
        )

    def _inverse_document_frequency(self, term: str) -> float:
        document_count = len(self._documents)
        document_frequency = self._document_frequencies[term]
        return math.log(
            1 + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._documents or self._average_length == 0:
            return []

        # Repeated query words do not multiply evidence in this basic variant.
        query_terms = dict.fromkeys(tokenize(question))
        scored: list[tuple[float, int, _IndexedChunk]] = []
        for position, document in enumerate(self._documents):
            score = 0.0
            length_ratio = document.length / self._average_length
            for term in query_terms:
                frequency = document.term_frequencies[term]
                if frequency == 0:
                    continue
                saturation = (
                    frequency * (self.k1 + 1)
                    / (frequency + self.k1 * (1 - self.b + self.b * length_ratio))
                )
                score += self._inverse_document_frequency(term) * saturation
            if score > 0:
                scored.append((score, position, document))

        # Corpus position provides deterministic ordering for exact score ties.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [RetrievedChunk(
            chunk_id=document.chunk_id,
            text=document.text,
            page=document.page,
            source=document.source,
            score=score,
        ) for score, _, document in scored[:top_k]]
