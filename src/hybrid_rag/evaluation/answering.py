"""Reproducible extractive answer baseline for offline evaluation."""

import re

from hybrid_rag.evaluation.metrics import normalized_tokens
from hybrid_rag.retrieval.base import RetrievedChunk


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "which", "why", "with",
}


class ExtractiveAnswerer:
    """Select context sentences by query-token overlap; this is not an LLM."""

    def __init__(self, max_sentences: int = 2) -> None:
        if type(max_sentences) is not int or max_sentences <= 0:
            raise ValueError("max_sentences must be a positive integer")
        self.max_sentences = max_sentences

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        query_tokens = set(normalized_tokens(question)) - _STOP_WORDS
        sentences: list[tuple[int, int, str]] = []
        seen: set[str] = set()
        position = 0
        for chunk in chunks:
            for sentence in _SENTENCE_SPLIT.split(chunk.text.replace("\n", " ")):
                sentence = " ".join(sentence.split()).strip()
                normalized = sentence.casefold()
                if not sentence or normalized in seen:
                    continue
                seen.add(normalized)
                sentence_tokens = set(normalized_tokens(sentence)) - _STOP_WORDS
                overlap = len(query_tokens & sentence_tokens)
                if overlap:
                    sentences.append((overlap, position, sentence))
                position += 1
        sentences.sort(key=lambda item: (-item[0], item[1]))
        if not sentences:
            return "I do not know from the supplied context."
        return " ".join(sentence for _, _, sentence in sentences[:self.max_sentences])
