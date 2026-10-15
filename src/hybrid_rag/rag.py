"""Basic vector RAG orchestration without lexical search or reranking."""

from dataclasses import dataclass
import re

from hybrid_rag.generation.base import LanguageModel
from hybrid_rag.generation.prompting import SYSTEM_PROMPT, build_context, build_user_prompt
from hybrid_rag.retrieval.vector import RetrievedChunk, VectorRetriever


@dataclass(frozen=True)
class Citation:
    label: int
    source: str
    page: int
    chunk_id: str


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    context: str
    retrieved_chunks: tuple[RetrievedChunk, ...]
    citations: tuple[Citation, ...]


class RAGPipeline:
    """Connect retrieval, transparent context construction, and generation."""

    def __init__(self, retriever: VectorRetriever, llm: LanguageModel) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(self, question: str, top_k: int = 3) -> RAGResult:
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = build_context(chunks)
        user_prompt = build_user_prompt(question, context)
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        labels = dict.fromkeys(int(match) for match in re.findall(r"\[(\d+)\]", answer))
        citations = tuple(Citation(
            label=label,
            source=chunks[label - 1].source,
            page=chunks[label - 1].page,
            chunk_id=chunks[label - 1].chunk_id,
        ) for label in labels if 1 <= label <= len(chunks))
        return RAGResult(
            question=question,
            answer=answer,
            context=context,
            retrieved_chunks=tuple(chunks),
            citations=citations,
        )
