"""Basic vector RAG tests with real local Qdrant and a controlled LLM."""

import pytest
from qdrant_client import QdrantClient

from hybrid_rag.generation.prompting import SYSTEM_PROMPT, build_context
from hybrid_rag.ingestion.indexing import index_document
from hybrid_rag.rag import RAGPipeline
from hybrid_rag.retrieval.vector import RetrievedChunk, VectorRetriever
from hybrid_rag.storage.qdrant_store import QdrantStore


class FakeEmbedder:
    dimension = 2
    model_name = "test"

    def embed_documents(self, texts):
        return [[1.0, 0.0] if "investigation" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "Keep the notes until the investigation ends [1]."


@pytest.fixture
def pipeline():
    client = QdrantClient(":memory:")
    store = QdrantStore(client, "rag", 2, "test-space")
    chunks = [
        {"chunk_id": "random-1", "text": "Keep notes until the investigation ends.",
         "page": 7, "source": "policy.pdf"},
        {"chunk_id": "random-2", "text": "The office kitchen closes at six.",
         "page": 2, "source": "handbook.pdf"},
    ]
    index_document(chunks, "policy-set", FakeEmbedder(), store)
    llm = RecordingLLM()
    yield RAGPipeline(VectorRetriever(FakeEmbedder(), store), llm), llm
    client.close()


def test_pipeline_retrieves_builds_context_generates_and_cites(pipeline):
    rag, llm = pipeline
    result = rag.answer("What happens to investigation notes?", top_k=1)

    assert result.answer == "Keep the notes until the investigation ends [1]."
    assert len(result.retrieved_chunks) == 1
    assert result.retrieved_chunks[0].source == "policy.pdf"
    assert result.citations[0].source == "policy.pdf"
    assert result.citations[0].page == 7
    assert result.context == (
        "[1]\n"
        "source: policy.pdf\n"
        "page: 7\n"
        f"chunk_id: {result.retrieved_chunks[0].chunk_id}\n"
        "text:\nKeep notes until the investigation ends."
    )
    system_prompt, user_prompt = llm.calls[0]
    assert system_prompt == SYSTEM_PROMPT
    assert result.context in user_prompt
    assert "What happens to investigation notes?" in user_prompt


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_rejected(pipeline, top_k):
    rag, llm = pipeline
    with pytest.raises(ValueError):
        rag.answer("question", top_k=top_k)
    assert llm.calls == []


@pytest.mark.parametrize("question", ["", " \n"])
def test_blank_question_rejected_before_generation(pipeline, question):
    rag, llm = pipeline
    with pytest.raises(ValueError):
        rag.answer(question)
    assert llm.calls == []


def test_empty_context_is_explicit():
    assert build_context([]) == "(No context was retrieved.)"


def test_context_labels_follow_retrieval_order():
    chunks = [
        RetrievedChunk("a", "first", 1, "a.pdf", .9),
        RetrievedChunk("b", "second", 3, "b.pdf", .8),
    ]
    context = build_context(chunks)
    assert context.index("[1]") < context.index("[2]")
    assert "source: b.pdf\npage: 3" in context


def test_only_labels_used_by_answer_become_citations(pipeline):
    rag, llm = pipeline
    llm.generate = lambda system, user: "The relevant policy is here [1], not [99]."
    result = rag.answer("question", top_k=2)
    assert [citation.label for citation in result.citations] == [1]
    assert len(result.retrieved_chunks) == 2
