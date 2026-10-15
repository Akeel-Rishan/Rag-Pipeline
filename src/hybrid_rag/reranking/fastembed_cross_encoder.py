"""Local ONNX cross-encoder scoring through FastEmbed."""

import math
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder


class FastEmbedCrossEncoder:
    """Score the question jointly with every candidate document."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        *,
        cache_dir: str | Path = ".cache/fastembed",
        batch_size: int = 16,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        supported = {model["model"] for model in TextCrossEncoder.list_supported_models()}
        if model_name not in supported:
            raise ValueError(f"Unsupported cross-encoder model: {model_name}")
        self._model_name = model_name
        self.batch_size = batch_size
        self._model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=2,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, question: str, documents: list[str]) -> list[float]:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            raise ValueError("question must not be blank")
        if not isinstance(documents, list):
            raise TypeError("documents must be a list of strings")
        if any(not isinstance(document, str) or not document.strip()
               for document in documents):
            raise ValueError("documents must contain nonblank strings")
        if not documents:
            return []
        scores = [float(score) for score in self._model.rerank(
            question, documents, batch_size=self.batch_size)]
        if len(scores) != len(documents):
            raise RuntimeError("Cross-encoder returned the wrong number of scores")
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Cross-encoder returned a nonfinite score")
        return scores
