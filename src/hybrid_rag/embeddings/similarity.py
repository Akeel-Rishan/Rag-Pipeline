"""Cosine similarity using explicit arithmetic, not a vector library."""

import math
from collections.abc import Sequence


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compare directions; reject undefined or incompatible numeric inputs."""
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("Vectors must be nonempty and have equal dimensions")
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("Vector coordinates must be finite")

    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    if length_a == 0 or length_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    denominator = length_a * length_b
    if not math.isfinite(dot_product) or not math.isfinite(denominator):
        raise ValueError("Coordinates are too large for this simple implementation")
    if denominator == 0:
        raise ValueError("Coordinates are too small for this simple implementation")
    score = dot_product / denominator
    return max(-1.0, min(1.0, score))  # Limit floating-point rounding drift.
