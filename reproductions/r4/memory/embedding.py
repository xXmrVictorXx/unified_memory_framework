"""Embedding interface for R4's SEM axis.

R4's paper doesn't name the embedding model. We treat it as an injectable
callable ``(text: str) -> Sequence[float]``. Tests use ``MockEmbedding``
(``reproductions._common.mocks``); production use wires up a real model
(``sentence-transformers``, OpenAI ``text-embedding-3-small``, etc.).

A small vector-math module is provided here in pure stdlib so the cosine
similarity needed for deduplication (Eq. 5) and semantic retrieval works
without numpy.
"""
from __future__ import annotations

from typing import Callable, Sequence

# An embedding function maps text to a fixed-dim L2-normalised float vector.
EmbeddingFn = Callable[[str], Sequence[float]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return sum(x * x for x in a) ** 0.5


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. Treats zero-norm vectors as dissimilar."""
    na = _norm(a)
    nb = _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# --------------------------------------------------------------------------- #
# Default embedding (fallback when none is injected)
# --------------------------------------------------------------------------- #
_DEFAULT_FN: EmbeddingFn = None  # type: ignore[assignment]


def get_default_embedding() -> EmbeddingFn:
    """Return the process-wide default embedding function.

    Lazily initialised to :class:`~reproductions._common.mocks.MockEmbedding`
    so reproductions work out of the box. Override via
    :func:`set_default_embedding` for production use.
    """
    global _DEFAULT_FN
    if _DEFAULT_FN is None:
        from ..._common.mocks import MockEmbedding  # local import to avoid cycle

        _DEFAULT_FN = MockEmbedding(dim=64)
    return _DEFAULT_FN


def set_default_embedding(fn: EmbeddingFn) -> None:
    """Override the default embedding function."""
    global _DEFAULT_FN
    _DEFAULT_FN = fn


__all__ = [
    "EmbeddingFn",
    "cosine_similarity",
    "euclidean_distance",
    "get_default_embedding",
    "set_default_embedding",
]
