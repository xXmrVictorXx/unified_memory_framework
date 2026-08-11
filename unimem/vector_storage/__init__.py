"""``unimem.vector_storage`` — on-demand vector database backends.

Used by methods that need vector retrieval (R4's semantic embedding
similarity, video clip embeddings, ...). The framework does NOT attach a
VectorStorage to every :class:`~unimem.core.module.MemoryModule`; it is an
explicit dependency instantiated where needed.

Public API:

* :class:`VectorStorage` — the ABC every backend implements.
* :class:`InMemoryVectorStorage` — pure-stdlib fallback (brute force).
* :func:`create_vector_storage` — factory driven by a config dict.

The Qdrant backend lives in :mod:`unimem.vector_storage.qdrant_backend`
and is imported lazily.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import VectorStorage
from .memory_backend import InMemoryVectorStorage


def create_vector_storage(config: Optional[Dict[str, Any]] = None) -> VectorStorage:
    """Build a :class:`VectorStorage` from a config dict.

    Recognised keys:

    * ``backend``: ``"memory"`` (default) | ``"qdrant"``
    * Qdrant-specific: ``host``, ``port``, ``api_key``, ``collection_prefix``
    """
    cfg = dict(config or {})
    backend = str(cfg.get("backend", "memory")).lower()
    if backend == "memory":
        return InMemoryVectorStorage()
    if backend == "qdrant":
        try:
            from .qdrant_backend import QdrantVectorStorage  # lazy import
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "Qdrant backend requires the `qdrant-client` package; "
                "install via `pip install qdrant-client`"
            ) from exc
        return QdrantVectorStorage(
            host=cfg.get("host"),
            port=cfg.get("port"),
            api_key=cfg.get("api_key"),
            collection_prefix=cfg.get("collection_prefix"),
        )
    raise ValueError(f"Unknown vector storage backend: {backend!r}")


__all__ = [
    "VectorStorage",
    "InMemoryVectorStorage",
    "create_vector_storage",
]
