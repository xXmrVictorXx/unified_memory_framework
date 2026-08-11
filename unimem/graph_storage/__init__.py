"""``unimem.graph_storage`` — pluggable graph-database backends.

Public API:

* :class:`GraphStorage` — the ABC every backend implements.
* :class:`InMemoryGraphStorage` — pure-stdlib fallback backend.
* :func:`create_graph_storage` — factory driven by a config dict.
* Time-index helpers in :mod:`unimem.graph_storage.time_index`.

The Neo4j backend lives in :mod:`unimem.graph_storage.neo4j_backend` and is
imported lazily — only required when ``backend == "neo4j"``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import GraphStorage
from .memory_backend import InMemoryGraphStorage


def create_graph_storage(config: Optional[Dict[str, Any]] = None) -> GraphStorage:
    """Build a :class:`GraphStorage` from a config dict.

    Recognised keys:

    * ``backend``: ``"memory"`` (default) | ``"neo4j"``
    * Neo4j-specific: ``uri``, ``user``, ``password``, ``database``
    """
    cfg = dict(config or {})
    backend = str(cfg.get("backend", "memory")).lower()
    if backend == "memory":
        return InMemoryGraphStorage()
    if backend == "neo4j":
        try:
            from .neo4j_backend import Neo4jGraphStorage  # lazy import
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "Neo4j backend requires the `neo4j` package; "
                "install via `pip install neo4j`"
            ) from exc
        return Neo4jGraphStorage(
            uri=cfg.get("uri"),
            user=cfg.get("user"),
            password=cfg.get("password"),
            database=cfg.get("database"),
        )
    raise ValueError(f"Unknown graph storage backend: {backend!r}")


__all__ = [
    "GraphStorage",
    "InMemoryGraphStorage",
    "create_graph_storage",
]
