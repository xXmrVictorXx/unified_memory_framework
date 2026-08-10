"""R4 reproduction — 4D spatio-temporal knowledge database.

Reproduces the memory architecture from
*"R4: Retrieval-Augmented Reasoning for Vision-Language Models in 4D
Spatio-Temporal Space"* (arXiv 2512.15940).

Paper spec recap (Section 3.1):

    D = { M, {O_j}_{j=1..N} }
    O_j^t = { SEM, SPA, TEM }

* ``SEM`` — natural-language description + vector embedding (semantic index)
* ``SPA`` — 3D centroid + extent in world coords, linked into SLAM map M (spatial index)
* ``TEM`` — observation timestamps in a columnar timeseries DB (temporal index)
* ``M``  — MapAnything-style SLAM map (only its "special points" interface used here)

We deliberately mirror unimem's own `MultiAxisIndex` philosophy here — R4 was
one of the key inspirations for that data structure. Each ObjectRecord is a
glorified `MemoryEntry` with the three axes already populated; the
`R4KnowledgeDatabase` is essentially a `MemoryModule` whose read() does the
three-axis retrieval that R4 calls "retrieval-augmented 4D reasoning".
"""
from __future__ import annotations

from .memory.embedding import EmbeddingFn, get_default_embedding
from .memory.knowledge_db import R4KnowledgeDatabase
from .memory.object_record import ObjectRecord, SemanticAxis, SpatialAxis, TemporalAxis
from .pipeline import R4Pipeline, R4PipelineResult

__all__ = [
    "ObjectRecord",
    "SemanticAxis",
    "SpatialAxis",
    "TemporalAxis",
    "R4KnowledgeDatabase",
    "EmbeddingFn",
    "get_default_embedding",
    "R4Pipeline",
    "R4PipelineResult",
]
