"""Type aliases used across the unimem framework.

All aliases are intentionally permissive (``Any``-flavoured) so that downstream
EQA/VLA systems can plug in arbitrary structured payloads without fighting the
type system. Concrete modules are free to narrow these in their own code.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

# A semantic key is a free-form token (word, lemma, phrase, tag, ...).
SemanticKey = str

# A spatial key is a numeric vector (2D pixel, 3D metric, 6-DoF pose, ...).
# Stored as a tuple so it is hashable and can live in sets/dicts.
SpatialKey = Tuple[float, ...]

# A temporal key is a single scalar timestamp (seconds since epoch / episode).
TemporalKey = float

# Arbitrary structured data attached to an entry (features, embeddings,
# bounding boxes, image hashes, ...). The framework never inspects it.
Payload = Any

# Free-form metadata bag (provenance, confidence, source modality, ...).
Metadata = Dict[str, Any]

# Convenience: the union of all three key kinds for terse signatures.
AnyKey = Union[SemanticKey, SpatialKey, TemporalKey]

__all__ = [
    "SemanticKey",
    "SpatialKey",
    "TemporalKey",
    "Payload",
    "Metadata",
    "AnyKey",
    "List",
    "Dict",
    "Tuple",
]
