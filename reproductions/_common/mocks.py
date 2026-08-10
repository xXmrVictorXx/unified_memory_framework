"""Deterministic mock LLM / VLM / embedding / vision-tool clients.

These are *not* meant to produce realistic outputs — they exist so the
reproduction pipelines can run end-to-end in unit tests without GPU, API
keys, or model weights. Each mock records its calls so tests can assert on
the interaction trace.

Swap-in points for real models:

* LLM: any callable ``(prompt: str, **kw) -> str``
* VLM: any callable ``(prompt: str, images: list, **kw) -> str``
* Embedding: any callable ``(text: str) -> Sequence[float]``
* Vision tools (caption/detect/track): any callable matching the signature

Reproductions accept these as constructor / function arguments, so wiring
up a real client never requires touching pipeline code.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Recording mixin
# --------------------------------------------------------------------------- #
class RecordedCalls:
    """Mixin that records every call to a method named in ``_recorded``."""

    _recorded: Tuple[str, ...] = ()

    def __init__(self) -> None:
        self.calls: Dict[str, List[Tuple[Tuple[Any, ...], Dict[str, Any]]]] = {}
        for name in self._recorded:
            self.calls[name] = []

    def _record(self, name: str, args: tuple, kwargs: dict) -> None:
        self.calls.setdefault(name, []).append((args, kwargs))

    @property
    def call_count(self) -> int:
        return sum(len(v) for v in self.calls.values())


# --------------------------------------------------------------------------- #
# Mock LLM
# --------------------------------------------------------------------------- #
class MockLLM(RecordedCalls):
    """Deterministic LLM stub.

    Behaviour:

    * If ``pattern_map`` matches, the corresponding canned reply wins.
    * Else if ``responses`` queue is non-empty, pop the next one.
    * Else return ``default_response`` verbatim.

    Special prompts can be routed via ``pattern_map``: regex patterns that
    select a canned reply. Used by pipelines that expect structured output
    (JSON, hypotheses, etc.).
    """

    _recorded = ("__call__",)

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        pattern_map: Optional[Sequence[Tuple[str, str]]] = None,
        default_response: str = "[mock-llm]",
    ) -> None:
        super().__init__()
        self._queue: List[str] = list(responses) if responses else []
        self._patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(pat), resp) for pat, resp in (pattern_map or [])
        ]
        self.default_response = default_response

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        self._record("__call__", (prompt,), kwargs)
        for pat, resp in self._patterns:
            if pat.search(prompt):
                return resp
        if self._queue:
            return self._queue.pop(0)
        return self.default_response


# --------------------------------------------------------------------------- #
# Mock VLM
# --------------------------------------------------------------------------- #
class MockVLM(RecordedCalls):
    """Deterministic VLM stub. Mirrors :class:`MockLLM` but accepts images."""

    _recorded = ("__call__",)

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        pattern_map: Optional[Sequence[Tuple[str, str]]] = None,
        default_response: str = "[mock-vlm]",
    ) -> None:
        super().__init__()
        self._queue: List[str] = list(responses) if responses else []
        self._patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(pat), resp) for pat, resp in (pattern_map or [])
        ]
        self.default_response = default_response

    def __call__(self, prompt: str, images: Optional[Sequence[Any]] = None, **kwargs: Any) -> str:
        self._record("__call__", (prompt,), {**kwargs, "n_images": len(images) if images else 0})
        for pat, resp in self._patterns:
            if pat.search(prompt):
                return resp
        if self._queue:
            return self._queue.pop(0)
        return self.default_response


# --------------------------------------------------------------------------- #
# Mock embedding
# --------------------------------------------------------------------------- #
class MockEmbedding(RecordedCalls):
    """Maps text to a fixed-dimensional float vector deterministically.

    Useful for testing R4-style cosine-similarity retrieval without a real
    embedding model. Output dimension is configurable (default 16 — small
    enough to be cheap, large enough to avoid collisions in tests).
    """

    _recorded = ("__call__",)

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = int(dim)

    def __call__(self, text: str) -> List[float]:
        self._record("__call__", (text,), {})
        # Hash-based: stable across runs, well-distributed across texts.
        h = hashlib.sha512(text.encode("utf-8")).digest()
        # Stretch hash to dim length.
        out = []
        i = 0
        while len(out) < self.dim:
            byte = h[i % len(h)]
            out.append((byte / 127.5) - 1.0)  # in [-1, 1]
            i += 1
        # L2-normalise so cosine sim is just dot product.
        norm = sum(x * x for x in out) ** 0.5 or 1.0
        return [x / norm for x in out]


# --------------------------------------------------------------------------- #
# Mock vision tools (for VideoHV-Agent)
# --------------------------------------------------------------------------- #
class MockVisionTools(RecordedCalls):
    """Replaces VideoHV-Agent's caption/detect/track tool calls.

    Each tool takes frames and returns a canned description. The mock keeps
    them as no-ops with predictable output so tests can verify tool *use*
    without doing real vision work.
    """

    _recorded = ("caption", "detect", "track")

    def __init__(
        self,
        captioner: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__()
        self._captioner = captioner or (lambda _: "[mock-caption]")

    def caption(self, frames: Sequence[Any], question: Optional[str] = None) -> str:
        self._record("caption", (list(frames),), {"question": question})
        return self._captioner(question or "default")

    def detect(self, frames: Sequence[Any], query: str) -> List[Dict[str, Any]]:
        self._record("detect", (list(frames),), {"query": query})
        return [{"label": query, "score": 0.9, "frame": 0}]

    def track(self, frames: Sequence[Any], query: str) -> List[Dict[str, Any]]:
        self._record("track", (list(frames),), {"query": query})
        return [{"label": query, "frame_range": [0, len(frames) if frames else 0]}]


__all__ = [
    "RecordedCalls",
    "MockLLM",
    "MockVLM",
    "MockEmbedding",
    "MockVisionTools",
]
