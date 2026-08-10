"""Shared utilities for the reproductions package.

Reproductions of real EQA/VLA methods need LLM, VLM, and embedding calls.
To keep tests runnable without GPU/API keys, every reproduction accepts
*injectable callables* for these dependencies. This module provides simple
deterministic mocks; users can swap in real OpenAI/HF/local-model clients
at runtime.
"""
from __future__ import annotations

from .mocks import (
    MockEmbedding,
    MockLLM,
    MockVLM,
    MockVisionTools,
    RecordedCalls,
)

__all__ = [
    "MockLLM",
    "MockVLM",
    "MockEmbedding",
    "MockVisionTools",
    "RecordedCalls",
]
