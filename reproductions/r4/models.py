"""R4 model adapters — thin wrappers around QwenRunner (loaded once) that
match R4Pipeline's ``VLMFn`` and ``DecompFn`` signatures.

R4Pipeline expects:

* ``vlm(prompt: str, images: list = None, **kw) -> str``
* ``decomposer(query: str, live_perception: Any) -> dict`` — returns keys
  ``k_sem`` / ``k_spa_centroid`` / ``k_spa_radius`` / ``k_t_min`` / ``k_t_max``

QwenRunner (from ``reproductions/clivis/models.py``) already satisfies the
VLM signature (``vlm(prompt, images=[path]) -> str``) and the LLM
signature (``llm(prompt) -> str``), so we just re-export it.

The decomposer is implemented as a VLM-driven query analyser: given the
question (and optionally the live perception summary), the LLM is asked
to output the retrieval keys as JSON.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

# Re-export QwenRunner so callers don't need to know it lives in clivis/
from reproductions.clivis.models import QwenRunner  # noqa: F401


def make_vlm_decomposer(llm):
    """Build a ``DecompFn`` that uses the LLM to break a question into
    R4's three retrieval axes.

    Returns a callable ``(query: str, live_perception: Any) -> dict`` whose
    output has keys:
    ``k_sem`` (list[str]),
    ``k_spa_centroid`` (tuple | None),
    ``k_spa_radius`` (float | None),
    ``k_t_min`` (float | None),
    ``k_t_max`` (float | None).
    """

    def decomposer(query: str, live_perception: Any) -> Dict[str, Any]:
        prompt = (
            "You are decomposing a question for retrieval from a 4D spatio-temporal "
            "memory of objects observed in a video.\n\n"
            f"Question: {query}\n\n"
            "Output JSON with these keys:\n"
            '  "k_sem": list of 1-3 significant nouns from the question\n'
            '  "k_spa_radius": a spatial search radius in metres (or null)\n'
            '  "k_t_min": earliest observation timestamp in seconds (or null)\n'
            '  "k_t_max": latest observation timestamp in seconds (or null)\n'
            "Do NOT include k_spa_centroid (we don't know the camera-relative "
            "position from the question text alone).\n"
            "Output JSON only, no explanation."
        )
        raw = llm(prompt, max_new_tokens=200)
        parsed = _safe_json_extract(raw) or {}

        k_sem = parsed.get("k_sem") or []
        if isinstance(k_sem, str):
            k_sem = [k_sem]
        k_sem = [str(s).strip().lower() for s in k_sem if s][:3]

        k_radius = parsed.get("k_spa_radius")
        try:
            k_radius = float(k_radius) if k_radius is not None else None
        except (TypeError, ValueError):
            k_radius = None

        k_t_min = parsed.get("k_t_min")
        k_t_max = parsed.get("k_t_max")
        try:
            k_t_min = float(k_t_min) if k_t_min is not None else None
        except (TypeError, ValueError):
            k_t_min = None
        try:
            k_t_max = float(k_t_max) if k_t_max is not None else None
        except (TypeError, ValueError):
            k_t_max = None

        return {
            "k_sem": k_sem,
            "k_spa_centroid": None,
            "k_spa_radius": k_radius,
            "k_t_min": k_t_min,
            "k_t_max": k_t_max,
        }

    return decomposer


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction — same logic as pipeline._safe_json_extract.

    Inlined here so this module has zero cross-pipeline coupling.
    """
    if not text:
        return None
    candidate = text.strip()
    try:
        v = json.loads(candidate)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fence:
        try:
            v = json.loads(fence.group(1))
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            candidate = fence.group(1)
    # Brace-counting
    obj = _extract_balanced(candidate, "{", "}")
    if obj is not None:
        try:
            v = json.loads(obj)
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _extract_balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    pos = 0
    while True:
        start = s.find(open_ch, pos)
        if start < 0:
            return None
        prefix = s[:start]
        if prefix.count("[") > prefix.count("]"):
            pos = start + 1
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        pos = start + 1


__all__ = ["QwenRunner", "make_vlm_decomposer"]
