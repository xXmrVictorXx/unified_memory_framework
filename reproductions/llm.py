"""Unified LLM/VLM interface for all reproductions.

This is the **only** file in the reproductions package that imports
HuggingFace ``transformers`` / ``torch`` / ``qwen_vl_utils``. Keeping the
heavy deps isolated means the rest of the package stays stdlib-only and
remains unit-testable without a GPU.

Public API
----------
* :class:`QwenRunner` — loads a single Qwen2.5-VL / Qwen3-VL model (4-bit)
  and exposes ``.llm(prompt)`` (text-only) and ``.vlm(prompt, images)``
  (video/image-aware) callables.
* :class:`QwenVisionTools` — VideoHV's ``.caption / .detect / .track``
  contract adapter.
* :func:`make_structured_llm` — normalises hypothesis-style LLM output to
  numeric prefixes (VideoHV).
* :func:`make_vlm_decomposer` — decomposes an R4 question into retrieval
  axes (k_sem / k_spa / k_t).
* :func:`make_embedding_fn` — loads BGE-m3 for the SEM-axis embedder.
* :func:`safe_json_extract` / :func:`extract_balanced` — robust JSON
  extraction from noisy LLM output (deduplicated; formerly copy-pasted in
  three files).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Default path on this server; override via QwenRunner(model_path=...).
DEFAULT_MODEL_PATH = "/home/eg4/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"

# Caps for vision-tower activations — keep small enough for shared GPUs.
# The pixel unit = (patch_size * merge_size)^2. Qwen2.5-VL uses 28^2=784,
# Qwen3-VL uses 32^2=1024. We auto-detect from preprocessor_config.json.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 128 * 28 * 28
DEFAULT_VIDEO_FPS = 0.5  # 1 frame every 2 seconds


# --------------------------------------------------------------------------- #
# JSON extraction utilities (deduplicated from pipeline.py / models.py)
# --------------------------------------------------------------------------- #
def extract_balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the first top-level balanced ``open_ch ... close_ch`` substring.

    Skips braces nested inside a ``[...]`` array so that for LLM JSON outputs
    like::

        ```json
        {                              <- we want this outer {...}
          "persons": [{"name": "x"}]   <- not this inner {...} inside the array
        }
        ```
    the outer object is returned, not the first inner dict.

    Returns ``None`` if no balanced substring is found.
    """
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
        in_string = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        pos = start + 1


def safe_json_extract(text: str) -> Optional[Any]:
    """Robust JSON extraction from LLM output.

    Handles:
    * Plain JSON
    * Markdown-fenced JSON (```json ... ``` or ``` ... ```)
    * JSON surrounded by prose / leading thoughts
    * Nested objects / arrays (via brace-counting, not regex)

    Returns the parsed JSON value (typically a dict), or ``None`` on failure.
    """
    if not text:
        return None
    candidate = text.strip()

    # Direct parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fence if present
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            candidate = fence.group(1)

    # Brace-counting: find the first balanced {...} substring
    balanced = extract_balanced(candidate, "{", "}")
    if balanced is not None:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
    return None


# --------------------------------------------------------------------------- #
# Model loading helpers
# --------------------------------------------------------------------------- #
def _detect_model_class(model_path: str):
    """Read ``config.json`` and return the appropriate transformers model class."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path) as f:
        config = json.load(f)
    arch = config.get("architectures", [""])[0]

    if arch == "Qwen3VLForConditionalGeneration":
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif arch == "Qwen2_5_VLForConditionalGeneration":
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        # Fallback: try auto-detection
        from transformers import AutoModelForImageTextToText as Cls  # type: ignore
    return Cls


def _detect_pixel_unit(model_path: str) -> int:
    """Return ``patch_size * merge_size`` from preprocessor_config.json.

    Falls back to 28 (Qwen2.5-VL default) if the file is missing or
    doesn't contain the fields.
    """
    pp_path = Path(model_path) / "preprocessor_config.json"
    if pp_path.exists():
        with open(pp_path) as f:
            pp = json.load(f)
        patch = pp.get("patch_size", 14)
        merge = pp.get("merge_size", 2)
        return patch * merge
    return 28


# --------------------------------------------------------------------------- #
# QwenRunner — the single shared model loader
# --------------------------------------------------------------------------- #
class QwenRunner:
    """Loads Qwen2.5-VL / Qwen3-VL once, exposes ``llm`` and ``vlm`` callables.

    Both callables match the signatures expected by the reproduction
    pipelines:

    * ``llm(prompt: str) -> str``
    * ``vlm(prompt: str, images: Optional[List[Any]] = None) -> str``
      where ``images`` is ``[video_path]`` or ``None``.

    The model is loaded in 4-bit quantization (bitsandbytes) so it fits on
    a shared multi-GPU server alongside other workloads (~6 GB total).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        video_fps: float = DEFAULT_VIDEO_FPS,
        max_memory_per_gpu: str = "7GB",
        cpu_offload: str = "32GB",
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
        )

        self._video_fps = float(video_fps)

        # Auto-detect pixel unit from the model's preprocessor config.
        pixel_unit = _detect_pixel_unit(model_path)
        if min_pixels is None:
            min_pixels = 256 * pixel_unit * pixel_unit
        if max_pixels is None:
            max_pixels = 128 * pixel_unit * pixel_unit
        self._max_pixels = int(max_pixels)

        model_cls = _detect_model_class(model_path)
        self._arch_name = model_cls.__name__

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        max_memory = {
            i: max_memory_per_gpu for i in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = cpu_offload

        t0 = time.time()
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._device = self.model.device
        load_dt = time.time() - t0
        per_gpu = [
            f"{torch.cuda.memory_allocated(i) / 1e9:.1f}G"
            for i in range(torch.cuda.device_count())
        ]
        print(f"[QwenRunner] {self._arch_name} from {model_path}")
        print(f"[QwenRunner] loaded in {load_dt:.1f}s "
              f"across {torch.cuda.device_count()} GPUs ({per_gpu}), "
              f"pixel_unit={pixel_unit}")

    # ------------------------------------------------------------------ #
    # LLM mode: text-only prompt -> text response
    # ------------------------------------------------------------------ #
    def llm(self, prompt: str, max_new_tokens: int = 1024) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()

    # ------------------------------------------------------------------ #
    # VLM mode: prompt + optional video file -> text response
    # ------------------------------------------------------------------ #
    def vlm(
        self,
        prompt: str,
        images: Optional[List[Any]] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """``images`` follows the CLiViS pipeline convention: a list whose
        first non-None element is a path (or file:// URL) to a video segment
        OR a single-frame image (``.jpg`` / ``.png`` / ``.jpeg``).

        The file type is detected from the extension: image files are sent
        as ``{"type": "image"}`` while everything else (``.mp4`` etc.) is
        sent as ``{"type": "video"}`` with the configured ``fps``.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        media_path: Optional[str] = None
        if images:
            for item in images:
                if item:
                    media_path = str(item)
                    break

        content: List[dict] = []
        if media_path:
            url = media_path if media_path.startswith("file://") else f"file://{media_path}"
            stripped = media_path.split("://")[-1].lower()
            if stripped.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                content.append({"type": "image", "image": url})
            else:
                content.append({
                    "type": "video",
                    "video": url,
                    "fps": self._video_fps,
                    "max_pixels": self._max_pixels,
                })
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        # Qwen3-VL's new video processor expects scalar fps, but
        # qwen_vl_utils returns a list (one entry per video). The frames
        # are already sampled by process_vision_info, so the processor
        # doesn't need fps again — pop it to avoid TypeError.
        if video_kwargs and "Qwen3" in self._arch_name:
            video_kwargs.pop("fps", None)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )[0].strip()


# --------------------------------------------------------------------------- #
# LLM output adapters
# --------------------------------------------------------------------------- #
# Matches: 0:, A:, <0>:, <A>:, 0., A., <0>., etc. at the start of a line.
_PREFIX_RE = re.compile(r"^<?([A-Za-z0-9])>?\s*[.:)]\s*", re.MULTILINE)


def make_structured_llm(runner: QwenRunner):
    """Wrap ``runner.llm`` so hypothesis-style output is parseable.

    Qwen2.5-VL-7B has two quirks that break the VideoHV pipeline's parser:

    * It wraps indices in angle brackets from the prompt template:
      ``<0>: text`` instead of ``0: text``.
    * It uses letter prefixes (``A:`` / ``B:``) for multiple-choice options.

    This wrapper strips angle brackets and converts letters to numbers so
    ``_parse_hypotheses`` sees clean ``0: text`` / ``1: text`` lines.
    """

    def _normalize(text: str) -> str:
        def _repl(m: "re.Match[str]") -> str:
            ch = m.group(1).upper()
            if "A" <= ch <= "Z":
                return f"{ord(ch) - ord('A')}: "
            return f"{ch}: "

        return _PREFIX_RE.sub(_repl, text)

    def llm(prompt: str, max_new_tokens: int = 512) -> str:
        resp = runner.llm(prompt, max_new_tokens=max_new_tokens)
        normalized = _normalize(resp)
        if os.environ.get("VIDEOHV_DEBUG"):
            print(f"[llm-debug] prompt={prompt[:80]!r}...")
            print(f"[llm-debug] raw={resp[:200]!r}")
            print(f"[llm-debug] normalized={normalized[:200]!r}")
        return normalized

    return llm


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
        parsed = safe_json_extract(raw) or {}
        if not isinstance(parsed, dict):
            parsed = {}

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


# --------------------------------------------------------------------------- #
# VLM adapters
# --------------------------------------------------------------------------- #
class QwenVisionTools:
    """Implements the ``.caption / .detect / .track`` contract expected by
    :class:`reproductions.videohv.pipeline.VideoHVPipeline` when
    ``vision_tools`` is wired.

    The pipeline's ``_verify_with_tools`` passes a list of ``entry_id``
    strings (e.g. ``["clip-0", "clip-1"]``) and a ``question`` (the
    distinguishing clue). This class resolves each id to the corresponding
    segment file via ``clip_media_map`` and asks the VLM to describe what
    is relevant to the clue.

    ``detect`` and ``track`` are stubbed to match the original VideoHV-Agent
    behaviour (defined but unused at inference time).
    """

    def __init__(
        self,
        runner: QwenRunner,
        clip_media_map: Optional[Dict[str, str]] = None,
        max_new_tokens_per_clip: int = 80,
    ) -> None:
        self._runner = runner
        self._clip_media_map: Dict[str, str] = dict(clip_media_map or {})
        self._max_tokens = int(max_new_tokens_per_clip)

    @property
    def clip_media_map(self) -> Dict[str, str]:
        return self._clip_media_map

    def register_clip(self, entry_id: str, media_path: str) -> None:
        self._clip_media_map[entry_id] = media_path

    # ------------------------------------------------------------------ #
    # Vision tool contract
    # ------------------------------------------------------------------ #
    def caption(
        self,
        frames: List[str],
        question: str = "",
    ) -> str:
        """Caption the requested clips with respect to ``question``.

        Each requested clip id that has a registered media file is sent to
        the VLM separately; the per-clip captions are concatenated so the
        downstream LLM verdict sees all evidence at once.
        """
        parts: List[str] = []
        for clip_id in frames:
            media = self._clip_media_map.get(clip_id)
            if not media:
                continue
            prompt = (
                f"Look at this video segment and describe what you see that "
                f"is relevant to: \"{question}\". Be concise (1-2 sentences)."
            )
            desc = self._runner.vlm(
                prompt, images=[media], max_new_tokens=self._max_tokens
            )
            parts.append(f"[{clip_id}] {desc}")
        if not parts:
            return "(no visual evidence available for requested clips)"
        return " ".join(parts)

    def detect(
        self,
        frames: List[str],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """Stub — original VideoHV-Agent defines but does not use detection."""
        return []

    def track(
        self,
        frames: List[str],
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """Stub — original VideoHV-Agent defines but does not use tracking."""
        return []


# --------------------------------------------------------------------------- #
# Embedding model (BGE-m3)
# --------------------------------------------------------------------------- #
def make_embedding_fn():
    """Build the SEM-axis embedder using a real sentence-transformer model.

    Uses BGE-m3 (1024-dim, multilingual, stored locally at
    ``~/.cache/modelscope/hub/models/BAAI/bge-m3``). Loaded once and cached
    on the first call.

    Falls back to a hash-based pseudo-embedder only if sentence-transformers
    or the model are unavailable (e.g. on a CI machine without the model).
    """
    model_path = "/home/eg4/.cache/modelscope/hub/models/BAAI/bge-m3"
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_path)
        # Warm up so the first real call doesn't pay the model-load cost.
        st.encode(["warmup"])
        print(f"[embedder] BGE-m3 loaded from {model_path} (dim={st.get_sentence_embedding_dimension()})")

        def embed(text: str):
            return st.encode(text, normalize_embeddings=True).tolist()

        return embed
    except (ImportError, OSError, Exception) as e:
        print(f"[embedder] WARNING: BGE-m3 unavailable ({e!r}); falling back to hash")
        import hashlib

        def embed(text: str):
            h = hashlib.sha512(text.encode("utf-8")).digest()
            dim = 64
            out = []
            i = 0
            while len(out) < dim:
                out.append((h[i % len(h)] / 127.5) - 1.0)
                i += 1
            norm = sum(x * x for x in out) ** 0.5 or 1.0
            return [x / norm for x in out]

        return embed


__all__ = [
    "QwenRunner",
    "QwenVisionTools",
    "make_structured_llm",
    "make_vlm_decomposer",
    "make_embedding_fn",
    "safe_json_extract",
    "extract_balanced",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_VIDEO_FPS",
]
