"""Real VLM/LLM runner for CLiViS — wraps a single Qwen2.5-VL-7B-Instruct
model as both the LLM (text-only) and VLM (video-aware) callable expected by
:class:`reproductions.clivis.pipeline.CLiViSPipeline`.

This file is the only place that touches HuggingFace transformers / CUDA.
Keeping it isolated means the rest of the package stays stdlib-only and
remains unit-testable without a GPU.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

# Default path on this server; override via QwenRunner(model_path=...).
DEFAULT_MODEL_PATH = "/mnt/my_hub2/models/Qwen2.5-VL-7B-Instruct"

# Caps for vision-tower activations — keep small enough for shared GPUs.
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 128 * 28 * 28
DEFAULT_VIDEO_FPS = 0.5  # 1 frame every 2 seconds


class QwenRunner:
    """Loads Qwen2.5-VL-7B once, exposes ``llm`` and ``vlm`` callables.

    Both callables match the signatures expected by
    :class:`reproductions.clivis.pipeline.CLiViSPipeline`:

    * ``llm(prompt: str) -> str``
    * ``vlm(prompt: str, images: Optional[List[Any]] = None) -> str``
      where ``images`` is ``[video_path]`` or ``None``.

    The model is loaded in 4-bit quantization (bitsandbytes) so it fits on
    a shared 4x A100 server alongside other workloads (~6 GB total).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        video_fps: float = DEFAULT_VIDEO_FPS,
        max_memory_per_gpu: str = "7GB",
        cpu_offload: str = "32GB",
    ) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        self._video_fps = float(video_fps)
        self._max_pixels = int(max_pixels)

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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        print(f"[QwenRunner] {model_path} loaded in {load_dt:.1f}s "
              f"across {torch.cuda.device_count()} GPUs ({per_gpu})")

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


__all__ = ["QwenRunner", "DEFAULT_MODEL_PATH"]
