"""Ivy-xDetector (IVY-FAKE): a Qwen2.5-VL-3B fine-tune for explainable AIGC image/video
detection (Pi3AI / AI-Safeguard; arXiv:2506.00979). Released checkpoint: AI-Safeguard/Ivy-Fake
on HuggingFace (3B). The model reasons in <think> and emits a <conclusion>real/fake</conclusion>
verdict; the paper samples video at 1 fps (max 6 frames per clip).

The weights carry no declared license; the Qwen2.5-VL-3B base is under the Qwen Research
License (non-commercial).
"""
from __future__ import annotations

from vidaudit.detectors.base import DetectorSpec
from vidaudit.detectors.mllm import MLLMDetector
from vidaudit.detectors.registry import register

# Upstream prompt uses em-dashes; written here with commas (no-em-dash style), tags unchanged.
_SYS = ("You are an AI-generated content detector. Classify the media as real or fake. "
        "Provide reasoning inside <think>...</think> tags. End with exactly one word, real "
        "or fake, wrapped in <conclusion>...</conclusion>.")


@register("ivy-xdetector")
class IvyXDetector(MLLMDetector):
    spec = DetectorSpec(
        name="Ivy-xDetector", backbone="Qwen2.5-VL-3B-Instruct", family="mllm",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="https://huggingface.co/AI-Safeguard/Ivy-Fake",
        license="weights license undeclared; Qwen2.5-VL-3B base under the Qwen Research "
                "License (non-commercial). Obtain from the original source.",
        paper="IVY-FAKE (Pi3AI / AI-Safeguard; arXiv:2506.00979)",
        notes="Qwen2.5-VL-3B fine-tune; <conclusion>real/fake</conclusion> verdict; paper "
              "samples video at 1 fps (max 6 frames).",
    )
    model_id = "AI-Safeguard/Ivy-Fake"
    hub = "hf"
    system_prompt = _SYS
    user_prompt = "Is this media real or fake?"
    answer_tags = ("<conclusion>", "</conclusion>")
