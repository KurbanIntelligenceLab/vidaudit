"""Skyra (JoeLeelyf et al., CVPR 2026; arXiv:2512.15693): a Qwen2.5-VL-7B MLLM that
reasons over sampled frames and emits a `<answer>real/fake</answer>` verdict. Weights
JoeLeelyf/Skyra-RL on HuggingFace, CC-BY-4.0, ungated.

Wrapped via the reusable `MLLMDetector` adapter: `score()` returns the soft p(generated)
from the verdict-token logits; `features()` is the pooled VLM hidden state. Heavy
(~16 GB, GPU); a real-weights run belongs on the cluster. The system/user prompt below
follows the repo's eval harness in spirit; confirm the exact wording in
`eval/inference_end2end/` before a published run (override via the prompt fields).
"""
from __future__ import annotations

from vidaudit.detectors.base import DetectorSpec
from vidaudit.detectors.mllm import MLLMDetector
from vidaudit.detectors.registry import register

_SYS = ("You are an expert AI video analyst. Inspect the frames for artifacts of AI "
        "generation: structural anomalies, texture jittering, unnatural blur, temporal "
        "inconsistency, and physically implausible motion. Reason step by step, then give "
        "a final verdict inside <answer> </answer> tags as exactly 'real' or 'fake'.")


@register("skyra")
class Skyra(MLLMDetector):
    spec = DetectorSpec(
        name="Skyra", backbone="Qwen2.5-VL-7B (SFT+RL)", family="mllm",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="https://huggingface.co/JoeLeelyf/Skyra-RL",
        license="CC-BY-4.0",
        paper="JoeLeelyf et al., CVPR 2026 (arXiv:2512.15693)",
        notes="Qwen2.5-VL-7B emitting a <answer>real/fake</answer> verdict; score() = soft "
              "p(generated) from the verdict-token logits, features() = pooled hidden state. "
              "Heavy (~16GB, GPU) -> run on the cluster; confirm the exact eval prompt.",
    )
    model_id = "JoeLeelyf/Skyra-RL"
    system_prompt = _SYS
    user_prompt = "Is this video real or fake? Reason step by step, then answer in <answer></answer>."
    answer_tags = ("<answer>", "</answer>")
