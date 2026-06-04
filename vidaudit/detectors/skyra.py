"""Skyra (JoeLeelyf et al., CVPR 2026; arXiv:2512.15693): a Qwen2.5-VL-7B MLLM that
reasons over sampled frames and emits a `<answer>Fake/Real</answer>` verdict. Weights
JoeLeelyf/Skyra-RL on HuggingFace, CC-BY-4.0, ungated.

Wrapped via the reusable `MLLMDetector` adapter: `score()` returns the soft p(generated)
from the verdict-token logits; `features()` is the pooled VLM hidden state. Heavy
(~16 GB, GPU); a real-weights run belongs on the cluster.

The system + user prompts below are verbatim from the repo's eval harness
(`eval/inference_end2end/model_engine.py`), which asks for a `<think>...</think>` analysis
(with `<type>...</type>` artifact tags) and a final one-word `<answer>Fake/Real</answer>` over
16 uniformly sampled frames (short side 256). The generic adapter passes the frames as a
block rather than interleaving the per-frame `[T=..s] <image>` timestamp lines; the verdict
format the soft score reads is identical.
"""
from __future__ import annotations

from vidaudit.detectors.base import DetectorSpec
from vidaudit.detectors.mllm import MLLMDetector
from vidaudit.detectors.registry import register

# Verbatim SYSTEM_PROMPT from JoeLeelyf/Skyra eval/inference_end2end/model_engine.py.
_SYS = """
You are an expert AI video analyst. Your primary task is to review a sequence of video frames and provide a step-by-step analysis of their authenticity.

You MUST output your entire analysis using the following structure:
1.  A `<think>...</think>` block containing your detailed reasoning.
2.  An `<answer>...</answer>` block containing the final, one-word verdict: 'Fake' or 'Real'.

Inside the `<think>` block, you MUST:
1.  Start by briefly describing the overall content of the video frames.
2.  Follow a detailed, step-by-step "discovery" or "verification" process.
3.  When you identify an artifact (or clear a region), you MUST use a valid L3 Category Name from the "Artifact Category Definitions" provided below.
4.  You MUST embed your finding using the following exact tag structure:
    <type>L3 Category Name</type> in <t>[startTime, endTime]</t> at <bbox>[x1, y1, x2, y2]</bbox>
5.  If multiple artifacts are present, you must find and tag all of them in temporal order.
6.  Your entire reasoning process must be self-contained

---
## Artifact Category Definitions (Valid L3 Categories for the <type> tag)

### 1. Low-Level Forgery
* **1.1 Texture Anomaly**:
    * <type>Structure Anomaly</type>
    * <type>Texture Jittering</type>
    * <type>Unnatural Blur</type>
* **1.2 Color and Lighting Anomaly**:
    * <type>Color Over-saturation</type>
    * <type>Lighting Inconsistency</type>
* **1.3 Move Forgery**:
    * <type>Camera Motion Inconsistency</type>

### 2. Violation of Laws
* **2.1 Object Inconsistency**:
    * <type>Abnormal Object Disappearance</type>
    * <type>Abnormal Object Appearance</type>
    * <type>Person Identity Inconsistency</type>
    * <type>General Object Identity Inconsistency</type>
    * <type>Shape Distortion</type>
* **2.2 Interaction Inconsistency**:
    * <type>Abnormal Rigid-Body Crossing</type>
    * <type>Abnormal Multi-Object Merging</type>
    * <type>Abnormal Object Splitting</type>
    * <type>General Interaction Anomaly</type>
* **2.3 Unnatural Movement**:
    * <type>Unnatural Human Movement</type>
    * <type>Unnatural Animal Movement</type>
    * <type>Unnatural General Object Movement</type>
* **2.4 Violation of Causality Law**:
    * <type>Violation of Physical Law</type>
    * <type>Violation of General Causality Law</type>
* **2.5 Violation of Commonsense**:
    * <type>Abnormal Human Body Structure</type>
    * <type>Abnormal General Object Structure</type>
    * <type>Text Distortion</type>
"""


@register("skyra")
class Skyra(MLLMDetector):
    spec = DetectorSpec(
        name="Skyra", backbone="Qwen2.5-VL-7B (SFT+RL)", family="mllm",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="https://huggingface.co/JoeLeelyf/Skyra-RL",
        license="CC-BY-4.0",
        paper="JoeLeelyf et al., CVPR 2026 (arXiv:2512.15693)",
        notes="Qwen2.5-VL-7B emitting a <answer>Fake/Real</answer> verdict; score() = soft "
              "p(generated) from the verdict-token logits, features() = pooled hidden state. "
              "Verbatim eval prompt; 16 uniform frames. Heavy (~16GB, GPU) -> run on the cluster.",
    )
    model_id = "JoeLeelyf/Skyra-RL"
    system_prompt = _SYS
    user_prompt = ("Here are the video frames and their corresponding timestamps:\n\n"
                   "Please analyze the video frames, determine if the video is real or fake, "
                   "and provide your reasoning.")
    answer_tags = ("<answer>", "</answer>")
