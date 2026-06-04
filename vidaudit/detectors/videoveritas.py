"""VideoVeritas (EricTan7 et al., ICML 2026; arXiv:2602.08828): a Qwen3-VL-8B MLLM that
reasons over frames and emits a `<answer>real/fake</answer>` verdict. Apache-2.0; weights
on ModelScope (EricTanh/VideoVeritas), so `pip install modelscope` is needed to resolve
them.

A config variant of the reusable `MLLMDetector` adapter (`hub="modelscope"`): `score()`
returns the soft p(generated) from the verdict-token logits, `features()` pools the VLM
hidden state. Heavy (~18 GB, GPU); a real-weights run belongs on the cluster.
"""
from __future__ import annotations

from vidaudit.detectors.base import DetectorSpec
from vidaudit.detectors.mllm import MLLMDetector
from vidaudit.detectors.registry import register


@register("videoveritas")
class VideoVeritas(MLLMDetector):
    spec = DetectorSpec(
        name="VideoVeritas", backbone="Qwen3-VL-8B", family="mllm",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="https://www.modelscope.cn/models/EricTanh/VideoVeritas",
        license="Apache-2.0",
        paper="EricTan7 et al., ICML 2026 (arXiv:2602.08828)",
        notes="Qwen3-VL-8B emitting a <answer>real/fake</answer> verdict; ModelScope-hosted "
              "(pip install modelscope); score() = soft p(generated) from the verdict-token "
              "logits. Heavy (~18GB, GPU) -> run on the cluster.",
    )
    model_id = "EricTanh/VideoVeritas"
    hub = "modelscope"
    # Verbatim eval prompt from EricTan7/VideoVeritas self_scripts/infer/infer_vllm.py
    # (the curly apostrophe and the newline after the first sentence are in the source).
    system_prompt = ("You are an expert video analyst.\n"
                     "Please think about the question as if you were a human pondering deeply. "
                     "It’s encouraged to include self-reflection or verification in the "
                     "reasoning process. Then, give a final verdict within <answer> </answer> tags.")
    user_prompt = "Is this video real or fake?"
    answer_tags = ("<answer>", "</answer>")
