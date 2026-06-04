"""Ivy-xDetector (IVY-FAKE; Pi3AI / AI-Safeguard, arXiv:2506.00979): an explainable
AIGC image/video detector that is a Qwen2.5-VL fine-tune. It reasons inside
`<think> </think>` and emits a `<conclusion>real/fake</conclusion>` verdict. The released
public checkpoint is the 3B variant `AI-Safeguard/Ivy-Fake` on HuggingFace (ungated,
~7.5 GB). The paper initializes from an Ivy-VL-LLaVA backbone at a sub-4B scale; the
public weights are concretely a Qwen2.5-VL-3B-Instruct fine-tune (not 7B).

Wrapped via the reusable `MLLMDetector` adapter (`hub="hf"`): `score()` returns the soft
p(generated) from the verdict-token logits at the `<conclusion>` step; `features()` pools
the VLM hidden state. Heavy (~7.5 GB, GPU); a real-weights run belongs on the cluster.

Licensing: the Ivy-Fake weights carry NO declared license on the HuggingFace repo, and the
Qwen2.5-VL-3B base is under the Qwen Research License (non-commercial). Treat as
non-commercial / research, obtain from the original source, and comply with those terms
(see the License section of the README). The paper samples video at 1 fps; this wrapper
samples a fixed number of frames uniformly (set `n_frames`), so pin it for a published run.
"""
from __future__ import annotations

from vidaudit.detectors.base import DetectorSpec
from vidaudit.detectors.mllm import MLLMDetector
from vidaudit.detectors.registry import register

_SYS = ("You are an AI-generated content detector. Classify the media as real or fake. "
        "Provide your reasoning inside <think> </think> tags, then end with exactly one "
        "word, either real or fake, wrapped in <conclusion> </conclusion> tags.")


@register("ivy-xdetector")
class IvyXDetector(MLLMDetector):
    spec = DetectorSpec(
        name="Ivy-xDetector", backbone="Qwen2.5-VL-3B-Instruct", family="mllm",
        published_weights=True, trainable=False, needs_gpu=True,
        weights_url="https://huggingface.co/AI-Safeguard/Ivy-Fake",
        license="weights license undeclared; Qwen2.5-VL-3B base under the Qwen Research "
                "License (non-commercial). Obtain from the original source.",
        paper="IVY-FAKE (Pi3AI / AI-Safeguard; arXiv:2506.00979)",
        notes="Qwen2.5-VL-3B fine-tune emitting <conclusion>real/fake</conclusion>; score() "
              "= soft p(generated) from the verdict-token logits, features() = pooled hidden "
              "state. Released checkpoint is 3B (not the paper's headline scale). Heavy "
              "(~7.5GB, GPU) -> run on the cluster; paper samples video at 1 fps.",
    )
    model_id = "AI-Safeguard/Ivy-Fake"
    hub = "hf"
    system_prompt = _SYS
    user_prompt = "Is this media real or fake?"
    answer_tags = ("<conclusion>", "</conclusion>")
