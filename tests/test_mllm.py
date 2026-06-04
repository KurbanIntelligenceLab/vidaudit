"""MLLM adapter: the verdict-parsing + soft-score logic (the bug-prone parts), and Skyra
registration. The 7-9B model I/O is not exercised here (no download); it is validated on
the cluster, like the NSG-VD wrapper."""
from __future__ import annotations

import pytest

from vidaudit.detectors.mllm import parse_verdict, verdict_prob


def test_parse_verdict():
    assert parse_verdict("<think>reasoning</think><answer>fake</answer>") == "fake"
    assert parse_verdict("...<answer> Real </answer>") == "real"
    assert parse_verdict("<answer>FAKE</answer>") == "fake"
    assert parse_verdict("no tags, but this video is fake") == "fake"
    assert parse_verdict("<answer>could be real or fake</answer>") is None      # both -> abstain
    assert parse_verdict("<answer>unsure</answer>") is None                      # neither
    assert parse_verdict("<answer>real</answer> ... <answer>fake</answer>") == "fake"   # last span wins
    assert parse_verdict("<conclusion>fake</conclusion>", tags=("<conclusion>", "</conclusion>")) == "fake"


def test_verdict_prob():
    assert verdict_prob(0.0, 0.0) == pytest.approx(0.5)
    assert verdict_prob(5.0, 0.0) > 0.99            # high fake logit -> high p(generated)
    assert verdict_prob(0.0, 5.0) < 0.01
    assert verdict_prob(1.0, 0.0) == pytest.approx(1 - verdict_prob(0.0, 1.0))


def test_skyra_registered_lazy():
    from vidaudit.detectors.registry import all_detectors, get
    import vidaudit.detectors  # noqa: F401  (trigger registration)
    assert "skyra" in all_detectors()
    d = get("skyra")            # no model download (loading is lazy)
    assert d.spec.family == "mllm" and d.spec.published_weights
    assert d.has_native_head and d.has_features
    assert d.model_id == "JoeLeelyf/Skyra-RL" and d.spec.license == "CC-BY-4.0"
