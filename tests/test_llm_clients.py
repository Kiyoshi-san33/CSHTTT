"""Kiểm tra Unified LLM Client, lọc tham số theo provider, tách <think> và chuẩn hóa LLMResponse."""
from __future__ import annotations

import warnings

from src.core.config import load_config
from src.llm.base import split_think
from src.llm.factory import get_llm
from src.llm.gemini_client import GeminiLLM
from src.llm.groq_client import GroqLLM
from src.llm.ollama_client import OllamaLLM


def test_split_think_extracts_reasoning():
    content, reasoning = split_think("<think>step 1: check brake disc\nstep 2: cast iron</think>\n1. Cast Iron: used.")
    assert content == "1. Cast Iron: used."
    assert reasoning == "step 1: check brake disc\nstep 2: cast iron"


def test_provider_supported_params_and_dropping():
    assert "reasoning_effort" in GroqLLM.SUPPORTED_PARAMS
    assert "reasoning_format" in GroqLLM.SUPPORTED_PARAMS
    assert "reasoning_effort" in GeminiLLM.SUPPORTED_PARAMS
    assert "think" not in GeminiLLM.SUPPORTED_PARAMS
    assert "think" in OllamaLLM.SUPPORTED_PARAMS

    cfg = load_config()
    spec = {
        "name": "dummy_test",
        "provider": "dummy",
        "model": "dummy-v1",
        "rpm": 6000,
        "params": {"reasoning_effort": "medium", "unsupported_xyz": 123},
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        llm = get_llm(cfg, spec)
        assert "unsupported_xyz" in llm.dropped_params
        assert "unsupported_xyz" not in llm.params
        assert llm.params["reasoning_effort"] == "medium"
        assert any("unsupported_xyz" in str(item.message) for item in w)

    resp = llm.chat("comma-separated list of materials for brake disc")
    assert resp.model_id == "dummy-v1"
    assert resp.provider == "dummy"
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert resp.finish_reason == "stop"
    assert resp.request_id.startswith("dummy-")

