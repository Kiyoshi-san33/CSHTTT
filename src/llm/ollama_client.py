"""Ollama local (OpenAI-compatible) — Qwen3 8B, CHỈ dùng cho phát triển/kiểm thử offline.

Cài đặt: https://ollama.com  ->  ollama pull qwen3:8b
Địa chỉ: biến OLLAMA_BASE_URL (mặc định http://localhost:11434/v1). Khối <think> của Qwen3
được tách khỏi câu trả lời và lưu vào trường reasoning.
"""
from __future__ import annotations

from .openai_compat import OpenAICompatLLM


class OllamaLLM(OpenAICompatLLM):
    provider = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_KEY_ENV = None
    REQUIRES_KEY = False
    SUPPORTED_PARAMS = frozenset({"think", "reasoning_effort", "top_p", "seed"})

