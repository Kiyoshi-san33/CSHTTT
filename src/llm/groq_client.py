"""Groq Cloud (OpenAI-compatible). Model A: openai/gpt-oss-120b; judge: qwen/qwen3.8-27b."""
from __future__ import annotations

from .openai_compat import OpenAICompatLLM


class GroqLLM(OpenAICompatLLM):
    provider = "groq"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_KEY_ENV = "GROQ_API_KEY"
    # reasoning_effort: GPT-OSS (low|medium|high), Qwen3 (none|default)
    # reasoning_format: model Qwen (parsed|raw|hidden); include_reasoning: GPT-OSS
    SUPPORTED_PARAMS = frozenset({"reasoning_effort", "reasoning_format", "include_reasoning",
                                  "top_p", "seed", "service_tier"})

