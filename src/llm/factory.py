"""Tạo client LLM từ spec trong config.yaml."""
from __future__ import annotations

from .base import BaseLLM

_REGISTRY = {
    "groq": ("groq_client", "GroqLLM"),
    "gemini": ("gemini_client", "GeminiLLM"),
    "gemini_native": ("gemini_client", "GeminiNativeLLM"),
    "ollama": ("ollama_client", "OllamaLLM"),
    "dummy": ("dummy_client", "DummyLLM"),
}


def get_llm(cfg: dict, spec: dict) -> BaseLLM:
    import importlib

    try:
        mod, cls = _REGISTRY[spec["provider"]]
    except KeyError as e:
        raise ValueError(f"Provider không hỗ trợ: {spec['provider']!r} (có: {', '.join(_REGISTRY)})") from e
    return getattr(importlib.import_module(f"{__package__}.{mod}"), cls)(cfg, spec)


def get_llms(cfg: dict, key: str = "generators", names: list[str] | None = None) -> list[BaseLLM]:
    specs = list(cfg.get(key) or [])
    if names:
        missing = set(names) - {s["name"] for s in specs}
        if missing:
            raise KeyError(f"Không có model {sorted(missing)} trong mục {key!r} của config")
        specs = [s for s in specs if s["name"] in names]
    return [get_llm(cfg, s) for s in specs]

