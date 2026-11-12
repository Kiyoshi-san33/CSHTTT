"""Google Gemini (Model B: gemini-3.8-flash).

- GeminiLLM          : qua OpenAI-compatible endpoint (mặc định, cùng định dạng log với Groq).
- GeminiNativeLLM    : adapter tùy chọn qua google-genai (provider: gemini_native) khi cần
                       tính năng riêng của Gemini (thinking_config, ...).
Lưu ý: dữ liệu gửi qua Free Tier có thể được Google dùng để cải thiện sản phẩm — chỉ gửi
corpus công khai / dữ liệu đã được phép.
"""
from __future__ import annotations

import os

from ..schemas.prediction import LLMResponse
from .base import BaseLLM, split_think
from .openai_compat import OpenAICompatLLM


class GeminiLLM(OpenAICompatLLM):
    provider = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DEFAULT_KEY_ENV = "GEMINI_API_KEY"
    SUPPORTED_PARAMS = frozenset({"reasoning_effort", "top_p", "seed"})

    def list_models(self) -> list[str] | None:
        # endpoint tương thích trả "models/gemini-..." -> chuẩn hóa bỏ tiền tố
        return sorted(m.id.removeprefix("models/") for m in self.client.models.list())


class GeminiNativeLLM(BaseLLM):
    provider = "gemini_native"
    SUPPORTED_PARAMS = frozenset({"thinking_budget", "thinking_level", "include_thoughts", "top_p", "seed"})

    def __init__(self, cfg: dict, spec: dict):
        super().__init__(cfg, spec)
        from google import genai

        key_env = self.provider_cfg.get("api_key_env", "GEMINI_API_KEY")
        key = os.getenv(key_env)
        if not key:
            raise RuntimeError(f"Thiếu API key: hãy đặt biến {key_env} trong file .env (model {self.name}).")
        self.client = genai.Client(api_key=key)

    def _complete(self, messages, max_tokens, temperature) -> LLMResponse:
        from google.genai import types

        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        contents = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        think_keys = {"thinking_budget", "thinking_level", "include_thoughts"}
        think = {k: v for k, v in self.params.items() if k in think_keys}
        other = {k: v for k, v in self.params.items() if k not in think_keys}
        conf = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_tokens, system_instruction=system,
            thinking_config=types.ThinkingConfig(**think) if think else None, **other)
        r = self.client.models.generate_content(model=self.model, contents=contents, config=conf)
        cand = (r.candidates or [None])[0]
        fr = getattr(cand, "finish_reason", None)
        um = r.usage_metadata
        content, thoughts = split_think(r.text or "")
        try:
            meta = r.model_dump(mode="json", exclude={"candidates"})
        except Exception:  # noqa: BLE001
            meta = {}
        return LLMResponse(
            content=content, reasoning=thoughts, model_id=self.model,
            served_model=getattr(r, "model_version", None), provider=self.provider,
            params=self._effective_params(max_tokens, temperature),
            prompt_tokens=getattr(um, "prompt_token_count", None),
            completion_tokens=getattr(um, "candidates_token_count", None),
            reasoning_tokens=getattr(um, "thoughts_token_count", None),
            total_tokens=getattr(um, "total_token_count", None),
            finish_reason=getattr(fr, "name", None) or (str(fr) if fr else None),
            request_id=getattr(r, "response_id", None), raw_metadata=meta)

    def list_models(self) -> list[str] | None:
        return sorted(m.name.removeprefix("models/") for m in self.client.models.list())

