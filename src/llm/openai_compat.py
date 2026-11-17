"""Adapter dùng chung cho các API tương thích OpenAI Chat Completions (thư viện openai)."""
from __future__ import annotations

import os

from ..schemas.prediction import LLMResponse
from .base import BaseLLM, split_think


def _get(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    val = getattr(obj, name, None)
    if val is None:
        extra = getattr(obj, "model_extra", None) or {}
        val = extra.get(name, default)
    return val


def _dump(obj) -> dict:
    try:
        return obj.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        return {}


class OpenAICompatLLM(BaseLLM):
    provider = "openai_compat"
    DEFAULT_BASE_URL: str | None = None
    DEFAULT_KEY_ENV: str | None = None
    REQUIRES_KEY = True

    def __init__(self, cfg: dict, spec: dict):
        super().__init__(cfg, spec)
        from openai import OpenAI

        pc = self.provider_cfg
        base_url = (os.getenv(pc["base_url_env"]) if pc.get("base_url_env") else None) \
            or pc.get("base_url") or self.DEFAULT_BASE_URL
        key_env = pc.get("api_key_env", self.DEFAULT_KEY_ENV)
        keys: list[str] = []
        if key_env:
            raw_val = os.getenv(key_env) or ""
            for k in raw_val.split(","):
                if k.strip():
                    keys.append(k.strip())
            for idx in range(2, 10):
                extra_k = (os.getenv(f"{key_env}_{idx}") or "").strip()
                if extra_k and extra_k not in keys:
                    keys.append(extra_k)
        if not keys:
            if self.REQUIRES_KEY:
                raise RuntimeError(f"Thiếu API key: hãy đặt biến {key_env} trong file .env (model {self.name}).")
            keys = ["not-needed"]
        self.base_url = base_url
        self._api_keys = keys
        self._key_idx = 0
        self._openai_cls = OpenAI
        self.client = OpenAI(base_url=base_url, api_key=self._api_keys[0], timeout=180, max_retries=0)

    def rotate_key(self) -> bool:
        """Chuyển sang API key dự phòng tiếp theo nếu người dùng khai báo nhiều key trong .env."""
        if len(getattr(self, "_api_keys", [])) <= 1:
            return False
        next_idx = self._key_idx + 1
        if next_idx >= len(self._api_keys):
            return False
        self._key_idx = next_idx
        self.client = self._openai_cls(
            base_url=self.base_url, api_key=self._api_keys[self._key_idx], timeout=180, max_retries=0
        )
        with self._lock:
            self._next_allowed = 0.0
        print(f"  [{self.name}] Đã tự động chuyển sang API key #{self._key_idx + 1}/{len(self._api_keys)}!")
        return True

    def _complete(self, messages, max_tokens, temperature) -> LLMResponse:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            extra_body=dict(self.params) or None,
        )
        return self._normalize(r, self._effective_params(max_tokens, temperature))

    def _normalize(self, r, params: dict) -> LLMResponse:
        choice = r.choices[0] if getattr(r, "choices", None) else None
        msg = _get(choice, "message")
        content, think = split_think(_get(msg, "content") or "")
        reasoning = _get(msg, "reasoning") or _get(msg, "reasoning_content") or think
        usage = getattr(r, "usage", None)
        details = _get(usage, "completion_tokens_details")
        meta = _dump(r)
        meta.pop("choices", None)
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            model_id=self.model,
            served_model=getattr(r, "model", None),
            provider=self.provider,
            params=params,
            prompt_tokens=_get(usage, "prompt_tokens"),
            completion_tokens=_get(usage, "completion_tokens"),
            reasoning_tokens=_get(details, "reasoning_tokens"),
            total_tokens=_get(usage, "total_tokens"),
            finish_reason=_get(choice, "finish_reason"),
            request_id=getattr(r, "_request_id", None) or getattr(r, "id", None),
            raw_metadata=meta,
        )

    def list_models(self) -> list[str] | None:
        return sorted(m.id for m in self.client.models.list())

