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
        self.num_keys = len(keys)
        self._key_idx = 0
        self._openai_cls = OpenAI
        self._clients = [OpenAI(base_url=base_url, api_key=k, timeout=180, max_retries=0) for k in keys]
        self._key_next_allowed = [0.0 for _ in keys]
        self._key_exhausted = [False for _ in keys]
        self._tls = __import__("threading").local()
        self.client = self._clients[0]

    def _wait(self):
        """Chọn API key rảnh sớm nhất (Load-Balancing qua tất cả các key còn hạn mức)."""
        import time

        while True:
            with self._lock:
                active = [i for i, ex in enumerate(self._key_exhausted) if not ex]
                if not active:
                    active = [0]
                best_i = min(active, key=lambda i: (self._key_next_allowed[i], i))
                now = time.time()
                wait_s = self._key_next_allowed[best_i] - now
                if wait_s <= 0:
                    self._key_next_allowed[best_i] = now + self.min_interval
                    self._tls.key_idx = best_i
                    return
            time.sleep(min(wait_s, 1.0))

    def _record_tokens(self, total_tokens: int) -> None:
        if not self.tpm or total_tokens <= 0:
            return
        import time

        idx = getattr(self._tls, "key_idx", 0)
        with self._lock:
            tpm_wait = 60.0 * float(total_tokens) / self.tpm
            self._key_next_allowed[idx] = max(self._key_next_allowed[idx], time.time() + tpm_wait)

    def _penalize_current_key(self, wait_s: float) -> None:
        import time

        idx = getattr(self._tls, "key_idx", 0)
        with self._lock:
            self._key_next_allowed[idx] = max(self._key_next_allowed[idx], time.time() + wait_s)

    def rotate_key(self) -> bool:
        """Đánh dấu API key hiện tại đã hết hạn mức ngày (TPD) và chuyển sang các key còn lại."""
        idx = getattr(self._tls, "key_idx", self._key_idx)
        with self._lock:
            self._key_exhausted[idx] = True
            remaining = [i for i, ex in enumerate(self._key_exhausted) if not ex]
            if not remaining:
                return False
            self._key_idx = remaining[0]
            self.client = self._clients[self._key_idx]
            print(f"  [{self.name}] Key #{idx + 1} chạm trần ngày (TPD) -> dùng tiếp {len(remaining)} key còn lại!")
            return True

    def _complete(self, messages, max_tokens, temperature) -> LLMResponse:
        idx = getattr(self._tls, "key_idx", self._key_idx)
        client = self._clients[idx]
        r = client.chat.completions.create(
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

