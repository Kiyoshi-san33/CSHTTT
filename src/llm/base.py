"""Lớp cơ sở cho mọi client LLM: giới hạn tốc độ, thử lại, lọc tham số, chuẩn hóa phản hồi.

Mỗi provider adapter khai báo SUPPORTED_PARAMS — các tham số riêng (ngoài temperature/max_tokens)
mà provider chấp nhận. Tham số không được hỗ trợ sẽ bị bỏ qua kèm cảnh báo, KHÔNG gửi đi,
vì các provider không hỗ trợ cùng một bộ tham số.
"""
from __future__ import annotations

import re
import threading
import time
import warnings
from abc import ABC, abstractmethod

from ..schemas.prediction import LLMResponse

_THINK = re.compile(r"<think>(.*?)</think>\s*", re.S | re.I)
_RETRY_AFTER_RE = re.compile(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", re.I)
FATAL_STATUS = {400, 401, 403, 404, 422}


def split_think(text: str) -> tuple[str, str | None]:
    """Tách khối <think>...</think> (Qwen3, DeepSeek...) khỏi câu trả lời."""
    if not text or "<think>" not in text.lower():
        return (text or "").strip(), None
    thoughts = "\n".join(m.strip() for m in _THINK.findall(text))
    rest = _THINK.sub("", text)
    if "<think>" in rest.lower():  # khối suy luận chưa đóng (bị cắt)
        k = rest.lower().index("<think>")
        thoughts = (thoughts + "\n" + rest[k + 7:]).strip()
        rest = rest[:k]
    return rest.strip(), thoughts or None


class LLMError(RuntimeError):
    pass


class BaseLLM(ABC):
    provider: str = "base"
    SUPPORTED_PARAMS: frozenset[str] = frozenset()

    def __init__(self, cfg: dict, spec: dict):
        self.cfg = cfg
        self.spec = dict(spec)
        self.name: str = spec["name"]
        self.model: str = spec["model"]
        self.provider_cfg: dict = (cfg.get("providers") or {}).get(spec["provider"], {}) or {}
        self.min_interval = 60.0 / float(spec.get("rpm", 60))
        tpm_val = spec.get("tpm")
        self.tpm: float | None = float(tpm_val) if tpm_val else None
        self._next_allowed = 0.0
        self._lock = threading.Lock()
        gen = cfg.get("generation", {})
        self.temperature = float(spec.get("temperature", gen.get("temperature", 0.0)))
        self.max_tokens = int(spec.get("max_tokens", gen.get("max_tokens", 4000)))
        self.max_attempts = int(gen.get("max_attempts", 8))
        raw = dict(spec.get("params") or {})
        self.params = {k: v for k, v in raw.items() if k in self.SUPPORTED_PARAMS}
        self.dropped_params = sorted(set(raw) - set(self.params))
        if self.dropped_params:
            warnings.warn(f"[{self.name}] provider {self.provider} không hỗ trợ tham số "
                          f"{self.dropped_params} -> bỏ qua", stacklevel=2)

    # ------------------------------------------------------------------ API
    def describe(self) -> dict:
        return {"name": self.name, "provider": self.provider, "model_id": self.model,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "params": self.params, "dropped_params": self.dropped_params}

    def chat(self, prompt: str, system: str | None = None, max_tokens: int | None = None,
             temperature: float | None = None) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        mt = int(max_tokens or self.max_tokens)
        temp = self.temperature if temperature is None else float(temperature)
        delay = 5.0
        last_err: Exception | None = None
        for attempt in range(self.max_attempts):
            self._wait()
            t0 = time.perf_counter()
            try:
                resp = self._complete(messages, mt, temp)
                resp.latency_s = round(time.perf_counter() - t0, 3)
                self._record_tokens((resp.prompt_tokens or 0) + (resp.completion_tokens or 0))
                if resp.truncated:
                    warnings.warn(f"[{self.name}] câu trả lời bị cắt (finish_reason={resp.finish_reason}, "
                                  f"max_tokens={mt}) — cân nhắc tăng max_tokens", stacklevel=2)
                return resp
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                if isinstance(status, int) and status in FATAL_STATUS:
                    raise LLMError(f"[{self.name}] lỗi không thử lại được ({status}): {str(e)[:300]}") from e
                msg = str(e)
                if "tokens per day (TPD)" in msg or "requests per day (RPD)" in msg:
                    if self.rotate_key():
                        continue
                    raise LLMError(
                        f"[{self.name}] Đã hết hạn mức 200.000 token/ngày (TPD) của gói Groq miễn phí trên key hiện tại. "
                        f"Cách xử lý: (1) Thêm GROQ_API_KEY_2=... của tài khoản Groq khác vào file .env để chạy tiếp ngay, "
                        f"hoặc (2) chạy trước model Gemini bằng lệnh: python -m src.experiments.runner --exp exp2 --models gemini_flash"
                    ) from e
                last_err = e
                m_wait = _RETRY_AFTER_RE.search(msg)
                if m_wait:
                    mins = float(m_wait.group(1) or 0)
                    secs = float(m_wait.group(2) or 0)
                    wait_s = mins * 60.0 + secs + 2.0
                else:
                    wait_s = delay
                    delay = min(delay * 2, 120)
                print(f"  [{self.name}] chạm giới hạn tốc độ API (lần {attempt + 1}) -> đợi {wait_s:.0f}s cho hồi hạn mức...")
                time.sleep(wait_s)
                with self._lock:
                    self._next_allowed = max(self._next_allowed, time.time() + self.min_interval)
        raise LLMError(f"[{self.name}] thất bại sau {self.max_attempts} lần thử: {last_err}")

    def list_models(self) -> list[str] | None:
        """Danh sách model ID provider đang cung cấp (None nếu không hỗ trợ)."""
        return None

    def rotate_key(self) -> bool:
        """Chuyển sang API key tiếp theo nếu có (mặc định False)."""
        return False

    # ------------------------------------------------------------- internal
    def _wait(self):
        with self._lock:
            now = time.time()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = time.time() + self.min_interval

    def _record_tokens(self, total_tokens: int) -> None:
        if not self.tpm or total_tokens <= 0:
            return
        with self._lock:
            tpm_wait = 60.0 * float(total_tokens) / self.tpm
            self._next_allowed = max(self._next_allowed, time.time() + tpm_wait)

    def _effective_params(self, max_tokens: int, temperature: float) -> dict:
        return {"temperature": temperature, "max_tokens": max_tokens, **self.params}

    @abstractmethod
    def _complete(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResponse:
        ...

