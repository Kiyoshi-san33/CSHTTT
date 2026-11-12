"""Schema chuẩn hóa cho phản hồi LLM, passage truy hồi và kết quả dự đoán."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LLMResponse(BaseModel):
    """Phản hồi đã chuẩn hóa từ mọi provider (Groq, Gemini, Ollama, dummy)."""
    content: str
    reasoning: str | None = None            # phần suy luận tách riêng (nếu provider trả về)
    model_id: str                           # model ID đã yêu cầu (được ghim trong config)
    served_model: str | None = None         # model ID provider báo đã phục vụ
    provider: str
    params: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    latency_s: float = 0.0
    finish_reason: str | None = None
    request_id: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() in {"length", "max_tokens"}


class RetrievedPassage(BaseModel):
    pid: int
    title: str
    text: str
    score: float
    rank: int                               # 1-based, sau RRF
    bm25_rank: int | None = None            # 1-based trong danh sách BM25 (None nếu không có)
    dense_rank: int | None = None


class RetrievalRecord(BaseModel):
    step: str                               # rag | cove_execute_<i>
    keyword_query: str
    dense_query: str
    passages: list[RetrievedPassage]


class LLMCall(BaseModel):
    step: str                               # answer | baseline | cove_plan | cove_execute_<i> | cove_final
    prompt_name: str
    prompt_version: str
    prompt: str
    system: str | None = None
    response: LLMResponse


class PredictionResult(BaseModel):
    component: str
    method: str                             # norag | rag | cove | few-shot | zero-shot
    answer: str
    materials: list[str] = Field(default_factory=list)
    baseline: str | None = None             # CoVe: câu trả lời sơ bộ
    verification: list[dict] = Field(default_factory=list)  # CoVe: [{q, a, pids}]
    calls: list[LLMCall] = Field(default_factory=list)
    retrievals: list[RetrievalRecord] = Field(default_factory=list)

    def usage(self) -> dict[str, int]:
        tot = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        for c in self.calls:
            for k in tot:
                tot[k] += getattr(c.response, k) or 0
        return tot

