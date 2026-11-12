"""Phương pháp No-RAG: dự đoán từ kiến thức nội tại của LLM, không truy xuất ngữ liệu."""
from __future__ import annotations

from ..core.prompts import FORMAT_HINT, QUESTION, load_prompt
from ..llm.base import BaseLLM
from ..schemas.prediction import LLMCall, PredictionResult
from .parsing import parse_explained


def run_no_rag(llm: BaseLLM, component: str, cfg: dict) -> PredictionResult:
    tpl = load_prompt("prediction")
    fmt = FORMAT_HINT if cfg.get("enforce_list_format", True) else ""
    q = QUESTION.format(component=component)
    prompt = tpl.render(question=q, fmt=fmt)
    resp = llm.chat(prompt)
    call = LLMCall(step="answer", prompt_name=tpl.name, prompt_version=tpl.version,
                   prompt=prompt, response=resp)
    return PredictionResult(
        component=component,
        method="norag",
        answer=resp.content,
        materials=parse_explained(resp.content),
        calls=[call],
    )

