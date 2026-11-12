"""Phương pháp RAG: truy xuất tài liệu bằng Hybrid Retriever và đưa bằng chứng vào prompt."""
from __future__ import annotations

from typing import Callable

from ..core.prompts import DENSE_QUERY, FORMAT_HINT, QUESTION, format_passages, load_prompt
from ..llm.base import BaseLLM
from ..schemas.prediction import LLMCall, PredictionResult, RetrievalRecord, RetrievedPassage
from .parsing import parse_explained


def run_rag(llm: BaseLLM, component: str, retrieve_fn: Callable[[str, str], list[RetrievedPassage]],
            cfg: dict, step_name: str = "answer") -> PredictionResult:
    tpl = load_prompt("rag")
    fmt = FORMAT_HINT if cfg.get("enforce_list_format", True) else ""
    q = QUESTION.format(component=component)
    dq = DENSE_QUERY.format(component=component)
    hits = retrieve_fn(component, dq)
    prompt = tpl.render(passages=format_passages(hits), question=q, fmt=fmt)
    resp = llm.chat(prompt)
    call = LLMCall(step=step_name, prompt_name=tpl.name, prompt_version=tpl.version,
                   prompt=prompt, response=resp)
    rec = RetrievalRecord(step="rag", keyword_query=component, dense_query=dq, passages=hits)
    return PredictionResult(
        component=component,
        method="rag",
        answer=resp.content,
        materials=parse_explained(resp.content),
        calls=[call],
        retrievals=[rec],
    )

