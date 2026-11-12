"""Phương pháp CoVe (Chain-of-Verification) gồm 4 bước:
  1. Baseline response (= RAG)
  2. Plan verification questions
  3. Execute verification độc lập với truy hồi riêng cho từng câu hỏi
  4. Final verified response
"""
from __future__ import annotations

from typing import Callable

from ..core.prompts import FORMAT_HINT, QUESTION, format_passages, load_prompt
from ..llm.base import BaseLLM
from ..schemas.prediction import LLMCall, PredictionResult, RetrievalRecord, RetrievedPassage
from .parsing import parse_explained, parse_questions
from .rag import run_rag


def run_cove(llm: BaseLLM, component: str, retrieve_fn: Callable[[str, str], list[RetrievedPassage]],
             cfg: dict) -> PredictionResult:
    q = QUESTION.format(component=component)
    fmt = FORMAT_HINT if cfg.get("enforce_list_format", True) else ""
    step_tokens = cfg.get("generation", {}).get("step_max_tokens", {})
    max_q = int(cfg.get("cove", {}).get("max_questions", 5))

    # Bước 1: Baseline (= RAG)
    base = run_rag(llm, component, retrieve_fn, cfg, step_name="baseline")
    calls = list(base.calls)
    retrievals = list(base.retrievals)

    # Bước 2: Lập kế hoạch câu hỏi kiểm chứng
    plan_tpl = load_prompt("cove_plan")
    plan_prompt = plan_tpl.render(question=q, baseline=base.answer)
    plan_resp = llm.chat(plan_prompt, max_tokens=step_tokens.get("cove_plan", 2000))
    calls.append(LLMCall(step="cove_plan", prompt_name=plan_tpl.name,
                         prompt_version=plan_tpl.version, prompt=plan_prompt, response=plan_resp))
    vqs = parse_questions(plan_resp.content, max_q)

    # Bước 3: Thực thi độc lập từng câu hỏi với passage truy hồi riêng
    exec_tpl = load_prompt("cove_execute")
    qa = []
    for idx, vq in enumerate(vqs):
        hits = retrieve_fn(vq, vq)
        retrievals.append(RetrievalRecord(step=f"cove_execute_{idx + 1}",
                                          keyword_query=vq, dense_query=vq, passages=hits))
        exec_prompt = exec_tpl.render(passages=format_passages(hits), vq=vq)
        exec_resp = llm.chat(exec_prompt, max_tokens=step_tokens.get("cove_execute", 1500))
        calls.append(LLMCall(step=f"cove_execute_{idx + 1}", prompt_name=exec_tpl.name,
                             prompt_version=exec_tpl.version, prompt=exec_prompt, response=exec_resp))
        qa.append({"q": vq, "a": exec_resp.content, "pids": [h.pid for h in hits]})

    # Bước 4: Tổng hợp câu trả lời đã kiểm chứng
    qa_txt = "\n".join(f"Q{i + 1}: {x['q']}\nA{i + 1}: {x['a']}" for i, x in enumerate(qa))
    final_tpl = load_prompt("cove_final")
    final_prompt = final_tpl.render(question=q, baseline=base.answer, qa=qa_txt, fmt=fmt)
    final_resp = llm.chat(final_prompt)
    calls.append(LLMCall(step="cove_final", prompt_name=final_tpl.name,
                         prompt_version=final_tpl.version, prompt=final_prompt, response=final_resp))

    return PredictionResult(
        component=component,
        method="cove",
        answer=final_resp.content,
        materials=parse_explained(final_resp.content),
        baseline=base.answer,
        verification=qa,
        calls=calls,
        retrievals=retrievals,
    )

