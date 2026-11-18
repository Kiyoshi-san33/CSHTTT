"""Phương pháp CoVe (Chain-of-Verification) gồm 4 bước:
  1. Baseline response (= RAG)
  2. Plan verification questions
  3. Execute verification độc lập với truy hồi riêng cho từng câu hỏi
  4. Final verified response
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ..core.prompts import FORMAT_HINT, QUESTION, format_passages, load_prompt
from ..llm.base import BaseLLM
from ..schemas.prediction import LLMCall, PredictionResult, RetrievalRecord, RetrievedPassage
from ..storage.repository import get_done_run_answer
from .parsing import parse_explained, parse_questions
from .rag import run_rag


def run_cove(llm: BaseLLM, component: str, retrieve_fn: Callable[[str, str], list[RetrievedPassage]],
             cfg: dict) -> PredictionResult:
    q = QUESTION.format(component=component)
    fmt = FORMAT_HINT if cfg.get("enforce_list_format", True) else ""
    step_tokens = cfg.get("generation", {}).get("step_max_tokens", {})
    max_q = int(cfg.get("cove", {}).get("max_questions", 5))

    # Bước 1: Baseline (= RAG). Nếu kết quả RAG của cùng (component, model, profile) đã có sẵn trong DB
    # thì tái sử dụng ngay (vì temperature=0.0 hoàn toàn tất định), tiết kiệm ~3.300 token & 1 lượt gọi LLM.
    profile = cfg.get("active_profile", "adapted")
    rag_run_id = f"{component}||{llm.name}||rag@{profile}"
    cached_baseline = get_done_run_answer(cfg, rag_run_id)
    if cached_baseline:
        base_hits = retrieve_fn(component, q)
        base_answer = cached_baseline
        calls: list[LLMCall] = []
        retrievals: list[RetrievalRecord] = [
            RetrievalRecord(step="baseline", keyword_query=component, dense_query=q, passages=base_hits)
        ]
    else:
        base = run_rag(llm, component, retrieve_fn, cfg, step_name="baseline")
        base_answer = base.answer
        calls = list(base.calls)
        retrievals = list(base.retrievals)

    # Bước 2: Lập kế hoạch câu hỏi kiểm chứng
    plan_tpl = load_prompt("cove_plan")
    plan_prompt = plan_tpl.render(question=q, baseline=base_answer)
    plan_resp = llm.chat(plan_prompt, max_tokens=step_tokens.get("cove_plan", 2000))
    calls.append(LLMCall(step="cove_plan", prompt_name=plan_tpl.name,
                         prompt_version=plan_tpl.version, prompt=plan_prompt, response=plan_resp))
    vqs = parse_questions(plan_resp.content, max_q)

    # Bước 3: Thực thi độc lập từng câu hỏi với passage truy hồi riêng
    # (Chạy song song trên nhiều luồng nếu người dùng cấu hình nhiều API key trong .env)
    exec_tpl = load_prompt("cove_execute")
    exec_max_tok = step_tokens.get("cove_execute", 1500)
    prepared = []
    for idx, vq in enumerate(vqs):
        hits = retrieve_fn(vq, vq)
        retrievals.append(RetrievalRecord(step=f"cove_execute_{idx + 1}",
                                          keyword_query=vq, dense_query=vq, passages=hits))
        exec_prompt = exec_tpl.render(passages=format_passages(hits), vq=vq)
        prepared.append((idx, vq, hits, exec_prompt))

    n_workers = max(1, min(len(prepared), getattr(llm, "num_keys", 1)))
    if n_workers > 1:
        def _exec_one(item):
            idx, vq, hits, exec_prompt = item
            resp = llm.chat(exec_prompt, max_tokens=exec_max_tok)
            return idx, vq, hits, exec_prompt, resp

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            exec_results = list(ex.map(_exec_one, prepared))
    else:
        exec_results = [
            (idx, vq, hits, exec_prompt, llm.chat(exec_prompt, max_tokens=exec_max_tok))
            for idx, vq, hits, exec_prompt in prepared
        ]

    qa = []
    for idx, vq, hits, exec_prompt, exec_resp in exec_results:
        calls.append(LLMCall(step=f"cove_execute_{idx + 1}", prompt_name=exec_tpl.name,
                             prompt_version=exec_tpl.version, prompt=exec_prompt, response=exec_resp))
        qa.append({"q": vq, "a": exec_resp.content, "pids": [h.pid for h in hits]})

    # Bước 4: Tổng hợp câu trả lời đã kiểm chứng
    qa_txt = "\n".join(f"Q{i + 1}: {x['q']}\nA{i + 1}: {x['a']}" for i, x in enumerate(qa))
    final_tpl = load_prompt("cove_final")
    final_prompt = final_tpl.render(question=q, baseline=base_answer, qa=qa_txt, fmt=fmt)
    final_resp = llm.chat(final_prompt)
    calls.append(LLMCall(step="cove_final", prompt_name=final_tpl.name,
                         prompt_version=final_tpl.version, prompt=final_prompt, response=final_resp))

    return PredictionResult(
        component=component,
        method="cove",
        answer=final_resp.content,
        materials=parse_explained(final_resp.content),
        baseline=base_answer,
        verification=qa,
        calls=calls,
        retrievals=retrievals,
    )

