"""Experiment I – Adapted Baseline (GPT-OSS 120B):
Dự đoán tối đa 5 vật liệu cho linh kiện xe (chỉ tên vật liệu, không giải thích),
sử dụng prompt Few-shot nguyên văn Phụ lục A (và Zero-shot để đối chiếu nội bộ).

Lưu ý: Đây là Adapted Baseline dùng GPT-OSS 120B, không phải tái lập trực tiếp Mixtral-8x22B.
"""
from __future__ import annotations

from ..core.prompts import load_prompt
from ..llm.base import BaseLLM
from ..schemas.prediction import LLMCall, PredictionResult
from .parsing import parse_list


def run_material_only(llm: BaseLLM, component: str, cfg: dict,
                      setting: str = "few-shot") -> PredictionResult:
    tpl_name = "exp1_fewshot" if setting == "few-shot" else "exp1_zeroshot"
    tpl = load_prompt(tpl_name)
    prompt = tpl.render(component=component)
    mt = cfg.get("generation", {}).get("step_max_tokens", {}).get("exp1", 2000)
    resp = llm.chat(prompt, max_tokens=mt)
    call = LLMCall(step="answer", prompt_name=tpl.name, prompt_version=tpl.version,
                   prompt=prompt, response=resp)
    return PredictionResult(
        component=component,
        method=setting,
        answer=resp.content,
        materials=parse_list(resp.content, k=5),
        calls=[call],
    )

