"""Bộ điều phối thực nghiệm (Experiment I & Experiment II).

  Experiment I  (Adapted Baseline - GPT-OSS 120B):
      python -m src.experiments.runner --exp exp1 [--limit 10]
  Experiment II (2 LLM x 3 phương pháp = 6 thực nghiệm, cùng tập test, cùng prompt, cùng điều kiện retrieval):
      python -m src.experiments.runner --exp exp2 [--methods norag rag cove] [--models gpt_oss_120b gemini_flash]
      python -m src.experiments.runner --exp exp2 --profile adapted --limit 3
      python -m src.experiments.runner --exp exp2 --dev --limit 3    # chạy thử với Ollama Qwen3 8B
"""
from __future__ import annotations

import argparse

from tqdm import tqdm

from ..core.config import load_config
from ..core.io import read_list
from ..core.prompts import all_prompt_versions
from ..llm.base import BaseLLM
from ..llm.factory import get_llms
from ..schemas.experiment import ExperimentSpec
from ..schemas.prediction import PredictionResult, RetrievedPassage
from ..storage import repository as R
from .cove import run_cove
from .material_only import run_material_only
from .no_rag import run_no_rag
from .rag import run_rag


class ExperimentRunner:
    """Thực thi và cache kết quả truy hồi để hai LLM dùng chung chính xác cùng tập passage."""

    def __init__(self, cfg: dict, top_k: int | None = None):
        self.cfg = cfg
        self.top_k = top_k or int(cfg["retrieval"]["final_top_k"])
        self._retriever = None
        self._cache: dict[tuple[str, str, int], list[RetrievedPassage]] = {}

    @property
    def retriever(self):
        if self._retriever is None:
            from ..rag.hybrid_retriever import HybridRetriever

            self._retriever = HybridRetriever(self.cfg)
        return self._retriever

    def retrieve(self, keyword_query: str, dense_query: str) -> list[RetrievedPassage]:
        key = (keyword_query, dense_query, self.top_k)
        if key not in self._cache:
            self._cache[key] = self.retriever.retrieve(keyword_query, dense_query, self.top_k)
        return self._cache[key]

    def retrieval_config(self, method: str) -> dict | None:
        if method in ("norag", "few-shot", "zero-shot"):
            return None
        rc = self.retriever.configuration()
        rc["final_top_k"] = self.top_k
        return rc

    def snapshot(self, llm: BaseLLM) -> dict:
        return {
            "active_profile": self.cfg.get("active_profile"),
            "llm": llm.describe(),
            "generation": self.cfg.get("generation", {}),
            "enforce_list_format": self.cfg.get("enforce_list_format", True),
            "retrieval": self.cfg.get("retrieval", {}),
            "cove": self.cfg.get("cove", {}),
            "prompt_versions": all_prompt_versions(),
        }

    def run_single(self, llm: BaseLLM, component: str, method: str, exp: str = "exp2") -> PredictionResult:
        if exp == "exp1":
            return run_material_only(llm, component, self.cfg, setting=method)
        if method == "norag":
            return run_no_rag(llm, component, self.cfg)
        if method == "rag":
            return run_rag(llm, component, self.retrieve, self.cfg)
        if method == "cove":
            return run_cove(llm, component, self.retrieve, self.cfg)
        raise ValueError(f"Phương pháp không hợp lệ: {method!r}")


def execute_matrix(cfg: dict, exp: str = "exp2", models: list[str] | None = None,
                   methods: list[str] | None = None, limit: int = 0, top_k: int | None = None,
                   dev: bool = False) -> dict[str, int]:
    runner = ExperimentRunner(cfg, top_k=top_k)
    if dev:
        llms = get_llms(cfg, "dev_generators", models)
    elif exp == "exp1":
        names = models or cfg.get("exp1_generators")
        llms = get_llms(cfg, "generators", names)
    else:
        llms = get_llms(cfg, "generators", models)

    if exp == "exp1":
        methods = methods or ["few-shot", "zero-shot"]
        comps = read_list(cfg["paths"]["test_exp1"])
    else:
        methods = methods or ["norag", "rag", "cove"]
        comps = read_list(cfg["paths"]["test_exp2"])
    if limit:
        comps = comps[:limit]

    profile = cfg.get("active_profile", "adapted")
    summary: dict[str, int] = {}
    for llm in llms:
        for method in methods:
            uses_rag = exp == "exp2" and method != "norag"
            spec = ExperimentSpec(
                exp=exp,
                model_name=llm.name,
                model_id=llm.model,
                provider=llm.provider,
                method=method,
                profile=profile if uses_rag else None,
                top_k=runner.top_k if uses_rag else None,
                is_dev=dev,
                config_snapshot=runner.snapshot(llm),
            )
            R.upsert_experiment(cfg, spec)
            done = R.done_run_ids(cfg, spec.id)
            n_new = 0
            for c in tqdm(comps, desc=f"{spec.id}"):
                rid = spec.run_id(c)
                if rid in done:
                    continue
                try:
                    res = runner.run_single(llm, c, method, exp=exp)
                    R.save_run(cfg, spec, res, runner.retrieval_config(method))
                    n_new += 1
                except Exception as e:  # noqa: BLE001
                    R.save_error_run(cfg, spec, c, str(e))
                    raise
            summary[spec.id] = n_new
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp2"], default="exp2")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--profile", choices=["adapted", "reproduction"], default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev", action="store_true", help="chạy bằng dev_generators (Ollama Qwen3 8B)")
    a = ap.parse_args()
    cfg = load_config(profile=a.profile)
    summary = execute_matrix(cfg, exp=a.exp, models=a.models, methods=a.methods,
                             limit=a.limit, top_k=a.top_k, dev=a.dev)
    for k, v in summary.items():
        print(f"  {k}: +{v} run mới")


if __name__ == "__main__":
    main()

