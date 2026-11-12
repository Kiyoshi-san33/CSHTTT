"""Truy cập dữ liệu: experiment, output, lượt gọi LLM, nhãn đánh giá, xếp hạng."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import delete, func, select

from ..schemas.experiment import ExperimentSpec
from ..schemas.prediction import LLMCall, PredictionResult
from .db import session_scope
from .models import Annotation, Experiment, LLMCallRow, Ranking, RetrievalRow, Run


# ------------------------------------------------------------------ experiments & runs
def upsert_experiment(cfg: dict, spec: ExperimentSpec) -> None:
    with session_scope(cfg) as s:
        e = s.get(Experiment, spec.id) or Experiment(id=spec.id)
        e.exp, e.model_name, e.model_id, e.provider = spec.exp, spec.model_name, spec.model_id, spec.provider
        e.method, e.method_tag, e.profile, e.top_k = spec.method, spec.method_tag, spec.profile, spec.top_k
        e.dense_backend = spec.config_snapshot.get("retrieval", {}).get("dense_backend") if spec.profile else None
        e.is_dev, e.config_snapshot = spec.is_dev, spec.config_snapshot
        s.merge(e)


def done_run_ids(cfg: dict, experiment_id: str) -> set[str]:
    with session_scope(cfg) as s:
        q = select(Run.id).where(Run.experiment_id == experiment_id, Run.status == "done")
        return set(s.scalars(q))


def _call_row(call: LLMCall, run_id: str | None, purpose: str) -> LLMCallRow:
    r = call.response
    return LLMCallRow(
        run_id=run_id, purpose=purpose, step=call.step, prompt_name=call.prompt_name,
        prompt_version=call.prompt_version, prompt=call.prompt, system=call.system, content=r.content,
        reasoning=r.reasoning, model_id=r.model_id, served_model=r.served_model, provider=r.provider,
        params=r.params, prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
        reasoning_tokens=r.reasoning_tokens, total_tokens=r.total_tokens, latency_s=r.latency_s,
        finish_reason=r.finish_reason, request_id=r.request_id, raw_metadata=r.raw_metadata,
        created_at=r.created_at)


def save_run(cfg: dict, spec: ExperimentSpec, res: PredictionResult, retrieval_config: dict | None) -> str:
    rid = spec.run_id(res.component)
    u = res.usage()
    with session_scope(cfg) as s:
        old = s.get(Run, rid)
        if old is not None:
            s.delete(old)
            s.flush()
        run = Run(id=rid, experiment_id=spec.id, exp=spec.exp, component=res.component,
                  model_name=spec.model_name, method=spec.method, method_tag=spec.method_tag, answer=res.answer,
                  materials=res.materials, baseline=res.baseline, verification=res.verification,
                  retrieval_config=retrieval_config, status="done", **u)
        run.calls = [_call_row(c, rid, "generation") for c in res.calls]
        backend = (retrieval_config or {}).get("dense_backend")
        run.retrievals = [RetrievalRow(run_id=rid, step=rr.step, keyword_query=rr.keyword_query,
                                       dense_query=rr.dense_query, rank=p.rank, pid=p.pid, title=p.title,
                                       score=p.score, bm25_rank=p.bm25_rank, dense_rank=p.dense_rank,
                                       dense_backend=backend)
                          for rr in res.retrievals for p in rr.passages]
        s.add(run)
    return rid


def save_error_run(cfg: dict, spec: ExperimentSpec, component: str, error: str) -> None:
    rid = spec.run_id(component)
    with session_scope(cfg) as s:
        old = s.get(Run, rid)
        if old is not None and old.status == "done":
            return
        if old is not None:
            s.delete(old)
            s.flush()
        s.add(Run(id=rid, experiment_id=spec.id, exp=spec.exp, component=component, model_name=spec.model_name,
                  method=spec.method, method_tag=spec.method_tag, status="error", error=error[:4000]))


def log_call(cfg: dict, call: LLMCall, purpose: str, run_id: str | None = None) -> None:
    with session_scope(cfg) as s:
        s.add(_call_row(call, run_id, purpose))


def _run_dict(r: Run, e: Experiment | None = None) -> dict:
    d = {"id": r.id, "exp": r.exp, "component": r.component, "model": r.model_name, "method": r.method_tag,
         "base_method": r.method, "answer": r.answer, "materials": r.materials or [], "baseline": r.baseline,
         "verification": r.verification or [], "retrieval_config": r.retrieval_config,
         "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
         "reasoning_tokens": r.reasoning_tokens, "status": r.status, "error": r.error,
         "created_at": r.created_at}
    if e is not None:
        d.update(model_id=e.model_id, provider=e.provider, profile=e.profile, dense_backend=e.dense_backend,
                 is_dev=e.is_dev)
    return d


def list_runs(cfg: dict, exp: str, models: list[str] | None = None, methods: list[str] | None = None,
              include_dev: bool = False, status: str | None = "done") -> list[dict]:
    with session_scope(cfg) as s:
        q = select(Run, Experiment).join(Experiment, Run.experiment_id == Experiment.id).where(Run.exp == exp)
        if not include_dev:
            q = q.where(Experiment.is_dev.is_(False))
        if models:
            q = q.where(Run.model_name.in_(models))
        if methods:
            q = q.where(Run.method_tag.in_(methods))
        if status:
            q = q.where(Run.status == status)
        q = q.order_by(Run.component, Run.model_name, Run.method_tag)
        return [_run_dict(r, e) for r, e in s.execute(q)]


def exp1_items(cfg: dict, setting: str = "few-shot") -> list[dict]:
    """Output Exp I được đưa cho annotator (paper: chỉ few-shot)."""
    return list_runs(cfg, "exp1", methods=[setting])


def exp2_items(cfg: dict, methods: list[str] | None = None) -> list[dict]:
    names = [g["name"] for g in cfg.get("generators", [])]
    return list_runs(cfg, "exp2", models=names or None, methods=methods)


def run_detail(cfg: dict, run_id: str) -> dict | None:
    with session_scope(cfg) as s:
        r = s.get(Run, run_id)
        if r is None:
            return None
        d = _run_dict(r, r.experiment)
        d["calls"] = [{c.name: getattr(x, c.name) for c in LLMCallRow.__table__.columns} for x in r.calls]
        d["retrievals"] = [{c.name: getattr(x, c.name) for c in RetrievalRow.__table__.columns}
                           for x in r.retrievals]
        return d


def experiment_progress(cfg: dict) -> list[dict]:
    with session_scope(cfg) as s:
        q = (select(Experiment.id, Experiment.exp, Experiment.model_name, Experiment.model_id,
                    Experiment.method_tag, Experiment.profile, Experiment.dense_backend, Experiment.is_dev,
                    Run.status, func.count(Run.id), func.sum(Run.prompt_tokens), func.sum(Run.completion_tokens),
                    func.sum(Run.reasoning_tokens))
             .join(Run, Run.experiment_id == Experiment.id, isouter=True)
             .group_by(Experiment.id, Run.status))
        agg: dict = {}
        for row in s.execute(q):
            eid = row[0]
            d = agg.setdefault(eid, {"experiment": eid, "exp": row[1], "model": row[2], "model_id": row[3],
                                     "method": row[4], "profile": row[5], "dense_backend": row[6],
                                     "is_dev": row[7], "done": 0, "error": 0, "prompt_tokens": 0,
                                     "completion_tokens": 0, "reasoning_tokens": 0})
            if row[8]:
                d[row[8]] = d.get(row[8], 0) + row[9]
            d["prompt_tokens"] += row[10] or 0
            d["completion_tokens"] += row[11] or 0
            d["reasoning_tokens"] += row[12] or 0
        return sorted(agg.values(), key=lambda d: d["experiment"])


def delete_runs(cfg: dict, experiment_id: str, status: str | None = None) -> int:
    with session_scope(cfg) as s:
        ids = select(Run.id).where(Run.experiment_id == experiment_id)
        if status:
            ids = ids.where(Run.status == status)
        ids = list(s.scalars(ids))
        for rid in ids:
            s.delete(s.get(Run, rid))
        return len(ids)


# ------------------------------------------------------------------ annotations
def annotator_kind(cfg: dict, name: str) -> str:
    return "human" if name.startswith(cfg.get("evaluation", {}).get("human_prefix", "human_")) else "llm"


def save_label(cfg: dict, exp: str, item_id: str, annotator: str, label: dict, kind: str | None = None,
               judge_model_id: str | None = None) -> None:
    kind = kind or annotator_kind(cfg, annotator)
    with session_scope(cfg) as s:
        a = s.scalar(select(Annotation).where(Annotation.exp == exp, Annotation.item_id == item_id,
                                              Annotation.annotator == annotator))
        if a is None:
            a = Annotation(exp=exp, item_id=item_id, annotator=annotator)
            s.add(a)
        a.kind, a.judge_model_id, a.label = kind, judge_model_id, dict(label)
        a.updated_at = datetime.now(timezone.utc)


def save_ranking(cfg: dict, component: str, annotator: str, order: list[str], kind: str | None = None,
                 judge_model_id: str | None = None, exp: str = "exp2") -> None:
    kind = kind or annotator_kind(cfg, annotator)
    with session_scope(cfg) as s:
        r = s.scalar(select(Ranking).where(Ranking.exp == exp, Ranking.component == component,
                                           Ranking.annotator == annotator))
        if r is None:
            r = Ranking(exp=exp, component=component, annotator=annotator)
            s.add(r)
        r.kind, r.judge_model_id, r.order = kind, judge_model_id, list(order)
        r.updated_at = datetime.now(timezone.utc)


def load_labels(cfg: dict, exp: str, kind: str | None = None,
                annotators: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """{annotator: {item_id: label}}"""
    with session_scope(cfg) as s:
        q = select(Annotation).where(Annotation.exp == exp)
        if kind:
            q = q.where(Annotation.kind == kind)
        if annotators:
            q = q.where(Annotation.annotator.in_(annotators))
        out: dict = defaultdict(dict)
        for a in s.scalars(q):
            out[a.annotator][a.item_id] = a.label
        return dict(out)


def load_rankings(cfg: dict, kind: str | None = None, annotators: list[str] | None = None,
                  exp: str = "exp2") -> dict[str, dict[str, list]]:
    """{annotator: {component: [id tốt nhất, ..., id kém nhất]}}"""
    with session_scope(cfg) as s:
        q = select(Ranking).where(Ranking.exp == exp)
        if kind:
            q = q.where(Ranking.kind == kind)
        if annotators:
            q = q.where(Ranking.annotator.in_(annotators))
        out: dict = defaultdict(dict)
        for r in s.scalars(q):
            out[r.annotator][r.component] = r.order
        return dict(out)


def annotator_info(cfg: dict, exp: str | None = None) -> list[dict]:
    """Danh sách annotator + loại + judge model ID + khoảng ngày đánh giá + số nhãn."""
    with session_scope(cfg) as s:
        q = select(Annotation.exp, Annotation.annotator, Annotation.kind, Annotation.judge_model_id,
                   func.count(Annotation.id), func.min(Annotation.created_at), func.max(Annotation.updated_at)
                   ).group_by(Annotation.exp, Annotation.annotator, Annotation.kind, Annotation.judge_model_id)
        if exp:
            q = q.where(Annotation.exp == exp)
        return [{"exp": r[0], "annotator": r[1], "kind": r[2], "judge_model_id": r[3], "n_labels": r[4],
                 "first": r[5], "last": r[6]} for r in s.execute(q)]


def delete_annotations(cfg: dict, exp: str, annotator: str) -> None:
    with session_scope(cfg) as s:
        s.execute(delete(Annotation).where(Annotation.exp == exp, Annotation.annotator == annotator))
        if exp == "exp2":
            s.execute(delete(Ranking).where(Ranking.annotator == annotator))

