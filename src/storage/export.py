"""Xuất dữ liệu SQLite ra JSONL/CSV để đính kèm báo cáo và tái lập.

    python -m src.storage.export
Kết quả:
    results/predictions/exp1_runs.jsonl, exp2_runs.jsonl  (output + cấu hình truy hồi + token)
    results/predictions/llm_calls.jsonl                   (mọi lượt gọi LLM: prompt, raw output, usage)
    results/predictions/retrievals.csv
    results/annotations/exp1_labels.csv, exp2_labels.csv, exp2_rankings.csv
"""
from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import select

from ..core.config import load_config
from ..core.io import write_jsonl
from ..core.paths import results_dir
from . import repository as R
from .db import session_scope
from .models import Annotation, LLMCallRow, Ranking, RetrievalRow


def _rows(cfg, model):
    with session_scope(cfg) as s:
        cols = [c.name for c in model.__table__.columns]
        return [{c: getattr(x, c) for c in cols} for x in s.scalars(select(model))]


def export_all(cfg: dict) -> dict[str, int]:
    pred, ann = results_dir(cfg, "predictions"), results_dir(cfg, "annotations")
    out = {}
    for exp in ("exp1", "exp2"):
        out[f"{exp}_runs"] = write_jsonl(pred / f"{exp}_runs.jsonl",
                                         R.list_runs(cfg, exp, include_dev=True, status=None))
    out["llm_calls"] = write_jsonl(pred / "llm_calls.jsonl", _rows(cfg, LLMCallRow))
    pd.DataFrame(_rows(cfg, RetrievalRow)).to_csv(pred / "retrievals.csv", index=False, encoding="utf-8-sig")
    labels = _rows(cfg, Annotation)
    for exp in ("exp1", "exp2"):
        rows = []
        for a in (x for x in labels if x["exp"] == exp):
            lab = a.pop("label") or {}
            rows.append({**a, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in lab.items()}})
        pd.DataFrame(rows).to_csv(ann / f"{exp}_labels.csv", index=False, encoding="utf-8-sig")
        out[f"{exp}_labels"] = len(rows)
    rk = [{**r, "order": json.dumps(r["order"], ensure_ascii=False)} for r in _rows(cfg, Ranking)]
    pd.DataFrame(rk).to_csv(ann / "exp2_rankings.csv", index=False, encoding="utf-8-sig")
    out["exp2_rankings"] = len(rk)
    return out


def main():
    cfg = load_config()
    for k, v in export_all(cfg).items():
        print(f"{k:<15} {v}")
    print("->", results_dir(cfg, "predictions"), "|", results_dir(cfg, "annotations"))


if __name__ == "__main__":
    main()

