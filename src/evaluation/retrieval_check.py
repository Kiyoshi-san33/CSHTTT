"""Kiểm tra thủ công chất lượng truy hồi (paper: ~22% passage thực sự hữu ích trên 10 linh kiện đầu).

    python -m src.evaluation.retrieval_check [--n 10] [--profile adapted]
    python -m src.evaluation.retrieval_check --score
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..core.config import load_config
from ..core.io import read_list
from ..core.paths import results_dir
from ..core.prompts import DENSE_QUERY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--profile", choices=["adapted", "reproduction"], default=None)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    cfg = load_config(profile=a.profile)
    out = results_dir(cfg, "metrics") / f"retrieval_check_{cfg['active_profile']}.csv"
    if a.score:
        df = pd.read_csv(out)
        df = df[df["helpful"].notna()]
        print(f"{len(df)} passage đã chấm | hữu ích: {df['helpful'].mean() * 100:.1f}% "
              f"({df['helpful'].sum() / max(df['component'].nunique(), 1):.1f} passage/linh kiện) "
              f"| paper: 22% (2.2/linh kiện)")
        print(df.groupby("component")["helpful"].sum())
        return
    from ..rag.hybrid_retriever import HybridRetriever

    R = HybridRetriever(cfg)
    rows = []
    for c in read_list(cfg["paths"]["test_exp2"])[:a.n]:
        for h in R.retrieve(c, DENSE_QUERY.format(component=c)):
            rows.append({
                "component": c,
                "rank": h.rank,
                "pid": h.pid,
                "title": h.title,
                "score": h.score,
                "bm25_rank": h.bm25_rank,
                "dense_rank": h.dense_rank,
                "text": h.text,
                "helpful": "",
            })
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("Đã ghi", out, "-> điền cột helpful (1/0) rồi chạy lại với --score")


if __name__ == "__main__":
    main()

