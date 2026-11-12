"""CLI xây dựng chỉ mục BM25 và FAISS (hỗ trợ cả profile adapted=BGE và reproduction=DPR).

    python -m src.rag.indexing --all
    python -m src.rag.indexing --sparse
    python -m src.rag.indexing --dense --profile adapted
    python -m src.rag.indexing --dense --profile reproduction
"""
from __future__ import annotations

import argparse

from ..core.config import load_config
from .bm25_retriever import build_bm25_index
from .chunking import load_passages
from .dense_retriever import build_dense_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", action="store_true", help="chỉ tạo chỉ mục BM25")
    ap.add_argument("--dense", action="store_true", help="chỉ tạo chỉ mục FAISS")
    ap.add_argument("--all", action="store_true", help="tạo cả BM25 và FAISS")
    ap.add_argument("--profile", choices=["adapted", "reproduction"], default=None)
    a = ap.parse_args()
    cfg = load_config(profile=a.profile)
    passages = load_passages(cfg)
    print(f"Đã nạp {len(passages):,} passage | profile={cfg['active_profile']} "
          f"(dense_backend={cfg['retrieval']['dense_backend']})")
    do_all = a.all or (not a.sparse and not a.dense)
    if a.sparse or do_all:
        m = build_bm25_index(cfg, passages)
        print("BM25 xong:", m)
    if a.dense or do_all:
        m = build_dense_index(cfg, passages)
        print("FAISS xong:", m)


if __name__ == "__main__":
    main()

