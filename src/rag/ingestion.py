"""CLI tổng hợp cho bước nạp và trích xuất kho ngữ liệu Wikipedia.

    python -m src.rag.ingestion download
    python -m src.rag.ingestion extract [--limit-tasks 50]
    python -m src.rag.ingestion all
"""
from __future__ import annotations

import argparse

from ..core.config import load_config
from .wiki.download import run_download
from .wiki.extract import run_extract


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download", "extract", "all"], nargs="?", default="extract")
    ap.add_argument("--date", default=cfg["wiki"]["dump_date"])
    ap.add_argument("--mirror", default="https://dumps.wikimedia.org")
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--workers", type=int, default=cfg["wiki"]["workers"])
    a = ap.parse_args()
    if a.action in ("download", "all"):
        run_download(cfg, a.date, a.mirror)
    if a.action in ("extract", "all"):
        run_extract(cfg, a.limit_tasks, a.workers)


if __name__ == "__main__":
    main()

