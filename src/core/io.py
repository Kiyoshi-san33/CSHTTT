"""Đọc/ghi file văn bản, JSONL dùng chung."""
from __future__ import annotations

import json
from pathlib import Path

from .paths import resolve


def read_list(path: str | Path) -> list[str]:
    """Đọc danh sách (mỗi dòng 1 mục, bỏ dòng trống và dòng bắt đầu bằng #)."""
    path = resolve(path, mkdir=False)
    items = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(line)
    return items


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n

