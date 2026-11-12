"""Các hàm tách danh sách vật liệu, câu hỏi kiểm chứng và tách câu từ đầu ra LLM."""
from __future__ import annotations

import re

_ITEM = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*\**\s*([^:\n*]{2,80}?)\s*\**\s*[:–—-]\s", re.M)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*])|\n+")


def parse_explained(raw: str) -> list[str]:
    """Lấy danh sách vật liệu từ câu trả lời có giải thích dạng '1. **Steel**: ...'."""
    seen, out = set(), []
    for m in _ITEM.finditer(raw or ""):
        name = m.group(1).strip(" *_")
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def parse_list(raw: str, k: int = 5) -> list[str]:
    """Lấy tối đa k vật liệu từ danh sách phân tách bằng dấu phẩy (Experiment I)."""
    lines = [ln.strip() for ln in (raw or "").strip().splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    first = re.sub(r"^(materials?|answer)\s*:\s*", "", first, flags=re.I)
    items = [re.sub(r"[\*\.\s]+$", "", x).strip(" *-") for x in first.split(",")]
    seen, out = set(), []
    for it in items:
        if it and it.lower() not in seen:
            seen.add(it.lower())
            out.append(it)
    return out[:k]


def parse_questions(raw: str, k: int = 5) -> list[str]:
    """Tách tối đa k câu hỏi kiểm chứng từ đầu ra bước lập kế hoạch của CoVe."""
    qs = [re.sub(r"^\s*(?:Q?\d+[\.\):]|[-*•])\s*", "", l).strip().strip('"')
          for l in (raw or "").splitlines()]
    return [q for q in qs if q.endswith("?")][:k]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text or "") if s and s.strip()]

