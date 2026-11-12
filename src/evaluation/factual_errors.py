"""Task 2 (Experiment II): Tỉ lệ output chứa lỗi sự thật / hallucination."""
from __future__ import annotations

import numpy as np


def error_rate_mean(item_ids: list[str], anns: dict[str, dict[str, dict]]) -> float:
    """Trung bình theo từng cặp (output, annotator) có đánh dấu >= 1 câu sai."""
    vals = [
        bool((anns[a].get(iid) or {}).get("incorrect_sentences"))
        for iid in item_ids
        for a in anns
        if iid in anns[a]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def error_rate_any(item_ids: list[str], anns: dict[str, dict[str, dict]]) -> float:
    """Tỉ lệ output bị ít nhất 1 annotator đánh dấu >= 1 câu sai."""
    if not item_ids or not anns:
        return float("nan")
    vals = [
        any(bool((anns[a].get(iid) or {}).get("incorrect_sentences")) for a in anns if iid in anns[a])
        for iid in item_ids
    ]
    return float(np.mean(vals)) if vals else float("nan")

