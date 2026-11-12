"""Tính Precision@n: tỉ lệ vật liệu dự đoán nhận được >= n phiếu đồng ý là plausible."""
from __future__ import annotations

import numpy as np

from .coverage import coverage_at_n


def precision_at_n(votes: dict[str, np.ndarray], n: int) -> float:
    vs = [v for v in votes.values() if len(v) > 0]
    if not vs:
        return float("nan")
    all_v = np.concatenate(vs)
    return float(np.mean(all_v >= n)) if len(all_v) else float("nan")


def coverage_precision(votes: dict[str, np.ndarray], n: int) -> tuple[float, float]:
    return coverage_at_n(votes, n), precision_at_n(votes, n)

