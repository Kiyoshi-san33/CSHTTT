"""Tính số phiếu plausible cho từng vật liệu và chỉ số Coverage@n.

Định nghĩa trong paper:
- votes[item_id][j] = số annotator đánh dấu vật liệu thứ j là plausible (bỏ qua annotator chọn dont_know).
- Coverage@n = tỉ lệ output có ÍT NHẤT 1 vật liệu nhận được >= n phiếu đồng ý là plausible.
"""
from __future__ import annotations

import numpy as np


def votes_for(item_ids: list[str], anns: dict[str, dict[str, dict]], n_mat: dict[str, int]) -> dict[str, np.ndarray]:
    """votes[id] = mảng số annotator cho là plausible với từng vật liệu của output id."""
    out: dict[str, np.ndarray] = {}
    for iid in item_ids:
        k = int(n_mat.get(iid, 0))
        v = np.zeros(k, dtype=int)
        for a in anns.values():
            rec = a.get(iid) or {}
            pl = rec.get("plausible")
            if pl is not None and not rec.get("dont_know"):
                arr = np.array((list(pl) + [False] * k)[:k], dtype=int)
                v += arr
        out[iid] = v
    return out


def coverage_at_n(votes: dict[str, np.ndarray], n: int) -> float:
    vs = [v for v in votes.values() if len(v) > 0]
    if not vs:
        return float("nan")
    return float(np.mean([bool((v >= n).any()) for v in vs]))


def first_last_plausible(votes: dict[str, np.ndarray], n: int, which: str = "first") -> float:
    vs = [v for v in votes.values() if len(v) > 0]
    if not vs:
        return float("nan")
    idx = 0 if which == "first" else -1
    return float(np.mean([bool(v[idx] >= n) for v in vs]))

