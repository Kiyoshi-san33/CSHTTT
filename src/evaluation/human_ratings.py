"""Task 3, Task 4 (điểm Likert 1-5) và Task 5 (vị trí xếp hạng trung bình, 0 = tốt nhất)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def mean_rating(item_ids: list[str], anns: dict[str, dict[str, dict]], field: str) -> float:
    vals = [
        float(anns[a][iid][field])
        for iid in item_ids
        for a in anns
        if iid in anns[a] and field in anns[a][iid]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def rank_positions(rankings: dict[str, dict[str, list[str]]]) -> dict[str, list[int]]:
    """rankpos[run_id] = danh sách vị trí (0 = tốt nhất) qua các annotator."""
    out: dict[str, list[int]] = defaultdict(list)
    for comp_map in rankings.values():
        for order in comp_map.values():
            for pos, oid in enumerate(order):
                out[oid].append(pos)
    return dict(out)


def mean_rank_position(item_ids: list[str], rankpos: dict[str, list[int]]) -> float:
    vals = [x for iid in item_ids for x in rankpos.get(iid, [])]
    return float(np.mean(vals)) if vals else float("nan")

