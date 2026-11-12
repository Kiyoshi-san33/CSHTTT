"""Kiểm định thống kê Wilcoxon signed-rank cho Experiment II."""
from __future__ import annotations

import numpy as np


def wilcoxon_comparisons(items: list[dict], anns: dict[str, dict[str, dict]],
                         base_method: str = "norag") -> list[dict]:
    from scipy.stats import wilcoxon

    by_key = {(it["component"], it["model"], it["method"]): it["id"] for it in items}
    methods = sorted({it["method"] for it in items})
    models = sorted({it["model"] for it in items})
    comps = sorted({it["component"] for it in items})
    names = list(anns)
    out = []

    for me in methods:
        if me == base_method:
            continue
        for field in ("material_rating", "explanation_rating"):
            x, y = [], []
            for c in comps:
                for mo in models:
                    id_base = by_key.get((c, mo, base_method))
                    id_me = by_key.get((c, mo, me))
                    if not id_base or not id_me:
                        continue
                    for n in names:
                        rb = anns[n].get(id_base)
                        rm = anns[n].get(id_me)
                        if rb and rm and field in rb and field in rm:
                            x.append(float(rb[field]))
                            y.append(float(rm[field]))
            if len(x) > 5 and any(a != b for a, b in zip(x, y)):
                _, pv = wilcoxon(x, y)
                out.append({
                    "so sánh": f"{base_method} vs {me}",
                    "chỉ số": field,
                    "n cặp": len(x),
                    f"TB {base_method}": f"{np.mean(x):.2f}",
                    "TB phương pháp": f"{np.mean(y):.2f}",
                    "p-value": f"{pv:.4f}",
                })
    return out

