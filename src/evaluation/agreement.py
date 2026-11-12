"""Độ đồng thuận giữa các annotator (IAA Eq.2, Krippendorff's alpha, Kendall's tau)
và đối chiếu riêng giữa nhóm Người (Human) và Giám khảo LLM (LLM-as-Judge).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def iaa_eq2(anns: dict[str, dict[str, dict]], item_ids: list[str],
            n_mat: dict[str, int]) -> tuple[float, pd.DataFrame]:
    """IAA theo Eq.(2) của paper: tỉ lệ quyết định plausible trùng nhau trung bình theo cặp annotator."""
    names = list(anns)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in itertools.combinations(names, 2):
        same = tot = 0
        for iid in item_ids:
            A, B = anns[a].get(iid), anns[b].get(iid)
            if not A or not B or A.get("dont_know") or B.get("dont_know"):
                continue
            k = int(n_mat.get(iid, 0))
            if k <= 0:
                continue
            pa = (list(A.get("plausible") or []) + [False] * k)[:k]
            pb = (list(B.get("plausible") or []) + [False] * k)[:k]
            same += sum(bool(x) == bool(y) for x, y in zip(pa, pb))
            tot += k
        mat.loc[a, b] = mat.loc[b, a] = same / tot if tot else np.nan
    vals = [mat.loc[a, b] for a, b in itertools.combinations(names, 2)]
    mean_val = float(np.nanmean(vals)) if vals and not np.all(np.isnan(vals)) else float("nan")
    return mean_val, mat


def krippendorff_alpha(anns: dict[str, dict[str, dict]], item_ids: list[str], field: str) -> float:
    try:
        import krippendorff
    except ImportError:
        return float("nan")
    if len(anns) < 2 or not item_ids:
        return float("nan")
    data = [[(anns[a].get(i) or {}).get(field, np.nan) for i in item_ids] for a in anns]
    arr = np.array(data, dtype=float)
    if arr.shape[0] < 2 or np.all(np.isnan(arr)):
        return float("nan")
    try:
        return float(krippendorff.alpha(reliability_data=arr, level_of_measurement="ordinal"))
    except Exception:  # noqa: BLE001
        return float("nan")


def mean_kendall_tau(rankings: dict[str, dict[str, list[str]]]) -> float:
    from scipy.stats import kendalltau

    taus = []
    comps = set().union(*[set(r) for r in rankings.values()]) if rankings else set()
    for c in comps:
        rs = [r[c] for r in rankings.values() if c in r]
        for r1, r2 in itertools.combinations(rs, 2):
            common = sorted(set(r1) & set(r2))
            if len(common) < 2:
                continue
            t, _ = kendalltau([r1.index(i) for i in common], [r2.index(i) for i in common])
            if not np.isnan(t):
                taus.append(float(t))
    return float(np.mean(taus)) if taus else float("nan")


def human_llm_agreement(items: list[dict], human_anns: dict[str, dict[str, dict]],
                        llm_anns: dict[str, dict[str, dict]], min_human_agree: int = 2) -> dict:
    """So sánh kết quả đồng thuận của nhóm Người (>= min_human_agree) với từng LLM-judge.

    Không gộp phiếu LLM vào phiếu người; chỉ báo cáo mức độ khớp và các trường hợp bất đồng.
    """
    from sklearn.metrics import cohen_kappa_score

    by_id = {it["id"]: it for it in items}
    common_ids = [
        iid for iid in by_id
        if any(iid in h for h in human_anns.values()) and any(iid in j for j in llm_anns.values())
    ]
    if not common_ids or not human_anns or not llm_anns:
        return {"n_common_items": 0, "per_judge": [], "disagreements": []}

    eff_n = min(min_human_agree, len(human_anns))
    per_judge = []
    disagreements = []

    for jname, jmap in llm_anns.items():
        h_vec, j_vec = [], []
        h_r3, j_r3, h_r4, j_r4 = [], [], [], []
        for iid in common_ids:
            if iid not in jmap:
                continue
            it = by_id[iid]
            mats = it.get("materials") or []
            k = len(mats)
            # Đa số người trên từng vật liệu
            h_votes = np.zeros(k, dtype=int)
            n_voters = 0
            for hmap in human_anns.values():
                rec = hmap.get(iid)
                if rec and not rec.get("dont_know"):
                    n_voters += 1
                    pl = (list(rec.get("plausible") or []) + [False] * k)[:k]
                    h_votes += np.array(pl, dtype=int)
            if n_voters == 0:
                continue
            thresh = min(eff_n, n_voters)
            h_pl = [bool(v >= thresh) for v in h_votes]
            j_pl = [bool(x) for x in (list(jmap[iid].get("plausible") or []) + [False] * k)[:k]]
            for idx_m, (hm, jm) in enumerate(zip(h_pl, j_pl)):
                h_vec.append(int(hm))
                j_vec.append(int(jm))
                if hm != jm:
                    disagreements.append({
                        "item_id": iid,
                        "component": it["component"],
                        "model": it.get("model"),
                        "method": it.get("method"),
                        "material": mats[idx_m],
                        "human_plausible": hm,
                        "human_votes": f"{int(h_votes[idx_m])}/{n_voters}",
                        "judge": jname,
                        "judge_plausible": jm,
                    })
            # Rating Task 3 / Task 4 (Exp II)
            if "material_rating" in jmap[iid]:
                hr3 = [hmap[iid]["material_rating"] for hmap in human_anns.values()
                       if iid in hmap and "material_rating" in hmap[iid]]
                hr4 = [hmap[iid]["explanation_rating"] for hmap in human_anns.values()
                       if iid in hmap and "explanation_rating" in hmap[iid]]
                if hr3:
                    h_r3.append(float(np.mean(hr3)))
                    j_r3.append(float(jmap[iid]["material_rating"]))
                if hr4:
                    h_r4.append(float(np.mean(hr4)))
                    j_r4.append(float(jmap[iid]["explanation_rating"]))

        acc = float(np.mean(np.array(h_vec) == np.array(j_vec))) if h_vec else float("nan")
        kappa = float("nan")
        if len(h_vec) >= 2 and len(set(h_vec)) > 1 and len(set(j_vec)) > 1:
            kappa = float(cohen_kappa_score(h_vec, j_vec))
        mae_r3 = float(np.mean(np.abs(np.array(h_r3) - np.array(j_r3)))) if h_r3 else float("nan")
        mae_r4 = float(np.mean(np.abs(np.array(h_r4) - np.array(j_r4)))) if h_r4 else float("nan")
        per_judge.append({
            "judge": jname,
            "n_materials": len(h_vec),
            "agreement_pct": acc,
            "cohen_kappa": kappa,
            "mae_material_rating": mae_r3,
            "mae_explanation_rating": mae_r4,
        })

    return {"n_common_items": len(common_ids), "per_judge": per_judge, "disagreements": disagreements}

