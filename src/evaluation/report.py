"""Tổng hợp báo cáo thực nghiệm từ SQLite — tách bạch hoàn toàn:
  1. Đánh giá chính thức bởi Người (Human Evaluation — primary)
  2. Đánh giá bổ sung bởi Giám khảo LLM (LLM-as-Judge — supplementary)
  3. Độ đồng thuận giữa Người và Giám khảo LLM + danh sách bất đồng

    python -m src.evaluation.report
Kết quả:
    results/metrics/report.md, results/metrics/metrics.json,
    results/metrics/human_llm_disagreements.csv, results/plots/fig_*.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from ..core.config import load_config
from ..core.paths import results_dir
from ..storage import repository as R
from .agreement import human_llm_agreement, iaa_eq2, krippendorff_alpha, mean_kendall_tau
from .coverage import first_last_plausible, votes_for
from .factual_errors import error_rate_any, error_rate_mean
from .human_ratings import mean_rank_position, mean_rating, rank_positions
from .precision import coverage_precision
from .significance import wilcoxon_comparisons

PRIOR = pd.DataFrame(
    {"COVERAGE@1": ["73%", "93%"], "COVERAGE@3": ["40%", "73%"], "PRECISION@1": ["45%", "62%"],
     "PRECISION@3": ["14%", "28%"]}, index=["Schlipf (2022)", "Eichel et al. (2023)"])
PAPER_EXP1 = {"COVERAGE@1": "100%", "COVERAGE@3": "100%", "PRECISION@1": "97%", "PRECISION@3": "71%"}


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def _eval_exp1_group(items: list[dict], anns: dict[str, dict[str, dict]], label_title: str) -> tuple[list[str], dict]:
    if not items or not anns:
        return [f"### {label_title}\n\n_Chưa có nhãn cho nhóm này._\n"], {}
    ids = [i["id"] for i in items if all(i["id"] in anns[n] for n in anns)]
    if not ids:
        ids = [i["id"] for i in items if any(i["id"] in anns[n] for n in anns)]
    n_mat = {i["id"]: len(i["materials"]) for i in items}
    votes = votes_for(ids, anns, n_mat)
    N = len(anns)
    rows = {n: coverage_precision(votes, n) for n in range(1, N + 1)}
    ours = {
        "COVERAGE@1": pct(rows.get(1, (np.nan, np.nan))[0]),
        "COVERAGE@3": pct(rows.get(3, (np.nan, np.nan))[0]),
        "PRECISION@1": pct(rows.get(1, (np.nan, np.nan))[1]),
        "PRECISION@3": pct(rows.get(3, (np.nan, np.nan))[1]),
    }
    tbl = pd.concat([
        PRIOR,
        pd.DataFrame([PAPER_EXP1, ours],
                     index=["Paper (Mixtral-8x22B)", f"Adapted Baseline – GPT-OSS 120B ({label_title})"])
    ])
    iaa, iaa_mat = iaa_eq2(anns, ids, n_mat)
    md = [
        f"### {label_title} ({len(ids)} linh kiện, {N} annotator: {', '.join(anns)})\n",
        "#### Bảng 1 – So sánh với nghiên cứu trước (Adapted Baseline: GPT-OSS 120B, Few-shot Appendix A)\n",
        tbl.to_markdown(), "\n",
        "#### Coverage@n / Precision@n theo ngưỡng số annotator đồng ý\n",
        pd.DataFrame({f"@{n}": [pct(c), pct(pr)] for n, (c, pr) in rows.items()},
                     index=["COVERAGE", "PRECISION"]).to_markdown(), "\n",
        f"#### IAA (Eq. 2) = **{iaa:.2f}** (paper: 0.69)\n",
        (iaa_mat.round(2).to_markdown() if N >= 2 else "_Cần >= 2 annotator để tính ma trận IAA._"), "\n",
    ]
    info = {
        "n_items": len(ids),
        "annotators": list(anns),
        "iaa": iaa,
        "coverage": {n: rows[n][0] for n in rows},
        "precision": {n: rows[n][1] for n in rows},
    }
    return md, info


def _eval_exp2_group(cfg: dict, items: list[dict], anns: dict[str, dict[str, dict]],
                     rankings: dict[str, dict[str, list[str]]], label_title: str) -> tuple[list[str], dict]:
    if not items or not anns:
        return [f"### {label_title}\n\n_Chưa có nhãn cho nhóm này._\n"], {}
    ids = [i["id"] for i in items if all(i["id"] in anns[n] for n in anns)]
    if not ids:
        ids = [i["id"] for i in items if any(i["id"] in anns[n] for n in anns)]
    by_id = {i["id"]: i for i in items}
    n_mat = {i["id"]: len(i["materials"]) for i in items}
    N = len(anns)
    target_n = int(cfg.get("evaluation", {}).get("exp2", {}).get("agreement_n", 2))
    eff_n = min(target_n, N)  # @2 khi có >=2 người chấm; @1 nếu mới có 1 annotator
    votes = votes_for(ids, anns, n_mat)

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i in ids:
        groups[(by_id[i]["method"], by_id[i]["model"])].append(i)
    models = sorted({m for _, m in groups})
    order_key = lambda m: (0 if m.startswith("norag") else (1 if m.startswith("rag") else 2), m)
    methods = sorted({m for m, _ in groups}, key=order_key)
    rpos = rank_positions(rankings)

    def cell(fn):
        df = pd.DataFrame(index=methods, columns=models + ["Average"], dtype=object)
        for me in methods:
            all_ids = []
            for mo in models:
                g = groups.get((me, mo), [])
                df.loc[me, mo] = fn(g) if g else np.nan
                all_ids += g
            df.loc[me, "Average"] = fn(all_ids) if all_ids else np.nan
        return df

    cov_fn = lambda g, n: coverage_precision({i: votes[i] for i in g}, n)[0]
    prec_fn = lambda g, n: coverage_precision({i: votes[i] for i in g}, n)[1]

    t_cov = cell(lambda g: cov_fn(g, eff_n)).map(pct)
    t_prec = cell(lambda g: prec_fn(g, eff_n)).map(pct)
    t_err_mean = cell(lambda g: error_rate_mean(g, anns)).map(pct)
    t_err_any = cell(lambda g: error_rate_any(g, anns)).map(pct)
    t_r3 = cell(lambda g: mean_rating(g, anns, "material_rating")).map(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    t_r4 = cell(lambda g: mean_rating(g, anns, "explanation_rating")).map(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    t_rk = cell(lambda g: mean_rank_position(g, rpos)).map(
        lambda x: "—" if pd.isna(x) else f"{x:.2f}")

    ns_all = sorted({1, eff_n, N})
    t_all_n = pd.DataFrame(
        {f"COV@{n}": cell(lambda g, n=n: cov_fn(g, n))["Average"].map(pct) for n in ns_all}
        | {f"PREC@{n}": cell(lambda g, n=n: prec_fn(g, n))["Average"].map(pct) for n in ns_all}
    )
    t_fl = pd.DataFrame({
        "First material": cell(lambda g: first_last_plausible({i: votes[i] for i in g}, eff_n, "first"))["Average"].map(pct),
        "Last material": cell(lambda g: first_last_plausible({i: votes[i] for i in g}, eff_n, "last"))["Average"].map(pct),
    })

    iaa, _ = iaa_eq2(anns, ids, n_mat)
    a3 = krippendorff_alpha(anns, ids, "material_rating")
    a4 = krippendorff_alpha(anns, ids, "explanation_rating")
    tau = mean_kendall_tau(rankings)
    sig = wilcoxon_comparisons([by_id[i] for i in ids], anns, base_method="norag")

    note = "" if N >= target_n else f" _(hiện có {N} annotator nên tính @{eff_n}; cần >= {target_n} để tính @{target_n})_"
    md = [
        f"### {label_title} ({len(ids)} output, {N} annotator: {', '.join(anns)}; ngưỡng đồng ý = @{eff_n}){note}\n",
        f"#### Bảng 2a – Task 1: COVERAGE@{eff_n}\n", t_cov.to_markdown(), "\n",
        f"#### Bảng 2b – Task 1: PRECISION@{eff_n}\n", t_prec.to_markdown(), "\n",
        "#### Bảng 3a – Task 2: Tỉ lệ lỗi sự thật / Hallucination Rate (trung bình theo annotator)\n",
        t_err_mean.to_markdown(), "\n",
        "_Theo tiêu chí 'ít nhất 1 annotator đánh dấu lỗi':_\n", t_err_any.to_markdown(), "\n",
        "#### Bảng 4a – Task 3: Điểm chất lượng dự đoán vật liệu (1–5)\n", t_r3.to_markdown(), "\n",
        "#### Bảng 4b – Task 4: Điểm chất lượng lời giải thích (1–5)\n", t_r4.to_markdown(), "\n",
        "#### Bảng 3b – Task 5: Vị trí xếp hạng trung bình (0 = tốt nhất, thấp hơn = tốt hơn)\n",
        t_rk.to_markdown(), "\n",
        "#### Bảng 7 – Coverage@n & Precision@n theo các ngưỡng n\n", t_all_n.to_markdown(), "\n",
        f"#### Bảng 9 – PRECISION@{eff_n} của vật liệu đầu vs cuối danh sách\n", t_fl.to_markdown(), "\n",
        "#### Bảng 6 – Độ đồng thuận giữa các annotator trong nhóm\n",
        pd.DataFrame({
            "Chỉ số": ["IAA Eq.2 (Task 1)", "Krippendorff α (Task 3)", "Krippendorff α (Task 4)", "Kendall τ (Task 5)"],
            "Thực nghiệm": [f"{iaa:.2f}", f"{a3:.2f}", f"{a4:.2f}", f"{tau:.2f}"],
            "Paper": ["0.63", "-0.22", "-0.23", "0.24"],
        }).to_markdown(index=False), "\n",
        "#### Kiểm định Wilcoxon (No-RAG so với RAG / CoVe)\n",
        (pd.DataFrame(sig).to_markdown(index=False) if sig else "_Chưa đủ cặp dữ liệu để kiểm định._"), "\n",
    ]
    info = {
        "n_outputs": len(ids),
        "annotators": list(anns),
        "agreement_n": eff_n,
        "iaa": iaa,
        "alpha_task3": a3,
        "alpha_task4": a4,
        "kendall_tau": tau,
        "coverage": t_cov.to_dict(),
        "precision": t_prec.to_dict(),
        "factual_error_mean": t_err_mean.to_dict(),
        "factual_error_any": t_err_any.to_dict(),
        "material_rating": t_r3.to_dict(),
        "explanation_rating": t_r4.to_dict(),
        "avg_rank": t_rk.to_dict(),
        "wilcoxon": sig,
    }
    return md, info


def _save_plots(cfg: dict, exp1_info: dict, exp2_items: list[dict],
                exp2_anns: dict[str, dict[str, dict]], exp2_rankings: dict[str, dict[str, list[str]]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = results_dir(cfg, "plots")
    cov = exp1_info.get("coverage") or {}
    prec = exp1_info.get("precision") or {}
    if cov:
        ns = sorted(cov)
        x = np.arange(len(ns))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(x - 0.2, [100 * (cov[n] or 0) for n in ns], 0.4, label="COVERAGE@n")
        ax.bar(x + 0.2, [100 * (prec[n] or 0) for n in ns], 0.4, label="PRECISION@n")
        ax.set_xticks(x, [f"n={n}" for n in ns])
        ax.set_ylabel("%")
        ax.set_ylim(0, 105)
        ax.set_title("Exp I: Adapted Baseline (GPT-OSS 120B)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "fig_exp1.png", dpi=200)
        plt.close(fig)

    if exp2_items and exp2_anns:
        ids = [i["id"] for i in exp2_items if any(i["id"] in exp2_anns[n] for n in exp2_anns)]
        by_id = {i["id"]: i for i in exp2_items}
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for i in ids:
            groups[(by_id[i]["method"], by_id[i]["model"])].append(i)
        models = sorted({m for _, m in groups})
        methods = sorted({m for m, _ in groups})
        rpos = rank_positions(exp2_rankings)
        if models and methods:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
            specs = [
                ("Task 3: material rating ↑", lambda g: mean_rating(g, exp2_anns, "material_rating")),
                ("Task 4: explanation rating ↑", lambda g: mean_rating(g, exp2_anns, "explanation_rating")),
                ("Task 5: avg rank ↓", lambda g: mean_rank_position(g, rpos)),
            ]
            for ax, (title, fn) in zip(axes, specs):
                x = np.arange(len(methods))
                w = 0.8 / max(len(models), 1)
                for k, mo in enumerate(models):
                    vals = [fn(groups.get((me, mo), [])) for me in methods]
                    vals = [0.0 if (v is None or np.isnan(v)) else v for v in vals]
                    ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=mo)
                ax.set_xticks(x, methods, rotation=15)
                ax.set_title(title)
                if "rating" in title:
                    ax.set_ylim(1, 5)
            axes[0].legend()
            fig.tight_layout()
            fig.savefig(plots / "fig_exp2.png", dpi=200)
            plt.close(fig)


def generate_report(cfg: dict) -> tuple[str, dict]:
    md: list[str] = ["# Báo cáo thực nghiệm dự đoán vật liệu linh kiện ô tô\n"]
    metrics: dict = {"active_profile": cfg.get("active_profile")}

    stats_file = results_dir(cfg, "metrics") / "corpus_stats.json"
    if stats_file.exists():
        s = json.loads(stats_file.read_text(encoding="utf-8"))
        md += [
            "## 1. Kho ngữ liệu Wikipedia\n",
            pd.DataFrame(
                {"Paper (Wagner et al., 2026)": [32368, 29425, 2943, 4233, 458972],
                 "Thực nghiệm": [s["articles"], s["criterion_1"], s["criterion_2"],
                                 s["avg_article_words"], s["passages"]]},
                index=["Số bài viết", "Tiêu chí (1)", "Tiêu chí (2)", "Độ dài TB (từ)", "Số passage"]
            ).to_markdown(), "\n",
        ]
        metrics["corpus"] = s

    # --- Experiment I ---
    md.append("## 2. Experiment I – Adapted Baseline (GPT-OSS 120B, Few-shot)\n")
    md.append("> **Lưu ý phương pháp:** Sử dụng nguyên văn prompt Few-shot ở Phụ lục A nhưng chạy trên "
              "`openai/gpt-oss-120b` (Adapted Baseline), không phải bản tái lập trực tiếp `Mixtral-8x22B`.\n")
    e1_items = R.exp1_items(cfg)
    e1_human = R.load_labels(cfg, "exp1", kind="human")
    e1_llm = R.load_labels(cfg, "exp1", kind="llm")
    m_h1, info_h1 = _eval_exp1_group(e1_items, e1_human, "2.1 Đánh giá chính thức bởi Người (Human Evaluation)")
    m_j1, info_j1 = _eval_exp1_group(e1_items, e1_llm, "2.2 Đánh giá bổ sung bởi Giám khảo LLM (LLM-as-Judge)")
    md += m_h1 + m_j1
    hl1 = human_llm_agreement(e1_items, e1_human, e1_llm, min_human_agree=2)
    if hl1["per_judge"]:
        md += ["### 2.3 Độ đồng thuận Người ↔ Giám khảo LLM (Experiment I)\n",
               pd.DataFrame(hl1["per_judge"]).to_markdown(index=False), "\n"]
    metrics["exp1"] = {"human": info_h1, "llm_judge": info_j1, "human_llm_agreement": hl1}

    # --- Experiment II ---
    md.append("## 3. Experiment II – Dự đoán kèm giải thích (2 LLM × No-RAG / RAG / CoVe)\n")
    md.append("> **Quy tắc tách nguồn nhãn:** Các chỉ số chính thức (Coverage@2, Precision@2, Factual Error, "
              "Human Rating 1–5, Kendall's τ, Krippendorff's α) chỉ tính từ phiếu của **Người**. "
              "Kết quả của **LLM-as-Judge** (`qwen/qwen3.8-27b`) được trình bày ở mục riêng.\n")
    e2_items = R.exp2_items(cfg)
    e2_human = R.load_labels(cfg, "exp2", kind="human")
    e2_human_rk = R.load_rankings(cfg, kind="human")
    e2_llm = R.load_labels(cfg, "exp2", kind="llm")
    e2_llm_rk = R.load_rankings(cfg, kind="llm")

    m_h2, info_h2 = _eval_exp2_group(cfg, e2_items, e2_human, e2_human_rk,
                                     "3.1 Đánh giá chính thức bởi Người (Primary Human Evaluation)")
    m_j2, info_j2 = _eval_exp2_group(cfg, e2_items, e2_llm, e2_llm_rk,
                                     "3.2 Đánh giá bổ sung bởi Giám khảo LLM (Supplementary LLM-as-Judge)")
    md += m_h2 + m_j2

    hl2 = human_llm_agreement(e2_items, e2_human, e2_llm,
                              min_human_agree=int(cfg.get("evaluation", {}).get("exp2", {}).get("agreement_n", 2)))
    if hl2["per_judge"]:
        md += ["### 3.3 Độ đồng thuận Người ↔ Giám khảo LLM (Experiment II)\n",
               pd.DataFrame(hl2["per_judge"]).to_markdown(index=False), "\n"]
    metrics["exp2"] = {"human": info_h2, "llm_judge": info_j2, "human_llm_agreement": hl2}

    # Lưu danh sách các trường hợp bất đồng giữa Người và LLM-judge
    dis = (hl1.get("disagreements") or []) + (hl2.get("disagreements") or [])
    mdir = results_dir(cfg, "metrics")
    if dis:
        pd.DataFrame(dis).to_csv(mdir / "human_llm_disagreements.csv", index=False, encoding="utf-8-sig")

    # Biểu đồ: ưu tiên vẽ từ nhãn người; nếu chưa có nhãn người thì vẽ từ nhãn LLM-judge
    _save_plots(
        cfg,
        info_h1 or info_j1,
        e2_items,
        e2_human or e2_llm,
        e2_human_rk if e2_human else e2_llm_rk,
    )

    report_txt = "\n".join(md)
    (mdir / "report.md").write_text(report_txt, encoding="utf-8")
    (mdir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str, ensure_ascii=False),
                                       encoding="utf-8")
    return report_txt, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["adapted", "reproduction"], default=None)
    a = ap.parse_args()
    cfg = load_config(profile=a.profile)
    txt, _ = generate_report(cfg)
    print(txt)
    print("\n->", results_dir(cfg, "metrics") / "report.md")


if __name__ == "__main__":
    main()

