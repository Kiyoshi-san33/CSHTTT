"""Trang 4: Tổng hợp kết quả đánh giá (tách biệt Human Evaluation vs LLM-as-Judge vs Agreement)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.sidebar import load_active_config
from src.core.paths import results_dir
from src.evaluation.report import generate_report
from src.storage import repository as R

cfg = load_active_config()
st.header("📊 Kết quả Đánh giá & Độ đồng thuận")

ann_list = R.annotator_info(cfg)
if ann_list:
    st.subheader("Danh sách Annotator trong hệ thống")
    st.dataframe(pd.DataFrame(ann_list), use_container_width=True, hide_index=True)
else:
    st.info("Chưa có nhãn đánh giá nào. Hãy gán nhãn ở trang **Annotation** hoặc chạy `python -m src.evaluation.llm_judge --exp exp2`.")

if st.button("🔄 Tính toán & Cập nhật Báo cáo", type="primary"):
    report_md, _ = generate_report(cfg)
    st.session_state["report_md"] = report_md

report_path = results_dir(cfg, "metrics") / "report.md"
report_md = st.session_state.get("report_md") or (
    report_path.read_text(encoding="utf-8") if report_path.exists() else None
)
if report_md:
    st.markdown(report_md)

plots_dir = results_dir(cfg, "plots")
p1, p2 = plots_dir / "fig_exp1.png", plots_dir / "fig_exp2.png"
if p1.exists() or p2.exists():
    st.subheader("Biểu đồ")
    c1, c2 = st.columns(2)
    if p1.exists():
        c1.image(str(p1), caption="Experiment I")
    if p2.exists():
        c2.image(str(p2), caption="Experiment II")

dis_path = results_dir(cfg, "metrics") / "human_llm_disagreements.csv"
if dis_path.exists():
    st.subheader("Các trường hợp bất đồng giữa Người và Giám khảo LLM")
    df_dis = pd.read_csv(dis_path)
    st.dataframe(df_dis, use_container_width=True, hide_index=True)
    st.download_button("⬇ Tải CSV bất đồng (Human vs LLM)",
                       data=dis_path.read_bytes(),
                       file_name="human_llm_disagreements.csv",
                       mime="text/csv")

