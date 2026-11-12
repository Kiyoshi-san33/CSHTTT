"""Trang 2: Quản lý thực nghiệm, theo dõi tiến độ ma trận 2x3 + Exp I, duyệt từng run và xuất dữ liệu."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.passage_view import render_calls, render_passages
from app.components.sidebar import load_active_config
from src.storage import export as E, repository as R

cfg = load_active_config()
st.header("🧪 Tiến độ & Nhật ký Thực nghiệm")

prog = R.experiment_progress(cfg)
if not prog:
    st.info("Chưa có lượt chạy nào trong cơ sở dữ liệu. Chạy lệnh:\n"
            "`python -m src.experiments.runner --exp exp2 --limit 3`")
else:
    st.subheader("Tổng hợp theo Experiment")
    st.dataframe(pd.DataFrame(prog), use_container_width=True, hide_index=True)

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("📦 Xuất toàn bộ SQLite ra JSONL / CSV"):
        counts = E.export_all(cfg)
        st.success(f"Đã xuất: {counts}")

st.divider()
st.subheader("Tra cứu chi tiết từng Run")
exp_choice = st.radio("Chọn thí nghiệm", ["exp2", "exp1"], horizontal=True)
runs = R.list_runs(cfg, exp_choice, include_dev=True, status=None)
if runs:
    run_ids = [r["id"] for r in runs]
    selected_id = st.selectbox("Chọn Run ID", run_ids)
    detail = R.run_detail(cfg, selected_id)
    if detail:
        st.markdown(f"**Linh kiện:** `{detail['component']}` | **Model:** `{detail['model']}` "
                    f"(`{detail.get('model_id')}`) | **Phương pháp:** `{detail['method']}` "
                    f"| **Trạng thái:** `{detail['status']}`")
        st.markdown("**Vật liệu trích xuất:** " + (", ".join(detail["materials"]) or "_(trống)_"))
        st.text_area("Câu trả lời", detail["answer"], height=180)
        if detail.get("retrievals"):
            st.markdown("**Các passage đã truy hồi:**")
            render_passages(detail["retrievals"])
        if detail.get("calls"):
            st.markdown("**Các lượt gọi LLM:**")
            render_calls(detail["calls"])

