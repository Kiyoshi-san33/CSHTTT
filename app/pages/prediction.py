"""Trang 1: Dự đoán tương tác cho một linh kiện với No-RAG / RAG / CoVe."""
from __future__ import annotations

import streamlit as st

from app.components.passage_view import render_calls, render_passages
from app.components.sidebar import load_active_config
from src.core.config import all_generator_specs
from src.experiments.runner import ExperimentRunner
from src.llm.factory import get_llm

cfg = load_active_config()
st.header("🔍 Dự đoán vật liệu linh kiện (Interactive Playground)")

specs = all_generator_specs(cfg, include_dev=True) + [
    {"name": "dummy", "provider": "dummy", "model": "dummy-local", "rpm": 6000}
]
spec_by_name = {s["name"]: s for s in specs}

col1, col2, col3 = st.columns([2, 1.5, 1])
with col1:
    component = st.text_input("Tên linh kiện xe (tiếng Anh)", value="brake disc").strip()
with col2:
    model_name = st.selectbox("Model", list(spec_by_name))
with col3:
    method = st.selectbox("Phương pháp", ["norag", "rag", "cove"])

if st.button("▶ Chạy dự đoán", type="primary", disabled=not component):
    try:
        llm = get_llm(cfg, spec_by_name[model_name])
        runner = ExperimentRunner(cfg)
        with st.spinner(f"Đang chạy {model_name} / {method}..."):
            res = runner.run_single(llm, component, method, exp="exp2")
        st.subheader("Kết quả dự đoán")
        st.markdown(res.answer)
        st.markdown("**Danh sách vật liệu trích xuất:** " + (", ".join(res.materials) or "_(không trích xuất được)_"))

        if res.baseline:
            with st.expander("Bước 1 & 2–3 của CoVe (Baseline & Câu hỏi kiểm chứng)"):
                st.markdown("**Baseline (RAG):**")
                st.write(res.baseline)
                for idx, v in enumerate(res.verification, 1):
                    st.markdown(f"- **Q{idx}:** {v['q']}  \n  **A{idx}:** {v['a']} _(PIDs: {v['pids']})_")

        if res.retrievals:
            st.subheader("Tài liệu truy hồi (Top-10 RRF)")
            render_passages(res.retrievals[0].passages)

        st.subheader("Nhật ký lượt gọi LLM & Token Usage")
        render_calls(res.calls)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

