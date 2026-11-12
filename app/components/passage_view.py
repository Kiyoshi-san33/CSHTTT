"""Thành phần hiển thị danh sách passage truy hồi và nhật ký lượt gọi LLM trên Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_passages(passages: list):
    if not passages:
        st.caption("Không sử dụng tài liệu truy hồi (No-RAG).")
        return
    rows = []
    for p in passages:
        d = p if isinstance(p, dict) else p.model_dump()
        rows.append({
            "Hạng RRF": d.get("rank"),
            "PID": d.get("pid"),
            "Tiêu đề": d.get("title"),
            "Điểm RRF": d.get("score"),
            "Hạng BM25": d.get("bm25_rank"),
            "Hạng Dense": d.get("dense_rank"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Xem toàn văn các passage"):
        for p in passages:
            d = p if isinstance(p, dict) else p.model_dump()
            st.markdown(f"**#{d.get('rank')} [{d.get('pid')}] {d.get('title')}** "
                        f"_(RRF={d.get('score')}, BM25={d.get('bm25_rank')}, Dense={d.get('dense_rank')})_")
            st.write(d.get("text"))
            st.divider()


def render_calls(calls: list):
    if not calls:
        return
    rows = []
    for c in calls:
        if isinstance(c, dict):
            rows.append({
                "Bước": c.get("step"),
                "Prompt": c.get("prompt_version"),
                "Model": c.get("served_model") or c.get("model_id"),
                "Input tok": c.get("prompt_tokens"),
                "Output tok": c.get("completion_tokens"),
                "Reasoning tok": c.get("reasoning_tokens"),
                "Độ trễ (s)": c.get("latency_s"),
                "Finish": c.get("finish_reason"),
            })
        else:
            r = c.response
            rows.append({
                "Bước": c.step,
                "Prompt": c.prompt_version,
                "Model": r.served_model or r.model_id,
                "Input tok": r.prompt_tokens,
                "Output tok": r.completion_tokens,
                "Reasoning tok": r.reasoning_tokens,
                "Độ trễ (s)": r.latency_s,
                "Finish": r.finish_reason,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

