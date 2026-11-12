"""Sidebar chọn profile truy hồi dùng chung cho các trang Streamlit."""
from __future__ import annotations

import streamlit as st

from src.core.config import load_config


def load_active_config() -> dict:
    with st.sidebar:
        profile = st.selectbox(
            "Retrieval Profile",
            options=["adapted", "reproduction"],
            index=0,
            help="adapted: BGE-small + FAISS + bm25s + RRF | reproduction: DPR + FAISS + bm25s + RRF",
        )
    return load_config(profile=profile)

