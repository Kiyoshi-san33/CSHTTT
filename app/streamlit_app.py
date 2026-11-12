"""Điểm vào chính của giao diện Streamlit đa trang:

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.core.config import load_config  # noqa: E402
from src.core.paths import cfg_path, index_dir  # noqa: E402
from src.storage import repository as R  # noqa: E402

st.set_page_config(page_title="Vehicle Material Prediction – Research UI", layout="wide")


def home_page():
    cfg = load_config()
    st.title("🚗 Dự đoán Vật liệu Linh kiện Ô tô với LLM & Hybrid RAG")
    st.caption("Research Prototype — So sánh GPT-OSS 120B (Groq) và Gemini 3.8 Flash (Google) trên No-RAG / RAG / CoVe")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GROQ_API_KEY", "Đã cấu hình" if os.getenv("GROQ_API_KEY") else "Chưa đặt")
    c2.metric("GEMINI_API_KEY", "Đã cấu hình" if os.getenv("GEMINI_API_KEY") else "Chưa đặt")
    bm25_ok = (index_dir(cfg, "bm25") / "meta.json").exists()
    faiss_bge_ok = (index_dir(cfg, "faiss", "bge") / "index.faiss").exists()
    faiss_dpr_ok = (index_dir(cfg, "faiss", "dpr") / "index.faiss").exists()
    c3.metric("Chỉ mục BM25", "Sẵn sàng" if bm25_ok else "Chưa tạo")
    c4.metric("Chỉ mục FAISS", f"BGE={'OK' if faiss_bge_ok else '—'} | DPR={'OK' if faiss_dpr_ok else '—'}")

    st.markdown("""
### Cấu hình công nghệ cố định (Technology Baseline)
- **Main LLM A:** Groq `openai/gpt-oss-120b`
- **Main LLM B:** Google `gemini-3.8-flash`
- **Local Dev LLM:** Ollama `qwen3:8b` (`--dev`, chỉ phục vụ phát triển)
- **Supplementary Judge:** Groq `qwen/qwen3.8-27b` (tách riêng khỏi kết quả đánh giá của người)
- **Retrieval Profiles:**
  - `adapted` (mặc định): `BAAI/bge-small-en-v1.5` + FAISS `IndexFlatIP` + `bm25s` + RRF (200 từ/chunk, 50 từ chồng lấn, Top-50 + Top-50 → Top-10)
  - `reproduction`: `facebook/dpr-*-multiset-base` + FAISS `IndexFlatIP` + `bm25s` + RRF
""")
    passages_file = cfg_path(cfg, "passages", mkdir=False)
    st.info(f"File passages: `{passages_file}` ({'đã có' if passages_file.exists() else 'chưa tạo'}) | "
            f"Số experiment trong SQLite: **{len(R.experiment_progress(cfg))}**")


nav = st.navigation([
    st.Page(home_page, title="Tổng quan hệ thống", icon="🏠", default=True),
    st.Page("pages/prediction.py", title="1. Dự đoán (Playground)", icon="🔍"),
    st.Page("pages/experiments.py", title="2. Thực nghiệm (Runs & Logs)", icon="🧪"),
    st.Page("pages/annotation.py", title="3. Gán nhãn (Human Annotation)", icon="✍️"),
    st.Page("pages/evaluation.py", title="4. Đánh giá (Metrics & Plots)", icon="📊"),
])
nav.run()

