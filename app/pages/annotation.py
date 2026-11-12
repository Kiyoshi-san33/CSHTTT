"""Trang 3: Giao diện gán nhãn mù (Blind Human Annotation) cho Experiment I & Experiment II.

- Tự động thêm tiền tố `human_` vào mã người gán nhãn và lưu với `kind="human"`.
- Ẩn tên model và phương pháp; thứ tự linh kiện & câu trả lời được xáo trộn tất định theo mã annotator.
"""
from __future__ import annotations

import random

import streamlit as st

from app.components.sidebar import load_active_config
from src.experiments.parsing import split_sentences
from src.schemas.annotation import Exp1Label, Exp2Label
from src.storage import repository as R

cfg = load_active_config()
st.header("✍️ Đánh giá thủ công bởi Người (Blind Human Annotation)")

prefix = cfg.get("evaluation", {}).get("human_prefix", "human_")
c1, c2 = st.columns([2, 2])
with c1:
    raw_name = st.text_input("Mã người đánh giá (vd: an, binh, ...)", key="ann_name").strip()
with c2:
    exp = st.radio("Thí nghiệm cần gán nhãn", ["exp2", "exp1"], horizontal=True)

if not raw_name:
    st.info("Nhập mã người đánh giá ở trên để bắt đầu. Kết quả tự lưu vào SQLite (`database/experiments.db`).")
    st.stop()

annotator = raw_name if raw_name.startswith(prefix) else f"{prefix}{raw_name}"
st.caption(f"Đang ghi nhãn dưới mã: **`{annotator}`** (`kind=human`)")

# ----------------------------------------------------------------- EXP 1
if exp == "exp1":
    items = R.exp1_items(cfg)
    if not items:
        st.warning("Chưa có output cho Experiment I. Chạy: `python -m src.experiments.runner --exp exp1`")
        st.stop()
    random.Random(annotator).shuffle(items)
    existing = R.load_labels(cfg, "exp1", annotators=[annotator]).get(annotator, {})
    done = sum(i["id"] in existing for i in items)
    st.progress(done / max(len(items), 1), text=f"Đã gán nhãn {done}/{len(items)} linh kiện")
    idx = st.number_input("Linh kiện thứ", 1, len(items), min(done + 1, len(items))) - 1
    it = items[idx]
    prev = existing.get(it["id"], {})

    st.subheader(f"Linh kiện: **{it['component']}**")
    st.write("Task 1 – Chọn TẤT CẢ vật liệu hợp lý (plausible) mà linh kiện này (hoặc một bộ phận của nó) có thể làm từ:")
    prev_pl = prev.get("plausible") or [False] * len(it["materials"])
    plaus = [
        st.checkbox(m, value=bool(prev_pl[k]) if k < len(prev_pl) else False, key=f"e1_{it['id']}_{k}")
        for k, m in enumerate(it["materials"])
    ]
    dk = st.checkbox("Tôi không biết linh kiện này", value=bool(prev.get("dont_know", False)),
                     key=f"e1_dk_{it['id']}")
    if st.button("💾 Lưu nhãn Experiment I", type="primary"):
        lab = Exp1Label(plausible=plaus, dont_know=dk)
        R.save_label(cfg, "exp1", it["id"], annotator, lab.model_dump(), kind="human")
        st.success("Đã lưu vào SQLite.")
    st.stop()

# ----------------------------------------------------------------- EXP 2
items = R.exp2_items(cfg)
if not items:
    st.warning("Chưa có output cho Experiment II. Chạy: `python -m src.experiments.runner --exp exp2`")
    st.stop()

comps = list(dict.fromkeys(i["component"] for i in items))
random.Random(annotator).shuffle(comps)
existing_out = R.load_labels(cfg, "exp2", annotators=[annotator]).get(annotator, {})
existing_rk = R.load_rankings(cfg, annotators=[annotator]).get(annotator, {})

tab1, tab2 = st.tabs(["Task 1–4: Đánh giá từng câu trả lời", "Task 5: Xếp hạng các câu trả lời"])

with tab1:
    order = [
        o for c in comps
        for o in random.Random(annotator + c).sample(
            [i for i in items if i["component"] == c],
            k=sum(i["component"] == c for i in items),
        )
    ]
    done = sum(o["id"] in existing_out for o in order)
    st.progress(done / max(len(order), 1), text=f"Đã chấm {done}/{len(order)} câu trả lời")
    k = st.number_input("Câu trả lời thứ", 1, len(order), min(done + 1, len(order))) - 1
    it = order[k]
    prev = existing_out.get(it["id"], {})

    st.subheader(f"Linh kiện: **{it['component']}**")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Task 2 – Tick vào các câu chứa thông tin SAI sự thật (Factual Error / Hallucination):**")
        bad = []
        for s_i, s in enumerate(split_sentences(it["answer"])):
            if st.checkbox(s, value=s_i in (prev.get("incorrect_sentences") or []),
                           key=f"s_{it['id']}_{s_i}"):
                bad.append(s_i)
    with right:
        st.markdown("**Task 1 – Chọn vật liệu hợp lý (plausible):**")
        pp = prev.get("plausible") or [False] * len(it["materials"])
        plaus = [
            st.checkbox(m, value=bool(pp[j]) if j < len(pp) else False, key=f"p_{it['id']}_{j}")
            for j, m in enumerate(it["materials"])
        ]
        st.markdown("**Task 3 – Chất lượng & độ đầy đủ danh sách vật liệu (1–5):**")
        r3 = st.slider("Điểm vật liệu", 1, 5, int(prev.get("material_rating", 3)), key=f"r3_{it['id']}")
        st.markdown("**Task 4 – Chất lượng lời giải thích (1–5):**")
        r4 = st.slider("Điểm giải thích", 1, 5, int(prev.get("explanation_rating", 3)), key=f"r4_{it['id']}")
        cm = st.text_area("Ghi chú vật liệu thiếu / thông tin thừa (tùy chọn)",
                          prev.get("comment", ""), key=f"cm_{it['id']}")
        if st.button("💾 Lưu Task 1–4", type="primary"):
            lab = Exp2Label(plausible=plaus, incorrect_sentences=bad,
                            material_rating=r3, explanation_rating=r4, comment=cm)
            R.save_label(cfg, "exp2", it["id"], annotator, lab.model_dump(), kind="human")
            st.success("Đã lưu vào SQLite.")

with tab2:
    c = st.selectbox("Chọn linh kiện để xếp hạng", comps)
    outs = [i for i in items if i["component"] == c]
    random.Random(annotator + "rank" + c).shuffle(outs)
    labels = [chr(65 + j) for j in range(len(outs))]
    st.caption("Xếp hạng tất cả câu trả lời theo chất lượng NỘI DUNG (bỏ qua định dạng). 1 = tốt nhất.")
    cols = st.columns(max(len(outs), 1))
    prev_rank = existing_rk.get(c, [])
    ranks = {}
    for col, l, o in zip(cols, labels, outs):
        with col:
            st.markdown(f"### Phương án {l}")
            default = prev_rank.index(o["id"]) + 1 if o["id"] in prev_rank else labels.index(l) + 1
            ranks[o["id"]] = st.number_input("Hạng", 1, len(outs), default, key=f"rk_{c}_{l}")
            st.markdown(o["answer"])
    if st.button("💾 Lưu xếp hạng Task 5", type="primary"):
        if len(set(ranks.values())) != len(outs):
            st.error("Mỗi phương án phải nhận một thứ hạng khác nhau.")
        else:
            ordered_ids = sorted(ranks, key=ranks.get)
            R.save_ranking(cfg, c, annotator, ordered_ids, kind="human")
            st.success("Đã lưu xếp hạng vào SQLite.")

