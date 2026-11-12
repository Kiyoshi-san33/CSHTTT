"""Bộ truy hồi lai (Hybrid Retriever): BM25 + Dense (FAISS) + Reciprocal Rank Fusion (RRF).

- BM25 nhận truy vấn từ khóa (tên linh kiện), trả top-50.
- Dense retriever (BGE hoặc DPR theo profile) nhận câu hỏi đầy đủ, trả top-50.
- RRF gộp hai danh sách với hằng số k=60: score(d) = sum 1 / (k + 1 + rank_0based) = sum 1 / (61 + rank_0based).
"""
from __future__ import annotations

from ..core.config import load_config
from ..schemas.prediction import RetrievedPassage
from .bm25_retriever import BM25Retriever
from .chunking import load_passages
from .dense_retriever import DenseRetriever


def rrf_fuse(rankings: list[list[int]], rrf_k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion khớp công thức DocumentJoiner của Haystack: 1 / (rrf_k + 1 + rank_0based)."""
    fused: dict[int, float] = {}
    base = rrf_k + 1
    for ranking in rankings:
        for rank0, pid in enumerate(ranking):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (base + rank0)
    return fused


class HybridRetriever:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.r = self.cfg["retrieval"]
        self.passages = load_passages(self.cfg)
        self.bm25 = BM25Retriever(self.cfg)
        self.dense = DenseRetriever(self.cfg)

    def configuration(self) -> dict:
        return {
            "profile": self.r.get("profile", "adapted"),
            "dense_backend": self.r.get("dense_backend", "bge"),
            "sparse_top_k": int(self.r["sparse_top_k"]),
            "dense_top_k": int(self.r["dense_top_k"]),
            "final_top_k": int(self.r["final_top_k"]),
            "join_mode": self.r.get("join_mode", "rrf"),
            "rrf_k": int(self.r.get("rrf_k", 60)),
            "bm25": self.bm25.meta,
            "dense": self.dense.meta,
        }

    def retrieve(self, keyword_query: str, question: str, top_k: int | None = None) -> list[RetrievedPassage]:
        r = self.r
        top_k = int(top_k or r["final_top_k"])
        s = self.bm25.retrieve(keyword_query, r["sparse_top_k"])
        d = self.dense.retrieve(question, r["dense_top_k"])
        s_rank = {pid: idx + 1 for idx, pid in enumerate(s)}
        d_rank = {pid: idx + 1 for idx, pid in enumerate(d)}

        if r.get("join_mode", "rrf") == "rrf":
            fused = rrf_fuse([s, d], int(r.get("rrf_k", 60)))
        else:
            fused = {}
            for rank0, pid in enumerate([x for pair in zip(s, d) for x in pair]):
                fused.setdefault(pid, float(-rank0))

        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        out: list[RetrievedPassage] = []
        for idx, pid in enumerate(best):
            p = self.passages[pid]
            out.append(RetrievedPassage(
                pid=pid,
                title=p["title"],
                text=p["text"],
                score=round(float(fused[pid]), 6),
                rank=idx + 1,
                bm25_rank=s_rank.get(pid),
                dense_rank=d_rank.get(pid),
            ))
        return out


if __name__ == "__main__":
    import sys

    comp = " ".join(sys.argv[1:]) or "brake disc"
    R = HybridRetriever()
    for h in R.retrieve(comp, f"What materials does the vehicle component '{comp}' consist of?"):
        print(f"\n[#{h.rank} pid={h.pid}] {h.title} (score={h.score}, bm25={h.bm25_rank}, dense={h.dense_rank})\n"
              f"{h.text[:300]}...")

