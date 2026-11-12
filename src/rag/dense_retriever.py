"""Truy hồi dày đặc (Dense Retrieval) với FAISS CPU (IndexFlatIP).

- Profile `adapted`     : BAAI/bge-small-en-v1.5 (384 chiều, vector chuẩn hóa L2 -> IP = cosine).
- Profile `reproduction`: facebook/dpr-*-multiset-base (768 chiều, IP giống DPR gốc).
- Hỗ trợ encode theo từng shard để có thể dừng và chạy tiếp nếu bị ngắt.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import faiss
import numpy as np
from tqdm import tqdm

from ..core.paths import index_dir
from .embeddings import get_embedder


def build_dense_index(cfg: dict, passages: list[dict]) -> dict:
    r = cfg["retrieval"]
    backend = r.get("dense_backend", "bge")
    out = index_dir(cfg, "faiss", backend)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    embedder = get_embedder(r, mode="passage")
    titles = [p["title"] for p in passages]
    texts = [p["text"] for p in passages]
    n = len(texts)
    S = int(r.get("shard_size", 10000))
    B = int(r.get("embed_batch_size", 64))
    n_shards = max((n + S - 1) // S, 1)

    t0, done_now = time.time(), 0
    for s in range(n_shards):
        f = shard_dir / f"shard_{s:05d}.npy"
        if f.exists():
            continue
        lo, hi = s * S, min(n, (s + 1) * S)
        vecs = []
        for i in tqdm(range(lo, hi, B), desc=f"{backend} shard {s + 1}/{n_shards}", leave=False):
            j = min(hi, i + B)
            vecs.append(embedder.encode_passages(titles[i:j], texts[i:j], batch_size=B))
        arr = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, embedder.dim), dtype=np.float32)
        np.save(f, arr)
        done_now += hi - lo
        elapsed = max(time.time() - t0, 1e-6)
        rate = done_now / elapsed
        print(f"shard {s + 1}/{n_shards} xong | {rate:.0f} passage/s")

    embs = np.concatenate([np.load(shard_dir / f"shard_{s:05d}.npy") for s in range(n_shards)], axis=0)
    embs = np.ascontiguousarray(embs, dtype=np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(out / "index.faiss"))

    meta = {
        **embedder.metadata(),
        "profile": r.get("profile", "adapted"),
        "faiss_index_type": "IndexFlatIP",
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "num_passages": int(index.ntotal),
        "dense_top_k": int(r["dense_top_k"]),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class DenseRetriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        r = cfg["retrieval"]
        self.backend = r.get("dense_backend", "bge")
        self.dir = index_dir(cfg, "faiss", self.backend)
        idx_path = self.dir / "index.faiss"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Chưa có FAISS index cho backend {self.backend!r} tại {idx_path}. "
                f"Chạy: python -m src.rag.indexing --dense --profile {r.get('profile', 'adapted')}"
            )
        self.index = faiss.read_index(str(idx_path))
        self.embedder = get_embedder(r, mode="query")
        meta_path = self.dir / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else embedder_meta(self)

    def retrieve(self, question: str, k: int | None = None) -> list[int]:
        k = int(k or self.cfg["retrieval"]["dense_top_k"])
        k_eff = min(k, int(self.index.ntotal))
        if k_eff <= 0:
            return []
        q = np.ascontiguousarray(self.embedder.encode_query(question).reshape(1, -1), dtype=np.float32)
        _, idx = self.index.search(q, k_eff)
        return [int(i) for i in idx[0] if i >= 0]


def embedder_meta(ret: DenseRetriever) -> dict:
    return {**ret.embedder.metadata(), "num_passages": int(ret.index.ntotal)}

