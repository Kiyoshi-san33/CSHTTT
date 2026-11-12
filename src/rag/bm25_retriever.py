"""Truy hồi thưa BM25 bằng thư viện bm25s (thay cho OpenSearch BM25 trong paper).

Ghi lại đầy đủ: thư viện, phiên bản, method, k1, b, stopwords, stemmer, số passage và thời điểm tạo.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from ..core.paths import index_dir


def bm25_config(cfg: dict) -> dict:
    import bm25s

    bcfg = cfg.get("retrieval", {}).get("bm25", {})
    return {
        "implementation": "bm25s",
        "version": getattr(bm25s, "__version__", "unknown"),
        "method": bcfg.get("method", "lucene"),
        "k1": float(bcfg.get("k1", 1.5)),
        "b": float(bcfg.get("b", 0.75)),
        "stopwords": bcfg.get("stopwords", "en"),
        "stemmer": bcfg.get("stemmer", "english"),
        "tokenizer": "bm25s.tokenize(lower=True, stopwords, SnowballStemmer)",
        "sparse_top_k": int(cfg["retrieval"]["sparse_top_k"]),
    }


def build_bm25_index(cfg: dict, passages: list[dict]) -> dict:
    import bm25s
    # pyrefly: ignore[missing-import]
    import Stemmer

    out = index_dir(cfg, "bm25")
    meta = bm25_config(cfg)
    stemmer = Stemmer.Stemmer(meta["stemmer"])
    corpus = [f"{p['title']}. {p['text']}" for p in passages]
    tok = bm25s.tokenize(corpus, stopwords=meta["stopwords"], stemmer=stemmer, show_progress=True)
    bm = bm25s.BM25(method=meta["method"], k1=meta["k1"], b=meta["b"])
    bm.index(tok, show_progress=True)
    bm.save(str(out))
    meta.update(num_passages=len(passages), built_at=datetime.now(timezone.utc).isoformat())
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class BM25Retriever:
    def __init__(self, cfg: dict):
        import bm25s
        # pyrefly: ignore[missing-import]
        import Stemmer

        self.cfg = cfg
        self.dir = index_dir(cfg, "bm25")
        self.meta = bm25_config(cfg)
        meta_file = self.dir / "meta.json"
        if meta_file.exists():
            self.meta.update(json.loads(meta_file.read_text(encoding="utf-8")))
        self.stemmer = Stemmer.Stemmer(self.meta["stemmer"])
        self.bm25 = bm25s.BM25.load(str(self.dir))
        self._bm25s = bm25s

    def retrieve(self, query: str, k: int | None = None) -> list[int]:
        k = int(k or self.cfg["retrieval"]["sparse_top_k"])
        q = self._bm25s.tokenize([query], stopwords=self.meta["stopwords"],
                                 stemmer=self.stemmer, show_progress=False)
        n_docs = getattr(self.bm25, "scores", {}).get("num_docs", k)
        k_eff = min(k, int(n_docs)) if n_docs else k
        idx, _ = self.bm25.retrieve(q, k=k_eff, show_progress=False)
        return [int(i) for i in idx[0]]

