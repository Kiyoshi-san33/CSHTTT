"""Bộ mã hóa vector (Embedding backends):
  - BGEEmbedder  : BAAI/bge-small-en-v1.5 (384 chiều, chuẩn hóa L2, query có instruction prefix) — profile adapted
  - DPREmbedder  : facebook/dpr-*-multiset-base (768 chiều, inner product) — profile reproduction
  - HashEmbedder : nhúng băm từ vựng 64 chiều chạy tức thì, tất định — dùng cho unit/smoke test
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    backend: str = "base"
    dim: int = 0
    normalized: bool = False

    @abstractmethod
    def encode_passages(self, titles: list[str], texts: list[str], batch_size: int = 32) -> np.ndarray:
        ...

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        ...

    def metadata(self) -> dict:
        return {"backend": self.backend, "dim": self.dim, "normalized": self.normalized}


class BGEEmbedder(BaseEmbedder):
    backend = "bge"
    dim = 384
    normalized = True

    def __init__(self, rcfg: dict):
        from sentence_transformers import SentenceTransformer

        self.model_name = rcfg.get("bge_model", "BAAI/bge-small-en-v1.5")
        self.query_instruction = rcfg.get(
            "bge_query_instruction", "Represent this sentence for searching relevant passages: ")
        self.max_seq_len = int(rcfg.get("bge_max_seq_len", 512))
        self.model = SentenceTransformer(self.model_name)
        self.model.max_seq_length = self.max_seq_len
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode_passages(self, titles: list[str], texts: list[str], batch_size: int = 64) -> np.ndarray:
        docs = [f"{t}. {x}" for t, x in zip(titles, texts)]
        vec = self.model.encode(docs, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        q = f"{self.query_instruction}{query}"
        vec = self.model.encode([q], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vec[0], dtype=np.float32)

    def metadata(self) -> dict:
        return {**super().metadata(), "model": self.model_name,
                "query_instruction": self.query_instruction, "max_seq_len": self.max_seq_len}


class DPREmbedder(BaseEmbedder):
    backend = "dpr"
    dim = 768
    normalized = False

    def __init__(self, rcfg: dict, mode: str = "both"):
        import torch
        from transformers import (
            DPRContextEncoder,
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoder,
            DPRQuestionEncoderTokenizerFast,
        )

        self._torch = torch
        self.ctx_name = rcfg.get("ctx_encoder", "facebook/dpr-ctx_encoder-multiset-base")
        self.q_name = rcfg.get("question_encoder", "facebook/dpr-question_encoder-multiset-base")
        self.max_len_passage = int(rcfg.get("max_seq_len_passage", 256))
        self.max_len_query = int(rcfg.get("max_seq_len_query", 64))
        self.ctx_tok = self.ctx_enc = None
        self.q_tok = self.q_enc = None
        if mode in ("both", "passage"):
            self.ctx_tok = DPRContextEncoderTokenizerFast.from_pretrained(self.ctx_name)
            self.ctx_enc = DPRContextEncoder.from_pretrained(self.ctx_name).eval()
        if mode in ("both", "query"):
            self.q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(self.q_name)
            self.q_enc = DPRQuestionEncoder.from_pretrained(self.q_name).eval()

    def encode_passages(self, titles: list[str], texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.ctx_enc is None:
            from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

            self.ctx_tok = DPRContextEncoderTokenizerFast.from_pretrained(self.ctx_name)
            self.ctx_enc = DPRContextEncoder.from_pretrained(self.ctx_name).eval()
        out = []
        for i in range(0, len(texts), batch_size):
            j = min(len(texts), i + batch_size)
            enc = self.ctx_tok(titles[i:j], texts[i:j], truncation=True, max_length=self.max_len_passage,
                               padding=True, return_tensors="pt")
            with self._torch.inference_mode():
                out.append(self.ctx_enc(**enc).pooler_output.numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if self.q_enc is None:
            from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

            self.q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(self.q_name)
            self.q_enc = DPRQuestionEncoder.from_pretrained(self.q_name).eval()
        with self._torch.inference_mode():
            enc = self.q_tok(query, return_tensors="pt", truncation=True, max_length=self.max_len_query)
            vec = self.q_enc(**enc).pooler_output[0].numpy().astype(np.float32)
        return vec

    def metadata(self) -> dict:
        return {**super().metadata(), "ctx_encoder": self.ctx_name, "question_encoder": self.q_name,
                "max_seq_len_passage": self.max_len_passage, "max_seq_len_query": self.max_len_query}


class HashEmbedder(BaseEmbedder):
    """Nhúng băm từ vựng nhẹ cho kiểm thử offline (không tải model)."""
    backend = "hash"
    dim = 64
    normalized = True

    def __init__(self, rcfg: dict | None = None):
        self.dim = int((rcfg or {}).get("hash_dim", 64))

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            tok = tok.strip(".,;:!?\"'()[]{}")
            if not tok:
                continue
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 31) == 0 else -1.0
            v[idx] += sign
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def encode_passages(self, titles: list[str], texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(f"{t} {x}") for t, x in zip(titles, texts)], axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        return self._vec(query)


def get_embedder(rcfg: dict, mode: str = "both") -> BaseEmbedder:
    backend = rcfg.get("dense_backend", "bge")
    if backend == "bge":
        return BGEEmbedder(rcfg)
    if backend == "dpr":
        return DPREmbedder(rcfg, mode=mode)
    if backend == "hash":
        return HashEmbedder(rcfg)
    raise ValueError(f"dense_backend không hợp lệ: {backend!r} (chọn: bge | dpr | hash)")

