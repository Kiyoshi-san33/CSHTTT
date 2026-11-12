"""Pipeline RAG: ingestion, chunking, embeddings (BGE/DPR), BM25, FAISS và Hybrid RRF."""
from .hybrid_retriever import HybridRetriever, rrf_fuse  # noqa: F401

