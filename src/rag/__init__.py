"""Pipeline RAG: ingestion, chunking, embeddings (BGE/DPR), BM25, FAISS và Hybrid RRF."""


def __getattr__(name: str):
    if name in ("HybridRetriever", "rrf_fuse"):
        from . import hybrid_retriever

        return getattr(hybrid_retriever, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


