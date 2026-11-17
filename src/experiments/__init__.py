"""Thực nghiệm dự đoán vật liệu: No-RAG, RAG, CoVe và Experiment I (material-only)."""


def __getattr__(name: str):
    if name in ("ExperimentRunner", "execute_matrix"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


