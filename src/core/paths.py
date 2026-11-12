"""Giải quyết đường dẫn: tương đối so với thư mục gốc dự án, hoặc tuyệt đối (dùng trong test)."""
from __future__ import annotations

from pathlib import Path

from .config import ROOT


def resolve(rel: str | Path, mkdir: bool = True) -> Path:
    """Trả về đường dẫn tuyệt đối; tạo thư mục cha nếu cần."""
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cfg_path(cfg: dict, key: str, *sub: str, mkdir: bool = True) -> Path:
    """Đường dẫn cfg["paths"][key] (/ các phần con)."""
    path = Path(cfg["paths"][key]).joinpath(*sub)
    return resolve(path, mkdir=mkdir)


def results_dir(cfg: dict, sub: str) -> Path:
    """results/<sub>/ (predictions, annotations, metrics, plots, logs)."""
    d = cfg_path(cfg, "results", sub, "x").parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_dir(cfg: dict, kind: str, backend: str | None = None) -> Path:
    """indexes/bm25/ hoặc indexes/faiss/<backend>/."""
    parts = [kind] + ([backend] if backend else [])
    d = cfg_path(cfg, "indexes", *parts, "x").parent
    d.mkdir(parents=True, exist_ok=True)
    return d

