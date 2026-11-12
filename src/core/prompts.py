"""Nạp prompt từ thư mục prompts/*.txt và gắn phiên bản (name@sha256[:8]).

Nội dung prompt lấy nguyên văn từ Phụ lục A, B của paper khi có; phần paper không công bố
đầy đủ (RAG, thực thi kiểm chứng, câu trả lời cuối của CoVe) được viết theo mô tả trong Mục 5.1.
File .txt dùng cú pháp str.format: {biến}; dấu ngoặc nhọn thật được viết là {{ }}.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import ROOT

PROMPT_DIR = ROOT / "prompts"

# Câu hỏi/truy vấn ngắn dùng chung (Phụ lục B)
QUESTION = 'In the automotive context, which materials is the component "{component}" made out of?'
DENSE_QUERY = "What materials does the vehicle component '{component}' consist of?"
FORMAT_HINT = '\nFormat each material as a numbered line: "<number>. <Material>: <brief explanation>".'


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str

    @property
    def version(self) -> str:
        return f"{self.name}@{hashlib.sha256(self.text.encode('utf-8')).hexdigest()[:8]}"

    def render(self, **kw) -> str:
        return self.text.format(**kw)


@lru_cache(maxsize=None)
def _load(directory: str, name: str) -> Prompt:
    f = Path(directory) / f"{name}.txt"
    text = f.read_text(encoding="utf-8")
    if text.endswith("\n"):          # bỏ đúng 1 ký tự xuống dòng cuối file
        text = text[:-1]
    return Prompt(name, text)


def load_prompt(name: str, directory: str | Path | None = None) -> Prompt:
    return _load(str(directory or PROMPT_DIR), name)


def all_prompt_versions(directory: str | Path | None = None) -> dict[str, str]:
    d = Path(directory or PROMPT_DIR)
    return {f.stem: load_prompt(f.stem, d).version for f in sorted(d.glob("*.txt"))}


def format_passages(hits) -> str:
    def g(h, k):
        return h[k] if isinstance(h, dict) else getattr(h, k)
    return "\n\n".join(f"Passage {i + 1} ({g(h, 'title')}):\n{g(h, 'text')}" for i, h in enumerate(hits))

