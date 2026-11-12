"""Schema mô tả một experiment (1 model x 1 phương pháp x 1 profile truy hồi)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExperimentSpec(BaseModel):
    exp: str                                # exp1 | exp2
    model_name: str
    model_id: str
    provider: str
    method: str                             # norag | rag | cove | few-shot | zero-shot
    profile: str | None = None              # adapted | reproduction (None nếu không truy hồi)
    top_k: int | None = None
    is_dev: bool = False
    config_snapshot: dict[str, Any] = Field(default_factory=dict)

    @property
    def method_tag(self) -> str:
        """Tên biến thể dùng trong ID output: norag | rag@adapted | cove@reproduction_k3 ..."""
        if self.exp == "exp1" or self.method == "norag":
            return self.method
        tag = f"{self.method}@{self.profile}"
        default_k = self.config_snapshot.get("retrieval", {}).get("final_top_k")
        if self.top_k and default_k and self.top_k != default_k:
            tag += f"_k{self.top_k}"
        return tag

    @property
    def id(self) -> str:
        return f"{self.exp}|{self.model_name}|{self.method_tag}"

    def run_id(self, component: str) -> str:
        return f"{component}||{self.model_name}||{self.method_tag}"

