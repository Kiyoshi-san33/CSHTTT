"""Schema nhãn đánh giá — dùng chung cho người (Streamlit) và LLM-judge.

Exp I  : Task 1 (plausible từng vật liệu) + "không biết".
Exp II : Task 1 plausible, Task 2 câu sai sự thật, Task 3 điểm vật liệu (1-5),
         Task 4 điểm giải thích (1-5), Task 5 xếp hạng các output của cùng linh kiện.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Exp1Label(BaseModel):
    plausible: list[bool]
    dont_know: bool = False


class Exp2Label(BaseModel):
    plausible: list[bool]
    incorrect_sentences: list[int] = Field(default_factory=list)
    material_rating: int
    explanation_rating: int
    comment: str = ""

    @field_validator("material_rating", "explanation_rating")
    @classmethod
    def _clip(cls, v: int) -> int:
        return min(5, max(1, int(v)))


def fix_len(lst, n: int) -> list[bool]:
    lst = [bool(x) for x in (lst or [])][:n]
    return lst + [False] * (n - len(lst))

