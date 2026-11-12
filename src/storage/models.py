"""Bảng SQLite (SQLAlchemy 2.0) cho experiment, output, lượt gọi LLM, truy hồi và nhãn đánh giá."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """1 model x 1 phương pháp x 1 profile truy hồi (vd exp2|gpt_oss_120b|rag@adapted)."""
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    exp: Mapped[str] = mapped_column(String, index=True)
    model_name: Mapped[str] = mapped_column(String)
    model_id: Mapped[str] = mapped_column(String)
    provider: Mapped[str] = mapped_column(String)
    method: Mapped[str] = mapped_column(String)          # norag | rag | cove | few-shot | zero-shot
    method_tag: Mapped[str] = mapped_column(String)      # norag | rag@adapted | cove@reproduction ...
    profile: Mapped[str | None] = mapped_column(String, nullable=True)
    dense_backend: Mapped[str | None] = mapped_column(String, nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_dev: Mapped[bool] = mapped_column(Boolean, default=False)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    runs: Mapped[list[Run]] = relationship(back_populates="experiment")


class Run(Base):
    """Một output: 1 linh kiện trong 1 experiment. ID = '<linh kiện>||<model>||<method_tag>'."""
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    exp: Mapped[str] = mapped_column(String, index=True)
    component: Mapped[str] = mapped_column(String, index=True)
    model_name: Mapped[str] = mapped_column(String)
    method: Mapped[str] = mapped_column(String)
    method_tag: Mapped[str] = mapped_column(String)
    answer: Mapped[str] = mapped_column(Text, default="")
    materials: Mapped[list] = mapped_column(JSON, default=list)
    baseline: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification: Mapped[list] = mapped_column(JSON, default=list)
    retrieval_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="done")   # done | error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    calls: Mapped[list[LLMCallRow]] = relationship(cascade="all, delete-orphan")
    retrievals: Mapped[list[RetrievalRow]] = relationship(cascade="all, delete-orphan")


class LLMCallRow(Base):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id"), index=True, nullable=True)
    purpose: Mapped[str] = mapped_column(String, default="generation")   # generation | judge | playground
    step: Mapped[str] = mapped_column(String)
    prompt_name: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String)
    prompt: Mapped[str] = mapped_column(Text)
    system: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_id: Mapped[str] = mapped_column(String)
    served_model: Mapped[str | None] = mapped_column(String, nullable=True)
    provider: Mapped[str] = mapped_column(String)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_s: Mapped[float] = mapped_column(Float, default=0.0)
    finish_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RetrievalRow(Base):
    __tablename__ = "retrievals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    step: Mapped[str] = mapped_column(String)
    keyword_query: Mapped[str] = mapped_column(Text)
    dense_query: Mapped[str] = mapped_column(Text)
    rank: Mapped[int] = mapped_column(Integer)
    pid: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    bm25_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dense_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dense_backend: Mapped[str | None] = mapped_column(String, nullable=True)


class Annotation(Base):
    """Nhãn của 1 annotator (người: kind=human; LLM-judge: kind=llm) cho 1 output."""
    __tablename__ = "annotations"
    __table_args__ = (UniqueConstraint("exp", "item_id", "annotator"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exp: Mapped[str] = mapped_column(String, index=True)
    item_id: Mapped[str] = mapped_column(String, index=True)
    annotator: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)                      # human | llm
    judge_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    label: Mapped[dict] = mapped_column(JSON)                      # Exp1Label / Exp2Label
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Ranking(Base):
    """Task 5 (Exp II): thứ tự output của 1 linh kiện, tốt nhất -> kém nhất."""
    __tablename__ = "rankings"
    __table_args__ = (UniqueConstraint("exp", "component", "annotator"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exp: Mapped[str] = mapped_column(String, default="exp2")
    component: Mapped[str] = mapped_column(String, index=True)
    annotator: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)
    judge_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    order: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

