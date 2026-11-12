"""Kết nối SQLite (WAL để nhiều tab Streamlit + runner ghi đồng thời)."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..core.paths import cfg_path
from .models import Base


@lru_cache(maxsize=None)
def _engine(url: str):
    eng = create_engine(url, future=True, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(eng, "connect")
    def _pragma(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    return eng


def engine_for(cfg: dict):
    path = cfg_path(cfg, "database")
    return _engine(f"sqlite:///{path.as_posix()}")


@contextmanager
def session_scope(cfg: dict):
    S = sessionmaker(bind=engine_for(cfg), expire_on_commit=False, future=True)
    s: Session = S()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

