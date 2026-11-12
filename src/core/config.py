"""Đọc cấu hình dự án và áp dụng profile truy hồi (reproduction | adapted).

- File cấu hình: tham số `path`, biến môi trường VMR_CONFIG (dùng cho kiểm thử) hoặc config.yaml.
- Profile: tham số `profile`, biến môi trường VMR_PROFILE hoặc khóa `active_profile`.
Sau khi áp dụng, cfg["retrieval"] = retrieval chung + retrieval_profiles[profile]
và cfg["retrieval"]["profile"] = tên profile (được ghi vào mỗi lượt thực nghiệm).
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_profile(cfg: dict, profile: str | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    profiles = cfg.get("retrieval_profiles") or {}
    name = profile or cfg.get("active_profile") or "adapted"
    if profiles and name not in profiles:
        raise KeyError(f"Profile truy hồi không tồn tại: {name!r} (có: {', '.join(profiles)})")
    cfg["retrieval"] = deep_merge(cfg.get("retrieval", {}), profiles.get(name, {}))
    cfg["retrieval"]["profile"] = name
    cfg["active_profile"] = name
    return cfg


def load_config(path: str | Path | None = None, profile: str | None = None) -> dict:
    path = Path(path or os.getenv("VMR_CONFIG") or (ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return apply_profile(cfg, profile or os.getenv("VMR_PROFILE"))


def all_generator_specs(cfg: dict, include_dev: bool = True) -> list[dict]:
    specs = list(cfg.get("generators", []))
    if include_dev:
        specs += list(cfg.get("dev_generators", []))
    return specs


def find_spec(cfg: dict, name: str) -> dict:
    for key in ("generators", "dev_generators", "judges"):
        for s in cfg.get(key, []) or []:
            if s["name"] == name:
                return s
    raise KeyError(f"Không tìm thấy model {name!r} trong config")

