"""Kiểm tra model trước khi chạy thực nghiệm/đánh giá đầy đủ.

    python -m src.llm.check                     # generators + judges: model ID có trong danh sách provider?
    python -m src.llm.check --judges --ping     # gửi 1 request nhỏ thật để xác nhận dùng được
    python -m src.llm.check --dev               # Ollama qwen3:8b
Kết quả được lưu vào results/logs/model_check_<ngày>.json (ghi lại ngày kiểm tra cho model Preview).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ..core.config import load_config
from ..core.paths import results_dir
from .factory import get_llm


def check_spec(cfg: dict, spec: dict, ping: bool) -> dict:
    row = {"name": spec["name"], "provider": spec["provider"], "model_id": spec["model"],
           "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        llm = get_llm(cfg, spec)
        row["dropped_params"] = llm.dropped_params
        try:
            models = llm.list_models()
            row["listed"] = None if models is None else spec["model"] in models
        except Exception as e:  # noqa: BLE001
            row["listed"] = None
            row["list_error"] = str(e)[:200]
        if ping:
            r = llm.chat("Reply with the single word OK.", max_tokens=1000)
            row.update(ping_ok=bool(r.content), served_model=r.served_model, latency_s=r.latency_s,
                       finish_reason=r.finish_reason, usage=[r.prompt_tokens, r.completion_tokens,
                                                             r.reasoning_tokens])
    except Exception as e:  # noqa: BLE001
        row["error"] = str(e)[:300]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", action="store_true")
    ap.add_argument("--judges", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--ping", action="store_true", help="gửi request thật (tốn rất ít token)")
    a = ap.parse_args()
    cfg = load_config()
    keys = [k for k, on in (("generators", a.generators), ("judges", a.judges), ("dev_generators", a.dev)) if on]
    keys = keys or ["generators", "judges"]
    rows = [check_spec(cfg, s, a.ping) for k in keys for s in cfg.get(k) or []]
    for r in rows:
        ok = "LỖI" if r.get("error") else ("OK" if r.get("listed") in (True, None) else "KHÔNG CÓ TRONG DANH SÁCH")
        print(f"{r['name']:<20} {r['provider']:<8} {r['model_id']:<28} {ok}  "
              f"{r.get('error') or r.get('list_error') or ''}")
    out = results_dir(cfg, "logs") / f"model_check_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()

