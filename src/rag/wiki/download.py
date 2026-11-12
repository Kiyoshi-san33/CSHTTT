"""Tải Wikipedia dump tiếng Anh (bản multistream + index) - có hỗ trợ tải tiếp khi bị ngắt."""
from __future__ import annotations

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

from ...core.config import load_config
from ...core.paths import resolve


DEFAULT_HEADERS = {
    "User-Agent": (
        "VehicleMaterialReproductionResearch/1.0 "
        "(Academic reproduction of Wagner et al.; python-requests)"
    )
}


def download_file(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, headers=DEFAULT_HEADERS, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("Content-Length", 0))
    have = out.stat().st_size if out.exists() else 0
    if total and have >= total:
        print(f"Đã có đủ: {out.name} ({total / 1e9:.2f} GB)")
        return
    headers = dict(DEFAULT_HEADERS)
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(out, mode) as f, tqdm(total=total, initial=have, unit="B", unit_scale=True,
                                        desc=out.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def run_download(cfg: dict, date: str | None = None, mirror: str = "https://dumps.wikimedia.org") -> Path:
    d = date or cfg["wiki"]["dump_date"]
    base = f"{mirror}/enwiki/{d}"
    out_dir = resolve(cfg["wiki"]["dump_dir"] + "/x").parent
    for name in (f"enwiki-{d}-pages-articles-multistream-index.txt.bz2",
                 f"enwiki-{d}-pages-articles-multistream.xml.bz2"):
        download_file(f"{base}/{name}", out_dir / name)
    print("Xong. File nằm trong", out_dir)
    return out_dir


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=cfg["wiki"]["dump_date"])
    ap.add_argument("--mirror", default="https://dumps.wikimedia.org")
    a = ap.parse_args()
    run_download(cfg, a.date, a.mirror)


if __name__ == "__main__":
    main()

