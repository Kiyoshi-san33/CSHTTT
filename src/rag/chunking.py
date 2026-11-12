"""Cắt bài viết thành đoạn (passage) 200 từ, chồng lấn 50 từ (giống paper).

    python -m src.rag.chunking
"""
from __future__ import annotations

import json
from collections import Counter

from tqdm import tqdm

from ..core.config import load_config
from ..core.paths import cfg_path, results_dir


def split_words(words: list[str], length: int, overlap: int):
    step = max(length - overlap, 1)
    for start in range(0, max(len(words) - overlap, 1), step):
        chunk = words[start:start + length]
        if chunk:
            yield " ".join(chunk)


def build_passages(cfg: dict) -> dict:
    r = cfg["retrieval"]
    src, dst = cfg_path(cfg, "articles", mkdir=False), cfg_path(cfg, "passages")
    seen, n_art, n_pas, crit = set(), 0, 0, Counter()
    lengths = []
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="articles"):
            a = json.loads(line)
            if a["id"] in seen:
                continue
            seen.add(a["id"])
            n_art += 1
            crit[a["criterion"]] += 1
            words = a["text"].split()
            lengths.append(len(words))
            for k, chunk in enumerate(split_words(words, r["passage_words"], r["passage_overlap"])):
                fout.write(json.dumps({"pid": n_pas, "title": a["title"], "chunk": k, "text": chunk},
                                      ensure_ascii=False) + "\n")
                n_pas += 1
    avg = sum(lengths) / max(len(lengths), 1)
    stats = {"articles": n_art, "criterion_1": crit[1], "criterion_2": crit[2],
             "avg_article_words": round(avg), "passages": n_pas,
             "passage_words": r["passage_words"], "passage_overlap": r["passage_overlap"]}
    (results_dir(cfg, "metrics") / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_passages(cfg: dict) -> list[dict]:
    path = cfg_path(cfg, "passages", mkdir=False)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append({"pid": d.get("pid", len(out)), "title": d["title"], "text": d["text"]})
    return out


def main():
    cfg = load_config()
    stats = build_passages(cfg)
    print(stats)


if __name__ == "__main__":
    main()

