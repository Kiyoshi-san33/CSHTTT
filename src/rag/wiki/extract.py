"""Trích xuất & lọc bài viết Wikipedia liên quan đến ô tô (song song nhiều nhân CPU)."""
from __future__ import annotations

import argparse
import bz2
import glob
import html
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

# pyrefly: ignore[missing-import]
import ahocorasick
from tqdm import tqdm

from ...core.config import load_config
from ...core.io import read_list
from ...core.paths import cfg_path, resolve
from .clean import clean_wikitext

_PAGE = re.compile(r"<page>(.*?)</page>", re.S)
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_NS = re.compile(r"<ns>(\d+)</ns>")
_ID = re.compile(r"<id>(\d+)</id>")
_TEXT = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
_MATERIAL = re.compile(r"\bmaterials?\b", re.I)
_AUTO = re.compile(r"\b(automotive|automobiles?|vehicles?|cars?|trucks?|motorcycles?|engines?|motor vehicles?)\b",
                   re.I)

G: dict = {}


def _init(dump_path: str, components: list[str], crit: dict):
    A = ahocorasick.Automaton()
    for c in components:
        A.add_word(c, c)
    A.make_automaton()
    G.update(dump=dump_path, A=A, crit=crit)


def _components_in(lower: str) -> set[str]:
    found = set()
    for end, comp in G["A"].iter(lower):
        start = end - len(comp) + 1
        before = lower[start - 1] if start > 0 else " "
        after = lower[end + 1] if end + 1 < len(lower) else " "
        if after == "s" and (end + 2 >= len(lower) or not lower[end + 2].isalnum()):
            after = " "
        if not before.isalnum() and not after.isalnum():
            found.add(comp)
    return found


def _process(task):
    task_id, ranges = task
    crit = G["crit"]
    out = []
    with open(G["dump"], "rb") as f:
        f.seek(ranges[0][0])
        blob = f.read(ranges[-1][1] - ranges[0][0])
    base = ranges[0][0]
    for s, e in ranges:
        try:
            xml = bz2.decompress(blob[s - base:e - base]).decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for pm in _PAGE.finditer(xml):
            page = pm.group(1)
            ns = _NS.search(page)
            if not ns or ns.group(1) != "0" or "<redirect" in page:
                continue
            tm = _TEXT.search(page)
            if not tm:
                continue
            raw = html.unescape(tm.group(1))
            lower = raw.lower()
            comps = _components_in(lower)
            c1 = len(comps) >= crit["min_components"]
            c2 = False
            if not c1:
                if len(_MATERIAL.findall(raw)) >= crit["min_material_mentions"] and \
                        len(_AUTO.findall(raw)) >= crit["min_auto_mentions"]:
                    c2 = True
            if not (c1 or c2):
                continue
            text = clean_wikitext(raw)
            if len(text.split()) < 50:
                continue
            out.append({
                "id": _ID.search(page).group(1),
                "title": html.unescape(_TITLE.search(page).group(1)),
                "criterion": 1 if c1 else 2,
                "components": sorted(comps),
                "text": text,
            })
    return task_id, out


def load_offsets(index_path: Path) -> list[int]:
    offs = set()
    with bz2.open(index_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            k = line.find(":")
            if k > 0:
                offs.add(int(line[:k]))
    return sorted(offs)


def run_extract(cfg: dict, limit_tasks: int = 0, workers: int | None = None) -> Path:
    w = cfg["wiki"]
    workers = workers or w["workers"]
    dump_dir = resolve(w["dump_dir"] + "/x").parent
    dumps = sorted(glob.glob(str(dump_dir / "*-pages-articles-multistream.xml.bz2")))
    idxs = sorted(glob.glob(str(dump_dir / "*-pages-articles-multistream-index.txt.bz2")))
    if not dumps or not idxs:
        raise SystemExit("Chưa có dump. Chạy: python -m src.rag.ingestion download")
    dump, idx = dumps[-1], idxs[-1]
    print("Dump:", dump)

    offs = load_offsets(Path(idx))
    offs.append(os.path.getsize(dump))
    ranges = list(zip(offs[:-1], offs[1:]))
    k = w["streams_per_task"]
    tasks = [(i, ranges[j:j + k]) for i, j in enumerate(range(0, len(ranges), k))]
    if limit_tasks:
        tasks = tasks[:limit_tasks]

    out_path = cfg_path(cfg, "articles")
    done_path = out_path.with_suffix(".done")
    done = set()
    if done_path.exists():
        done = {int(x) for x in done_path.read_text().split()}
    todo = [t for t in tasks if t[0] not in done]
    print(f"{len(ranges):,} stream | {len(tasks):,} task | còn {len(todo):,} task | {workers} tiến trình")

    components = [c.lower() for c in read_list(cfg["paths"]["components_filter"])
                  if len(re.split(r"[\s\-]+", c.strip())) >= 2]
    print(f"{len(components)} linh kiện đa từ dùng để lọc")

    n1 = n2 = 0
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(dump, components, w)) as pool, \
            open(out_path, "a", encoding="utf-8") as fout, open(done_path, "a") as fdone:
        bar = tqdm(pool.imap_unordered(_process, todo, chunksize=1), total=len(todo), unit="task")
        for task_id, arts in bar:
            for art in arts:
                fout.write(json.dumps(art, ensure_ascii=False) + "\n")
                n1 += art["criterion"] == 1
                n2 += art["criterion"] == 2
            fout.flush()
            fdone.write(f"{task_id}\n")
            fdone.flush()
            bar.set_postfix(crit1=n1, crit2=n2)
    print(f"Xong sau {(time.time() - t0) / 60:.1f} phút. Thêm: tiêu chí(1)={n1:,}, tiêu chí(2)={n2:,}")
    return out_path


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--workers", type=int, default=cfg["wiki"]["workers"])
    a = ap.parse_args()
    run_extract(cfg, a.limit_tasks, a.workers)


if __name__ == "__main__":
    main()

