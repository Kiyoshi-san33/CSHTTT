"""LLM-as-a-Judge (đánh giá BỔ SUNG, tách riêng khỏi đánh giá của người).

Giám khảo độc lập mặc định: Groq `qwen/qwen3.8-27b` (Preview).
Ghi vào SQLite với `kind="llm"` và `judge_model_id` được ghim cố định cùng thời điểm chấm.

    python -m src.evaluation.llm_judge --exp exp1
    python -m src.evaluation.llm_judge --exp exp2 [--judges judge_qwen38_27b] [--limit 5]
"""
from __future__ import annotations

import argparse
import json
import random
import re

from tqdm import tqdm

from ..core.config import load_config
from ..core.prompts import load_prompt
from ..experiments.parsing import split_sentences
from ..llm.base import BaseLLM
from ..llm.factory import get_llms
from ..schemas.annotation import Exp1Label, Exp2Label, fix_len
from ..schemas.prediction import LLMCall
from ..storage import repository as R


def parse_json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        raise ValueError("Không tìm thấy JSON trong phản hồi của judge")
    return json.loads(m.group(0))


def _ask(cfg: dict, llm: BaseLLM, tpl_name: str, render_kw: dict, check_fn,
         run_id: str | None = None) -> dict:
    sys_tpl = load_prompt("judge_system")
    tpl = load_prompt(tpl_name)
    prompt = tpl.render(**render_kw)
    mt = cfg.get("generation", {}).get("step_max_tokens", {}).get("judge", 3000)
    for attempt in range(3):
        resp = llm.chat(prompt, system=sys_tpl.text, max_tokens=mt, temperature=0.0)
        R.log_call(cfg, LLMCall(step=f"{tpl_name}_{attempt + 1}", prompt_name=tpl.name,
                                prompt_version=tpl.version, prompt=prompt,
                                system=sys_tpl.text, response=resp),
                   purpose="judge", run_id=run_id)
        try:
            d = parse_json(resp.content)
            check_fn(d)
            return d
        except Exception:  # noqa: BLE001
            prompt += "\n\nIMPORTANT: answer with valid JSON exactly in the requested format."
    raise RuntimeError(f"[{llm.name}] không trả về JSON hợp lệ cho {tpl_name}")


def judge_exp1(cfg: dict, llm: BaseLLM, limit: int = 0) -> int:
    existing = R.load_labels(cfg, "exp1", annotators=[llm.name]).get(llm.name, {})
    items = R.exp1_items(cfg)[:limit or None]
    n_new = 0
    for it in tqdm(items, desc=f"{llm.name} exp1"):
        if it["id"] in existing:
            continue
        k = len(it["materials"])
        mats = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(it["materials"])) or "(none)"
        d = _ask(cfg, llm, "judge_exp1", {"component": it["component"], "mats": mats},
                 lambda x: isinstance(x.get("plausible"), list), run_id=it["id"])
        lab = Exp1Label(plausible=fix_len(d.get("plausible"), k),
                        dont_know=bool(d.get("dont_know", False)))
        R.save_label(cfg, "exp1", it["id"], llm.name, lab.model_dump(),
                     kind="llm", judge_model_id=llm.model)
        n_new += 1
    return n_new


def judge_exp2(cfg: dict, llm: BaseLLM, limit: int = 0) -> tuple[int, int]:
    existing_out = R.load_labels(cfg, "exp2", annotators=[llm.name]).get(llm.name, {})
    existing_rk = R.load_rankings(cfg, annotators=[llm.name]).get(llm.name, {})
    items = R.exp2_items(cfg)
    comps = list(dict.fromkeys(i["component"] for i in items))[:limit or None]
    items = [i for i in items if i["component"] in comps]

    n_out = n_rk = 0
    for it in tqdm(items, desc=f"{llm.name} exp2 task1-4"):
        if it["id"] in existing_out:
            continue
        k = len(it["materials"])
        sents = "\n".join(f"[{i}] {s}" for i, s in enumerate(split_sentences(it["answer"])))
        mats = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(it["materials"])) or "(none)"
        d = _ask(cfg, llm, "judge_exp2", {"component": it["component"], "sents": sents, "mats": mats},
                 lambda x: (int(x["material_rating"]), int(x["explanation_rating"])), run_id=it["id"])
        bad = sorted({int(x) for x in (d.get("incorrect_sentences") or []) if str(x).isdigit()})
        lab = Exp2Label(
            plausible=fix_len(d.get("plausible"), k),
            incorrect_sentences=bad,
            material_rating=int(d["material_rating"]),
            explanation_rating=int(d["explanation_rating"]),
            comment=str(d.get("comment", ""))[:500],
        )
        R.save_label(cfg, "exp2", it["id"], llm.name, lab.model_dump(),
                     kind="llm", judge_model_id=llm.model)
        n_out += 1

    for c in tqdm(comps, desc=f"{llm.name} exp2 task5"):
        if c in existing_rk:
            continue
        outs = [i for i in items if i["component"] == c]
        if not outs:
            continue
        random.Random(f"{llm.name}{c}").shuffle(outs)
        labels = [chr(65 + k) for k in range(len(outs))]
        answers = "\n\n".join(f"=== Answer {l} ===\n{o['answer']}" for l, o in zip(labels, outs))
        d = _ask(cfg, llm, "judge_rank",
                 {"component": c, "n": len(outs), "labels": ", ".join(labels), "answers": answers},
                 lambda x: isinstance(x.get("ranking"), list))
        order = [re.sub(r"(?i)^answer\s*", "", str(x)).strip().upper() for x in d["ranking"]]
        order = [l for l in dict.fromkeys(order) if l in labels]
        order += [l for l in labels if l not in order]
        ranked_ids = [outs[labels.index(l)]["id"] for l in order]
        R.save_ranking(cfg, c, llm.name, ranked_ids, kind="llm", judge_model_id=llm.model)
        n_rk += 1
    return n_out, n_rk


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["exp1", "exp2"], required=True)
    ap.add_argument("--judges", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for llm in get_llms(cfg, "judges", a.judges):
        if a.exp == "exp1":
            n = judge_exp1(cfg, llm, a.limit)
            print(f"[{llm.name}] Exp1: +{n} nhãn")
        else:
            no, nr = judge_exp2(cfg, llm, a.limit)
            print(f"[{llm.name}] Exp2: +{no} output, +{nr} xếp hạng")


if __name__ == "__main__":
    main()

