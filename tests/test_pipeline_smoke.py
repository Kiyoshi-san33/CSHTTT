"""End-to-end smoke test cho toàn bộ kiến trúc mới trong thư mục tạm (tmp_path):
  Wiki dump giả -> extract -> chunking -> BM25 + FAISS index -> Exp I & Exp II (2 LLM x 3 phương pháp)
  -> Human annotations (3 người cho Exp II, 4 người cho Exp I) + LLM-as-Judge (tách riêng)
  -> Report + Export JSONL/CSV.
"""
from __future__ import annotations

import bz2
from pathlib import Path

import yaml

from src.core.config import load_config
from src.evaluation.llm_judge import judge_exp1, judge_exp2
from src.evaluation.report import generate_report
from src.experiments.runner import execute_matrix
from src.llm.factory import get_llms
from src.rag.bm25_retriever import build_bm25_index
from src.rag.chunking import build_passages, load_passages
from src.rag.dense_retriever import build_dense_index
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.wiki.extract import run_extract
from src.storage import export as E, repository as R

ARTICLES = [
    ("Disc brake", "A '''disc brake''' is a type of [[brake]] that uses [[brake caliper|calipers]] to squeeze pairs of "
     "pads against a disc. The '''brake disc''' is usually made of [[gray iron|grey cast iron]]. "
     "Brake pads are made of ceramic or semi-metallic material.<ref>Smith 2001</ref> {{Infobox|x=y}} "
     "High-performance cars use carbon-ceramic brake discs. The brake pad material affects wear. " * 6),
    ("Spark plug", "A '''spark plug''' delivers current from the [[ignition coil]] to the combustion chamber. "
     "The insulator is made of sintered alumina (aluminium oxide) ceramic; the centre electrode uses nickel alloy, "
     "copper core, platinum or iridium tips. The shell is steel. " * 8),
    ("Banana", "The banana is an elongated, edible fruit produced by several kinds of large herbaceous plants. " * 20),
    ("Brake Disc", "#REDIRECT [[Disc brake]]"),
    ("Piston", "A '''piston''' is a component of reciprocating engines. Engine pistons are made of aluminium alloy, "
     "cast or forged. The piston ring is made of cast iron or steel. The connecting rod links the piston to the "
     "crankshaft. Material choice for car engines depends on temperature. Materials used in vehicles: " * 6),
]


def _page(i: int, title: str, text: str) -> str:
    redirect = '<redirect title="x" />' if text.startswith("#REDIRECT") else ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f"<page><title>{title}</title><ns>0</ns><id>{i}</id>{redirect}<revision><text bytes=\"1\">"
            f"{text}</text></revision></page>\n")


def _write_fake_dump(dump_dir: Path):
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump = dump_dir / "enwiki-test-pages-articles-multistream.xml.bz2"
    index_lines, off = [], 0
    with open(dump, "wb") as f:
        head = bz2.compress(b"<mediawiki><siteinfo></siteinfo>\n")
        f.write(head)
        off = len(head)
        for s in range(0, len(ARTICLES), 2):
            chunk = "".join(_page(i + 1, *ARTICLES[i]) for i in range(s, min(s + 2, len(ARTICLES))))
            data = bz2.compress(chunk.encode("utf-8"))
            for i in range(s, min(s + 2, len(ARTICLES))):
                index_lines.append(f"{off}:{i + 1}:{ARTICLES[i][0]}")
            f.write(data)
            off += len(data)
        f.write(bz2.compress(b"</mediawiki>"))
    with bz2.open(dump_dir / "enwiki-test-pages-articles-multistream-index.txt.bz2", "wt", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")


def test_full_pipeline_smoke(tmp_path: Path):
    _write_fake_dump(tmp_path / "wiki_dump")
    base_cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    base_cfg["generators"] = [
        {"name": "gpt_oss_120b", "provider": "dummy", "model": "openai/gpt-oss-120b", "rpm": 60000},
        {"name": "gemini_flash", "provider": "dummy", "model": "gemini-3.8-flash", "rpm": 60000},
    ]
    base_cfg["exp1_generators"] = ["gpt_oss_120b"]
    base_cfg["judges"] = [
        {"name": "judge_qwen38_27b", "provider": "dummy", "model": "qwen/qwen3.8-27b", "rpm": 60000},
    ]
    base_cfg["wiki"].update(dump_dir=str(tmp_path / "wiki_dump"), workers=1, streams_per_task=1)
    base_cfg["paths"].update(
        articles=str(tmp_path / "corpus" / "articles.jsonl"),
        passages=str(tmp_path / "chunks" / "passages.jsonl"),
        indexes=str(tmp_path / "indexes"),
        database=str(tmp_path / "database" / "experiments.db"),
        results=str(tmp_path / "results"),
    )
    base_cfg["retrieval"].update(sparse_top_k=3, dense_top_k=3, final_top_k=2,
                                 shard_size=4, passage_words=60, passage_overlap=10)
    base_cfg["retrieval_profiles"]["adapted"]["dense_backend"] = "hash"

    cfg_file = tmp_path / "config_smoke.yaml"
    cfg_file.write_text(yaml.safe_dump(base_cfg, allow_unicode=True), encoding="utf-8")
    cfg = load_config(cfg_file, profile="adapted")

    # 1. Trích xuất & cắt passage
    run_extract(cfg, workers=1)
    stats = build_passages(cfg)
    assert stats["articles"] == 3
    assert stats["passages"] > 5

    # 2. Lập chỉ mục BM25 + FAISS
    passages = load_passages(cfg)
    bm25_meta = build_bm25_index(cfg, passages)
    faiss_meta = build_dense_index(cfg, passages)
    assert bm25_meta["num_passages"] == len(passages)
    assert faiss_meta["num_passages"] == len(passages)

    ret = HybridRetriever(cfg)
    hits = ret.retrieve("brake disc", "What materials does the vehicle component 'brake disc' consist of?")
    assert len(hits) == 2

    # 3. Chạy Experiment I (2 linh kiện) & Experiment II (2 linh kiện x 2 LLM x 3 phương pháp = 12 runs)
    s1 = execute_matrix(cfg, exp="exp1", limit=2)
    assert sum(s1.values()) == 4  # 2 linh kiện x (few-shot, zero-shot)
    s2 = execute_matrix(cfg, exp="exp2", limit=2)
    assert sum(s2.values()) == 12  # 2 linh kiện x 2 LLM x 3 phương pháp

    # Chạy lại lần nữa phải bỏ qua các run đã hoàn thành (idempotent checkpoint)
    s2_again = execute_matrix(cfg, exp="exp2", limit=2)
    assert sum(s2_again.values()) == 0

    # 4. Gán nhãn bởi Người (4 người cho Exp I, 3 người cho Exp II)
    for idx_h in range(1, 5):
        hname = f"human_{idx_h}"
        for it in R.exp1_items(cfg):
            R.save_label(cfg, "exp1", it["id"], hname,
                         {"plausible": [True] * len(it["materials"]), "dont_know": False}, kind="human")
    e2_items = R.exp2_items(cfg)
    for idx_h in range(1, 4):
        hname = f"human_{idx_h}"
        for it in e2_items:
            R.save_label(cfg, "exp2", it["id"], hname, {
                "plausible": [True] * len(it["materials"]),
                "incorrect_sentences": [],
                "material_rating": 4,
                "explanation_rating": 4,
                "comment": "",
            }, kind="human")
        for comp in {i["component"] for i in e2_items}:
            ids_c = [i["id"] for i in e2_items if i["component"] == comp]
            R.save_ranking(cfg, comp, hname, ids_c, kind="human")

    # 5. Chạy LLM-as-Judge (supplementary, tách biệt với human)
    judge_llm = get_llms(cfg, "judges")[0]
    assert judge_exp1(cfg, judge_llm, limit=2) == 2
    no, nr = judge_exp2(cfg, judge_llm, limit=2)
    assert no == 12 and nr == 2

    # 6. Tổng hợp báo cáo & xuất dữ liệu
    report_md, metrics = generate_report(cfg)
    assert "COVERAGE@2" in report_md
    assert "PRECISION@2" in report_md
    assert metrics["exp2"]["human"]["agreement_n"] == 2
    assert len(metrics["exp2"]["human"]["annotators"]) == 3
    assert metrics["exp2"]["llm_judge"]["annotators"] == ["judge_qwen38_27b"]

    exported = E.export_all(cfg)
    assert exported["exp2_runs"] == 12
    assert exported["llm_calls"] > 12

