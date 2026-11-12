"""Kiểm tra hệ thống nạp prompt, phiên bản SHA-256, cấu hình profile và hàm parse."""
from __future__ import annotations

from src.core.config import load_config
from src.core.prompts import all_prompt_versions, load_prompt
from src.experiments.parsing import parse_explained, parse_list, parse_questions, split_sentences


def test_retrieval_profiles():
    cfg_adapted = load_config(profile="adapted")
    assert cfg_adapted["active_profile"] == "adapted"
    assert cfg_adapted["retrieval"]["dense_backend"] == "bge"
    assert cfg_adapted["retrieval"]["bge_model"] == "BAAI/bge-small-en-v1.5"
    assert cfg_adapted["retrieval"]["passage_words"] == 200
    assert cfg_adapted["retrieval"]["passage_overlap"] == 50
    assert cfg_adapted["retrieval"]["sparse_top_k"] == 50
    assert cfg_adapted["retrieval"]["dense_top_k"] == 50
    assert cfg_adapted["retrieval"]["final_top_k"] == 10

    cfg_repro = load_config(profile="reproduction")
    assert cfg_repro["active_profile"] == "reproduction"
    assert cfg_repro["retrieval"]["dense_backend"] == "dpr"
    assert "facebook/dpr-ctx_encoder-multiset-base" in cfg_repro["retrieval"]["ctx_encoder"]


def test_configured_models_respect_technical_review():
    cfg = load_config()
    gen_models = {g["model"] for g in cfg["generators"]}
    assert gen_models == {"openai/gpt-oss-120b", "gemini-3.8-flash"}
    dev_models = {g["model"] for g in cfg["dev_generators"]}
    assert dev_models == {"qwen3:8b"}
    judge_models = {j["model"] for j in cfg["judges"]}
    assert "qwen/qwen3.8-27b" in judge_models
    # Không sử dụng các model đã bị loại bỏ
    all_models = gen_models | dev_models | judge_models
    assert "llama-3.3-70b-versatile" not in all_models
    assert "qwen/qwen3-32b" not in all_models


def test_prompts_load_and_version():
    versions = all_prompt_versions()
    expected = {
        "exp1_fewshot", "exp1_zeroshot", "prediction", "rag",
        "cove_plan", "cove_execute", "cove_final",
        "judge_system", "judge_exp1", "judge_exp2", "judge_rank",
    }
    assert expected.issubset(set(versions))
    p = load_prompt("exp1_fewshot")
    assert p.version.startswith("exp1_fewshot@")
    rendered = p.render(component="brake disc")
    assert "brake disc" in rendered
    assert "grey cast iron, carbon-ceramic composite, ceramic" in rendered


def test_parsing_helpers():
    raw_exp2 = (
        "1. **Grey Cast Iron**: widely used for brake discs due to thermal conductivity.\n"
        "2. **Carbon-Ceramic Composite**: used in high-performance sports cars.\n"
        "3. Grey cast iron: duplicate material should be deduplicated."
    )
    assert parse_explained(raw_exp2) == ["Grey Cast Iron", "Carbon-Ceramic Composite"]
    assert parse_list("Materials: steel, aluminium alloy, cast iron, steel, rubber, copper, glass", k=5) == [
        "steel", "aluminium alloy", "cast iron", "rubber", "copper"
    ]
    qs = parse_questions("1. Is cast iron used in brake discs?\n2. Statement without question mark.\n3. Is ceramic suitable?", k=5)
    assert qs == ["Is cast iron used in brake discs?", "Is ceramic suitable?"]
    sents = split_sentences("First sentence. Second sentence! Third sentence?")
    assert len(sents) == 3

