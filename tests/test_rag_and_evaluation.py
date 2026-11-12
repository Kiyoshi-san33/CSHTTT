"""Kiểm tra chunking 200/50, RRF fusion, Coverage@2/Precision@2 và tách biệt Human vs LLM-judge."""
from __future__ import annotations

import numpy as np

from src.evaluation.agreement import human_llm_agreement, iaa_eq2, krippendorff_alpha, mean_kendall_tau
from src.evaluation.coverage import votes_for
from src.evaluation.factual_errors import error_rate_any, error_rate_mean
from src.evaluation.precision import coverage_precision
from src.rag.chunking import split_words
from src.rag.hybrid_retriever import rrf_fuse


def test_split_words_200_50():
    words = [f"w{i}" for i in range(450)]
    chunks = list(split_words(words, 200, 50))
    # step = 150 -> starts at 0 (0..200), 150 (150..350), 300 (300..450)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 200
    assert len(chunks[1].split()) == 200
    assert len(chunks[2].split()) == 150
    # 50 từ cuối của chunk 0 trùng với 50 từ đầu của chunk 1
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


def test_rrf_fuse_matches_haystack_formula():
    sparse = [10, 20, 30]
    dense = [20, 40, 10]
    fused = rrf_fuse([sparse, dense], rrf_k=60)
    # pid 20: rank 1 trong sparse -> 1/62, rank 0 trong dense -> 1/61
    assert np.isclose(fused[20], 1.0 / 62 + 1.0 / 61)
    # pid 10: rank 0 trong sparse -> 1/61, rank 2 trong dense -> 1/63
    assert np.isclose(fused[10], 1.0 / 61 + 1.0 / 63)
    assert fused[20] > fused[10] > fused[40] > fused[30]


def test_coverage_precision_at_2_and_human_llm_separation():
    items = [
        {"id": "brake disc||gpt_oss_120b||rag@adapted", "component": "brake disc",
         "model": "gpt_oss_120b", "method": "rag@adapted", "materials": ["cast iron", "wood"]},
        {"id": "piston||gpt_oss_120b||rag@adapted", "component": "piston",
         "model": "gpt_oss_120b", "method": "rag@adapted", "materials": ["plastic", "glass"]},
    ]
    ids = [i["id"] for i in items]
    n_mat = {i["id"]: len(i["materials"]) for i in items}

    # 3 người đánh giá:
    # - brake disc: cast iron được 3/3 đồng ý, wood được 1/3 đồng ý
    # - piston: plastic được 1/3 đồng ý, glass được 0/3 đồng ý
    human_anns = {
        "human_1": {
            ids[0]: {"plausible": [True, True], "incorrect_sentences": [], "material_rating": 5, "explanation_rating": 4},
            ids[1]: {"plausible": [True, False], "incorrect_sentences": [0], "material_rating": 2, "explanation_rating": 2},
        },
        "human_2": {
            ids[0]: {"plausible": [True, False], "incorrect_sentences": [], "material_rating": 4, "explanation_rating": 4},
            ids[1]: {"plausible": [False, False], "incorrect_sentences": [0], "material_rating": 2, "explanation_rating": 3},
        },
        "human_3": {
            ids[0]: {"plausible": [True, False], "incorrect_sentences": [], "material_rating": 5, "explanation_rating": 5},
            ids[1]: {"plausible": [False, False], "incorrect_sentences": [], "material_rating": 1, "explanation_rating": 2},
        },
    }
    votes = votes_for(ids, human_anns, n_mat)
    cov2, prec2 = coverage_precision(votes, 2)
    # Coverage@2: chỉ brake disc có >=1 vật liệu đạt >=2 phiếu -> 1/2 = 0.5
    # Precision@2: chỉ 1/4 vật liệu (cast iron) đạt >=2 phiếu -> 1/4 = 0.25
    assert np.isclose(cov2, 0.5)
    assert np.isclose(prec2, 0.25)

    assert np.isclose(error_rate_any(ids, human_anns), 0.5)
    assert np.isclose(error_rate_mean(ids, human_anns), 2.0 / 6.0)

    iaa, _ = iaa_eq2(human_anns, ids, n_mat)
    assert 0.5 <= iaa <= 1.0
    alpha = krippendorff_alpha(human_anns, ids, "material_rating")
    assert alpha > 0.5

    rankings = {
        "human_1": {"brake disc": [ids[0], ids[1]]},
        "human_2": {"brake disc": [ids[0], ids[1]]},
        "human_3": {"brake disc": [ids[0], ids[1]]},
    }
    assert np.isclose(mean_kendall_tau(rankings), 1.0)

    # Giám khảo LLM chấm lệch 1 vật liệu (plastic = True trong khi đa số người = False)
    llm_anns = {
        "judge_qwen38_27b": {
            ids[0]: {"plausible": [True, False], "incorrect_sentences": [], "material_rating": 5, "explanation_rating": 4},
            ids[1]: {"plausible": [True, False], "incorrect_sentences": [0], "material_rating": 3, "explanation_rating": 2},
        }
    }
    hl = human_llm_agreement(items, human_anns, llm_anns, min_human_agree=2)
    assert hl["n_common_items"] == 2
    assert len(hl["per_judge"]) == 1
    assert np.isclose(hl["per_judge"][0]["agreement_pct"], 0.75)  # khớp 3/4 vật liệu
    assert len(hl["disagreements"]) == 1
    assert hl["disagreements"][0]["material"] == "plastic"

