import pytest

from rag.evaluation import (
    average,
    average_by,
    detect_language,
    evaluate_question,
    load_eval_questions,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_expected_documents_in_top_k():
    retrieved = ["APP-08", "APP-08", "APP-07", "REC-01"]
    assert recall_at_k(retrieved, ["APP-08"], k=1) == 1.0
    assert recall_at_k(retrieved, ["APP-08", "APP-07"], k=2) == 0.5  # 2e morceau = même document
    assert recall_at_k(retrieved, ["APP-08", "APP-07"], k=3) == 1.0
    assert recall_at_k(retrieved, ["SUP-01"], k=4) == 0.0


def test_reciprocal_rank_uses_first_relevant_chunk():
    assert reciprocal_rank(["APP-08", "REC-01"], ["APP-08"]) == 1.0
    assert reciprocal_rank(["REC-01", "REC-02", "APP-08"], ["APP-08", "REC-02"]) == 0.5
    assert reciprocal_rank(["REC-01"], ["APP-08"]) == 0.0


def test_evaluate_question_and_averages_by_group():
    base = {"question": "?", "style": "standard"}
    k_values = (1, 2)
    rows = [
        # recall@1 = 1, recall@2 = 1, rr = 1
        evaluate_question({**base, "id": "Q1", "type": "factuelle", "sources": ["A"]}, ["A", "B"], k_values),
        # recall@1 = 0, recall@2 = 1, rr = 0,5
        evaluate_question({**base, "id": "Q2", "type": "factuelle", "sources": ["A"]}, ["B", "A"], k_values),
        # 1 source sur 2 retrouvée : recall@1 = recall@2 = 0,5, rr = 1
        evaluate_question({**base, "id": "Q3", "type": "multi_docs", "sources": ["A", "C"]}, ["C", "B"], k_values),
    ]
    assert average(rows, k_values) == pytest.approx({"n": 3, "recall@1": 1.5 / 3, "recall@2": 2.5 / 3, "mrr": 2.5 / 3})
    by_type = average_by(rows, "type", k_values)
    assert by_type["factuelle"] == pytest.approx({"n": 2, "recall@1": 0.5, "recall@2": 1.0, "mrr": 0.75})
    assert by_type["multi_docs"]["recall@2"] == 0.5


def test_detect_language_on_real_answers():
    assert detect_language("You need a minimum of **1,500 points** to redeem a gift card. [REC-04]") == "en"
    assert detect_language("Pour quitter le panel, vous pouvez vous rendre dans Profil > Mon compte [CPT-05].") == "fr"


def test_load_eval_questions_keeps_only_split_questions_with_sources():
    dev = load_eval_questions("dev")
    test = load_eval_questions("test")
    assert dev and test
    assert all(q["split"] == "dev" and q["sources"] for q in dev)
    assert not {q["id"] for q in dev} & {q["id"] for q in test}
