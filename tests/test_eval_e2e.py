"""Tests de la logique de l'évaluation de bout en bout (sans LLM) : métriques et seuils de la CI."""
from eval_e2e import check_thresholds, summarize

from rag import config


def row(comportement="repondre", **overrides):
    base = {
        "id": "Q1", "comportement": comportement, "style": "standard", "answer": "Réponse [REC-01].",
        "action": "repondre", "expected_action": "repondre", "invalid_citation": False, "latency_s": 1.0,
        "fidele": True, "correcte": True, "comportement_ok": True, "erreur_juge": False,
    }
    return base | overrides


def test_summarize_excludes_unreadable_verdicts_and_counts_them():
    rows = [row(), row(fidele=False), row(erreur_juge=True, fidele=False)]
    metrics = summarize(rows)
    assert metrics["faithfulness"] == 0.5  # la ligne en erreur n'est pas comptée
    assert round(metrics["judge_error_rate"], 3) == 0.333


def test_summarize_checks_language_of_english_questions():
    rows = [row(style="anglais", answer="You need 1,500 points to redeem a gift card."),
            row(style="anglais", answer="Pour quitter le panel, vous pouvez écrire au support.")]
    assert summarize(rows)["language_ok.anglais"] == 0.5


def perfect_metrics() -> dict[str, float]:
    return {name: 1.0 for name in config.CI_THRESHOLDS} | {"invalid_citation_rate": 0.0}


def test_thresholds_pass_on_perfect_metrics():
    assert check_thresholds(perfect_metrics()) == []


def test_thresholds_fail_on_a_single_security_regression():
    metrics = perfect_metrics() | {"behavior_ok.refuser": 5 / 6}
    failures = check_thresholds(metrics)
    assert len(failures) == 1 and "behavior_ok.refuser" in failures[0]


def test_thresholds_fail_on_invented_citations():
    assert check_thresholds(perfect_metrics() | {"invalid_citation_rate": 2 / 67})
