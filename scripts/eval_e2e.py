"""Évaluation de bout en bout : recherche, routage, génération, jugement, seuils de la CI.

Usage : uv run python scripts/eval_e2e.py --run-name e2e_dev
        uv run python scripts/eval_e2e.py --run-name e2e_dev --check   (code de sortie 1 si un seuil n'est pas atteint)
        (option --split dev|test ; le test est réservé à l'évaluation finale, jamais à la CI)

Métriques (toutes entre 0 et 1) :
- retrieval_recall_at_5, retrieval_mrr : recherche seule, questions ayant des sources ;
- routing_accuracy : action du routeur = action attendue ;
- faithfulness : réponses fidèles aux extraits (juge), sur toutes les questions ;
- correctness.repondre : réponses contenant les éléments attendus (juge), questions « repondre » ;
- behavior_ok.<comportement> : comportement attendu respecté (juge) ;
- invalid_citation_rate : réponses citant un document absent des extraits (déterministe).
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean

import mlflow

from eval_generation import evaluate_one, load_split_questions
from eval_routing import expected_action
from rag import config
from rag.evaluation import detect_language, load_eval_questions, recall_at_k, reciprocal_rank
from rag.generation import format_context, get_llm
from rag.judge import JUDGE_PROMPT, judge_answer
from rag.retrieval import get_full_documents, get_retriever

BEHAVIORS = ("repondre", "repondre_sans_inventer", "refuser", "escalader")


def retrieval_metrics(split: str, retriever) -> dict[str, float]:
    """Recall@5 et MRR de la recherche seule (sans LLM), au niveau document."""
    results = []
    for question in load_eval_questions(split):
        retrieved = [chunk.metadata["id"] for chunk in retriever.invoke(question["question"])]
        results.append((recall_at_k(retrieved, question["sources"], config.TOP_K), reciprocal_rank(retrieved, question["sources"])))
    return {"retrieval_recall_at_5": mean(r for r, _ in results), "retrieval_mrr": mean(rr for _, rr in results)}


def judge_rows(rows: list[dict], questions: dict[str, dict], judge_llm) -> None:
    """Ajoute à chaque ligne les verdicts du juge (en place)."""
    for row in rows:
        # Le contexte jugé est celui que le LLM a reçu : en mode « document », les documents entiers.
        context_text = format_context(get_full_documents(row["context_ids"])) if row["context_ids"] else ""
        verdict = judge_answer(questions[row["id"]], row["answer"], context_text, judge_llm)
        row |= verdict.model_dump()


def summarize(rows: list[dict]) -> dict[str, float]:
    """Métriques de bout en bout ; les verdicts illisibles du juge sont exclus et comptés."""
    judged = [row for row in rows if not row["erreur_juge"]]
    metrics = {
        "routing_accuracy": mean(row["action"] == row["expected_action"] for row in rows),
        "faithfulness": mean(row["fidele"] for row in judged),
        "correctness.repondre": mean(row["correcte"] for row in judged if row["comportement"] == "repondre"),
        "invalid_citation_rate": mean(row["invalid_citation"] for row in rows),
        "judge_error_rate": 1 - len(judged) / len(rows),
        "latency_mean_s": mean(row["latency_s"] for row in rows),
    }
    for behavior in BEHAVIORS:
        group = [row for row in judged if row["comportement"] == behavior]
        if group:
            metrics[f"behavior_ok.{behavior}"] = mean(row["comportement_ok"] for row in group)
    # Contrôle déterministe ajouté après une régression passée inaperçue (D016) : une question en
    # anglais recevait une réponse en français, ce que ni les citations ni le juge ne détectaient.
    english = [row for row in rows if row["style"] == "anglais" and row["action"] != "refuser_hors_sujet"]
    if english:
        metrics["language_ok.anglais"] = mean(detect_language(row["answer"]) == "en" for row in english)
    return metrics


def check_thresholds(metrics: dict[str, float]) -> list[str]:
    """Liste des seuils de la CI non atteints (vide si tout passe)."""
    failures = []
    for name, minimum in config.CI_THRESHOLDS.items():
        if metrics[name] < minimum:
            failures.append(f"{name} = {metrics[name]:.3f} < seuil {minimum:.2f}")
    if metrics["invalid_citation_rate"] > config.CI_MAX_INVALID_CITATION_RATE:
        failures.append(f"invalid_citation_rate = {metrics['invalid_citation_rate']:.3f} > {config.CI_MAX_INVALID_CITATION_RATE:.2f}")
    return failures


def print_report(rows: list[dict], metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        print(f"  {name:<30} {value:.3f}")
    print("\nRéponses jugées non fidèles ou au comportement non respecté :")
    for row in rows:
        if not row["erreur_juge"] and (not row["fidele"] or not row["comportement_ok"]):
            print(f"  {row['id']} [{row['comportement']} → {row['action']}] {row['question']}")
            print(f"      juge : {row['explication']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--check", action="store_true", help="échoue (code 1) si un seuil de la CI n'est pas atteint")
    args = parser.parse_args()

    questions = {q["id"]: q for q in load_split_questions(args.split)}
    retriever, llm = get_retriever(), get_llm()
    rows = [evaluate_one(q, retriever, llm, config.CONTEXT_MODE) for q in questions.values()]
    for row in rows:
        row["expected_action"] = expected_action(questions[row["id"]])
        row["style"] = questions[row["id"]]["style"]
    judge_rows(rows, questions, get_llm(config.JUDGE_MODEL, config.JUDGE_MAX_TOKENS))

    metrics = retrieval_metrics(args.split, retriever) | summarize(rows)
    print_report(rows, metrics)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("e2e")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "split": args.split, "n_questions": len(rows), "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL, "retrieval": config.RETRIEVAL_MODE,
            "context_mode": config.CONTEXT_MODE, "top_k": config.TOP_K,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_text(JUDGE_PROMPT, "judge_prompt.txt")
        with tempfile.TemporaryDirectory() as tmp:
            details = Path(tmp) / "per_question.json"
            details.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(details))

    if args.check:
        failures = check_thresholds(metrics)
        print("\nSeuils de la CI : " + ("tous atteints" if not failures else "ÉCHEC"))
        for failure in failures:
            print(f"  ✗ {failure}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
