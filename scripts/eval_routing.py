"""Évaluation du routage (garde-fou + routeur LLM) : action et catégorie, contre les annotations.

Usage : uv run python scripts/eval_routing.py --run-name router_v1
        (option --split dev|test ; le test est réservé à l'évaluation finale)

L'action attendue se déduit des annotations :
- comportement « repondre » ou « repondre_sans_inventer » → repondre
- « escalader » → escalader
- « refuser » d'une question hors sujet → refuser_hors_sujet ; sinon → refuser_donnees_personnelles
"""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import mlflow

from rag import config
from rag.generation import get_llm
from rag.guardrails import scan_pii
from rag.pipeline import decide_route
from rag.router import ROUTER_PROMPT

ACTIONS = ("repondre", "escalader", "refuser_hors_sujet", "refuser_donnees_personnelles")


def expected_action(question: dict) -> str:
    if question["comportement"] in ("repondre", "repondre_sans_inventer"):
        return "repondre"
    if question["comportement"] == "escalader":
        return "escalader"
    return "refuser_hors_sujet" if question["type"] == "hors_sujet" else "refuser_donnees_personnelles"


def evaluate(questions: list[dict], llm) -> list[dict]:
    rows = []
    for question in questions:
        safe_question, pii_types = scan_pii(question["question"])
        route = decide_route(safe_question, pii_types, llm)
        rows.append({
            "id": question["id"],
            "question": question["question"],
            "expected_action": expected_action(question),
            "action": route.action,
            "expected_category": question["categorie"],
            "category": route.categorie,
            "reason": route.raison,
            "fallback": route.fallback,
        })
    return rows


def recall_by_action(rows: list[dict]) -> dict[str, float]:
    """Pour chaque action attendue, part des questions correctement routées (rappel par classe)."""
    result = {}
    for action in ACTIONS:
        group = [row for row in rows if row["expected_action"] == action]
        if group:
            result[action] = sum(row["action"] == action for row in group) / len(group)
    return result


def print_report(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    metrics = {
        "action_accuracy": sum(r["action"] == r["expected_action"] for r in rows) / n,
        "category_accuracy": sum(r["category"] == r["expected_category"] for r in rows) / n,
        "fallback_rate": sum(r["fallback"] for r in rows) / n,
    }
    for name, value in metrics.items():
        print(f"{name:<20} {value:.3f}")

    print("\nMatrice de confusion des actions (lignes : attendu, colonnes : obtenu)")
    confusion = Counter((r["expected_action"], r["action"]) for r in rows)
    short = {a: a[:12] for a in ACTIONS}
    print(f"  {'':<30}" + "".join(f"{short[a]:>14}" for a in ACTIONS))
    for expected in ACTIONS:
        print(f"  {expected:<30}" + "".join(f"{confusion[(expected, got)]:>14}" for got in ACTIONS))

    print("\nErreurs d'action :")
    for r in rows:
        if r["action"] != r["expected_action"]:
            print(f"  {r['id']} attendu {r['expected_action']}, obtenu {r['action']} | {r['question']}")
            print(f"      raison donnée : {r['reason']}")
    print("\nErreurs de catégorie :")
    for r in rows:
        if r["category"] != r["expected_category"]:
            print(f"  {r['id']} attendu {r['expected_category']}, obtenu {r['category']} | {r['question']}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    lines = config.QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    questions = [q for q in (json.loads(line) for line in lines if line) if q["split"] == args.split]
    rows = evaluate(questions, get_llm())
    metrics = print_report(rows)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("routing")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({"split": args.split, "n_questions": len(rows), "llm_model": config.LLM_MODEL})
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"recall.{action}": value for action, value in recall_by_action(rows).items()})
        mlflow.log_text(ROUTER_PROMPT, "router_prompt.txt")
        with tempfile.TemporaryDirectory() as tmp:
            details = Path(tmp) / "per_question.json"
            details.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(details))


if __name__ == "__main__":
    main()
