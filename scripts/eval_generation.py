"""Évaluation de la génération ancrée, avec des métriques déterministes (sans LLM-juge).

Usage : uv run python scripts/eval_generation.py --run-name gen_baseline
        (option --split dev|test ; le test est réservé à l'évaluation finale)

Pour chaque question du split, on lance le pipeline complet (recherche + LLM) et on mesure,
par comportement attendu :
- answered              : le LLM a répondu (pas de marqueur « hors documentation ») ;
- cites_a_source        : la réponse cite au moins un document ;
- cites_expected_source : elle cite au moins un des documents sources annotés ;
- invalid_citation      : elle cite un document absent des extraits fournis (citation inventée).
Attendu : answered élevé pour « repondre », bas pour « repondre_sans_inventer ». Les questions
« refuser » et « escalader » sont mesurées pour servir de point de départ à la phase 5.

La justesse et la fidélité des réponses sont évaluées en phase 6 (LLM-juge) ; ici, le détail
de chaque réponse est enregistré dans MLflow pour relecture humaine.
"""
import argparse
import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import mlflow

from rag import config
from rag.generation import SYSTEM_PROMPT, get_llm
from rag.pipeline import answer_question
from rag.retrieval import get_retriever

BOOLEAN_METRICS = ("answered", "cites_a_source", "cites_expected_source", "invalid_citation")


def load_split_questions(split: str) -> list[dict]:
    """Toutes les questions du split, y compris celles sans source (pièges, hors sujet)."""
    lines = config.QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    return [q for q in (json.loads(line) for line in lines if line) if q["split"] == split]


def evaluate_one(question: dict, retriever, llm, context_mode: str) -> dict:
    """Lance le pipeline sur une question et calcule ses indicateurs."""
    start = time.perf_counter()
    answer = answer_question(question["question"], retriever=retriever, llm=llm, context_mode=context_mode)
    return {
        "id": question["id"],
        "type": question["type"],
        "comportement": question["comportement"],
        "question": question["question"],
        "sources": question["sources"],
        "reponse_attendue": question["reponse_attendue"],
        "answer": answer.text,
        "context_ids": answer.context_ids,
        "cited_ids": answer.cited_ids,
        "action": answer.action,
        "latency_s": round(time.perf_counter() - start, 2),
        "answered": not answer.not_in_docs,
        "cites_a_source": bool(answer.cited_ids),
        "cites_expected_source": bool(set(answer.cited_ids) & set(question["sources"])),
        "invalid_citation": bool(answer.invalid_citations),
    }


def rates_by_behavior(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Part de questions vérifiant chaque indicateur, par comportement attendu."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["comportement"]].append(row)
    return {
        behavior: {"n": len(group), **{m: mean(row[m] for row in group) for m in BOOLEAN_METRICS}}
        for behavior, group in sorted(groups.items())
    }


def print_report(rows: list[dict], table: dict[str, dict[str, float]]) -> None:
    print(f"\n  {'comportement':<24} {'n':>3} " + " ".join(f"{m:>22}" for m in BOOLEAN_METRICS))
    for behavior, rates in table.items():
        print(f"  {behavior:<24} {rates['n']:>3} " + " ".join(f"{rates[m]:>22.2f}" for m in BOOLEAN_METRICS))
    print(f"\nLatence moyenne : {mean(row['latency_s'] for row in rows):.1f} s")
    print("\nActions décidées par le routeur, par comportement attendu :")
    for behavior in table:
        actions = Counter(row["action"] for row in rows if row["comportement"] == behavior)
        print(f"  {behavior:<24} {dict(actions)}")

    # Les cas à relire en priorité : réponse refusée à tort, invention probable, citation inventée.
    suspects = [
        row for row in rows
        if (row["comportement"] == "repondre" and not row["answered"])
        or (row["comportement"] == "repondre_sans_inventer" and row["answered"])
        or row["invalid_citation"]
    ]
    print(f"\nRéponses à relire en priorité : {len(suspects)}")
    for row in suspects:
        print(f"\n  {row['id']} [{row['comportement']}] {row['question']}")
        print(f"    attendu : {row['reponse_attendue']}")
        print(f"    obtenu  : {row['answer']}")


def log_to_mlflow(run_name: str, split: str, context_mode: str, rows: list[dict], table: dict[str, dict[str, float]]) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("generation")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "split": split,
            "n_questions": len(rows),
            "llm_model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "retrieval": config.RETRIEVAL_MODE,
            "top_k": config.TOP_K,
            "context_mode": context_mode,
        })
        for behavior, rates in table.items():
            mlflow.log_metrics({f"{m}.{behavior}": rates[m] for m in BOOLEAN_METRICS})
        mlflow.log_metric("latency_mean_s", mean(row["latency_s"] for row in rows))
        # Le prompt fait partie du « modèle » : on le versionne avec chaque run.
        mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")
        with tempfile.TemporaryDirectory() as tmp:
            details = Path(tmp) / "per_question.json"
            details.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(details))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--context-mode", default=config.CONTEXT_MODE, choices=["section", "document"])
    args = parser.parse_args()

    retriever, llm = get_retriever(), get_llm()
    rows = [evaluate_one(q, retriever, llm, args.context_mode) for q in load_split_questions(args.split)]
    table = rates_by_behavior(rows)
    print_report(rows, table)
    log_to_mlflow(args.run_name, args.split, args.context_mode, rows, table)


if __name__ == "__main__":
    main()
