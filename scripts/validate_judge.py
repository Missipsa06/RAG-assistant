"""Validation du LLM-juge : accord avec des verdicts humains (eval/judge_labels.jsonl).

Usage : uv run python scripts/validate_judge.py

Les réponses jugées sont relues dans les runs MLflow de l'expérience « generation ». Le contexte
fourni au juge est reconstruit à l'identique : documents entiers en mode « document », sinon les
morceaux retrouvés par la recherche (déterministe, même index).
"""
import json

import mlflow

from rag import config
from rag.generation import format_context, get_llm
from rag.judge import judge_answer
from rag.retrieval import get_full_documents, get_retriever

LABELS_PATH = config.ROOT / "eval/judge_labels.jsonl"
VERDICTS = ("fidele", "correcte", "comportement_ok")


def load_run_rows(run_name: str) -> tuple[dict[str, dict], str]:
    """Réponses d'un run de génération (par id de question) et son mode de contexte."""
    runs = mlflow.search_runs(experiment_names=["generation"], filter_string=f"tags.mlflow.runName = '{run_name}'")
    run = runs.iloc[0]
    path = mlflow.artifacts.download_artifacts(run_id=run.run_id, artifact_path="per_question.json")
    rows = {row["id"]: row for row in json.load(open(path, encoding="utf-8"))}
    return rows, run.get("params.context_mode") or "section"  # les premiers runs n'enregistraient pas ce paramètre


def main() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    questions = {q["id"]: q for q in map(json.loads, config.QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()) if q}
    labels = [json.loads(line) for line in LABELS_PATH.read_text(encoding="utf-8").splitlines() if line]
    judge_llm, retriever = get_llm(config.JUDGE_MODEL, config.JUDGE_MAX_TOKENS), get_retriever()

    agreements = {verdict: [] for verdict in VERDICTS}
    runs_cache = {}
    for label in labels:
        if label["run"] not in runs_cache:
            runs_cache[label["run"]] = load_run_rows(label["run"])
        rows, context_mode = runs_cache[label["run"]]
        row = rows[label["id"]]
        if context_mode == "document":
            context = get_full_documents(row["context_ids"])
        else:
            context = retriever.invoke(row["question"])
        verdict = judge_answer(questions[label["id"]], row["answer"], format_context(context), judge_llm)

        for name in VERDICTS:
            if label[name] is not None:
                agree = getattr(verdict, name) == label[name]
                agreements[name].append(agree)
                if not agree:
                    print(f"DÉSACCORD {label['run']} {label['id']} {name} : humain {label[name]}, juge {getattr(verdict, name)}")
                    print(f"    note humaine : {label['note']}\n    juge : {verdict.explication}")

    print()
    for name, values in agreements.items():
        print(f"Accord {name:<16} {sum(values)}/{len(values)}")
    total = [agree for values in agreements.values() for agree in values]
    print(f"Accord global          {sum(total)}/{len(total)} = {sum(total) / len(total):.0%}")


if __name__ == "__main__":
    main()
