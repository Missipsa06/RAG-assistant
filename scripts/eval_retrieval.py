"""Évaluation de la recherche : recall@k et MRR, ventilés par type et par style de question.

Usage : uv run python scripts/eval_retrieval.py --run-name baseline_vector --retrieval vector
        uv run python scripts/eval_retrieval.py --run-name hybrid_w05 --retrieval hybrid --vector-weight 0.5
        (option --split dev|test ; le test est réservé à l'évaluation finale)

Chaque exécution est enregistrée dans MLflow (paramètres, métriques, détail par question).
"""
import argparse
import json
import tempfile
from pathlib import Path

import mlflow
from langchain_core.runnables import Runnable

from rag import config
from rag.evaluation import average, average_by, evaluate_question, load_eval_questions, metric_names
from rag.retrieval import BM25_ANALYZERS, RETRIEVAL_MODES, get_retriever


def run_retrieval(questions: list[dict], retriever: Runnable) -> list[dict]:
    """Interroge le retriever pour chaque question et calcule ses métriques.

    Le retriever doit renvoyer k = max(EVAL_K_VALUES) morceaux : le top 1 et le top 3 sont
    les débuts de cette même liste, inutile de relancer la recherche pour chaque k.
    """
    rows = []
    for question in questions:
        chunks = retriever.invoke(question["question"])
        rows.append(evaluate_question(question, [chunk.metadata["id"] for chunk in chunks]))
    return rows


def print_table(title: str, table: dict[str, dict[str, float]]) -> None:
    names = metric_names()
    print(f"\n{title}")
    print(f"  {'':<20} {'n':>3} " + " ".join(f"{name:>9}" for name in names))
    for value, metrics in table.items():
        print(f"  {value:<20} {metrics['n']:>3} " + " ".join(f"{metrics[name]:>9.3f}" for name in names))


def print_failures(rows: list[dict]) -> None:
    """Questions dont un document attendu manque dans le top k : la matière de l'analyse d'erreurs."""
    k = max(config.EVAL_K_VALUES)
    failures = [row for row in rows if row[f"recall@{k}"] < 1]
    print(f"\nQuestions avec un document attendu absent du top {k} : {len(failures)}")
    for row in failures:
        print(f"  {row['id']} [{row['type']}/{row['style']}] {row['question']}")
        print(f"      attendu {row['sources']} | retrouvé {row['retrieved']}")


def mlflow_name(metric: str) -> str:
    """MLflow n'accepte pas « @ » dans un nom de métrique : recall@5 → recall_at_5."""
    return metric.replace("@", "_at_")


def log_to_mlflow(run_name: str, params: dict, rows: list[dict]) -> None:
    """Enregistre le run : paramètres du pipeline, métriques globales et par groupe, détail."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            **params,
            "n_questions": len(rows),
            "embedding_model": config.EMBEDDING_MODEL,
            "chunking": "markdown_headers",
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
        })
        overall = average(rows)
        mlflow.log_metrics({mlflow_name(name): overall[name] for name in metric_names()})
        for field in ("type", "style"):
            for value, metrics in average_by(rows, field).items():
                mlflow.log_metrics({f"{mlflow_name(name)}.{field}.{value}": metrics[name] for name in metric_names()})
        with tempfile.TemporaryDirectory() as tmp:
            details = Path(tmp) / "per_question.json"
            details.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(details))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--retrieval", default=config.RETRIEVAL_MODE, choices=RETRIEVAL_MODES)
    parser.add_argument("--vector-weight", type=float, default=config.VECTOR_WEIGHT)
    parser.add_argument("--bm25-analyzer", default=config.BM25_ANALYZER, choices=list(BM25_ANALYZERS))
    args = parser.parse_args()

    params = {"split": args.split, "retrieval": args.retrieval}
    if args.retrieval in ("bm25", "hybrid"):
        params["bm25_analyzer"] = args.bm25_analyzer
    if args.retrieval == "hybrid":
        params |= {"vector_weight": args.vector_weight, "fusion_candidates": config.FUSION_CANDIDATES, "rrf_c": config.RRF_C}

    retriever = get_retriever(
        args.retrieval, k=max(config.EVAL_K_VALUES), vector_weight=args.vector_weight, bm25_analyzer=args.bm25_analyzer
    )
    rows = run_retrieval(load_eval_questions(args.split), retriever)
    print_table("Global", {"tous": average(rows)})
    print_table("Par type", average_by(rows, "type"))
    print_table("Par style", average_by(rows, "style"))
    print_failures(rows)
    log_to_mlflow(args.run_name, params, rows)


if __name__ == "__main__":
    main()
