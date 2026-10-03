"""Métriques d'évaluation de la recherche, calculées au niveau des documents.

Les annotations du jeu d'évaluation indiquent les documents sources attendus (ex. REC-04),
pas les sections : un morceau est donc jugé pertinent s'il appartient à un document attendu.

Convention : k compte des **morceaux** (ce que le LLM recevra), pas des documents distincts.
Si deux des 5 morceaux viennent du même document, le LLM ne voit que 4 documents.
"""
import json
import re
from collections import defaultdict
from statistics import mean

from rag import config


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """Part des documents attendus présents dans les k premiers morceaux (entre 0 et 1).

    Pour une question à 2 sources dont une seule est retrouvée, le recall vaut 0,5.
    """
    found = set(retrieved_doc_ids[:k]) & set(expected_doc_ids)
    return len(found) / len(expected_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """1 / rang du premier morceau pertinent (1 si en tête, 0,5 si 2e...), 0 si aucun.

    Moyenné sur toutes les questions, cela donne le MRR : il récompense le fait de
    trouver le bon document tôt, pas seulement de le trouver.
    """
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1 / rank
    return 0.0


def evaluate_question(question: dict, retrieved_doc_ids: list[str], k_values=config.EVAL_K_VALUES) -> dict:
    """Résultat détaillé d'une question : ses attributs, ce qui a été retrouvé, ses métriques."""
    row = {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "style": question["style"],
        "sources": question["sources"],
        "retrieved": retrieved_doc_ids,
    }
    for k in k_values:
        row[f"recall@{k}"] = recall_at_k(retrieved_doc_ids, question["sources"], k)
    row["mrr"] = reciprocal_rank(retrieved_doc_ids, question["sources"])
    return row


def metric_names(k_values=config.EVAL_K_VALUES) -> list[str]:
    return [f"recall@{k}" for k in k_values] + ["mrr"]


def average(rows: list[dict], k_values=config.EVAL_K_VALUES) -> dict[str, float]:
    """Moyenne de chaque métrique sur un ensemble de questions, avec leur nombre `n`."""
    return {"n": len(rows), **{name: mean(row[name] for row in rows) for name in metric_names(k_values)}}


def average_by(rows: list[dict], field: str, k_values=config.EVAL_K_VALUES) -> dict[str, dict[str, float]]:
    """Moyennes ventilées par valeur d'un champ (ex. field="type" → une ligne par type)."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {value: average(group, k_values) for value, group in sorted(groups.items())}


# Mots très fréquents, quasi exclusifs à chaque langue : de quoi distinguer une réponse en
# anglais d'une réponse en français, sans dépendance (questions « anglais » du jeu d'évaluation).
_FRENCH_MARKERS = {"le", "la", "les", "des", "vous", "votre", "est", "pour", "dans", "une", "et", "sur", "pas"}
_ENGLISH_MARKERS = {"the", "you", "your", "is", "to", "for", "in", "and", "of", "can", "not", "be", "on"}


def detect_language(text: str) -> str:
    """« en » ou « fr » selon les mots courants les plus présents (heuristique, pour l'évaluation)."""
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    english = sum(word in _ENGLISH_MARKERS for word in words)
    french = sum(word in _FRENCH_MARKERS for word in words)
    return "en" if english > french else "fr"


def load_eval_questions(split: str) -> list[dict]:
    """Questions d'un split qui ont au moins un document source.

    Les questions sans source (hors sujet, informations absentes du corpus) sont exclues :
    le recall n'y a pas de sens. Elles sont évaluées sur le comportement (phases 4 à 6).
    """
    lines = config.QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line]
    return [q for q in questions if q["split"] == split and q["sources"]]
