"""Découpage figé des questions d'évaluation en dev (~80 %) et test (~20 %).

Usage : uv run python scripts/split_questions.py (depuis la racine du projet)

Le split est écrit dans eval/questions.jsonl (champ "split"). Une question qui a déjà
un split le garde : relancer le script après ajout de questions ne fait jamais passer
une question de test en dev (ni l'inverse).
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "eval/questions.jsonl"

# 20 % : compromis entre un test assez gros pour mesurer quelque chose et un dev assez
# gros pour régler les paramètres. Avec 86 questions, c'est déjà peu (voir decisions.md).
TEST_RATIO = 0.2
SEED = 42

# Reformulations d'une même demande : elles doivent tomber dans le même split, sinon
# régler le prompt sur l'une revient à « voir » l'autre (fuite du test vers le dev).
# Toute nouvelle reformulation doit être ajoutée ici avec sa question d'origine en premier.
PARAPHRASES: list[tuple[str, ...]] = [
    ("Q003", "Q080"),  # seuil d'échange de points (FR / EN)
    ("Q006", "Q084"),  # bonus d'assiduité (standard / fautes)
    ("Q025", "Q083"),  # mot de passe oublié (standard / fautes)
    ("Q039", "Q076"),  # suppression des données (factuelle / escalade)
]


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Lit le fichier JSONL des questions."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def build_groups(questions: list[dict]) -> list[list[dict]]:
    """Regroupe les reformulations ; chaque autre question forme un groupe à elle seule."""
    by_id = {q["id"]: q for q in questions}
    grouped_ids = {qid for group in PARAPHRASES for qid in group}
    groups = [[by_id[qid] for qid in group] for group in PARAPHRASES]
    groups += [[q] for q in questions if q["id"] not in grouped_ids]
    return groups


def stratum(group: list[dict]) -> tuple[str, ...]:
    """Strate d'un groupe, pour que le test ressemble au dev sur les axes qu'on mesurera.

    - Styles non standard (familier, fautes, anglais) : strate à part, car rares (8 sur 86),
      un tirage au hasard pourrait n'en mettre aucun en test.
    - Questions auxquelles il faut répondre : strate par catégorie, pour évaluer le routage
      sur toutes les catégories (stratifier par type seul laissait 1 question
      « récompenses » sur 23 en test).
    - Autres types (piège, hors sujet, sécurité, escalade) : strate par type.
    """
    first = group[0]
    if any(q["style"] != "standard" for q in group):
        return ("non_standard",)
    if first["type"] in {"factuelle", "multi_docs"}:
        return ("repondable", first["categorie"])
    return (first["type"],)


def assign_splits(questions: list[dict], test_ratio: float = TEST_RATIO, seed: int = SEED) -> None:
    """Ajoute le champ "split" ("dev" ou "test") aux questions qui n'en ont pas, en place."""
    rng = random.Random(seed)
    unsplit_by_stratum: dict[tuple[str, bool], list[list[dict]]] = defaultdict(list)

    for group in build_groups(questions):
        existing = {q["split"] for q in group if "split" in q}
        if len(existing) > 1:
            raise ValueError(f"Reformulations dans des splits différents : {[q['id'] for q in group]}")
        if existing:
            # Une nouvelle reformulation hérite du split de sa question d'origine.
            split = existing.pop()
            for q in group:
                q["split"] = split
        else:
            unsplit_by_stratum[stratum(group)].append(group)

    for key in sorted(unsplit_by_stratum):
        groups = unsplit_by_stratum[key]
        rng.shuffle(groups)
        n_test = round(len(groups) * test_ratio)
        for rank, group in enumerate(groups):
            for q in group:
                q["split"] = "test" if rank < n_test else "dev"


def main() -> None:
    questions = load_questions()
    assign_splits(questions)
    QUESTIONS_PATH.write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf-8"
    )
    print("split :", dict(Counter(q["split"] for q in questions)))
    for field in ("type", "style", "comportement", "categorie"):
        counts = Counter((q[field], q["split"]) for q in questions)
        print(f"\n{field} (dev / test) :")
        for value in sorted({q[field] for q in questions}):
            print(f"  {value:<24} {counts[(value, 'dev')]:>3} / {counts[(value, 'test')]:>2}")


if __name__ == "__main__":
    main()
