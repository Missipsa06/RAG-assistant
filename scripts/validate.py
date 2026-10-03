"""Contrôle de cohérence du corpus et du jeu de test.
Usage : python scripts/validate.py (depuis la racine du projet)
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"id", "titre", "categorie", "version", "date_maj"}
COMPORTEMENTS = {"repondre", "repondre_sans_inventer", "refuser", "escalader"}


def parse_frontmatter(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path.name} : frontmatter manquant")
    meta = {}
    for line in lines[1:]:
        if line == "---":
            return meta
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    raise ValueError(f"{path.name} : frontmatter non fermé")


def main() -> int:
    errors = []
    docs = {}
    for path in sorted((ROOT / "corpus").rglob("*.md")):
        meta = parse_frontmatter(path)
        missing = REQUIRED - meta.keys()
        if missing:
            errors.append(f"{path.name} : champs manquants {missing}")
        if meta.get("categorie") != path.parent.name:
            errors.append(f"{path.name} : catégorie != dossier")
        if meta["id"] in docs:
            errors.append(f"id dupliqué : {meta['id']}")
        docs[meta["id"]] = path

    questions = [json.loads(l) for l in (ROOT / "eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l]
    ids = Counter(q["id"] for q in questions)
    errors += [f"question dupliquée : {i}" for i, n in ids.items() if n > 1]
    cited = set()
    for q in questions:
        if q.get("split") not in {"dev", "test"}:
            errors.append(f"{q['id']} : split manquant (lancer scripts/split_questions.py)")
        if q["comportement"] not in COMPORTEMENTS:
            errors.append(f"{q['id']} : comportement inconnu {q['comportement']}")
        for s in q["sources"]:
            cited.add(s)
            if s not in docs:
                errors.append(f"{q['id']} : source inexistante {s}")
        if q["comportement"] == "repondre" and not q["sources"]:
            errors.append(f"{q['id']} : réponse attendue sans source")

    print(f"Documents : {len(docs)} | Questions : {len(questions)}")
    for field in ("split", "type", "comportement", "categorie", "style"):
        print(f"\n{field} :", dict(Counter(q[field] for q in questions)))
    uncovered = sorted(set(docs) - cited)
    print("\nDocuments jamais cités :", uncovered or "aucun")

    if errors:
        print("\nERREURS :", *errors, sep="\n- ")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
