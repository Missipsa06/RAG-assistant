"""Chaîne complète question → réponse : garde-fous, routage, recherche, génération ancrée.

    question ─► garde-fou (masque les données personnelles) ─► routeur ─┬─ refuser_hors_sujet → message fixe
                                                                        ├─ refuser_donnees_personnelles → génération + consigne
                                                                        ├─ escalader → génération + fiches support + consigne
                                                                        └─ repondre → génération
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from rag import config
from rag.generation import (
    ESCALATION_INSTRUCTION,
    PERSONAL_DATA_INSTRUCTION,
    Answer,
    generate_answer,
    get_llm,
)
from rag.guardrails import BLOCKING_PII, OFF_TOPIC_ANSWER, scan_pii
from rag.retrieval import expand_to_full_documents, get_full_documents, get_retriever
from rag.router import Route, route_question

# Fiches ajoutées au contexte en cas d'escalade : comment joindre le support et faire une
# réclamation. La recherche ne les ramène pas toujours (ex. « Je veux parler à un humain »).
SUPPORT_DOC_IDS = ("SUP-01", "SUP-02")

INSTRUCTIONS = {
    "escalader": ESCALATION_INSTRUCTION,
    "refuser_donnees_personnelles": PERSONAL_DATA_INSTRUCTION,
}


def decide_route(safe_question: str, pii_types: set[str], llm: BaseChatModel) -> Route:
    """Route décidée par le LLM, sauf si un garde-fou déterministe impose le refus."""
    if pii_types & BLOCKING_PII:
        # Pas besoin du LLM : la présence de coordonnées bancaires suffit à décider.
        return Route(categorie="rgpd", action="refuser_donnees_personnelles", raison="coordonnées bancaires détectées")
    return route_question(safe_question, llm)


def answer_question(
    question: str,
    retriever: Runnable | None = None,
    llm: BaseChatModel | None = None,
    context_mode: str = config.CONTEXT_MODE,
) -> Answer:
    """Répond à une question. Retriever et LLM sont injectables, pour les tests et l'évaluation.

    context_mode : "section" (morceaux retrouvés) ou "document" (documents entiers de ces morceaux).
    """
    llm = llm or get_llm()
    # Les données personnelles ne quittent jamais la machine : seule la version masquée est
    # envoyée au routeur et au LLM de génération.
    safe_question, pii_types = scan_pii(question)
    route = decide_route(safe_question, pii_types, llm)

    if route.action == "refuser_hors_sujet":
        return Answer(OFF_TOPIC_ANSWER, [], [], [], False, route.action, route.categorie)

    retriever = retriever or get_retriever()
    chunks = retriever.invoke(safe_question)
    if context_mode == "document":
        chunks = expand_to_full_documents(chunks)
    if route.action == "escalader":
        retrieved_ids = [chunk.metadata["id"] for chunk in chunks]
        chunks += [doc for doc in get_full_documents(SUPPORT_DOC_IDS) if doc.metadata["id"] not in retrieved_ids]

    answer = generate_answer(safe_question, chunks, llm, INSTRUCTIONS.get(route.action))
    answer.action, answer.category = route.action, route.categorie
    return answer
