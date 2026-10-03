"""API HTTP de l'assistant (FastAPI).

Lancement : uv run uvicorn rag.api:app --port 8000   (documentation interactive : /docs)

- POST /ask    : une question → la réponse, l'action décidée et les sources citées.
- GET  /health : le service est-il prêt ?
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag import config
from rag.generation import get_llm
from rag.pipeline import answer_question
from rag.retrieval import get_full_documents, get_retriever

# Journal d'Uvicorn : nos lignes apparaissent avec les siennes, sans configuration supplémentaire.
logger = logging.getLogger("uvicorn.error")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS, examples=["Combien valent 1000 points ?"])


class Source(BaseModel):
    id: str
    titre: str


class AskResponse(BaseModel):
    answer: str
    action: Literal["repondre", "escalader", "refuser_hors_sujet", "refuser_donnees_personnelles"]
    category: str | None
    sources: list[Source]  # documents cités dans la réponse
    not_in_docs: bool  # l'information demandée n'est pas dans la documentation
    latency_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge une seule fois, au démarrage, ce qui est coûteux : modèle d'embeddings, index, LLM.

    Les charger à chaque requête ajouterait plusieurs secondes par question.
    """
    app.state.retriever = get_retriever()
    app.state.llm = get_llm()
    yield


app = FastAPI(title="Assistant panélistes PanelConso", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Répond dès que le démarrage (modèle, index, LLM) est terminé."""
    return {"status": "ok", "llm_model": config.LLM_MODEL, "retrieval": config.RETRIEVAL_MODE}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Répond à une question de panéliste.

    Fonction synchrone (pas async) : le pipeline fait des appels bloquants (embeddings, LLM) ;
    FastAPI l'exécute alors dans un pool de threads, sans bloquer les autres requêtes.
    """
    start = time.perf_counter()
    try:
        answer = answer_question(payload.question, retriever=request.app.state.retriever, llm=request.app.state.llm)
    except Exception as error:  # panne du LLM, quota épuisé, réseau...
        # On journalise le type d'erreur, jamais la question (elle peut contenir des données personnelles).
        logger.error("Échec du pipeline : %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Assistant momentanément indisponible, réessayez dans quelques instants.")
    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info("action=%s categorie=%s latence_ms=%d", answer.action, answer.category, latency_ms)
    # Seules les citations de documents réellement fournis au LLM sont montrées comme sources :
    # une citation inventée (identifiant inexistant ou absent des extraits) n'est pas une source.
    cited_documents = get_full_documents(doc_id for doc_id in answer.cited_ids if doc_id in answer.context_ids)
    return AskResponse(
        answer=answer.text,
        action=answer.action,
        category=answer.category,
        sources=[Source(id=doc.metadata["id"], titre=doc.metadata["titre"]) for doc in cited_documents],
        not_in_docs=answer.not_in_docs,
        latency_ms=latency_ms,
    )
