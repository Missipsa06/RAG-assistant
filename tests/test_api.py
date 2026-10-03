"""Tests de l'API avec un faux pipeline : ni modèle d'embeddings, ni index, ni LLM."""
import pytest
from fastapi.testclient import TestClient

from rag import api, config
from rag.generation import Answer


@pytest.fixture
def client(monkeypatch):
    """Client de test ; le démarrage ne charge rien de coûteux et le pipeline est simulé."""
    monkeypatch.setattr(api, "get_retriever", lambda: "faux retriever")
    monkeypatch.setattr(api, "get_llm", lambda: "faux llm")

    def fake_answer_question(question, retriever, llm):
        return Answer(
            text="Il faut 1 500 points minimum [REC-04].",
            context_ids=["REC-04", "SUP-01"],
            cited_ids=["REC-04", "APP-99"],  # APP-99 : citation inventée, absente des extraits
            invalid_citations=["APP-99"],
            not_in_docs=False,
            action="repondre",
            category="recompenses",
        )

    monkeypatch.setattr(api, "answer_question", fake_answer_question)
    with TestClient(api.app) as test_client:  # « with » déclenche le démarrage (lifespan)
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_ask_returns_answer_action_and_only_real_sources(client):
    response = client.post("/ask", json={"question": "À partir de combien de points ?"})
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "repondre" and body["category"] == "recompenses"
    # La citation inventée APP-99 n'est pas présentée comme une source.
    assert body["sources"] == [{"id": "REC-04", "titre": "Échanger ses points"}]
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize("question", ["", "x" * (config.MAX_QUESTION_CHARS + 1)])
def test_ask_rejects_empty_or_too_long_questions(client, question):
    assert client.post("/ask", json={"question": question}).status_code == 422


def test_pipeline_failure_gives_503_without_leaking_details(client, monkeypatch):
    def broken_pipeline(question, retriever, llm):
        raise RuntimeError("quota épuisé, clé sk-or-secrète")

    monkeypatch.setattr(api, "answer_question", broken_pipeline)
    response = client.post("/ask", json={"question": "Bonjour"})
    assert response.status_code == 503
    assert "sk-or" not in response.text  # aucun détail interne renvoyé à l'utilisateur
