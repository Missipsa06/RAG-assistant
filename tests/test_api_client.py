"""Tests du client HTTP de l'interface (transport simulé) et affichage de l'interface Streamlit."""
import httpx
import pytest
from api_client import AssistantUnavailable, ask_api, format_sources
from streamlit.testing.v1 import AppTest

from rag import config

RESPONSE = {
    "answer": "Il faut 1 500 points [REC-04].", "action": "repondre", "category": "recompenses",
    "sources": [{"id": "REC-04", "titre": "Échanger ses points"}], "not_in_docs": False, "latency_ms": 800,
}


def client_returning(status_code: int, payload: dict) -> httpx.Client:
    """Client httpx dont toutes les requêtes reçoivent la réponse indiquée, sans réseau."""
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status_code, json=payload)))


def test_ask_api_returns_the_api_response():
    assert ask_api("Combien de points ?", "http://api", client_returning(200, RESPONSE)) == RESPONSE


def test_ask_api_turns_errors_into_readable_messages():
    with pytest.raises(AssistantUnavailable, match="momentanément"):
        ask_api("?", "http://api", client_returning(503, {"detail": "Assistant momentanément indisponible."}))
    with pytest.raises(AssistantUnavailable, match="trop longue"):
        ask_api("?", "http://api", client_returning(422, {"detail": []}))


def test_ask_api_handles_unreachable_api():
    def unreachable(request):
        raise httpx.ConnectError("connexion refusée")

    with pytest.raises(AssistantUnavailable, match="Impossible de joindre"):
        ask_api("?", "http://api", httpx.Client(transport=httpx.MockTransport(unreachable)))


def test_format_sources():
    assert format_sources(RESPONSE["sources"]) == "- **REC-04** : Échanger ses points"


def test_streamlit_page_renders_without_error():
    page = AppTest.from_file(str(config.ROOT / "app/streamlit_app.py")).run()
    assert not page.exception
    assert "PanelConso" in page.title[0].value
