"""Client de l'API de l'assistant, utilisé par l'interface Streamlit.

Séparé de l'interface pour être testable sans lancer Streamlit.
"""
import httpx

# Libellés affichés selon l'action décidée par l'assistant.
ACTION_LABELS = {
    "repondre": None,  # cas normal : pas de bandeau
    "escalader": "🧑‍💼 Cette demande nécessite un conseiller : voir les moyens de contact ci-dessus.",
    "refuser_hors_sujet": "ℹ️ Question hors du périmètre de l'assistant.",
    "refuser_donnees_personnelles": "🔒 Aucune donnée personnelle n'est communiquée ni conservée.",
}

# Le premier appel peut être lent (réveil du service, appel au LLM) : marge large.
TIMEOUT_SECONDS = 60


class AssistantUnavailable(Exception):
    """L'API ne répond pas ou renvoie une erreur : message à afficher tel quel."""


def ask_api(question: str, api_url: str, client: httpx.Client | None = None) -> dict:
    """Envoie la question à l'API et renvoie sa réponse JSON.

    `client` est injectable pour les tests (transport simulé, sans réseau).
    """
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        response = client.post(f"{api_url}/ask", json={"question": question})
    except httpx.HTTPError:
        raise AssistantUnavailable("Impossible de joindre l'assistant. Vérifiez que l'API est démarrée.")
    if response.status_code == 422:
        raise AssistantUnavailable("Question vide ou trop longue.")
    if response.status_code != 200:
        raise AssistantUnavailable(response.json().get("detail", "L'assistant a rencontré une erreur."))
    return response.json()


def format_sources(sources: list[dict]) -> str:
    """Liste Markdown des documents cités, ex. « - **REC-04** : Échanger ses points »."""
    return "\n".join(f"- **{source['id']}** : {source['titre']}" for source in sources)
