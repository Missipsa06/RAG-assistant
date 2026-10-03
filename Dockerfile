# Image unique de l'assistant : API FastAPI (commande par défaut) et interface Streamlit.
# Construction : docker build -t rag-assistant .
# La clé d'API n'est JAMAIS dans l'image : elle est fournie au lancement (docker compose, .env).
FROM python:3.12-slim

# uv, même version qu'en développement, pour installer exactement les versions de uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/

# Utilisateur non-root (bonne pratique de sécurité ; exigé par Hugging Face Spaces).
RUN useradd --create-home --uid 1000 app
WORKDIR /app
RUN chown app:app /app
USER app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# 1. Dépendances seules : cette couche, la plus longue, reste en cache tant que uv.lock ne change pas.
#    --no-dev : ni pytest ni MLflow (outils d'évaluation), inutiles au service.
#    Le cache de téléchargement de uv est monté pendant le build, pas copié dans l'image
#    (sans ce montage, il ajoutait ~2 Go à l'image).
COPY --chown=app:app pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000 \
    uv sync --locked --no-dev --no-install-project

# 2. Code et corpus, puis installation du package « rag ».
COPY --chown=app:app src ./src
COPY --chown=app:app app ./app
COPY --chown=app:app corpus ./corpus
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=1000 \
    uv sync --locked --no-dev

# 3. Téléchargement du modèle d'embeddings et construction de l'index pendant le build :
#    le conteneur démarre ensuite sans rien télécharger ni calculer.
RUN python -m rag.ingestion
# Le modèle est désormais dans l'image : au démarrage, ne plus interroger le Hub Hugging Face
# (démarrage indépendant du réseau ; observé sans ce réglage : appel au Hub à chaque lancement).
ENV HF_HUB_OFFLINE=1

EXPOSE 8000 8501
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
