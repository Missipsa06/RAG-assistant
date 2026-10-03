"""Interface de chat de l'assistant (Streamlit). Elle appelle l'API, sans importer le pipeline.

Lancement : uv run streamlit run app/streamlit_app.py   (l'API doit tourner, voir rag/api.py)
"""
import os

import streamlit as st
from api_client import ACTION_LABELS, AssistantUnavailable, ask_api, format_sources

from rag import config

API_URL = os.environ.get("API_URL", config.API_URL)
EXAMPLE_QUESTIONS = [
    "Combien valent 1000 points ?",
    "L'appli affiche l'erreur E302, je fais quoi ?",
    "Je veux parler à un humain",
    "How do I leave the panel?",
]

st.set_page_config(page_title="Assistant panélistes", page_icon="🛒")
st.title("🛒 Assistant des panélistes PanelConso")
st.caption("Démonstration : entreprise et documentation **fictives**. Ne saisissez aucune donnée personnelle réelle.")

with st.sidebar:
    st.subheader("À propos")
    st.write(
        "L'assistant répond à partir de la documentation du panel (application, points, compte, "
        "données personnelles), cite ses sources, et oriente vers un conseiller si nécessaire."
    )
    st.subheader("Exemples")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, width="stretch"):
            st.session_state.pending_question = example

if "messages" not in st.session_state:
    st.session_state.messages = []  # historique : {"role", "content", "response"}


def show_assistant_extras(response: dict) -> None:
    """Bandeau selon l'action décidée, puis sources citées (repliables)."""
    if label := ACTION_LABELS.get(response["action"]):
        st.info(label)
    if response["sources"]:
        with st.expander(f"Sources ({len(response['sources'])})"):
            st.markdown(format_sources(response["sources"]))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("response"):
            show_assistant_extras(message["response"])

question = st.chat_input("Posez votre question...") or st.session_state.pop("pending_question", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche dans la documentation..."):
                response = ask_api(question, API_URL)
        except AssistantUnavailable as error:
            st.error(str(error))
        else:
            st.markdown(response["answer"])
            show_assistant_extras(response)
            st.session_state.messages.append({"role": "assistant", "content": response["answer"], "response": response})
