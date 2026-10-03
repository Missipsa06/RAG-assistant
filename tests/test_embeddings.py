"""Tests sur le vrai modèle (téléchargé une fois, ~470 Mo, puis en cache)."""
import math

import pytest

from rag.embeddings import get_embeddings


@pytest.fixture(scope="module")
def embeddings():
    return get_embeddings()


def test_vectors_have_384_dimensions_and_unit_norm(embeddings):
    vector = embeddings.embed_query("Combien valent 1000 points ?")
    assert len(vector) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, abs_tol=1e-4)


def test_query_and_passage_prefixes_give_different_vectors(embeddings):
    text = "Mot de passe oublié"
    assert embeddings.embed_query(text) != embeddings.embed_documents([text])[0]


def test_question_is_closer_to_relevant_passage_even_in_english(embeddings):
    passages = [
        "## Mot de passe oublié\nCliquez sur « Mot de passe oublié » sur l'écran de connexion.",
        "## Batterie\nLa ScanBox se recharge avec un câble USB-C.",
    ]
    password_vec, scanbox_vec = embeddings.embed_documents(passages)
    question_vec = embeddings.embed_query("I forgot my password")

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # vecteurs normalisés

    assert cosine(question_vec, password_vec) > cosine(question_vec, scanbox_vec)
