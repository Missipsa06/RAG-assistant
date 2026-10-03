"""Modèle d'embeddings : transforme un texte en vecteur de 384 nombres."""
from langchain_huggingface import HuggingFaceEmbeddings

from rag import config


def get_embeddings() -> HuggingFaceEmbeddings:
    """Modèle E5 local (CPU), avec des préfixes différents pour les questions et les passages.

    normalize_embeddings=True ramène chaque vecteur à une longueur de 1 : le produit
    scalaire devient alors la similarité cosinus, et toutes les distances se valent pour classer.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"prompt": config.PASSAGE_PREFIX, "normalize_embeddings": True},
        query_encode_kwargs={"prompt": config.QUERY_PREFIX, "normalize_embeddings": True},
    )
