"""Recherche des morceaux de documents pertinents pour une question.

Trois modes, tous exposés comme un Runnable LangChain (`.invoke(question) → list[Document]`) :
- "vector" : similarité d'embeddings (le sens) ;
- "bm25"   : correspondance de mots (les termes exacts) ;
- "hybrid" : fusion des deux classements par Reciprocal Rank Fusion (RRF).
"""
import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

import snowballstemmer
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

from rag import config
from rag.embeddings import get_embeddings
from rag.ingestion import load_corpus

RETRIEVAL_MODES = ("vector", "bm25", "hybrid")

# Mots vides français, écrits sans accents car filtrés après normalize_for_bm25. Ils
# apparaissent partout et noient le signal des mots utiles. Volontairement absents de la
# liste : « pas », « plus », « jamais » (une négation change le sens : « ne s'allume plus »).
FRENCH_STOPWORDS = frozenset("""
    a au aux avec ce c ca ces cet cette d dans de des du elle elles en es est et etre
    il ils j je l la le les leur leurs lui m ma me mes moi mon n ne nos notre nous on ou
    par pour qu que quel quelle quels quelles qui quoi s sa se ses si son sont sur t ta
    te tes toi ton tu un une vos votre vous y ai as avons avez ont suis
""".split())
_FRENCH_STEMMER = snowballstemmer.stemmer("french")


def load_vector_store(embeddings: Embeddings | None = None) -> Chroma:
    """Ouvre l'index Chroma construit par `python -m rag.ingestion`.

    Les embeddings passés doivent être ceux de l'indexation : des vecteurs issus de deux
    modèles différents ne sont pas comparables.
    """
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(f"Index absent ({config.CHROMA_DIR}) : lancer `uv run python -m rag.ingestion`")
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_vector_retriever(k: int = config.TOP_K, store: Chroma | None = None) -> VectorStoreRetriever:
    """Retriever LangChain de recherche vectorielle seule : les k morceaux les plus proches.

    On passe par un retriever (plutôt que store.similarity_search) car c'est l'interface
    commune à toutes les recherches LangChain : BM25 et la recherche hybride s'utilisent
    exactement de la même façon, via `.invoke(question)`.
    """
    return (store or load_vector_store()).as_retriever(search_kwargs={"k": k})


def normalize_for_bm25(text: str) -> list[str]:
    """Découpe un texte en mots comparables : minuscules, sans accents, sans ponctuation.

    BM25 compare des mots à l'identique. Sans ce nettoyage, « Bancaire », « bancaire. » et
    « bancaire » seraient trois mots différents, et une question tapée sans accents
    (« supprimer mes donnees ») ne retrouverait pas « données ».
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())  # « é » → « e » + accent séparé
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.findall(r"\w+", without_accents)


def analyze_french_for_bm25(text: str) -> list[str]:
    """normalize_for_bm25, puis suppression des mots vides et racinisation française.

    La racinisation (stemming) coupe les terminaisons : « bancaires » → « bancair »,
    « déménagement » → « demenag », pour que les variantes d'un même mot se correspondent.
    Elle s'applique après la suppression des accents : ainsi « données » et « donnees »
    (question tapée sans accents) donnent la même racine « donne ».
    """
    words = [word for word in normalize_for_bm25(text) if word not in FRENCH_STOPWORDS]
    return _FRENCH_STEMMER.stemWords(words)


BM25_ANALYZERS = {"basic": normalize_for_bm25, "french": analyze_french_for_bm25}


def load_chunks_from_store(store: Chroma) -> list[Document]:
    """Relit tous les morceaux stockés dans Chroma.

    BM25 est construit à partir de ces morceaux-là (plutôt qu'en relisant le corpus) pour
    que les deux recherches portent exactement sur les mêmes morceaux, avec les mêmes chunk_id.
    """
    stored = store.get()
    return [Document(page_content=text, metadata=metadata) for text, metadata in zip(stored["documents"], stored["metadatas"])]


def get_bm25_retriever(
    chunks: list[Document], k: int = config.TOP_K, analyzer: str = config.BM25_ANALYZER
) -> BM25Retriever:
    """Retriever lexical BM25, construit en mémoire (quelques millisecondes pour 80 morceaux).

    Le même prétraitement (`analyzer`) s'applique aux morceaux et aux questions.
    """
    return BM25Retriever.from_documents(chunks, k=k, preprocess_func=BM25_ANALYZERS[analyzer])


def get_hybrid_retriever(
    k: int = config.TOP_K,
    vector_weight: float = config.VECTOR_WEIGHT,
    store: Chroma | None = None,
    bm25_analyzer: str = config.BM25_ANALYZER,
) -> Runnable:
    """Fusion RRF pondérée des recherches vectorielle et BM25, tronquée aux k meilleurs morceaux.

    Chaque retriever propose FUSION_CANDIDATES morceaux ; EnsembleRetriever additionne pour
    chaque morceau poids / (rang + RRF_C) sur les deux listes, puis trie. Il renvoie l'union
    des deux listes (jusqu'à 2 × FUSION_CANDIDATES) : on garde nous-mêmes les k premiers, pour
    donner au LLM (et comparer à la baseline) le même nombre de morceaux qu'en vectoriel seul.
    """
    store = store or load_vector_store()
    ensemble = EnsembleRetriever(
        retrievers=[
            get_vector_retriever(config.FUSION_CANDIDATES, store),
            get_bm25_retriever(load_chunks_from_store(store), config.FUSION_CANDIDATES, bm25_analyzer),
        ],
        weights=[vector_weight, 1 - vector_weight],
        c=config.RRF_C,
        # Identifie un morceau par son chunk_id (par défaut : par son texte). Deux sections
        # au texte identique dans deux documents différents ne seraient pas fusionnées à tort.
        id_key="chunk_id",
    )
    return ensemble | RunnableLambda(lambda chunks: chunks[:k])


def get_retriever(
    mode: str = config.RETRIEVAL_MODE,
    k: int = config.TOP_K,
    vector_weight: float = config.VECTOR_WEIGHT,
    bm25_analyzer: str = config.BM25_ANALYZER,
) -> Runnable:
    """Point d'entrée unique : le retriever du mode demandé, renvoyant k morceaux."""
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Mode de recherche inconnu : {mode!r} (attendu : {RETRIEVAL_MODES})")
    store = load_vector_store()
    if mode == "vector":
        return get_vector_retriever(k, store)
    if mode == "bm25":
        return get_bm25_retriever(load_chunks_from_store(store), k, bm25_analyzer)
    return get_hybrid_retriever(k, vector_weight, store, bm25_analyzer)


@lru_cache(maxsize=1)
def _documents_by_id() -> dict[str, Document]:
    """Documents entiers du corpus, indexés par identifiant (lus une seule fois)."""
    return {document.metadata["id"]: document for document in load_corpus()}


def expand_to_full_documents(chunks: list[Document]) -> list[Document]:
    """Remplace les morceaux retrouvés par les documents entiers dont ils proviennent.

    Technique du « document parent » : on cherche avec de petits morceaux (précis), mais on
    donne au LLM tout le document (complet). Ordre de la recherche conservé, sans doublon.
    """
    return get_full_documents(chunk.metadata["id"] for chunk in chunks)


def get_full_documents(doc_ids: Iterable[str]) -> list[Document]:
    """Documents entiers correspondant aux identifiants, sans doublon, dans l'ordre donné."""
    documents = _documents_by_id()
    return [documents[doc_id] for doc_id in dict.fromkeys(doc_ids)]
