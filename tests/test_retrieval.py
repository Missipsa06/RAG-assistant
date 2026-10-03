import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag import config
from rag.ingestion import build_index, load_corpus, split_into_chunks
from rag.retrieval import (
    analyze_french_for_bm25,
    expand_to_full_documents,
    get_bm25_retriever,
    get_hybrid_retriever,
    get_retriever,
    get_vector_retriever,
    load_chunks_from_store,
    load_vector_store,
    normalize_for_bm25,
)


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    """Index Chroma du vrai corpus, mais avec de faux embeddings : on teste la mécanique, pas la pertinence."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    embeddings = DeterministicFakeEmbedding(size=8)
    build_index(split_into_chunks(load_corpus()), embeddings)
    return load_vector_store(embeddings)


def test_missing_index_gives_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="rag.ingestion"):
        load_vector_store(DeterministicFakeEmbedding(size=8))


def test_vector_retriever_returns_k_chunks_with_metadata(fake_store):
    chunks = get_vector_retriever(k=3, store=fake_store).invoke("Combien de points pour un article ?")
    assert len(chunks) == 3
    assert all({"id", "categorie", "chunk_id"} <= chunk.metadata.keys() for chunk in chunks)


def test_normalize_for_bm25_lowercases_removes_accents_and_punctuation():
    assert normalize_for_bm25("Données bancaires. Erreur E302 !") == ["donnees", "bancaires", "erreur", "e302"]


def test_french_analyzer_drops_stopwords_and_matches_word_variants():
    assert analyze_french_for_bm25("Est-ce que mes données bancaires") == ["donne", "bancair"]
    # Question tapée sans accents ou au singulier : mêmes racines que le texte du corpus.
    assert analyze_french_for_bm25("donnee bancaire") == analyze_french_for_bm25("Données bancaires")
    # Les négations sont gardées : elles changent le sens.
    assert "plus" in analyze_french_for_bm25("elle ne s'allume plus")


def test_chunks_reloaded_from_store_match_the_index(fake_store):
    chunks = load_chunks_from_store(fake_store)
    assert len(chunks) == 80
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == 80


def test_bm25_finds_exact_terms_without_accents(fake_store):
    bm25 = get_bm25_retriever(load_chunks_from_store(fake_store), k=1)
    # Question sans accent, mot présent tel quel dans une seule section du corpus.
    assert bm25.invoke("scanbox batterie")[0].metadata["chunk_id"] == "APP-10#2"


def test_hybrid_returns_k_unique_chunks(fake_store):
    chunks = get_hybrid_retriever(k=5, vector_weight=0.5, store=fake_store).invoke("carte bancaire")
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(ids) == 5  # tronqué : l'union des deux listes en contient jusqu'à 2 × FUSION_CANDIDATES
    assert len(set(ids)) == 5  # un morceau trouvé par les deux retrievers n'apparaît qu'une fois


def test_hybrid_with_full_bm25_weight_follows_bm25_ranking(fake_store):
    bm25_top = get_bm25_retriever(load_chunks_from_store(fake_store), k=3).invoke("carte bancaire")
    hybrid_top = get_hybrid_retriever(k=3, vector_weight=0.0, store=fake_store).invoke("carte bancaire")
    assert [c.metadata["chunk_id"] for c in hybrid_top] == [c.metadata["chunk_id"] for c in bm25_top]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        get_retriever("keyword")


def test_expand_to_full_documents_keeps_order_and_removes_duplicates():
    chunks = [
        Document("## Procédure", metadata={"id": "APP-05", "chunk_id": "APP-05#1"}),
        Document("## Canaux", metadata={"id": "SUP-01", "chunk_id": "SUP-01#1"}),
        Document("# Produit inconnu", metadata={"id": "APP-05", "chunk_id": "APP-05#0"}),
    ]
    documents = expand_to_full_documents(chunks)
    assert [doc.metadata["id"] for doc in documents] == ["APP-05", "SUP-01"]
    # Le document entier contient toutes ses sections, y compris celle qui manquait (Q009).
    assert "# Produit inconnu" in documents[0].page_content and "## Procédure" in documents[0].page_content
