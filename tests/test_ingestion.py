from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag import config
from rag.ingestion import build_index, load_corpus, parse_markdown_file, split_into_chunks

SAMPLE_BODY = """# Mot de passe et connexion

Introduction courte.

## Mot de passe oublié
Cliquez sur « Mot de passe oublié ».

## Compte bloqué
Attendez 30 minutes.
"""


def sample_document() -> Document:
    return Document(page_content=SAMPLE_BODY, metadata={"id": "CPT-01", "categorie": "compte"})


def test_parse_markdown_file_separates_frontmatter_and_body(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("---\nid: X-01\ntitre: Un titre : avec deux-points\n---\n# Corps\n", encoding="utf-8")
    metadata, body = parse_markdown_file(path)
    assert metadata == {"id": "X-01", "titre": "Un titre : avec deux-points"}
    assert body == "# Corps\n"


def test_load_corpus_reads_every_document_with_its_metadata():
    documents = load_corpus()
    assert len(documents) == 27
    for document in documents:
        assert {"id", "titre", "categorie", "version", "date_maj", "source"} <= document.metadata.keys()
        assert document.metadata["source"].startswith(document.metadata["categorie"] + "/")


def test_split_cuts_on_sections_and_keeps_document_metadata():
    chunks = split_into_chunks([sample_document()])
    assert [chunk.metadata.get("section") for chunk in chunks] == [None, "Mot de passe oublié", "Compte bloqué"]
    assert all(chunk.metadata["id"] == "CPT-01" and chunk.metadata["categorie"] == "compte" for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == ["CPT-01#0", "CPT-01#1", "CPT-01#2"]
    # strip_headers=False : le titre de section reste dans le texte embarqué.
    assert chunks[2].page_content.startswith("## Compte bloqué")


def test_split_real_corpus_covers_every_document_with_unique_ids():
    documents = load_corpus()
    chunks = split_into_chunks(documents)
    assert {chunk.metadata["id"] for chunk in chunks} == {doc.metadata["id"] for doc in documents}
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(len(chunk.page_content) <= config.CHUNK_SIZE for chunk in chunks)


def test_build_index_stores_chunks_with_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    # Faux embeddings (vecteurs pseudo-aléatoires) : on teste le stockage, pas la pertinence.
    store = build_index(split_into_chunks([sample_document()]), DeterministicFakeEmbedding(size=8))
    stored = store.get()
    assert sorted(stored["ids"]) == ["CPT-01#0", "CPT-01#1", "CPT-01#2"]
    assert all(metadata["categorie"] == "compte" for metadata in stored["metadatas"])
