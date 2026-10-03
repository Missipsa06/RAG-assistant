"""Ingestion du corpus : lecture des fichiers Markdown, découpage par section, indexation.

Usage : uv run python -m rag.ingestion   (reconstruit l'index Chroma de zéro)
"""
import shutil
import statistics
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from rag import config
from rag.embeddings import get_embeddings


def parse_markdown_file(path: Path) -> tuple[dict[str, str], str]:
    """Sépare le frontmatter (bloc « clé: valeur » entre deux lignes « --- ») du corps Markdown."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name} : frontmatter manquant")
    _, frontmatter, body = text.split("---\n", 2)
    metadata = {}
    for line in frontmatter.splitlines():
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata, body


def load_corpus(corpus_dir: Path = config.CORPUS_DIR) -> list[Document]:
    """Charge chaque fichier du corpus en un Document LangChain (corps + métadonnées)."""
    documents = []
    for path in sorted(corpus_dir.rglob("*.md")):
        metadata, body = parse_markdown_file(path)
        metadata["source"] = path.relative_to(corpus_dir).as_posix()
        documents.append(Document(page_content=body, metadata=metadata))
    return documents


def split_into_chunks(documents: list[Document]) -> list[Document]:
    """Découpe chaque document par section Markdown ; chaque morceau hérite des métadonnées du document.

    Les métadonnées (id, catégorie, version...) sont recopiées sur chaque morceau : c'est
    ce qui permettra de citer la source d'une réponse et de filtrer par catégorie.
    """
    # strip_headers=False : le titre de section reste dans le texte, donc il est pris en
    # compte par l'embedding (« ## Compte bloqué » est un signal fort pour la recherche).
    header_splitter = MarkdownHeaderTextSplitter(config.HEADERS_TO_SPLIT_ON, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = []
    for document in documents:
        sections = size_splitter.split_documents(header_splitter.split_text(document.page_content))
        for position, section in enumerate(sections):
            section.metadata = {
                **document.metadata,
                **section.metadata,
                "chunk_id": f"{document.metadata['id']}#{position}",
            }
            chunks.append(section)
    return chunks


def build_index(chunks: list[Document], embeddings: Embeddings) -> Chroma:
    """Recrée de zéro la collection Chroma sur disque à partir des morceaux.

    On supprime l'ancien index plutôt que de le mettre à jour : il se reconstruit en
    quelques secondes, et on évite ainsi de garder des morceaux d'une ancienne version.
    """
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    # Distance par défaut de Chroma : L2. Les vecteurs étant normalisés (voir embeddings.py),
    # L2 et cosinus donnent exactement le même classement.
    return Chroma.from_documents(
        chunks,
        embeddings,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )


def main() -> None:
    start = time.perf_counter()
    documents = load_corpus()
    chunks = split_into_chunks(documents)
    build_index(chunks, get_embeddings())
    lengths = [len(chunk.page_content) for chunk in chunks]
    print(f"{len(documents)} documents -> {len(chunks)} morceaux")
    print(f"Taille (caractères) : min {min(lengths)}, médiane {statistics.median(lengths):.0f}, max {max(lengths)}")
    print(f"Index écrit dans {config.CHROMA_DIR} en {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
