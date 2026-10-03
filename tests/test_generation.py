import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage

from rag import config
from rag.generation import (
    NOT_IN_DOCS_MARKER,
    build_messages,
    format_context,
    generate_answer,
    get_llm,
    parse_answer,
    unique_doc_ids,
)

CHUNKS = [
    Document("## Options\nCartes cadeaux sous 5 jours ouvrés.", metadata={"id": "REC-04", "titre": "Échanger ses points", "section": "Options"}),
    Document("# Échanger ses points\nMinimum 1 500 points.", metadata={"id": "REC-04", "titre": "Échanger ses points"}),
    Document("## Canaux\nFormulaire, email.", metadata={"id": "SUP-01", "titre": "Contacter le support", "section": "Canaux"}),
]


def test_context_labels_each_chunk_with_its_citable_id_and_heading():
    context = format_context(CHUNKS)
    assert context.startswith("[REC-04] Échanger ses points › Options\n## Options")
    assert "[REC-04] Échanger ses points\n# Échanger" in context  # pas de section : titre seul
    assert "[SUP-01] Contacter le support › Canaux" in context


def test_unique_doc_ids_keeps_retrieval_order():
    assert unique_doc_ids(CHUNKS) == ["REC-04", "SUP-01"]


def test_messages_put_rules_in_system_and_extracts_with_question_in_user():
    system, user = build_messages("Combien de points ?", CHUNKS)
    assert isinstance(system, SystemMessage) and "UNIQUEMENT" in system.content
    assert "[SUP-01]" in user.content and user.content.endswith("Question : Combien de points ?")


def test_parse_answer_extracts_citations_and_flags_invented_ones():
    answer = parse_answer("Minimum 1 500 points [REC-04]. Voir aussi [REC-04] et [APP-99].", ["REC-04", "SUP-01"])
    assert answer.cited_ids == ["REC-04", "APP-99"]
    assert answer.invalid_citations == ["APP-99"]
    assert not answer.not_in_docs


def test_parse_answer_detects_and_removes_not_in_docs_marker():
    answer = parse_answer(f"Cette information n'est pas dans la documentation. {NOT_IN_DOCS_MARKER}", ["REC-04"])
    assert answer.not_in_docs
    assert NOT_IN_DOCS_MARKER not in answer.text
    assert answer.cited_ids == []


def test_generate_answer_with_fake_llm():
    llm = FakeListChatModel(responses=["Il faut 1 500 points minimum [REC-04]."])
    answer = generate_answer("À partir de combien de points ?", CHUNKS, llm)
    assert answer.text == "Il faut 1 500 points minimum [REC-04]."
    assert answer.context_ids == ["REC-04", "SUP-01"]
    assert answer.cited_ids == ["REC-04"] and answer.invalid_citations == []


def test_instruction_is_added_to_the_user_prompt_only_when_given():
    _, user = build_messages("Ma ScanBox est en panne", CHUNKS, instruction="Oriente vers le support.")
    assert "Consigne particulière : Oriente vers le support." in user.content
    _, user_without = build_messages("Ma ScanBox est en panne", CHUNKS)
    assert "Consigne particulière" not in user_without.content


def test_missing_api_key_gives_an_explicit_error(monkeypatch):
    # Variable présente mais vide : load_dotenv ne l'écrase pas, get_llm doit refuser.
    monkeypatch.setenv(config.LLM_API_KEY_ENV, "")
    with pytest.raises(RuntimeError, match=config.LLM_API_KEY_ENV):
        get_llm()
