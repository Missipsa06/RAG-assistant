"""Tests du pipeline complet, avec un faux retriever et un faux LLM (aucun appel réseau)."""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.guardrails import OFF_TOPIC_ANSWER
from rag.pipeline import answer_question

ROUTE_ANSWER = '{"categorie": "technique", "action": "repondre", "raison": "x"}'
ROUTE_ESCALATE = '{"categorie": "technique", "action": "escalader", "raison": "x"}'
ROUTE_OFF_TOPIC = '{"categorie": "hors_sujet", "action": "refuser_hors_sujet", "raison": "x"}'
CHUNK = Document("## Batterie\nRechargez en USB-C.", metadata={"id": "APP-10", "titre": "Lecteur ScanBox", "section": "Batterie"})


class RecordingLLM:
    """Faux LLM : renvoie les réponses prévues, dans l'ordre, et garde les messages reçus."""

    def __init__(self, responses: list[str]):
        self.responses, self.received = list(responses), []

    def invoke(self, messages):
        self.received.append(messages)
        return AIMessage(self.responses.pop(0))


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def invoke(self, question):
        self.calls += 1
        return [CHUNK]


def test_answer_route_runs_retrieval_and_generation():
    llm, retriever = RecordingLLM([ROUTE_ANSWER, "Rechargez-la en USB-C [APP-10]."]), FakeRetriever()
    answer = answer_question("Batterie ScanBox ?", retriever=retriever, llm=llm, context_mode="section")
    assert answer.action == "repondre" and answer.category == "technique"
    assert answer.cited_ids == ["APP-10"] and retriever.calls == 1


def test_off_topic_is_refused_without_retrieval_nor_generation():
    llm, retriever = RecordingLLM([ROUTE_OFF_TOPIC]), FakeRetriever()
    answer = answer_question("Une recette de crêpes ?", retriever=retriever, llm=llm)
    assert answer.text == OFF_TOPIC_ANSWER and answer.action == "refuser_hors_sujet"
    assert retriever.calls == 0 and len(llm.received) == 1  # seul le routeur a été appelé


def test_escalation_adds_support_documents_and_instruction():
    llm = RecordingLLM([ROUTE_ESCALATE, "Contactez le support [SUP-01]."])
    answer = answer_question("Ma ScanBox est morte", retriever=FakeRetriever(), llm=llm, context_mode="section")
    assert answer.action == "escalader"
    assert answer.context_ids == ["APP-10", "SUP-01", "SUP-02"]
    generation_prompt = llm.received[1][1].content
    assert "Consigne particulière" in generation_prompt and "conseiller" in generation_prompt


@pytest.mark.parametrize("card", ["4970 1234 5678 9012", "4970 XXXX XXXX 1234"])
def test_bank_card_is_never_sent_to_the_llm_and_forces_refusal(card):
    llm = RecordingLLM(["Ne communiquez jamais vos coordonnées bancaires."])  # aucune réponse de routeur prévue
    answer = answer_question(f"Voici ma carte {card}", retriever=FakeRetriever(), llm=llm, context_mode="section")
    assert answer.action == "refuser_donnees_personnelles"
    assert len(llm.received) == 1  # le routeur LLM n'a pas été appelé : garde-fou déterministe
    sent = " ".join(message.content for message in llm.received[0])
    assert card not in sent and "[CARTE_BANCAIRE]" in sent
