from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.router import ROUTER_PROMPT, parse_route, route_question


def test_parse_route_reads_json_even_with_surrounding_text():
    route = parse_route('Voici : {"categorie": "technique", "action": "repondre", "raison": "scan"} ')
    assert (route.categorie, route.action, route.fallback) == ("technique", "repondre", False)


def test_parse_route_falls_back_to_answer_on_invalid_output():
    for raw in ["pas de JSON", '{"categorie": "technique"}', '{"categorie": "x", "action": "supprimer"}']:
        route = parse_route(raw)
        assert route.action == "repondre" and route.fallback


def test_route_question_with_fake_llm():
    llm = FakeListChatModel(responses=['{"categorie": "hors_sujet", "action": "refuser_hors_sujet", "raison": "météo"}'])
    assert route_question("Quel temps fera-t-il ?", llm).action == "refuser_hors_sujet"


def test_router_prompt_lists_every_action():
    for action in ["repondre", "escalader", "refuser_hors_sujet", "refuser_donnees_personnelles"]:
        assert action in ROUTER_PROMPT
