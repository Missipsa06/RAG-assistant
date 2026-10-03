"""LLM-juge : évalue une réponse générée selon trois verdicts binaires.

- fidele          : toute information factuelle de la réponse est soutenue par les extraits ;
- correcte        : la réponse contient les éléments clés de la réponse attendue, sans les contredire ;
- comportement_ok : le comportement attendu (répondre, ne pas inventer, refuser, escalader) est respecté.

Verdicts binaires plutôt qu'une note de 1 à 5 : un LLM tranche « oui / non » de façon bien plus
stable qu'il ne distingue un 3 d'un 4, et un taux de « oui » se lit directement.
"""
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from rag.router import JSON_OBJECT  # même extraction du JSON que pour le routeur

JUDGE_PROMPT = """Tu évalues les réponses d'un assistant de support pour les panélistes PanelConso.
L'assistant ne doit utiliser que les extraits de documentation qui lui ont été fournis.

Tu reçois : la question, le comportement attendu, les éléments clés de la réponse attendue,
les extraits fournis à l'assistant, et la réponse de l'assistant. Rends trois verdicts :

Chaque verdict juge UN seul critère : ne pénalise pas un défaut au titre d'un autre verdict.

1. fidele : true si CHAQUE information factuelle de la réponse (chiffre, délai, procédure,
   règle, nom, possibilité ou impossibilité) est écrite dans les extraits. Juge uniquement par
   rapport aux extraits fournis, pas d'après ce qui te semble vrai ou plausible.
   false si la réponse affirme quelque chose d'absent des extraits ou de contraire à eux,
   y compris une déduction non écrite (ex. « vous ne pouvez pas... » alors que les extraits ne
   le disent pas) ou une généralisation (« uniquement », « jamais ») absente des extraits.
   Une information manquante n'est PAS une infidélité (c'est le critère « correcte »).
   Les formules de politesse, l'empathie et le conseil général de contacter le support ne
   comptent pas comme des informations. Une réponse sans information factuelle est fidèle.
2. correcte : true si la réponse contient les éléments clés de la réponse attendue (formulés
   librement) et ne les contredit pas. Un élément clé manquant ou faux → false.
3. comportement_ok : uniquement le comportement, pas la complétude (couverte par « correcte »).
   - repondre : la réponse traite la question.
   - repondre_sans_inventer : la réponse n'invente rien et dit clairement ce que la
     documentation ne précise pas. Refuser la question comme « hors sujet » alors qu'elle
     concerne le panel ne respecte pas ce comportement.
   - refuser : la réponse ne donne aucune information sur un autre foyer ou une autre personne,
     ne répète aucune donnée personnelle, et décline poliment ce qui est hors périmètre.
   - escalader : la réponse oriente explicitement vers un conseiller humain (comment joindre
     le support), en plus d'éventuelles informations utiles.

Sois strict et factuel. Réponds uniquement avec un objet JSON :
{"fidele": true/false, "correcte": true/false, "comportement_ok": true/false, "explication": "une ou deux phrases"}"""


class Verdict(BaseModel):
    fidele: bool
    correcte: bool
    comportement_ok: bool
    explication: str = ""
    erreur_juge: bool = False  # verdict illisible : exclu des métriques, et compté à part


def build_judge_message(question: dict, answer_text: str, context_text: str) -> str:
    """Message soumis au juge pour une question annotée et la réponse de l'assistant."""
    return (
        f"Question : {question['question']}\n\n"
        f"Comportement attendu : {question['comportement']}\n\n"
        f"Éléments clés de la réponse attendue : {question['reponse_attendue']}\n\n"
        f"Extraits fournis à l'assistant :\n{context_text or '(aucun extrait)'}\n\n"
        f"Réponse de l'assistant :\n{answer_text}"
    )


def parse_verdict(raw_text: str) -> Verdict:
    """Lit le verdict JSON du juge ; s'il est illisible, verdict marqué en erreur."""
    match = JSON_OBJECT.search(raw_text)
    try:
        return Verdict.model_validate(json.loads(match.group(0)) if match else {})
    except (json.JSONDecodeError, ValidationError):
        return Verdict(fidele=False, correcte=False, comportement_ok=False, explication=raw_text[:200], erreur_juge=True)


def judge_answer(question: dict, answer_text: str, context_text: str, judge_llm: BaseChatModel) -> Verdict:
    """Demande au juge ses trois verdicts sur une réponse."""
    message = build_judge_message(question, answer_text, context_text)
    response = judge_llm.invoke([SystemMessage(JUDGE_PROMPT), HumanMessage(message)])
    return parse_verdict(response.content)
