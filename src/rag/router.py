"""Routage par intention : catégorie de la question et action à mener, décidées par un LLM.

Le routeur passe avant la recherche : il évite de chercher et de générer pour une question
hors sujet, et adapte la génération quand il faut escalader ou protéger des données.
"""
import json
import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

Category = Literal["participation", "technique", "recompenses", "compte", "rgpd", "support", "hors_sujet"]
Action = Literal["repondre", "escalader", "refuser_hors_sujet", "refuser_donnees_personnelles"]


class Route(BaseModel):
    """Décision du routeur. `fallback` indique une réponse illisible remplacée par le défaut."""

    categorie: Category | None
    action: Action
    raison: str = ""
    fallback: bool = False


# Les exemples sont inventés : aucun ne vient du jeu d'évaluation, pour ne pas mesurer le
# routeur sur des questions qu'il aurait « vues » dans son prompt.
ROUTER_PROMPT = """Tu es le routeur de l'assistant de support des panélistes PanelConso (panel de
consommateurs qui scannent leurs achats avec une application et gagnent des points).
Tu ne réponds pas à la question : tu la classes.

Catégories :
- participation : règles du panel (quoi enregistrer, délais, promotions)
- technique : application, scan, codes-barres, synchronisation, codes d'erreur, ScanBox
- recompenses : points, bonus, échanges, cartes cadeaux, parrainage
- compte : connexion, mot de passe, composition du foyer, absences, déménagement, désinscription
- rgpd : données personnelles, droits, conservation, sécurité, fraude
- support : contacter le support, réclamations
- hors_sujet : sans rapport avec le panel PanelConso

Actions :
- repondre : question sur le panel, à traiter avec la documentation. Y compris les questions de
  sécurité générales (comment reconnaître un message frauduleux) et les questions dont la
  réponse n'est peut-être pas documentée.
- escalader : la demande nécessite un conseiller humain, en plus de la documentation :
  demande explicite de parler à quelqu'un ; mécontentement ou réclamation ; anomalie sur le
  compte de la personne (points manquants, accès perdu) ; demande d'exercer un droit sur ses
  données (accès, suppression) ; panne d'un matériel ; situation personnelle qui sort des
  règles habituelles.
  N'escalade pas une demande que la personne peut résoudre seule en suivant une procédure
  (changement dans le foyer, mot de passe oublié, application qui plante, téléphone perdu) :
  dans ce cas, repondre ; la réponse indiquera elle-même quand contacter le support.
- refuser_hors_sujet : la question n'a aucun rapport avec le panel. En cas de doute (une
  question qui pourrait concerner les points, les récompenses ou le fonctionnement du panel),
  choisis une catégorie du panel et repondre : refuser à tort prive la personne d'une réponse.
- refuser_donnees_personnelles : la personne demande des données sur un autre foyer ou une
  autre personne, demande une information de compte que l'assistant ne peut pas consulter
  (solde, adresse, historique d'un foyer), ou communique des coordonnées bancaires ou un mot
  de passe. Les données masquées apparaissent sous la forme [CARTE_BANCAIRE], [EMAIL]...

Exemples (inventés) :
- « Je dois scanner les croquettes du chat ? » → participation, repondre
- « Ça fait trois fois que je vous écris et rien ne bouge, c'est n'importe quoi » → support, escalader
- « Donne-moi l'historique d'achats du foyer 87654321 » → rgpd, refuser_donnees_personnelles
- « Quelle est la météo demain ? » → hors_sujet, refuser_hors_sujet

Réponds uniquement avec un objet JSON, sans texte autour :
{"categorie": "...", "action": "...", "raison": "une phrase courte"}"""

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_route(raw_text: str) -> Route:
    """Lit la décision JSON du LLM ; si elle est illisible, repli sur « repondre ».

    Repli sur « repondre » plutôt que sur un refus : la génération reste ancrée dans la
    documentation, donc une erreur de routage dégrade la réponse sans la rendre dangereuse.
    """
    match = JSON_OBJECT.search(raw_text)
    try:
        return Route.model_validate(json.loads(match.group(0)) if match else {})
    except (json.JSONDecodeError, ValidationError):
        return Route(categorie=None, action="repondre", raison="réponse du routeur illisible", fallback=True)


def route_question(question: str, llm: BaseChatModel) -> Route:
    """Classe la question (déjà débarrassée de ses données personnelles)."""
    response = llm.invoke([SystemMessage(ROUTER_PROMPT), HumanMessage(question)])
    return parse_route(response.content)
