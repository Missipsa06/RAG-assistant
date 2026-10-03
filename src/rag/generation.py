"""Génération ancrée : le LLM répond à partir des seuls extraits retrouvés, en citant ses sources."""
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_litellm import ChatLiteLLM

from rag import config

# Marqueur ajouté par le LLM quand les extraits ne contiennent pas l'information. On détecte
# un marqueur plutôt qu'une phrase (« Je ne trouve pas... ») car le LLM répond dans la langue
# de la question : une phrase fixe ne serait pas reconnue en anglais. Il est retiré avant affichage.
NOT_IN_DOCS_MARKER = "[HORS_DOC]"

SYSTEM_PROMPT = f"""Tu es l'assistant de support des panélistes PanelConso.
Tu réponds en t'appuyant UNIQUEMENT sur les extraits de documentation fournis.

Règles :
1. N'utilise aucune connaissance extérieure aux extraits. N'invente jamais de chiffre, de
   délai, de nom d'enseigne, de numéro de téléphone ni de procédure.
2. Après chaque information, cite l'identifiant du document source entre crochets, par
   exemple [REC-04]. N'utilise que les identifiants présents dans les extraits.
3. Si les extraits ne contiennent AUCUNE information utile pour répondre, dis-le simplement,
   propose de contacter le support, et termine ta réponse par le marqueur {NOT_IN_DOCS_MARKER}.
   N'ajoute jamais ce marqueur si tu donnes une réponse, même partielle.
4. Si les extraits ne répondent qu'à une partie de la question, réponds à cette partie et
   précise clairement ce qui n'est pas couvert par la documentation (sans marqueur).
5. Réponds dans la langue de la question, même si les extraits sont en français (question en
   anglais → réponse en anglais), en quelques phrases claires, en vouvoyant.
6. Ne demande jamais de données personnelles (mot de passe, coordonnées bancaires, adresse,
   numéro de foyer) et ne répète jamais celles que la personne aurait communiquées."""

# Consignes ajoutées à la demande selon l'action décidée par le routeur (phase 5).
ESCALATION_INSTRUCTION = """Cette demande nécessite l'intervention d'un conseiller. Commence par
reconnaître la situation de la personne en une phrase, donne ce que la documentation permet de
faire, puis indique clairement comment joindre le support (canaux et horaires). S'il s'agit d'un
mécontentement ou d'une réclamation, explique aussi la procédure de réclamation."""

PERSONAL_DATA_INSTRUCTION = """Cette demande touche à des données personnelles. Ne communique
aucune information sur un foyer ou une personne : tu n'as accès à aucun compte. Si la personne a
communiqué des coordonnées bancaires ou un mot de passe (masqués sous la forme [CARTE_BANCAIRE]...),
rappelle-lui de ne jamais les transmettre. Indique ensuite, si la documentation le permet,
comment elle peut obtenir elle-même l'information sur son propre compte."""

CITATION_PATTERN = re.compile(r"\[([A-Z]+-\d+)\]")


@dataclass
class Answer:
    """Réponse générée, avec ce qu'il faut pour l'afficher et l'évaluer."""

    text: str  # réponse affichable (marqueur retiré)
    context_ids: list[str]  # documents fournis au LLM, dans l'ordre de la recherche
    cited_ids: list[str]  # documents cités dans la réponse, sans doublon
    invalid_citations: list[str]  # citations d'un document qui n'était pas dans les extraits
    not_in_docs: bool  # le LLM a signalé que l'information est absente des extraits
    action: str = "repondre"  # action décidée par le routeur (phase 5)
    category: str | None = None  # catégorie décidée par le routeur


def get_llm(model: str = config.LLM_MODEL, max_tokens: int = config.LLM_MAX_TOKENS) -> ChatLiteLLM:
    """LLM appelé via LiteLLM (génération par défaut, juge en phase 6).

    La clé d'API est lue dans .env, jamais dans le code.
    """
    load_dotenv(config.ROOT / ".env")
    if not os.environ.get(config.LLM_API_KEY_ENV):
        raise RuntimeError(f"Clé d'API manquante : renseigner {config.LLM_API_KEY_ENV} dans le fichier .env")
    return ChatLiteLLM(
        model=model,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=max_tokens,
        max_retries=config.LLM_MAX_RETRIES,
        request_timeout=config.LLM_TIMEOUT_SECONDS,
    )


def unique_doc_ids(chunks: list[Document]) -> list[str]:
    """Identifiants des documents des morceaux, sans doublon, dans l'ordre de la recherche."""
    return list(dict.fromkeys(chunk.metadata["id"] for chunk in chunks))


def format_context(chunks: list[Document]) -> str:
    """Présente les morceaux au LLM, chacun précédé de son identifiant citable et de son titre.

    L'identifiant entre crochets est exactement ce que le LLM doit recopier pour citer.
    """
    blocks = []
    for chunk in chunks:
        title = chunk.metadata["titre"]
        section = chunk.metadata.get("section")
        heading = f"{title} › {section}" if section else title
        blocks.append(f"[{chunk.metadata['id']}] {heading}\n{chunk.page_content.strip()}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, chunks: list[Document], instruction: str | None = None) -> list:
    """Messages envoyés au LLM : les règles (système), puis les extraits, la consigne éventuelle et la question."""
    user_prompt = f"Extraits de la documentation :\n\n{format_context(chunks)}\n\n"
    if instruction:
        user_prompt += f"Consigne particulière : {instruction}\n\n"
    user_prompt += f"Question : {question}"
    return [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_prompt)]


def parse_answer(raw_text: str, context_ids: list[str]) -> Answer:
    """Extrait les citations et le marqueur « hors documentation » de la réponse brute du LLM."""
    cited_ids = list(dict.fromkeys(CITATION_PATTERN.findall(raw_text)))
    return Answer(
        text=raw_text.replace(NOT_IN_DOCS_MARKER, "").strip(),
        context_ids=context_ids,
        cited_ids=cited_ids,
        invalid_citations=[doc_id for doc_id in cited_ids if doc_id not in context_ids],
        not_in_docs=NOT_IN_DOCS_MARKER in raw_text,
    )


def generate_answer(
    question: str, chunks: list[Document], llm: BaseChatModel, instruction: str | None = None
) -> Answer:
    """Demande au LLM une réponse ancrée dans les morceaux fournis."""
    response = llm.invoke(build_messages(question, chunks, instruction))
    return parse_answer(response.content, unique_doc_ids(chunks))
