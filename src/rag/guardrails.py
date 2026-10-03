"""Garde-fous déterministes, appliqués avant tout appel à un LLM.

Les données personnelles saisies par un panéliste (carte bancaire, IBAN, email, téléphone)
sont masquées avant d'être envoyées au routeur et au LLM de génération : ces services sont
externes, et l'assistant n'a besoin d'aucune de ces données pour répondre. Des expressions
régulières plutôt qu'un LLM : c'est instantané, gratuit, testable, et ça fonctionne même si
le LLM est indisponible.
"""
import re

# Les motifs sont appliqués dans cet ordre, chacun sur le texte déjà masqué par les précédents :
# l'IBAN d'abord (il contient des groupes de 4 chiffres qu'on prendrait pour une carte), puis la
# carte (dont un morceau ressemblerait à un téléphone).
PII_PATTERNS: dict[str, re.Pattern] = {
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"),
    # 3 à 5 groupes de 4 chiffres (ou de X masquant des chiffres), séparés ou non.
    "CARTE_BANCAIRE": re.compile(r"\b(?:\d{4}|[xX*]{4})(?:[ -]?(?:\d{4}|[xX*]{4})){2,4}\b"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "TELEPHONE": re.compile(r"(?:\+33\s?|\b0)[1-9](?:[ .-]?\d{2}){4}\b"),
}

# Types de données dont la seule présence impose un refus : l'assistant doit rappeler de ne
# jamais les communiquer (corpus RGPD-03), quel que soit l'avis du routeur.
BLOCKING_PII = {"CARTE_BANCAIRE", "IBAN"}

OFF_TOPIC_ANSWER = (
    "Je suis l'assistant des panélistes PanelConso : je réponds uniquement aux questions sur "
    "votre participation au panel (application et scan, points et récompenses, compte, "
    "données personnelles, contact du support). N'hésitez pas à me poser une question sur ces sujets."
)


def scan_pii(text: str) -> tuple[str, set[str]]:
    """Masque les données personnelles et liste leurs types.

    Renvoie le texte où chaque donnée est remplacée par son type entre crochets (ex. « [EMAIL] »)
    et l'ensemble des types trouvés (ex. {"EMAIL"}).
    """
    found = set()
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.add(pii_type)
            text = pattern.sub(f"[{pii_type}]", text)
    return text, found
