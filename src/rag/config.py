"""Paramètres du pipeline RAG, centralisés ici avec la justification de chaque valeur.

Toute valeur qu'on peut vouloir régler vit dans ce fichier, jamais en dur dans le code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- Données ---------------------------------------------------------------------------
CORPUS_DIR = ROOT / "corpus"
QUESTIONS_PATH = ROOT / "eval/questions.jsonl"

# --- Découpage (chunking) --------------------------------------------------------------
# On coupe sur les titres Markdown : chaque section « ## » traite d'un seul sujet (ex.
# « Mot de passe oublié », « Compte bloqué »), c'est la frontière sémantique naturelle
# du corpus. Le titre « # » est conservé en métadonnée.
HEADERS_TO_SPLIT_ON = [("#", "titre_doc"), ("##", "section")]

# Filet de sécurité pour une section trop longue. Les sections actuelles font 100 à 400
# caractères, donc ce second découpage ne se déclenche pas sur le corpus actuel ; il
# protège contre un futur document sans sous-titres. 1 000 caractères ≈ 250 tokens, bien
# en dessous de la limite de 512 tokens du modèle d'embeddings (au-delà, le texte est tronqué).
CHUNK_SIZE = 1000
# Chevauchement de 10 % : une phrase coupée à la frontière reste lisible dans les deux
# morceaux. Plus serait redondant sur des sections aussi courtes.
CHUNK_OVERLAP = 100

# --- Embeddings ------------------------------------------------------------------------
# Multilingue (corpus FR, questions FR/EN), entraîné pour la recherche question → passage,
# petit (118 M paramètres, vecteurs de 384 dimensions), tourne sur CPU. Voir docs/decisions.md (D005).
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Les modèles E5 ont été entraînés avec ces préfixes : les omettre dégrade la recherche.
# Ils indiquent au modèle si le texte est une question courte ou un passage à retrouver.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# --- Vector store ----------------------------------------------------------------------
CHROMA_DIR = ROOT / "data/chroma"
COLLECTION_NAME = "panelconso_corpus"

# --- Recherche -------------------------------------------------------------------------
# Nombre de morceaux renvoyés, donc donnés au LLM en phase 4. 5 morceaux × ~170 caractères
# ≈ 850 caractères de contexte : largement assez court pour le prompt, et assez pour couvrir
# les questions à 2 ou 3 documents sources (11 questions sur 62 en dev). Valeur à confirmer
# par la courbe recall@1/3/5 de l'évaluation.
TOP_K = 5

# Mode de recherche utilisé par défaut : "vector", "bm25" ou "hybrid".
# "vector" : meilleur recall@5 du dev (0,922, contre 0,898 pour l'hybride français). L'hybride
# français classe mieux (recall@1 0,777 contre 0,723) mais les écarts portent sur 1 à 5
# questions : non significatif. À réévaluer de bout en bout en phase 6 (voir D009, D010).
RETRIEVAL_MODE = "vector"

# Prétraitement du texte pour BM25 :
# - "basic"  : minuscules, sans accents, sans ponctuation ;
# - "french" : "basic" + suppression des mots vides + racinisation Snowball (« bancaires »
#   → « bancair »), pour que les variantes d'un même mot se correspondent.
# "french" : meilleur sur toutes les métriques du dev (BM25 seul : recall@5 0,796 → 0,884,
# MRR 0,728 → 0,789), voir D010.
BM25_ANALYZER = "french"

# Recherche hybride : chaque retriever propose ce nombre de candidats avant fusion. Plus
# large que TOP_K pour qu'un morceau classé 7e par un retriever mais 1er par l'autre puisse
# remonter dans le top 5 fusionné. 10 = 1/8 du corpus (80 morceaux) : au-delà, on ne fait
# qu'ajouter des morceaux non pertinents aux deux listes.
FUSION_CANDIDATES = 10
# Poids de la recherche vectorielle dans la fusion (BM25 reçoit 1 - VECTOR_WEIGHT).
# Avec l'analyseur français, 0,5 donne le meilleur classement (recall@1 0,777, MRR 0,890,
# contre 0,712 et 0,864 à 0,7). Avec l'analyseur basique, 0,7 était meilleur (voir D009, D010).
VECTOR_WEIGHT = 0.5
# Constante k de la Reciprocal Rank Fusion : score = poids / (rang + RRF_C). 60 est la valeur
# de l'article d'origine (Cormack et al., 2009), reprise par défaut partout. Elle écrase les
# écarts entre rangs (1er : 1/61, 5e : 1/65) : un morceau présent dans les deux listes bat
# presque toujours un morceau premier d'une seule liste. On ne la règle pas : trop peu de
# questions pour régler deux paramètres de fusion sans sur-apprendre le dev.
RRF_C = 60

# Contexte donné au LLM :
# - "section"  : les TOP_K morceaux retrouvés, tels quels ;
# - "document" : les documents entiers dont proviennent ces morceaux (sans doublon). Les
#   documents font 0,4 à 1,3 Ko : au plus ~6 Ko de contexte. Vise le cas « bon document,
#   mauvaise section » observé en phase 4 (Q009, Q015, voir D012).
# "document" : corrige Q009 (hallucination) et Q015 sur le dev, sans perte ailleurs ; contexte
# moyen 2 800 caractères au lieu de 1 250 (≈ 700 tokens), latence inchangée (D013).
CONTEXT_MODE = "document"

# --- Génération ------------------------------------------------------------------------
# Modèle appelé via LiteLLM (préfixe « openrouter/ » = API OpenRouter, clé OPENROUTER_API_KEY
# dans .env). Mistral Medium 3.1 : très bon en français et pour suivre des consignes strictes
# (s'en tenir aux extraits, citer, dire « je ne sais pas »), fiable à l'usage (5/5 appels,
# 0,8 s). Mistral Large, choisi d'abord, était saturé (0/5) ; NVIDIA NIM inutilisable. Voir D011.
LLM_MODEL = "openrouter/mistralai/mistral-medium-3.1"
LLM_API_KEY_ENV = "OPENROUTER_API_KEY"
# Température 0 : on veut la réponse la plus probable, pas de créativité. Cela rend aussi
# les évaluations plus reproductibles (sans garantir un déterminisme parfait côté API).
LLM_TEMPERATURE = 0.0
# Réponses de support : quelques phrases suffisent. La limite évite aussi qu'une réponse
# qui dérive ne coûte cher. ~400 tokens ≈ 300 mots.
LLM_MAX_TOKENS = 400
# Nouvelles tentatives automatiques (attente croissante) en cas d'erreur réseau ou de quota :
# le quota de Mistral via OpenRouter est partagé et renvoie parfois des erreurs 429 passagères
# (observé lors du premier essai). 6 tentatives laissent environ une minute pour que ça passe.
LLM_MAX_RETRIES = 6
LLM_TIMEOUT_SECONDS = 60

# --- Juge (évaluation de la génération, phase 6) ---------------------------------------
# Un modèle d'une autre famille que celui qui génère : un LLM a tendance à juger plus
# favorablement des textes proches des siens (biais d'auto-préférence). Gemini 2.5 Flash :
# juge reconnu, ~0,06 $ par évaluation du dev, pas de tokens de réflexion via OpenRouter
# (mesuré : 0). Même clé OpenRouter que la génération. Température 0 (LLM_TEMPERATURE). Voir D016.
JUDGE_MODEL = "openrouter/google/gemini-2.5-flash"
# Le verdict est un petit JSON (3 booléens + une explication courte).
JUDGE_MAX_TOKENS = 300

# --- Service (phase 7) -----------------------------------------------------------------
# Longueur maximale d'une question acceptée par l'API. Les questions du jeu d'évaluation font
# moins de 150 caractères ; 1 000 laisse de la marge pour une description détaillée, tout en
# bornant le coût d'un appel et en limitant les abus (textes collés en masse).
MAX_QUESTION_CHARS = 1000
# Adresse de l'API appelée par l'interface Streamlit. En local : http://localhost:8000 ; dans
# Docker Compose, la variable d'environnement API_URL pointe vers le service « api ».
API_URL = "http://localhost:8000"

# --- Seuils bloquants de la CI (calculés sur le dev, jamais sur le test : D003) ----------
# Valeurs mesurées sur le dev (runs e2e_dev_v1 et e2e_dev_v2) moins une marge, car le
# générateur ET le juge varient d'une exécution à l'autre : sur un groupe de n questions, une
# seule bascule déplace la métrique de 1/n. Les seuils détectent une régression, pas un bruit.
CI_THRESHOLDS = {
    # Recherche : déterministe (mesuré 0,922 / 0,862) ; marge d'environ une question.
    "retrieval_recall_at_5": 0.90,
    "retrieval_mrr": 0.84,
    # Routage (67 questions, mesuré 0,955) : tolère 2 bascules.
    "routing_accuracy": 0.92,
    # Juge (mesuré 1,000 et 0,94 à 0,96) : tolère 3 à 4 bascules.
    "faithfulness": 0.95,
    "correctness.repondre": 0.88,
    "behavior_ok.repondre": 0.95,
    # Sécurité : aucun écart toléré sur les refus (6 questions, mesuré 1,000).
    "behavior_ok.refuser": 1.00,
    # Petits groupes (5 et 6 questions, mesurés 0,667 à 1,000) : seuils bas, une bascule = 17 à 20 points.
    "behavior_ok.escalader": 0.80,
    "behavior_ok.repondre_sans_inventer": 0.66,
    # Langue : régression déjà survenue une fois (D016), contrôle déterministe sur 3 questions.
    "language_ok.anglais": 1.00,
}
# Une citation inventée sur 67 réponses (1,5 %) est tolérée ; deux ne le sont pas.
CI_MAX_INVALID_CITATION_RATE = 0.02

# --- Évaluation ------------------------------------------------------------------------
# Valeurs de k rapportées : k=1 dit si le meilleur morceau est le bon, k=TOP_K dit si le
# LLM aura l'information sous les yeux ; k=3 montre la forme de la courbe entre les deux.
EVAL_K_VALUES = (1, 3, TOP_K)
# Base locale MLflow (un fichier SQLite, ignoré par git). Consulter avec :
# uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "retrieval"
