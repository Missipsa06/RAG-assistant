# Journal des décisions techniques

Une entrée par décision : contexte, options, choix, raison, résultat mesuré.

---

## D001 - Le dataset devient la racine du repo (2026-09-30)

- **Contexte** : le dataset était livré dans `panel-rag-dataset.zip`, avec un dossier parent.
- **Options** : (a) garder le zip, (b) décompresser dans `panel-rag-dataset/`, (c) décompresser à la racine.
- **Choix** : (c) `corpus/`, `eval/`, `scripts/` à la racine de `RAG-assistant/`.
- **Raison** : git ne versionne pas le contenu d'un zip (pas de diff sur les corrections du
  corpus) ; `validate.py` attend `corpus/` et `eval/` à côté de `scripts/` ; un niveau de
  dossier en moins dans tous les chemins.
- **Résultat** : `validate.py` passe depuis la racine (27 documents, 86 questions).

---

## D002 - Jeu de test figé, stratifié, sans fuite entre reformulations (2026-09-30)

- **Contexte** : on va régler des paramètres (taille des chunks, k, poids hybrides) et des
  prompts en regardant les erreurs sur les questions. Un score mesuré sur ces mêmes
  questions est optimiste : il mesure en partie notre ajustement à ces questions, pas la
  qualité sur des questions nouvelles (même principe que le sur-apprentissage).
- **Options** :
  1. Tirage aléatoire simple de 20 %.
  2. Tirage stratifié par type de question.
  3. Tirage stratifié par catégorie (questions répondables), type (pièges, hors sujet,
     sécurité, escalade) et style (non standard à part), avec regroupement des reformulations.
- **Choix** : option 3, dans `scripts/split_questions.py`, graine fixe (42), split écrit
  dans `eval/questions.jsonl` (champ `split`).
- **Raison** :
  - Le tirage simple peut ne mettre aucune question rare en test (8 questions en style
    non standard, 4 hors sujet).
  - L'option 2, essayée d'abord, laissait 1 question « récompenses » sur 23 en test et
    aucune en « support » : le routage aurait été évalué sur un test déséquilibré.
  - 4 paires de questions sont des reformulations (ex. Q003 / Q080, même question en
    français et en anglais). Si l'une est en dev et l'autre en test, régler le système sur
    la première revient à voir la seconde. Elles sont forcées dans le même split.
  - Un split déjà attribué n'est jamais modifié : quand le jeu passera à 150-200
    questions, les nouvelles reformulations hériteront du split de leur question d'origine.
- **Résultat** : 67 dev / 19 test (22 %). En test : 12 factuelles, 1 multi-docs, 2 pièges,
  1 hors sujet, 1 sécurité, 2 escalades ; toutes les catégories sauf « support » (4
  questions au total, aucune tirée).
- **Limite assumée** : 19 questions, c'est peu. Une question vaut ~5 points de
  pourcentage, donc un écart de moins de 10 points entre deux systèmes sur le test n'est
  pas interprétable. Le test sert au chiffre final, le dev (67) sert à comparer les variantes.

---

## D003 - Seuils bloquants de la CI calculés sur le dev (2026-09-30)

- **Contexte** : la CI (phase 6) doit bloquer les régressions.
- **Options** : seuils sur le test, ou seuils sur le dev.
- **Choix** : dev. Le test est évalué uniquement pour le chiffre final du README.
- **Raison** : une CI rouge sur le test pousse à corriger jusqu'à ce qu'elle passe, donc à
  régler le système sur le test, qui perd alors sa valeur. Avec 19 questions, le signal
  serait aussi trop bruité pour bloquer une fusion (1 question ≈ 5 points).
- **Résultat** : à mesurer en phase 6.

---

## D004 - LangChain comme framework RAG (2026-09-30)

- **Contexte** : ingestion, recherche et génération peuvent s'écrire à la main ou via un framework.
- **Options** : code à la main, LangChain, LlamaIndex (LangGraph n'est pas une alternative :
  c'est une couche au-dessus de LangChain pour les flux à états, à envisager en phase 5).
- **Choix** : LangChain, en utilisant ses briques (découpage, vector store, retrievers)
  plutôt que ses chaînes toutes faites, pour que chaque étape reste visible dans le code.
- **Raison** : choix de l'auteur du projet ; LangChain est le framework le plus demandé
  dans les offres visées et couvre toutes les phases 1 à 4 avec la stack autorisée.
  LlamaIndex est plus « clé en main » (moins de contrôle sur chaque étape) et hors de la
  stack autorisée.
- **Résultat** : à mesurer en phases 2-3.

---

## D005 - Modèle d'embeddings : multilingual-e5-small, local (2026-09-30)

- **Contexte** : corpus en français, questions en français (parfois familier ou avec fautes)
  et en anglais ; déploiement visé sur CPU gratuit (Hugging Face Spaces).
- **Options** : multilingual-e5-small, bge-m3, paraphrase-multilingual-MiniLM, API d'embeddings.
- **Choix** : `intfloat/multilingual-e5-small` via `langchain-huggingface`, préfixes
  `query: ` / `passage: `, vecteurs normalisés.
- **Raison** : multilingue, entraîné pour la recherche question → passage, 118 M paramètres
  (tourne sur CPU). bge-m3 est meilleur mais ~5× plus lourd ; MiniLM est entraîné pour
  comparer des phrases de même nature ; une API ajoute coût et dépendance réseau.
- **Résultat** : 384 dimensions, norme 1, question anglaise plus proche du bon passage
  français (tests). Qualité de recherche : à mesurer en phase 2 ; bge-m3 est la première
  alternative à tester si le recall déçoit.

---

## D006 - Vector store : Chroma local persistant (2026-09-30)

- **Contexte** : 80 morceaux à indexer ; besoin de persister l'index et, en phase 5, de
  filtrer par catégorie.
- **Options** : Chroma, FAISS, matrice numpy, base serveur (Qdrant, pgvector).
- **Choix** : Chroma via `langchain-chroma`, dossier `data/chroma/` (ignoré par git).
- **Raison** : persistance sur disque, filtre par métadonnées natif, pas de serveur. FAISS n'a
  pas de filtre natif et sa sauvegarde LangChain passe par `pickle`. À cette échelle, le
  choix ne change pas la qualité ni la vitesse.
- **Résultat** : index reconstruit de zéro en 18,5 s (dont chargement du modèle).

---

## D007 - Découpage par section Markdown (2026-09-30)

- **Contexte** : 27 documents courts (0,4 à 1,3 Ko), chacun structuré en sections « ## »
  d'un seul sujet.
- **Options** : par section, document entier, taille fixe, découpage sémantique.
- **Choix** : `MarkdownHeaderTextSplitter` sur `#` et `##` (titres conservés dans le texte),
  puis `RecursiveCharacterTextSplitter` (1 000 caractères, overlap 100) en filet de sécurité.
  Chaque morceau hérite des métadonnées du document + `section` + `chunk_id`.
- **Raison** : les sections sont les frontières de sens posées par les rédacteurs ;
  1 000 caractères ≈ 250 tokens reste sous la limite de 512 tokens du modèle.
- **Résultat** : 80 morceaux, 48 à 750 caractères (médiane 166) ; le second découpage ne se
  déclenche jamais. **Risque observé** : 10 morceaux < 100 caractères, certains privés de
  contexte (ex. `CPT-05#2`, « ## ScanBox / Renvoyez-la sous 30 jours » : rien ne dit qu'il
  s'agit de quitter le panel). Impact à mesurer en phase 2.

---

## D008 - Protocole d'évaluation de la recherche et baseline vectorielle (2026-09-30)

- **Contexte** : il faut une baseline chiffrée avant toute amélioration de la recherche.
- **Options** : métriques au niveau document ou section ; k en morceaux ou en documents
  distincts ; recall, hit rate, precision, nDCG.
- **Choix** : recall@1/3/5 et MRR au niveau **document**, k compté en **morceaux** (ce que
  le LLM reçoit), questions sans source exclues, dev uniquement (62 questions), chaque run
  tracé dans MLflow (`mlflow.db`, expérience `retrieval`).
- **Raison** : les annotations sont au niveau document ; le recall répond à « le LLM
  aura-t-il l'information ? » ; la précision est plafonnée (1 à 3 documents pertinents pour k = 5).
- **Résultat (run `baseline_vector`)** : recall@1 0,723, recall@3 0,852, **recall@5 0,922**,
  **MRR 0,862**. Factuelles : recall@5 0,988 (41 questions). Escalade : 0,500 (5 questions).
  8 questions avec un document attendu manquant au top 5 ; analyse :
  - 5 sur 8 ratent un document « secondaire » que la question ne mentionne pas : le support
    (SUP-01 ×3, SUP-02), les règles de participation (PAN-01). Q074 « Je veux parler à un
    humain » ne partage aucun mot avec SUP-01 : relève du routage (phase 5) plus que de la recherche.
  - Q068 rate RGPD-03 alors que « carte bancaire » y figure mot pour mot (section
    « Informations bancaires ») : échec typique de la recherche vectorielle sur une
    correspondance exacte ; hypothèse à tester avec BM25 en phase 3.
  - Certains morceaux génériques reviennent souvent comme bruit (APP-07 dans 5 échecs, dont
    `APP-07#1` « Forcer la synchronisation ») et des doublons du même document occupent des
    places du top 5.

---

## D009 - Recherche hybride testée, non retenue ; le vectoriel reste le mode par défaut (2026-09-30)

- **Contexte** : hypothèse issue de D008 : BM25 rattraperait les correspondances exactes
  ratées par la recherche vectorielle (ex. « carte bancaire », Q068).
- **Options** : vectoriel seul, BM25 seul, hybride RRF avec poids vectoriel 0,3 / 0,5 / 0,7.
- **Implémentation** : `BM25Retriever` (langchain-community, rank-bm25) avec normalisation
  (minuscules, accents, ponctuation) ; `EnsembleRetriever` (langchain-classic), 10 candidats
  par retriever, c = 60, dédoublonnage par `chunk_id`, **troncature explicite à k = 5**
  (EnsembleRetriever renvoie l'union complète des listes).
- **Résultat (dev, 62 questions, runs MLflow)** :

  | Run | recall@1 | recall@3 | recall@5 | MRR |
  |---|---|---|---|---|
  | baseline_vector | **0,723** | 0,852 | **0,922** | **0,862** |
  | bm25_only | 0,575 | 0,758 | 0,796 | 0,728 |
  | hybrid_w03 | 0,688 | 0,831 | 0,844 | 0,815 |
  | hybrid_w05 | 0,720 | 0,836 | 0,892 | 0,851 |
  | hybrid_w07 | 0,720 | 0,868 | 0,901 | 0,858 |

- **Choix** : `RETRIEVAL_MODE = "vector"`. Le mode hybride est gardé (et testé) pour les
  expériences suivantes, avec `VECTOR_WEIGHT = 0.7`, le meilleur des poids testés.
- **Raison (analyse d'erreurs)** :
  - BM25 est faible en français sans racinisation ni mots vides : Q005 « mes points
    **expirent** » ne retrouve pas « **Expiration** des points » ; son top 5 est rempli de
    morceaux qui ne partagent que « est », « ce », « que », « mes » avec la question.
  - La RRF à c = 60 favorise le consensus : sur Q068, BM25 classe `RGPD-03#2` 2e, mais ce
    morceau (absent du top 10 vectoriel, score $\frac{0{,}5}{62} \approx 0{,}0081$) est battu
    par `APP-01#2`, 7e en vectoriel et 3e en BM25 ($\frac{0{,}5}{67} + \frac{0{,}5}{63} \approx 0{,}0154$).
  - Tendance monotone : plus le poids vectoriel est élevé, meilleur est le résultat.
- **Risque noté** : `langchain-community` est annoncé en fin de vie ; remplacement possible
  par `rank_bm25` utilisé directement (~15 lignes).
- **Prochaine expérience proposée** : BM25 avec mots vides et racinisation française, avant
  de conclure définitivement sur l'hybride.

---

## D010 - BM25 avec analyseur français (mots vides + racinisation Snowball) (2026-09-30)

- **Contexte** : D009 attribue l'échec de l'hybride à la faiblesse de BM25 en français.
- **Options** : analyseur `basic` (minuscules, accents, ponctuation) ou `french` (+ ~80 mots
  vides + racinisation Snowball, dépendance `snowballstemmer` validée par l'auteur).
- **Implémentation** : `analyze_french_for_bm25()` dans `retrieval.py`. Accents retirés
  **avant** la racinisation (« données » et « donnees » → « donne » ; dans l'ordre inverse :
  « don » et « donne »). Négations (« pas », « plus », « jamais ») gardées.
- **Résultat (dev, 62 questions)** :

  | Run | recall@1 | recall@3 | recall@5 | MRR |
  |---|---|---|---|---|
  | baseline_vector | 0,723 | 0,852 | **0,922** | 0,862 |
  | bm25_only (basic) | 0,575 | 0,758 | 0,796 | 0,728 |
  | **bm25_fr** | 0,608 | 0,863 | 0,884 | 0,789 |
  | **hybrid_fr_w05** | **0,777** | **0,884** | 0,898 | **0,890** |
  | hybrid_fr_w07 | 0,712 | 0,884 | 0,906 | 0,864 |

  Comparaison appariée hybrid_fr_w05 / baseline : recall@1 meilleur sur 5 questions, moins
  bon sur 1 ; recall@5 meilleur sur 1 (Q013), moins bon sur 2 (Q070 « adresse du foyer »,
  Q085 « lappli plante tout le tps »).
- **Choix** : `BM25_ANALYZER = "french"` (meilleur sur toutes les métriques de BM25) et
  `VECTOR_WEIGHT = 0.5`. **`RETRIEVAL_MODE` reste "vector"** : c'est lui qui maximise le
  recall@5, critère principal fixé en D008 (le LLM aura-t-il l'information ?).
- **Raison** : l'hybride français classe mieux (+5,4 points de recall@1), mais tous les écarts
  portent sur 1 à 5 questions : non significatif sur 62 questions, et 5 variantes ont été
  essayées sur le même dev (risque de retenir un gagnant par hasard). À écart non
  significatif, on garde la solution la plus simple (sans dépendance en fin de vie).
- **Constat honnête** : la racinisation ne corrige pas le cas qui motivait l'expérience :
  Snowball laisse « expirent » intact (alors que « expiration » → « expir »). Le gain sur Q005
  vient de la suppression des mots vides, qui retire le bruit que BM25 injectait.
- **Suite** : l'hybride français est le candidat à réévaluer **de bout en bout** en phase 6 :
  un meilleur premier morceau donne-t-il de meilleures réponses ?

---

## D011 - Fournisseur et modèle de génération : Mistral Medium 3.1 via OpenRouter (2026-09-30)

- **Contexte** : il faut un LLM accessible par LiteLLM, bon en français et fiable pour suivre
  des consignes strictes (s'en tenir aux extraits, citer, dire « je ne sais pas »).
- **Options essayées, dans l'ordre** :
  1. NVIDIA NIM, Mistral Large 2 : erreur 404 (modèle listé au catalogue mais non servi).
     Sur 15 modèles testés, 1 seul répondait (un modèle de raisonnement qui affiche sa
     réflexion dans la réponse), puis erreur 403 « Authorization failed » sur tous. Abandonné.
  2. OpenRouter, Mistral Large 2512 : quota partagé saturé chez son unique fournisseur,
     **0 appel réussi sur 5** (erreurs 429), même avec 6 nouvelles tentatives.
  3. OpenRouter, Mistral Medium 3.1 : **5/5**, 0,8 s ; Mistral Small 3.2 : 5/5, 3,3 s.
- **Choix** : `openrouter/mistralai/mistral-medium-3.1` (validé par l'auteur), température 0,
  400 tokens max, 6 nouvelles tentatives automatiques.
- **Raison** : fiabilité mesurée, rapidité, qualité en français ; coût d'une évaluation des
  67 questions du dev estimé à quelques centimes.
- **Leçon** : LiteLLM a permis de changer deux fois de fournisseur en modifiant deux lignes de
  `config.py`, sans toucher au reste du code.

---

## D012 - Génération ancrée : prompt, citations, marqueur « hors documentation » (2026-09-30)

- **Contexte** : phase 4, répondre à partir des seules sources, citer, dire « je ne sais pas ».
- **Choix** : prompt système en 5 règles ; extraits étiquetés `[ID] Titre › Section` ; citations
  `[REC-04]` vérifiées par regex contre les documents fournis ; marqueur `[HORS_DOC]` (plutôt
  qu'une phrase fixe, pour fonctionner quelle que soit la langue de la réponse) ; `ChatLiteLLM`
  (LangChain + LiteLLM).
- **Résultat (dev, 67 questions, run `gen_medium31_v1`)** :
  - « repondre » (50) : répond 94 %, cite une source 98 %, cite la source attendue 98 %,
    **0 citation inventée** ; latence moyenne 0,8 s.
  - « repondre_sans_inventer » (6) : **6/6 comportements corrects à la relecture humaine**
    (5 aveux d'absence ; Q057 répond correctement sans inventer de numéro).
  - « refuser » (6) et « escalader » (5) : réponses « hors documentation, contactez le
    support », inadaptées au hors sujet et à la sécurité ; point de départ de la phase 5.
- **Erreurs analysées** :
  - Q009 : **hallucination par inférence**. Seule l'introduction d'APP-05 a été retrouvée
    (pas la section « Procédure ») ; le LLM en déduit « vous ne pouvez pas enregistrer ce
    produit », ce qui contredit la documentation. Citation valide, réponse fausse.
  - Q015 : « je ne sais pas » justifié par le contexte reçu : la section « Plusieurs
    appareils dans le foyer » n'a pas été retrouvée, bien que le document APP-01 l'ait été.
  - Q058, Q073, Q075, Q085 : réponse partielle **et** marqueur ; le modèle confond « absent »
    et « partiel ».
- **Enseignements** : (1) le recall au niveau document surestime la qualité de recherche :
  bon document, mauvaise section ; (2) les indicateurs déterministes ne voient pas les
  inférences fausses : la fidélité doit être jugée (phase 6).
- **Pistes, à mesurer avant adoption** : donner au LLM le document entier des morceaux
  retrouvés (documents de 0,4 à 1,3 Ko : quelques Ko de contexte) ; préciser la règle du
  marqueur (seulement si aucune information utile).

---

## D013 - Contexte en documents entiers et règle du marqueur précisée (2026-10-01)

- **Contexte** : D012 identifie deux défauts : (1) bon document mais mauvaise section dans le
  contexte (Q009, hallucination ; Q015, faux « je ne sais pas ») ; (2) marqueur `[HORS_DOC]`
  ajouté aux réponses partielles.
- **Options testées** (dev, 67 questions, un seul changement à la fois) :

  | Run | Contexte | Prompt | « repondre » : a répondu | cite la source attendue | Q009 | Q015 |
  |---|---|---|---|---|---|---|
  | gen_medium31_v1 | sections | v1 | 0,94 | 0,98 | ❌ hallucination | ❌ « je ne sais pas » |
  | gen_doc_promptv1 | **documents** | v1 | 0,98 | **1,00** | ✅ | ✅ |
  | gen_doc_promptv2 | **documents** | **v2** | **1,00** | **1,00** | ✅ | ✅ |
  | gen_section_promptv2 | sections | **v2** | 1,00 | 1,00 | ❌ hallucination | ⚠️ réponse partielle honnête |

  0 citation inventée dans les 4 runs ; latence moyenne 0,7 à 0,8 s dans tous les cas.
- **Choix** : `CONTEXT_MODE = "document"` (technique du « document parent » : on cherche avec
  les sections, on donne au LLM le document entier) et prompt v2 (marqueur seulement si
  **aucune** information utile ; jamais avec une réponse, même partielle).
- **Raison** :
  - Seul le contexte en documents entiers corrige l'hallucination de Q009 : le prompt v2 seul
    (run C) la laisse intacte. La cause était bien le contexte incomplet, pas le prompt.
  - Le prompt v2 supprime le marqueur sur les réponses partielles (Q085, Q058) sans faire
    inventer : relecture des 6 questions « repondre_sans_inventer » toujours correcte.
  - Coût maîtrisé : contexte moyen 2 819 caractères au lieu de 1 249 (≈ 700 tokens au lieu
    de 300), 4,1 documents distincts en moyenne, max 4 298 caractères.
  - Effet de bord utile : le recall mesuré au niveau document (D008) correspond désormais
    exactement à ce que voit le LLM.
- **Défaut restant observé** : Q075 « Mes 3000 points ont disparu » : REC-05 (expiration) n'est
  pas retrouvé (rang 18 en recherche vectorielle), et le modèle affirme que les points peuvent
  être perdus « **uniquement** » dans les cas cités : généralisation abusive. À
  traiter par l'escalade (phase 5) et, côté recherche, par un reranking (piste notée).

---

## D014 - Garde-fous déterministes contre la collecte de données personnelles (2026-10-01)

- **Contexte** : phase 5, « aucune collecte de données personnelles ». Le routeur et le LLM de
  génération sont des services externes (OpenRouter, Mistral).
- **Options** : consigne dans le prompt seule ; regex maison ; bibliothèque (Presidio) ; LLM-juge.
- **Choix** : `scan_pii()` (regex : IBAN, carte bancaire, email, téléphone), appliqué **avant
  tout appel externe** ; seule la question masquée (`[CARTE_BANCAIRE]`...) quitte la machine.
  Carte bancaire ou IBAN → refus imposé, sans appeler le routeur. Règle 6 ajoutée au prompt de
  génération : ne jamais demander ni répéter de données personnelles.
- **Raison** : une garantie ne doit pas reposer sur un prompt ; un LLM-juge enverrait la donnée
  à l'extérieur pour la détecter ; Presidio est disproportionné pour 4 formats.
- **Résultat** : tests unitaires (formats détectés, nombres ordinaires non masqués) et test de
  pipeline qui vérifie que le numéro de carte n'atteint jamais le LLM. Sur le dev, 3/3 demandes
  de données personnelles refusées.
- **Limites** : formats non couverts (adresse postale, nom, numéro écrit en lettres) ; pas de
  défense dédiée contre l'injection de prompt (risque limité : l'assistant n'a accès à aucun
  compte ni action).

---

## D015 - Routeur par intention : LLM zero-shot à sortie JSON validée (2026-10-01)

- **Contexte** : répondre, escalader ou refuser selon la demande ; en phase 4, hors sujet et
  demandes sensibles recevaient des réponses inadaptées (« pas dans la documentation »).
- **Options** : règles par mots-clés ; classifieur entraîné ; LLM zero-shot ; routage dans le
  même appel que la génération.
- **Choix** : LLM zero-shot (Mistral Medium 3.1, température 0), sortie JSON validée par
  pydantic, repli sur « repondre » si illisible. 4 actions : repondre ; escalader (génération +
  fiches SUP-01/SUP-02 + consigne d'empathie et d'orientation) ; refuser_hors_sujet (message fixe,
  sans recherche ni génération) ; refuser_donnees_personnelles (génération + consigne de
  protection). La catégorie est journalisée mais ne filtre pas la recherche.
- **Raison** : pas assez d'exemples pour entraîner un classifieur ; les mots-clés ne résistent
  pas au langage familier ; un appel séparé permet d'évaluer le routage seul et de refuser sans
  générer. Hiérarchie des erreurs : un refus à tort est grave, une escalade en trop est bénigne
  (la réponse documentaire est donnée quand même).
- **Résultat (dev, 67 questions)** :

  | Version | Actions correctes | Escalades détectées | Refus corrects | Escalades en trop | Refus à tort |
  |---|---|---|---|---|---|
  | router_v1 | 89,6 % | 5/5 | 6/6 | 6 | 1 |
  | **router_v2** | **95,5 %** | 4/5 | 6/6 | 1 | 1 |

  Catégorie correcte : 86,6 % (désaccords de frontière, sans effet sur l'action). Taux de repli : 0 %.
- **Honnêteté sur le réglage** :
  - La règle ajoutée en v2 (« n'escalade pas ce que la personne peut résoudre seule ») cite des
    situations proches des erreurs du dev : le score du dev est donc optimiste.
  - En lisant les comportements attendus pour concevoir le routeur, j'ai affiché par erreur les
    4 questions du **jeu de test** de ces catégories (Q067, Q071, Q076, Q077). Leurs formulations
    n'ont pas été reprises (exemples du prompt inventés), mais le jeu de test n'est plus
    strictement « jamais vu » pour ces 4 questions ; à garder en tête en lisant les scores finaux.
- **Limites observées** : Q063 (« À quelle association vont les dons ? ») refusée à tort par les
  deux versions ; Q078 (accès au compte perdu) n'est plus escaladée en v2 ; Q054 escaladée alors
  qu'annotée « repondre » (annotation discutable).
- **Évaluation de bout en bout** (run `e2e_router_v2`, garde-fous + routeur + recherche +
  génération, documents entiers, prompt v2 + règle 6) :
  - « repondre » (50) : répond 100 %, cite la source attendue 100 %, 0 citation inventée :
    la règle 6 et le routage ne dégradent pas les réponses normales.
  - Relecture humaine des 11 questions « refuser » et « escalader » : 9 réponses conformes,
    2 incomplètes : Q075 n'évoque pas l'expiration des points (REC-05 non retrouvé par la
    recherche), Q079 propose l'échange de la ScanBox sans conseiller d'abord de vérifier la
    charge. Q078, non escaladée, reçoit malgré tout la bonne réponse (contacter le support,
    vérification d'identité).
  - Latence moyenne : 1,2 s (0,7 s sans routeur) : le routeur coûte un appel LLM par question.
  - Blocage rencontré : crédit OpenRouter épuisé (niveau gratuit), puis 10 $ ajoutés ; les
    modèles gratuits (50 appels par jour) ne suffisaient pas pour une évaluation (~134 appels).

---

## D016 - LLM-juge (Gemini 2.5 Flash), validé contre des verdicts humains (2026-10-01)

- **Contexte** : phase 6 ; les indicateurs déterministes (citations, marqueur) ne voient ni les
  inférences fausses (Q009) ni les réponses incomplètes.
- **Options** : métriques lexicales (BLEU, ROUGE), similarité d'embeddings, bibliothèques
  (Ragas, DeepEval), LLM-juge maison ; juge du même modèle que le générateur ou d'une autre famille.
- **Choix** : LLM-juge maison, `openrouter/google/gemini-2.5-flash` (choisi par l'auteur parmi
  Claude Haiku 4.5 et Gemini 2.5 Flash), température 0, 3 verdicts binaires (fidele, correcte,
  comportement_ok), sortie JSON validée, verdict illisible exclu et compté.
- **Raison** : autre famille que Mistral (biais d'auto-préférence) ; binaire plus stable qu'une
  note ; transparent et peu coûteux ; 0 token de réflexion mesuré via OpenRouter.
- **Validation (13 cas relus à la main, 36 verdicts)** : accord 86 % (grille v1) → 97 % (grille v2,
  « un verdict = un critère ; une omission n'est pas une infidélité ; une déduction non écrite
  est infidèle »). 97 % est optimiste : grille réglée sur ces mêmes cas. Deux exécutions de la
  validation donnent des verdicts identiques.
- **Résultats (dev, 3 exécutions de bout en bout)** : fidélité 0,985 à 1,000 ; correction des
  réponses « repondre » 0,94 à 0,96 ; tous les comportements à 1,000 sauf
  « repondre_sans_inventer » (0,667 à 1,000 selon l'exécution, 6 questions).
- **Ce que le juge a attrapé** : Q051 (« oui » faux : deux mois et demi > 60 jours) ; Q045
  (« médicaments sans distinction » au lieu de « sur ordonnance », dans un seul run).
- **Limites mesurées** : le juge varie sur les cas limites (Q063 jugée tantôt acceptable, tantôt
  non) ; le générateur varie malgré la température 0 (Q056, Q045) ; le juge laisse passer un
  « uniquement » abusif.
- **Régression découverte et corrigée** : depuis D013 (documents entiers, plus de texte
  français), Q082 « How do I leave the panel? » recevait une réponse en français ; aucune
  métrique ne le voyait. Correction : règle 5 du prompt (« même si les extraits sont en
  français ») ; ajout d'un contrôle déterministe `language_ok.anglais` (1,000 après correction).
- **Coût mesuré** : ~0,09 $ par évaluation complète (génération + juge).

---

## D017 - CI d'évaluation : deux workflows et seuils bloquants calculés sur le dev (2026-10-01)

- **Contexte** : bloquer les régressions de qualité, y compris silencieuses (exemple de la langue).
- **Choix** :
  - `tests.yml` (chaque push, gratuit) : validation des données + tests unitaires (faux LLM).
  - `eval.yml` (pull request et à la demande, ~0,09 $) : index reconstruit, `eval_e2e.py --split
    dev --check`, résultats MLflow en artefact ; clé OpenRouter en secret GitHub.
  - Seuils `CI_THRESHOLDS` dans `config.py` : valeur mesurée sur le dev moins une marge liée à la
    taille du groupe (une bascule déplace la métrique de $1/n$) ; aucun écart toléré sur les refus
    (sécurité) ni sur la langue (régression déjà survenue).
  - Reproductibilité : `uv sync --locked`, Python 3.12 épinglé, torch CPU sous Linux (supprime
    CUDA, `nvidia-*` et `triton` de l'installation), cache du modèle d'embeddings.
- **Raison** : le test reste hors de toute boucle de réglage (D003) ; l'évaluation est trop
  coûteuse pour chaque push ; les marges absorbent le bruit mesuré du générateur et du juge.
- **Résultat** : la commande de la CI passe en local (run `e2e_dev_ci_check`, code de sortie 0) ;
  la logique des seuils est couverte par `tests/test_eval_e2e.py` (53 tests au total). Les
  workflows n'ont pas encore tourné sur GitHub (dépôt distant à créer).
- **Limites** : faux positifs possibles (variabilité), dépendance à l'API (quota, panne), pas de
  secrets pour les pull requests venant d'un fork, seuils à remonter à chaque amélioration.

---

## D018 - Évaluation finale sur le jeu de test : résultats et analyse (2026-10-01)

- **Contexte** : fin de la phase 6 ; le jeu de test (19 questions, D002) est évalué **une seule
  fois** (run `e2e_test_final`), avec le système figé. Il ne sert plus ensuite à régler quoi que ce soit.
- **Résultats, comparés au dev** (intervalles de confiance de Wilson à 95 %) :

  | Métrique | Dev (67 questions) | Test (19 questions) |
  |---|---|---|
  | Recall@5 de la recherche | 0,922 | **0,765** |
  | MRR | 0,862 | **0,649** |
  | Routage (actions correctes) | 0,955 [0,88 ; 0,98] | **0,789** [0,57 ; 0,91] |
  | Fidélité (juge) | 0,985 [0,92 ; 1,00] | **0,842** [0,62 ; 0,94] |
  | Correction, questions « repondre » | 0,940 [0,84 ; 0,98] | **0,692** [0,42 ; 0,87] |
  | Comportements (repondre, sans inventer, refuser, escalader) | 0,667 à 1,000 | 1,000 |
  | Citations inventées | 0 | 0 |

- **Analyse des erreurs (question par question)** :
  - **Les 3 réponses infidèles du test sont 3 échecs de la recherche** : Q034 « points en attente »
    (REC-03 « Délais de crédit » non retrouvé), Q039 « supprimer toutes mes données » (RGPD-02
    « Exercer vos droits » non retrouvé), Q086 « wesh jpeux scanner le pain » (APP-04 non
    retrouvé). Sans le bon document, le générateur comble le vide malgré le prompt : même
    mécanisme que Q009 (D012). Causes : écart de vocabulaire et langage familier.
  - **Langage familier jamais mesuré pendant le réglage** : les 2 seules questions « familier »
    du jeu sont tombées dans le test (strate « non standard » de D002) ; le dev n'en contenait aucune.
  - **Le routage v2 ne généralise pas** : 3 escalades en trop réapparaissent (Q026, Q034, Q039) ;
    la règle ajoutée en v2 corrigeait surtout les erreurs du dev (sur-apprentissage annoncé en D015).
    Q077, l'une des 4 questions de test vues par erreur pendant la conception (D015), n'est pas
    escaladée : pas d'avantage visible lié à cette fuite.
- **Lecture honnête** : les intervalles se chevauchent, donc l'écart n'est pas statistiquement
  établi avec 19 questions ; mais il va dans le même sens sur toutes les métriques et s'explique
  par des causes identifiées. **Les chiffres du dev étaient optimistes ; ceux du test sont
  l'estimation à présenter, avec leur forte incertitude.**
- **Conséquences** :
  - Le jeu de test est désormais « consommé » : toute amélioration future devra être mesurée
    sur un nouveau jeu de test (ex. lors de l'extension à 150-200 questions, README du dataset).
  - Pistes prioritaires, dans l'ordre de l'impact attendu : (1) la recherche face aux écarts de
    vocabulaire et au langage familier (reranking, reformulation de la question avant
    recherche) ; (2) enrichir le dev en questions familières ; (3) revoir la règle anti-escalade du routeur.

---

## D019 - Mise en service : API FastAPI, interface Streamlit, image Docker (2026-10-02)

- **Contexte** : phase 7 ; exposer le pipeline et le rendre exécutable à l'identique partout.
- **Choix** :
  - API FastAPI (`src/rag/api.py`) : `POST /ask` (entrée validée par pydantic, 1 à 1 000
    caractères), `GET /health` ; modèle, index et LLM chargés une fois au démarrage (lifespan) ;
    point de terminaison synchrone (pipeline bloquant, exécuté dans un pool de threads) ; panne
    du LLM → 503 sans détail ; seules les citations de documents fournis au LLM sont renvoyées
    comme sources ; journal sans la question (données personnelles possibles).
  - Interface Streamlit (`app/`) qui **appelle l'API par HTTP** plutôt que d'importer le pipeline
    (architecture réaliste, parties testables séparément).
  - Une seule image Docker pour les deux services (`python:3.12-slim`, uv, `--locked`, torch CPU,
    utilisateur non-root, modèle et index construits au build, `HF_HUB_OFFLINE=1`, aucun
    secret dans l'image) ; `docker-compose.yml` avec healthcheck.
- **Résultats mesurés** :
  - 63 tests (dont API avec faux pipeline, client HTTP simulé, rendu Streamlit via AppTest).
  - API locale : démarrage ~42 s, première réponse 2,7 s.
  - Image : 6,97 Go → **4,08 Go** après deux corrections (cache uv monté au lieu d'être copié :
    ~2 Go ; MLflow déplacé dans les dépendances de développement).
  - Docker Compose : API saine ~43 s après démarrage ; escalade en 2,9 s, refus hors sujet en 0,9 s.
- **Écueils rencontrés** : ports 8000 et 8501 occupés par le relais WSL sur le poste (exposition
  sur 8010 et 8510) ; un test `curl` sous Windows envoyait un JSON mal encodé (accents), d'où une
  erreur 400 de l'API, à juste titre.
- **Limites** : pas d'authentification ni de limitation de débit (indispensable avant une
  mise en ligne publique) ; un seul processus ; pas de mémoire de conversation ; corpus figé
  dans l'image.
