# Corpus et jeu d'évaluation - Assistant RAG support panélistes

Données **entièrement fictives** (entreprise « PanelConso »). Aucune donnée interne réelle.

## Structure
```
corpus/            27 documents Markdown avec frontmatter (id, titre, categorie, version, date_maj)
  participation/   règles du panel
  technique/       appli, scan, synchro, codes d'erreur, ScanBox
  recompenses/     points, bonus, échanges, parrainage
  compte/          connexion, foyer, absences, désinscription
  rgpd/            données, droits, conservation
  support/         contact, réclamations
eval/questions.jsonl   86 questions annotées
scripts/validate.py    contrôle de cohérence (sources existantes, ids uniques, répartition)
```

## Schéma d'une question
| champ | valeurs |
|---|---|
| type | factuelle, multi_docs, piege_invention, hors_sujet, securite_donnees, escalade |
| style | standard, familier, fautes, anglais |
| categorie | label de routage attendu (dont hors_sujet) |
| comportement | repondre, repondre_sans_inventer, refuser, escalader |
| sources | ids des documents qui doivent être retrouvés |
| reponse_attendue | éléments clés que la réponse doit contenir |

## Prochaines étapes
1. Relire le corpus et corriger ce qui ne sonne pas juste.
2. Monter à 150-200 questions : reformulations des questions existantes (même sources), plus de fautes/familier/anglais, plus de pièges.
3. Mettre de côté ~20 % des questions comme jeu de test final, jamais utilisé pour régler les prompts.
