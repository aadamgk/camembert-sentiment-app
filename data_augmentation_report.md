# Rapport — Dataset augmenté Ynov (binaire)

**Date** : 2026-05-06
**Spec source** : `docs/superpowers/specs/2026-05-06-data-augmentation-ynov-design.md`
**Pivot vs spec initiale** : passage d'une cible 1000/1000/1000 (3 classes) à **197/197 binaire** pour rester dans un budget temps réaliste et éviter de générer une classe `neutre` 100% synthétique sans ancrage réel (le dataset original ne contenait que 2 vrais avis neutres).

## Résumé

| Métrique | Valeur |
|---|---|
| Total | **394 avis** |
| Classes | binaire `positif` / `negatif` |
| Distribution | 197 / 197 (parfaitement équilibré) |
| Vrais avis | 294 (74,6%) |
| Avis synthétiques | 100 (25,4%) |
| Doublons | 0 |

## Composition

| Source | Positif | Négatif | Total |
|---|---|---|---|
| `real` | 197 | 97 | 294 |
| `synthetic` | 0 | 100 | 100 |
| **Total** | **197** | **197** | **394** |

## Distribution des longueurs

| Sentiment | min | médiane | max |
|---|---|---|---|
| positif | 1 mot | 24 mots | 454 mots |
| negatif | 1 mot | 38 mots | 665 mots |

## Méthode

1. **Filtrage des vrais avis** : `avis_ynov_All_final.csv` (550 lignes) → drop des `comment` null + dedup → 381 vrais avis exploitables.
2. **Génération de 100 avis négatifs synthétiques** (batches `synthetic_negatif_01.csv` et `synthetic_negatif_02.csv`) en français avec accents corrects, calibrés sur le style des vrais avis Ynov :
   - Diversité campus : 11/11 campus mentionnés (Paris, Lyon, Bordeaux, Toulouse, Nantes, Lille, Aix, Sophia, Rennes, Strasbourg, Montpellier)
   - Diversité filières : 10 filières (Informatique, 3D, Audiovisuel, Marketing, Création Digitale, Game Design, Architecture, Tech & Data, Cyber, Business)
   - Distribution rating : 70/30 (rating=1 / rating=2)
   - Longueurs variées (1-446 mots)
3. **Combinaison négatifs** : 97 réels + 100 synthétiques = 197 négatifs.
4. **Échantillonnage positifs** : 197 vrais positifs tirés au hasard (random_state=42) parmi les 284 disponibles.
5. **Shuffle final** + écriture de `avis_ynov_augmented.csv`.

## Schéma du fichier

```
author, rating, sentiment_label, date, comment, source
```

- `source` : `"real"` ou `"synthetic"` (utile pour évaluation séparée lors du fine-tuning)
- Encoding : UTF-8 (accents français préservés)

## Limites connues

- **Pas de classe `neutre`** : seulement 2 vrais avis neutres dans le dataset original, pas assez pour ancrer une génération synthétique réaliste. Décision : passer en binaire.
- **Labels dérivés mécaniquement du rating** dans le dataset original (1-2 → négatif, 4-5 → positif). Cohérent mais non re-annoté manuellement.
- **Biais de génération** : 100 négatifs synthétiques produits par un seul générateur (Claude) → style potentiellement détectable. À évaluer en split train/test stratifié par `source`.
- **Taille modeste** : 394 avis suffisent pour un POC de fine-tuning, mais pour la production il faudrait viser 1000+ par classe (idéalement scrapés de vrais avis).

## Prochaines étapes (hors scope)

- Fine-tuning binaire d'un modèle français (CamemBERT recommandé) sur ce dataset.
- Évaluation séparée real/synthetic (split par `source`) pour mesurer le biais de génération.
- Si les résultats sont prometteurs : phase 2 d'augmentation pour atteindre 500/500 ou 1000/1000.
- Intégration dans `app.py` (remplacement du modèle Twitter XLM-RoBERTa actuel).
