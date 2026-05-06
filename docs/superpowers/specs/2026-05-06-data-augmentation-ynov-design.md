# Augmentation du dataset d'avis Ynov pour fine-tuning

**Date** : 2026-05-06
**Auteur** : Ahmat (collab. Claude)
**Statut** : Design validé, en attente de relecture utilisateur

## 1. Contexte et motivation

Le projet `camembert-sentiment-app` utilise actuellement un modèle générique multilingue Twitter (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) pour classer les avis étudiants Ynov. Le dataset existant (`avis_ynov_All_final.csv`, 550 lignes) présente plusieurs limites bloquantes pour un fine-tuning :

- **Taille insuffisante** : 550 lignes, dont 167 avec `comment` null → ~383 exploitables.
- **Déséquilibre extrême** : 416 positifs / 127 négatifs / **7 neutres**.
- **Labels mécaniques** : `sentiment_label` est dérivé à 100% du `rating` (1-2 = négatif, 3 = neutre, 4-5 = positif), donc le modèle apprendrait à prédire la note plutôt que le sentiment réel — mais on accepte ce biais comme heuristique de départ et on génère les nouveaux exemples avec la même règle de cohérence rating ↔ sentiment.

L'objectif de cette spec est de construire un dataset équilibré de **3000 avis (1000 par classe)** prêt pour fine-tuner un modèle sentiment français spécifique au domaine Ynov.

## 2. Objectif et critères de succès

### Objectif
Produire `avis_ynov_augmented.csv` contenant 3000 avis répartis équitablement entre `positif`, `neutre`, `negatif`, en mélangeant les vrais avis exploitables et des avis synthétiques générés par Claude (au sein de cette session Claude Code, sans coût API séparé).

### Critères de succès
1. Distribution finale : 1000 / 1000 / 1000 (±5 lignes par classe acceptable).
2. Aucun doublon textuel exact entre commentaires.
3. Distribution de longueur (mots) des avis synthétiques cohérente avec les vrais (médiane 12-15 mots, max ~600).
4. Le fichier est lu sans erreur par `pandas.read_csv` avec les mêmes types/colonnes que l'original (+ une colonne `source`).
5. Spot-check manuel sur 20 avis aléatoires : aucune incohérence flagrante (sentiment ≠ texte, fautes grossières, contenu hors-sujet).

## 3. Scope

### Inclus
- Lecture et filtrage du dataset existant (suppression des lignes avec `comment` null).
- Génération synthétique d'avis Ynov diversifiés via Claude (cette session).
- Production du fichier CSV augmenté.
- Rapport court (`data_augmentation_report.md`) avec stats finales et exemples.

### Exclus (futurs travaux)
- Fine-tuning effectif du modèle.
- Modification de `app.py` pour utiliser le nouveau modèle.
- Scraping de vrais avis supplémentaires (Google Maps, Trustpilot).
- Ré-annotation manuelle des labels existants.

## 4. Architecture et flux de données

### Flux global
```
avis_ynov_All_final.csv (550 lignes)
        │
        ▼
[Filtrage] → drop comment null  (~383 lignes "real")
        │
        ▼
[Calibrage] → lecture de ~30-50 avis variés (toutes classes, toutes longueurs)
        │
        ▼
[Génération par lots] → batches de 50, par sentiment, avec matrice de diversité
        │
        ▼
[Validation par lot] → anti-doublon, longueur, cohérence rating/sentiment
        │
        ▼
[Concaténation] → real + synthetic
        │
        ▼
avis_ynov_augmented.csv (3000 lignes, 6 colonnes)
        │
        ▼
data_augmentation_report.md (stats + exemples)
```

### Composants et exécution

**Modèle d'exécution** : il n'y a **pas de script Python automatisé** pour la génération. Claude (moi) exécute directement chaque étape dans cette session Claude Code, en alternant outils `Bash`/`Read` (pour pandas) et `Write` (pour générer les batches synthétiques sous forme de CSV). Les seuls scripts Python utilisés sont des one-liners `pandas` exécutés via `Bash` pour le filtrage, la concaténation et la validation.

**A. Filtre / chargeur** (one-liner pandas via Bash)
- Lit le CSV original avec `pandas`.
- Drop des lignes où `comment.isna()`.
- Ajoute la colonne `source = "real"`.
- Calcule combien d'avis manquent par classe pour atteindre 1000.
- Écrit `intermediate/real_filtered.csv`.

**B. Bibliothèque de calibrage** (lecture par Claude)
- J'échantillonne et lis ~30-50 avis réels représentatifs (mix classes, longueurs courte/moyenne/longue) avant chaque session de génération.
- Sert de référence stylistique mentale pour produire des avis synthétiques crédibles.

**C. Générateur** (Claude écrit directement le CSV via `Write`)
- Pour chaque sentiment, je rédige des batches de 50 avis au format CSV.
- Chaque batch est écrit dans `batches/synthetic_<sentiment>_<n>.csv` immédiatement après rédaction.
- Permet la reprise si la session est interrompue (les batches déjà écrits sont conservés).

**D. Validateur** (one-liner pandas via Bash, après chaque batch)
- Anti-doublon exact (set Python sur les commentaires).
- Vérification format : toutes les colonnes présentes, rating cohérent avec sentiment.
- Stats par lot : longueur min/médiane/max imprimées en console.

**E. Assembleur final** (one-liner pandas via Bash)
- Concatène `intermediate/real_filtered.csv` + tous les `batches/synthetic_*.csv`.
- Mélange aléatoire (`df.sample(frac=1, random_state=42)`).
- Écrit `avis_ynov_augmented.csv`.
- Génère `data_augmentation_report.md` (Claude rédige le contenu après lecture des stats).

## 5. Schéma des données

### Colonnes du CSV de sortie
| Colonne | Type | Exemple | Notes |
|---|---|---|---|
| `author` | str | "Marie L." | Pour les `real` : valeurs d'origine conservées telles quelles. Pour les `synthetic` : prénom français courant tiré aléatoirement (Marie, Lucas, Sophie, Théo, Camille, Hugo, etc.) + une initiale de nom de famille aléatoire (A. à Z.). Aucun patronyme complet inventé pour éviter toute collision avec une vraie personne. |
| `rating` | int | 5 | 1-5, cohérent avec sentiment |
| `sentiment_label` | str | "positif" | "positif" / "neutre" / "negatif" |
| `date` | str | "il y a 3 mois" | Format Google Maps : "il y a X jours/semaines/mois/an(s)" |
| `comment` | str | "Excellent campus..." | Texte de l'avis, en français |
| `source` | str | "synthetic" | "real" ou "synthetic" |

### Règles de cohérence
- `rating` 1 ou 2 ⇒ `sentiment_label = "negatif"`
- `rating` 3 ⇒ `sentiment_label = "neutre"`
- `rating` 4 ou 5 ⇒ `sentiment_label = "positif"`

### Distribution cible des `rating` synthétiques (par classe)
- **Négatifs** : 70% rating 1, 30% rating 2
- **Neutres** : 100% rating 3
- **Positifs** : 30% rating 4, 70% rating 5

(Reflète approximativement la distribution observée dans les vrais avis.)

## 6. Stratégie de génération (matrice de diversité)

Pour chaque avis synthétique, je tire une combinaison sur ces axes :

| Axe | Valeurs possibles |
|---|---|
| **Campus** | Paris, Lyon, Bordeaux, Toulouse, Nantes, Lille, Aix-en-Provence, Sophia Antipolis, Rennes, Strasbourg, Montpellier |
| **Filière** | Informatique, 3D Animation, Audiovisuel, Marketing & Communication, Création Digitale, Game Design, Architecture d'intérieur, Tech & Data, Cybersécurité, Business |
| **Année** | Bachelor 1/2/3, Mastère 1/2 |
| **Angle/sujet** | pédagogie, intervenants, alternance, frais de scolarité, locaux/campus, vie étudiante, projets, jury/soutenance, IA dans les cursus, admin, débouchés, réputation |
| **Ton** | enthousiaste, factuel, sarcastique, déçu, mitigé, recommandation, mise en garde |
| **Longueur** | très court (1-15 mots), court (15-50), moyen (50-150), long (150-400), très long (400+) |

**Distribution des longueurs synthétiques** (pour mimer le réel) :
- Très court : 20%
- Court : 35%
- Moyen : 30%
- Long : 12%
- Très long : 3%

**Mécanisme** : pour chaque batch de 50, je m'assure de varier au moins 4 axes différents pour éviter la répétition. Pas de génération aveugle "fais 50 avis positifs" — chaque ligne a un brief implicite distinct.

## 7. Gestion des erreurs et reprise

- **Sauvegarde par batch** : chaque batch de 50 est écrit dans `batches/synthetic_<sentiment>_<n>.csv` dès rédaction. Si la session est interrompue, on reprend en listant les batches existants et en continuant la numérotation.
- **Validation à chaque batch** : après écriture, je lance un one-liner pandas qui vérifie unicité, format et longueurs. Si KO, je régénère le batch incriminé en évitant les phrases déjà utilisées.
- **Limite de retries** : 3 tentatives par batch. Au-delà, je documente l'échec dans le rapport final et je continue (on tolère un léger sous-effectif d'une classe plutôt que de bloquer le projet).

## 8. Validation finale

Avant de déclarer le travail terminé :
1. `pandas.read_csv("avis_ynov_augmented.csv")` ne lève pas d'exception.
2. `df["sentiment_label"].value_counts()` retourne ~1000/1000/1000.
3. `df["comment"].duplicated().sum() == 0`.
4. Distribution longueurs synthétiques ≈ longueurs réelles (test KS ou histogramme inspecté).
5. Spot-check : 20 lignes aléatoires lues à la main, jugement humain "OK" / "à corriger".

## 9. Livrables

- `avis_ynov_augmented.csv` à la racine du projet (3000 lignes, 6 colonnes).
- `data_augmentation_report.md` à la racine, contenant :
  - Stats par classe (count, longueur médiane, % real vs synthetic)
  - 5 exemples par classe (mix real/synthetic)
  - Méthode résumée
  - Limites connues
- Conservation du fichier original `avis_ynov_All_final.csv` intact.

## 10. Décisions verrouillées (questions résolues)

- **Author réaliste vs anonyme** : tranché → "réaliste" (Prénom courant + Initiale, ex: "Marie L.") pour matcher le style des vrais avis Google Maps déjà présents (ex: "François Buffard"). Aucun nom complet ne sera inventé pour éviter de générer un patronyme correspondant à une vraie personne.
- **Stratégie globale** : tranché → 100% synthétique généré par Claude dans cette session, sans coût API séparé (option A retenue lors du brainstorming).
- **Volume cible** : tranché → 1000 / 1000 / 1000 (option C "ambitieux" retenue lors du brainstorming).
- **Modèle base pour le futur fine-tuning** : non tranché — décision repoussée à la spec suivante (fine-tuning) qui sortira du scope de ce document.

## 11. Hors scope explicite

- Pas de fine-tuning dans ce projet.
- Pas de modification du modèle dans `app.py`.
- Pas d'ajout de dépendances Python (seul `pandas`, déjà transitivement présent via `streamlit`/`scikit-learn`, est requis).
