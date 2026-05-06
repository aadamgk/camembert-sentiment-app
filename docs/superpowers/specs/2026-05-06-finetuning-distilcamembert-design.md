# Fine-tuning DistilCamemBERT et intégration comparative dans l'app

**Date** : 2026-05-06
**Auteur** : Ahmat (collab. Claude)
**Statut** : Design validé, en attente de relecture utilisateur
**Spec source du dataset** : `docs/superpowers/specs/2026-05-06-data-augmentation-ynov-design.md`

## 1. Contexte et motivation

L'app `camembert-sentiment-app` utilise actuellement `cardiffnlp/twitter-xlm-roberta-base-sentiment` — un modèle générique multilingue Twitter, pas adapté au domaine des avis étudiants Ynov (vocabulaire scolaire, formats Google Maps, longueurs variables).

L'utilisateur dispose maintenant d'un dataset binaire augmenté (`avis_ynov_augmented.csv`, 394 lignes équilibrées 197/197) prêt pour un fine-tuning. Il dispose aussi d'un repo HF existant (`Ahmat293/camembert-sentiment-ynov`) contenant un ancien modèle qu'il veut **conserver** pour pouvoir comparer empiriquement les performances dans l'app.

## 2. Objectif et critères de succès

### Objectif

Fine-tuner DistilCamemBERT sur le dataset augmenté, le pousser sur un nouveau repo HF, et modifier `app.py` pour permettre à l'utilisateur de switcher entre le modèle existant et le nouveau via un sélecteur dans l'UI.

### Critères de succès

1. Notebook Colab `finetune_ynov.ipynb` s'exécute de bout en bout sur un T4 gratuit sans erreur.
2. Modèle final accessible publiquement sur `Ahmat293/distilcamembert-ynov-augmented`.
3. Métriques sur le test set : accuracy ≥ 80%, F1 macro ≥ 0.80.
4. Évaluation séparée sur les sous-ensembles `real` et `synthetic` du test set, pour mesurer le biais de génération.
5. `streamlit run app.py` fonctionne avec un sélecteur de modèle visible et fonctionnel ; les deux modèles produisent des prédictions cohérentes.
6. Le repo `Ahmat293/camembert-sentiment-ynov` n'est pas modifié.

## 3. Scope

### Inclus

- Notebook Colab complet : install, load CSV, split, tokenize, train, evaluate, push HF Hub
- Création du repo HF `Ahmat293/distilcamembert-ynov-augmented`
- Modification ciblée de `app.py` :
  - Sélecteur de modèle (radio button) en haut de la page d'analyse
  - Chargement paresseux des deux modèles via `@st.cache_resource`
  - Adaptation du label_map pour gérer les deux schémas de sortie

### Exclus (futurs travaux)

- Hyperparameter tuning poussé (sweep, optuna)
- Quantization, ONNX export, optimisation d'inférence
- Évaluation cross-domain (avis d'autres écoles)
- Modification du modèle déjà présent sur `Ahmat293/camembert-sentiment-ynov`

## 4. Architecture

### Pipeline d'entraînement (notebook Colab)

```
┌─────────────────────────────────────────────────────────┐
│ Cell 1 — Install                                        │
│   pip install transformers datasets evaluate accelerate │
│           huggingface_hub scikit-learn                  │
├─────────────────────────────────────────────────────────┤
│ Cell 2 — Auth HF                                        │
│   notebook_login() → token HF en mémoire                │
├─────────────────────────────────────────────────────────┤
│ Cell 3 — Upload CSV                                     │
│   files.upload() → avis_ynov_augmented.csv              │
├─────────────────────────────────────────────────────────┤
│ Cell 4 — Load + Split (70/15/15 stratifié)              │
│   pandas → train_test_split stratifié sur               │
│   sentiment_label × source                              │
├─────────────────────────────────────────────────────────┤
│ Cell 5 — Tokenize                                       │
│   AutoTokenizer.from_pretrained(                        │
│       'cmarkea/distilcamembert-base'),                  │
│   max_length=512, truncation=True                       │
├─────────────────────────────────────────────────────────┤
│ Cell 6 — Model + Trainer                                │
│   AutoModelForSequenceClassification                    │
│   num_labels=2, id2label, label2id                      │
│   TrainingArguments + Trainer                           │
├─────────────────────────────────────────────────────────┤
│ Cell 7 — Train (5 epochs)                               │
│   trainer.train()                                       │
├─────────────────────────────────────────────────────────┤
│ Cell 8 — Evaluate                                       │
│   - Test set global : accuracy, F1, confusion matrix    │
│   - Subset real : mêmes métriques                       │
│   - Subset synthetic : mêmes métriques                  │
├─────────────────────────────────────────────────────────┤
│ Cell 9 — Push HF Hub                                    │
│   trainer.push_to_hub(                                  │
│     'Ahmat293/distilcamembert-ynov-augmented')          │
│   tokenizer.push_to_hub(...)                            │
└─────────────────────────────────────────────────────────┘
```

### Architecture app modifiée

```
┌─────────────────────────────────────────────────────────┐
│ Header                                                  │
│   Ynov Sentiment Analyser                               │
│   Subtitle : Powered by [model name selected]           │
├─────────────────────────────────────────────────────────┤
│ Section dataset (inchangée)                             │
├─────────────────────────────────────────────────────────┤
│ Section "Analyser un commentaire"                       │
│                                                         │
│   ┌─ NEW : Sélecteur de modèle ──────────────┐          │
│   │ ⦿ CamemBERT original (Ahmat293/camembert-│          │
│   │   sentiment-ynov)                         │          │
│   │ ⦾ DistilCamemBERT augmenté (Ahmat293/    │          │
│   │   distilcamembert-ynov-augmented)         │          │
│   └───────────────────────────────────────────┘          │
│                                                         │
│   Textarea + bouton (inchangés)                         │
│   Résultat affiché avec label correspondant au modèle   │
│   sélectionné                                           │
├─────────────────────────────────────────────────────────┤
│ Section historique (inchangée, mais ajoute la colonne   │
│ "modèle" dans chaque entrée)                            │
└─────────────────────────────────────────────────────────┘
```

## 5. Spécifications techniques

### Hyperparamètres d'entraînement

| Paramètre | Valeur |
|---|---|
| Model base | `cmarkea/distilcamembert-base` |
| Num labels | 2 |
| id2label | `{0: "negatif", 1: "positif"}` |
| label2id | `{"negatif": 0, "positif": 1}` |
| Max length | 512 |
| Batch size train | 16 |
| Batch size eval | 32 |
| Learning rate | 2e-5 |
| Num epochs | 5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Eval strategy | "epoch" |
| Save strategy | "epoch" |
| Load best at end | True |
| Metric for best model | "f1_macro" |
| Greater is better | True |

### Split du dataset

- 70% train / 15% validation / 15% test
- `train_test_split` deux fois (sklearn), stratifié sur la concaténation `sentiment_label + "_" + source`
- `random_state=42` pour reproductibilité
- À 394 lignes : ~276 train / ~58 val / ~60 test

### Métriques

- **Accuracy** (sklearn)
- **F1 macro** (moyenne non-pondérée des F1 par classe)
- **F1 par classe** (positif et négatif)
- **Matrice de confusion** (matplotlib + seaborn dans le notebook)
- **Évaluation croisée real vs synthetic** sur le test set : si la perf chute fortement sur `synthetic`, c'est que le modèle a overfit sur les vrais avis ; si elle chute sur `real`, c'est que le synthetic a tiré le modèle dans une mauvaise direction.

### Repo HF cible

- Nom : `Ahmat293/distilcamembert-ynov-augmented`
- Visibilité : **public** (à confirmer en cours d'exécution)
- Modèle pushé : poids fine-tunés + tokenizer + `config.json` avec `id2label`/`label2id`
- README généré automatiquement par `Trainer.push_to_hub()`

## 6. Modifications de `app.py`

### Constantes (en haut du fichier)

```python
MODELS = {
    "CamemBERT (original)": {
        "id": "Ahmat293/camembert-sentiment-ynov",
        # label_map dérivé en chargeant le modèle : la première étape de
        # l'implémentation sera d'inspecter `model.config.id2label` du repo
        # existant et de mapper ces labels vers "positif"/"négatif"/"neutre".
        # Si le modèle est multi-classes (3 sorties), on garde le mapping
        # complet ; si binaire, on map vers positif/négatif.
        "label_map": None,  # rempli au load
    },
    "DistilCamemBERT (augmenté)": {
        "id": "Ahmat293/distilcamembert-ynov-augmented",
        # Notre modèle à entraîner expose id2label = {0: "negatif", 1: "positif"}
        # via la config HF, donc le pipeline retourne "negatif" / "positif"
        # directement. Le label_map ici sert juste à mettre l'accent (négatif).
        "label_map": {"negatif": "négatif", "positif": "positif"},
    },
}
```

**Note d'implémentation** : le repo existant contient un modèle CamemBERT (probablement) + des fichiers sklearn (.pkl). Au chargement, on utilise uniquement le modèle transformer (sous-dossier `model/` du repo) via `pipeline(model="Ahmat293/camembert-sentiment-ynov")`. Si HF ne trouve pas la structure attendue à la racine du repo, on bascule sur `pipeline(model="Ahmat293/camembert-sentiment-ynov", subfolder="model")`. Cas à traiter dans le plan d'implémentation.

### Cache du chargement (modifié)

```python
@st.cache_resource
def load_model(model_id):
    return pipeline("sentiment-analysis", model=model_id)
```

(`model_id` devient un argument, le décorateur cache par valeur d'argument → les deux modèles sont chargés à la demande puis gardés en mémoire.)

### UI

Ajouter un `st.radio` dans la section "Analyser un commentaire" :

```python
selected_model_name = st.radio(
    "Modèle",
    options=list(MODELS.keys()),
    horizontal=True,
)
```

### Logique de prédiction

```python
config = MODELS[selected_model_name]
classifier = load_model(config["id"])
sentiment, confidence = predict(comment, classifier, config["label_map"])
```

### Historique

Ajouter une clé `"model"` dans chaque entrée pour distinguer les prédictions par modèle utilisé. Affichage : badge avec le nom du modèle.

## 7. Gestion des erreurs

- **Notebook** : si le push HF échoue (token invalide, quota), afficher un message clair et conserver le modèle en local pour permettre un retry manuel.
- **App** : si le chargement d'un modèle échoue (réseau, modèle non publié, quota Hub), afficher un message d'erreur sans crasher l'app, et désactiver le sélecteur pour ce modèle.

## 8. Validation finale

1. Le notebook Colab tourne complètement sur T4 gratuit sans OOM ni erreur.
2. `https://huggingface.co/Ahmat293/distilcamembert-ynov-augmented` est accessible publiquement.
3. Métriques test : accuracy ≥ 80%, F1 macro ≥ 0.80, confusion matrix montre une bonne diagonale.
4. `streamlit run app.py` s'ouvre sans erreur, les deux modèles chargent au premier clic.
5. Tester 3 commentaires (1 nettement positif, 1 nettement négatif, 1 ambigu) avec les deux modèles → résultats cohérents et différenciés (le nouveau modèle doit mieux performer sur le ton "avis Ynov").

## 9. Livrables

- `finetune_ynov.ipynb` à la racine du projet (committé)
- Modifications de `app.py` (committé)
- Repo HF `Ahmat293/distilcamembert-ynov-augmented` populé
- Pas de modification de `Ahmat293/camembert-sentiment-ynov`

## 10. Décisions verrouillées

- **Plateforme** : Colab gratuit (T4) — l'utilisateur n'a pas de GPU local
- **Modèle de base** : DistilCamemBERT (`cmarkea/distilcamembert-base`)
- **Repo HF cible** : nouveau repo `Ahmat293/distilcamembert-ynov-augmented`, repo existant intact
- **Mode UI** : sélecteur (option A — radio button), pas de side-by-side
- **Split** : 70/15/15 stratifié sentiment × source
- **Hyperparamètres** : 5 epochs, lr=2e-5, batch=16, weight_decay=0.01, warmup=0.1
- **Visibilité repo** : public (à confirmer au moment du push)

## 11. Hors scope explicite

- Pas de fine-tuning du modèle original `camembert-sentiment-ynov`
- Pas de réentraînement multi-classes (on reste binaire positif/négatif)
- Pas d'optimisation production (quantization, ONNX, batching)
- Pas de modification de l'UI au-delà du sélecteur (le reste du design Streamlit reste identique)
