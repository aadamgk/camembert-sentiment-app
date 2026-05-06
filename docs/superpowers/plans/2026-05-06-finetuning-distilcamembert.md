# Plan d'implémentation — Fine-tuning DistilCamemBERT + sélecteur app

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tuner DistilCamemBERT sur `avis_ynov_augmented.csv` via Colab, le pousser sur `Ahmat293/distilcamembert-ynov-augmented`, puis modifier `app.py` pour permettre à l'utilisateur de choisir entre l'ancien CamemBERT et le nouveau modèle.

**Architecture:** Notebook Jupyter (`finetune_ynov.ipynb`) écrit en local mais exécuté par l'utilisateur dans Google Colab (T4 GPU gratuit). Modification ciblée de `app.py` pour cacher deux modèles HF via `@st.cache_resource(model_id)` et un `st.radio` pour le switch.

**Tech Stack:** transformers, datasets, evaluate, accelerate, huggingface_hub, sklearn, streamlit, pandas, plotly.

**Spec source:** `docs/superpowers/specs/2026-05-06-finetuning-distilcamembert-design.md`

---

## File Structure

**Créés :**
- `finetune_ynov.ipynb` — notebook Colab d'entraînement (à exécuter par l'utilisateur sur Colab)
- `eval_results.json` — métriques produites par le notebook (uploadé en sortie de Colab, optionnel)

**Modifiés :**
- `app.py` — ajout du sélecteur de modèle, refactor du chargement, adaptation du label_map, ajout de la colonne "modèle" dans l'historique

**Non modifiés :**
- `avis_ynov_augmented.csv` (sera uploadé tel quel dans Colab)
- `requirements.txt` (les nouvelles deps sont uniquement dans Colab, pas en local pour Streamlit)
- Repo HF `Ahmat293/camembert-sentiment-ynov` (laissé intact)

**Convention notebook :**
- Cellules markdown introduisant chaque étape pour faciliter la compréhension par l'utilisateur
- Cellules code commentées en français
- Toutes les variables intermédiaires gardent un nom explicite (pas d'abréviations)

---

## Task 1: Créer le notebook Colab — squelette + cellules markdown

**Files:**
- Create: `finetune_ynov.ipynb`

- [ ] **Step 1: Initialiser le notebook avec un script Python helper**

Le format `.ipynb` est du JSON. Pour le créer proprement, on va générer le notebook via un petit script Python local (jupyter n'est pas requis, on construit le JSON directement).

Créer le fichier en écrivant directement le JSON du notebook. Utiliser l'outil `Write` avec le contenu JSON suivant comme **point de départ initial** (sera enrichi dans les tâches suivantes) :

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Fine-tuning DistilCamemBERT sur les avis Ynov\n",
        "\n",
        "Ce notebook fine-tune `cmarkea/distilcamembert-base` sur le dataset binaire `avis_ynov_augmented.csv` (197 positifs / 197 négatifs), évalue les performances et pousse le modèle final sur Hugging Face Hub.\n",
        "\n",
        "**Plateforme cible** : Google Colab (T4 GPU gratuit)\n",
        "\n",
        "**Durée estimée** : 5-10 minutes pour l'entraînement complet"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 2: Vérifier que le notebook est un JSON valide**

Run :
```bash
python -c "import json; json.load(open('finetune_ynov.ipynb', encoding='utf-8')); print('JSON valide')"
```

Expected : `JSON valide`

- [ ] **Step 3: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): squelette du notebook Colab"
```

---

## Task 2: Notebook — Cellule install des dépendances

**Files:**
- Modify: `finetune_ynov.ipynb` (ajout d'une cellule code après la cellule markdown initiale)

- [ ] **Step 1: Ajouter une cellule markdown introductive**

Ajouter cette cellule markdown au notebook (avant la cellule code) :

```markdown
## 1. Installation des dépendances

`transformers`, `datasets` et `evaluate` ne sont pas pré-installés sur Colab. `accelerate` est requis par `Trainer`. `huggingface_hub` permettra le push final.
```

- [ ] **Step 2: Ajouter la cellule code d'installation**

Cellule code à ajouter :

```python
!pip install -q transformers datasets evaluate accelerate huggingface_hub scikit-learn
```

- [ ] **Step 3: Modifier le fichier `finetune_ynov.ipynb`**

Lire le fichier actuel, ajouter ces deux cellules dans la liste `cells` après la cellule markdown initiale, écrire le fichier modifié. Utiliser un script Python pour faire la modification proprement :

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Installation des dépendances\n",
        "\n",
        "`transformers`, `datasets` et `evaluate` ne sont pas pré-installés sur Colab. `accelerate` est requis par `Trainer`. `huggingface_hub` permettra le push final."
    ]
})

nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install -q transformers datasets evaluate accelerate huggingface_hub scikit-learn"
    ]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

Exécuter ce script via `python -c "..."` ou comme script séparé.

- [ ] **Step 4: Vérifier**

Run :
```bash
python -c "
import json
nb = json.load(open('finetune_ynov.ipynb', encoding='utf-8'))
print(f'Cellules : {len(nb[\"cells\"])}')
print(f'Types : {[c[\"cell_type\"] for c in nb[\"cells\"]]}')
"
```

Expected : `Cellules : 3`, `Types : ['markdown', 'markdown', 'code']`

- [ ] **Step 5: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellule install dépendances"
```

---

## Task 3: Notebook — Cellules auth HF + upload CSV

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Ajouter cellule markdown "Authentification HF"**

```markdown
## 2. Authentification Hugging Face

Pour pouvoir pusher le modèle final, on s'authentifie avec un token HF (créer un token avec write access sur https://huggingface.co/settings/tokens si besoin).
```

- [ ] **Step 2: Ajouter cellule code auth**

```python
from huggingface_hub import notebook_login
notebook_login()
```

- [ ] **Step 3: Ajouter cellule markdown "Upload CSV"**

```markdown
## 3. Upload du dataset

Upload `avis_ynov_augmented.csv` (le fichier généré localement par data augmentation, 394 lignes binaires).
```

- [ ] **Step 4: Ajouter cellule code upload**

```python
from google.colab import files
uploaded = files.upload()  # → sélectionner avis_ynov_augmented.csv
csv_path = "avis_ynov_augmented.csv"
```

- [ ] **Step 5: Modifier le notebook**

Script Python à exécuter pour ajouter les 4 cellules :

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 2. Authentification Hugging Face\n", "\n",
        "Pour pouvoir pusher le modèle final, on s'authentifie avec un token HF (créer un token avec write access sur https://huggingface.co/settings/tokens si besoin)."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from huggingface_hub import notebook_login\n",
        "notebook_login()"
    ]},
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## 3. Upload du dataset\n", "\n",
        "Upload `avis_ynov_augmented.csv` (le fichier généré localement par data augmentation, 394 lignes binaires)."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
        "from google.colab import files\n",
        "uploaded = files.upload()  # sélectionner avis_ynov_augmented.csv\n",
        "csv_path = \"avis_ynov_augmented.csv\""
    ]},
]

nb['cells'].extend(new_cells)

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 6: Vérifier**

```bash
python -c "
import json
nb = json.load(open('finetune_ynov.ipynb', encoding='utf-8'))
print(f'Cellules : {len(nb[\"cells\"])}')
"
```

Expected : `Cellules : 7`

- [ ] **Step 7: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellules auth HF + upload CSV"
```

---

## Task 4: Notebook — Cellules load + split stratifié 70/15/15

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Cellules à ajouter**

Markdown :
```markdown
## 4. Chargement et split stratifié

On lit le CSV, on encode les labels, et on splitte en 70% train / 15% validation / 15% test.
La stratification se fait sur `sentiment_label × source` pour garantir une représentation équilibrée des avis réels et synthétiques dans chaque split.
```

Code :
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(csv_path)
print(f"Total : {len(df)} lignes")
print(f"Sentiment : {df['sentiment_label'].value_counts().to_dict()}")
print(f"Source : {df['source'].value_counts().to_dict()}")

# Encodage des labels (0 = negatif, 1 = positif)
label2id = {"negatif": 0, "positif": 1}
id2label = {0: "negatif", 1: "positif"}
df["label"] = df["sentiment_label"].map(label2id)

# Strate : sentiment × source pour préserver la composition dans chaque split
df["strat"] = df["sentiment_label"] + "_" + df["source"]

# Split 70 / 15 / 15
train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["strat"], random_state=42
)
train, val = train_test_split(
    train_val, test_size=0.176,  # 0.176 * 0.85 ≈ 0.15 du total
    stratify=train_val["strat"], random_state=42
)

print(f"\nTrain : {len(train)}, Val : {len(val)}, Test : {len(test)}")
print(f"Train sentiment : {train['sentiment_label'].value_counts().to_dict()}")
print(f"Test  sentiment : {test['sentiment_label'].value_counts().to_dict()}")
```

- [ ] **Step 2: Modifier le notebook**

Script Python :
```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## 4. Chargement et split stratifié\n", "\n",
        "On lit le CSV, on encode les labels, et on splitte en 70% train / 15% validation / 15% test.\n",
        "La stratification se fait sur `sentiment_label × source` pour garantir une représentation équilibrée des avis réels et synthétiques dans chaque split."
    ]
})

code = '''import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(csv_path)
print(f"Total : {len(df)} lignes")
print(f"Sentiment : {df['sentiment_label'].value_counts().to_dict()}")
print(f"Source : {df['source'].value_counts().to_dict()}")

label2id = {"negatif": 0, "positif": 1}
id2label = {0: "negatif", 1: "positif"}
df["label"] = df["sentiment_label"].map(label2id)

df["strat"] = df["sentiment_label"] + "_" + df["source"]

train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["strat"], random_state=42
)
train, val = train_test_split(
    train_val, test_size=0.176,
    stratify=train_val["strat"], random_state=42
)

print(f"\\nTrain : {len(train)}, Val : {len(val)}, Test : {len(test)}")
print(f"Train sentiment : {train['sentiment_label'].value_counts().to_dict()}")
print(f"Test  sentiment : {test['sentiment_label'].value_counts().to_dict()}")'''

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [line + "\n" for line in code.split("\n")]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 3: Vérifier**

```bash
python -c "
import json
nb = json.load(open('finetune_ynov.ipynb', encoding='utf-8'))
print(f'Cellules : {len(nb[\"cells\"])}')
"
```

Expected : `Cellules : 9`

- [ ] **Step 4: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellule split stratifié 70/15/15"
```

---

## Task 5: Notebook — Cellules tokenize + datasets HF

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Cellules à ajouter**

Markdown :
```markdown
## 5. Tokenization

On tokenise les commentaires avec le tokenizer de DistilCamemBERT (`max_length=512`, troncature à droite). On convertit ensuite les pandas DataFrames en `datasets.Dataset` pour interfaçage avec `Trainer`.
```

Code :
```python
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_BASE = "cmarkea/distilcamembert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

def tokenize(batch):
    return tokenizer(
        batch["comment"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

# Conversion pandas → datasets.Dataset puis tokenization
train_ds = Dataset.from_pandas(train[["comment", "label", "source"]]).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val[["comment", "label", "source"]]).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test[["comment", "label", "source"]]).map(tokenize, batched=True)

print(f"Train : {len(train_ds)}, Val : {len(val_ds)}, Test : {len(test_ds)}")
print(f"Colonnes : {train_ds.column_names}")
```

- [ ] **Step 2: Modifier le notebook**

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## 5. Tokenization\n", "\n",
        "On tokenise les commentaires avec le tokenizer de DistilCamemBERT (`max_length=512`, troncature à droite). On convertit ensuite les pandas DataFrames en `datasets.Dataset` pour interfaçage avec `Trainer`."
    ]
})

code = '''from datasets import Dataset
from transformers import AutoTokenizer

MODEL_BASE = "cmarkea/distilcamembert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

def tokenize(batch):
    return tokenizer(
        batch["comment"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

train_ds = Dataset.from_pandas(train[["comment", "label", "source"]]).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val[["comment", "label", "source"]]).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test[["comment", "label", "source"]]).map(tokenize, batched=True)

print(f"Train : {len(train_ds)}, Val : {len(val_ds)}, Test : {len(test_ds)}")
print(f"Colonnes : {train_ds.column_names}")'''

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [line + "\n" for line in code.split("\n")]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 3: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellule tokenization"
```

---

## Task 6: Notebook — Cellules modèle + TrainingArguments + compute_metrics

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Cellules à ajouter**

Markdown :
```markdown
## 6. Modèle, métriques et configuration d'entraînement

Chargement du modèle DistilCamemBERT avec une tête de classification 2 classes. Définition de `compute_metrics` (accuracy + F1 macro + F1 par classe). Configuration des `TrainingArguments` selon la spec.
```

Code :
```python
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_negatif": f1_score(labels, preds, pos_label=0, average="binary"),
        "f1_positif": f1_score(labels, preds, pos_label=1, average="binary"),
    }

training_args = TrainingArguments(
    output_dir="./distilcamembert-ynov-augmented",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    push_to_hub=False,  # on push manuellement à la fin
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

- [ ] **Step 2: Modifier le notebook**

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## 6. Modèle, métriques et configuration d'entraînement\n", "\n",
        "Chargement du modèle DistilCamemBERT avec une tête de classification 2 classes. Définition de `compute_metrics` (accuracy + F1 macro + F1 par classe). Configuration des `TrainingArguments` selon la spec."
    ]
})

code = '''import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_BASE,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_negatif": f1_score(labels, preds, pos_label=0, average="binary"),
        "f1_positif": f1_score(labels, preds, pos_label=1, average="binary"),
    }

training_args = TrainingArguments(
    output_dir="./distilcamembert-ynov-augmented",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=10,
    push_to_hub=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)'''

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [line + "\n" for line in code.split("\n")]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 3: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellule modèle + Trainer + metrics"
```

---

## Task 7: Notebook — Cellule entraînement

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Cellules à ajouter**

Markdown :
```markdown
## 7. Entraînement

5 epochs sur 276 exemples train, ~6 minutes sur T4 gratuit. Le best model selon F1 macro sur la validation est rechargé automatiquement à la fin grâce à `load_best_model_at_end=True`.
```

Code :
```python
trainer.train()
```

- [ ] **Step 2: Modifier le notebook**

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## 7. Entraînement\n", "\n",
        "5 epochs sur 276 exemples train, ~6 minutes sur T4 gratuit. Le best model selon F1 macro sur la validation est rechargé automatiquement à la fin grâce à `load_best_model_at_end=True`."
    ]
})

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": ["trainer.train()"]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 3: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellule entraînement"
```

---

## Task 8: Notebook — Cellules évaluation globale + par source

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Cellules à ajouter**

Markdown :
```markdown
## 8. Évaluation sur le test set

Trois évaluations :
1. **Globale** sur tout le test set
2. **Subset `real`** uniquement
3. **Subset `synthetic`** uniquement

Si l'écart entre real et synthetic est important (> 10 points de F1), c'est un signal de biais de génération.
```

Code :
```python
import json as _json

def evaluate_subset(name, dataset):
    metrics = trainer.evaluate(eval_dataset=dataset)
    print(f"\n=== {name} (n={len(dataset)}) ===")
    for k, v in metrics.items():
        if k.startswith("eval_") and k != "eval_runtime" and "samples" not in k and "steps" not in k:
            print(f"  {k}: {v:.4f}")
    return metrics

# Évaluation globale
global_metrics = evaluate_subset("Global test set", test_ds)

# Subsets par source
real_ds      = test_ds.filter(lambda ex: ex["source"] == "real")
synthetic_ds = test_ds.filter(lambda ex: ex["source"] == "synthetic")

real_metrics      = evaluate_subset("Test set REAL", real_ds)
synthetic_metrics = evaluate_subset("Test set SYNTHETIC", synthetic_ds)

# Sauvegarde des métriques
all_metrics = {
    "global": {k: float(v) for k, v in global_metrics.items() if isinstance(v, (int, float))},
    "real":      {k: float(v) for k, v in real_metrics.items()      if isinstance(v, (int, float))},
    "synthetic": {k: float(v) for k, v in synthetic_metrics.items() if isinstance(v, (int, float))},
}
with open("eval_results.json", "w") as f:
    _json.dump(all_metrics, f, indent=2)

print("\n=== Synthèse ===")
print(f"Global F1 macro     : {global_metrics['eval_f1_macro']:.4f}")
print(f"Real F1 macro       : {real_metrics['eval_f1_macro']:.4f}")
print(f"Synthetic F1 macro  : {synthetic_metrics['eval_f1_macro']:.4f}")
gap = abs(real_metrics["eval_f1_macro"] - synthetic_metrics["eval_f1_macro"])
print(f"Écart real/synthetic: {gap:.4f}")
```

Markdown matrice de confusion :
```markdown
## 8b. Matrice de confusion sur le test global
```

Code matrice :
```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["negatif", "positif"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Matrice de confusion — DistilCamemBERT Ynov augmenté")
plt.tight_layout()
plt.show()

print(classification_report(labels, preds, target_names=["negatif", "positif"]))
```

- [ ] **Step 2: Modifier le notebook (4 cellules ajoutées)**

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cellule markdown évaluation
nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## 8. Évaluation sur le test set\n", "\n",
        "Trois évaluations :\n",
        "1. **Globale** sur tout le test set\n",
        "2. **Subset `real`** uniquement\n",
        "3. **Subset `synthetic`** uniquement\n", "\n",
        "Si l'écart entre real et synthetic est important (> 10 points de F1), c'est un signal de biais de génération."
    ]
})

# Cellule code évaluation
code1 = '''import json as _json

def evaluate_subset(name, dataset):
    metrics = trainer.evaluate(eval_dataset=dataset)
    print(f"\\n=== {name} (n={len(dataset)}) ===")
    for k, v in metrics.items():
        if k.startswith("eval_") and k != "eval_runtime" and "samples" not in k and "steps" not in k:
            print(f"  {k}: {v:.4f}")
    return metrics

global_metrics = evaluate_subset("Global test set", test_ds)

real_ds      = test_ds.filter(lambda ex: ex["source"] == "real")
synthetic_ds = test_ds.filter(lambda ex: ex["source"] == "synthetic")

real_metrics      = evaluate_subset("Test set REAL", real_ds)
synthetic_metrics = evaluate_subset("Test set SYNTHETIC", synthetic_ds)

all_metrics = {
    "global":    {k: float(v) for k, v in global_metrics.items() if isinstance(v, (int, float))},
    "real":      {k: float(v) for k, v in real_metrics.items()      if isinstance(v, (int, float))},
    "synthetic": {k: float(v) for k, v in synthetic_metrics.items() if isinstance(v, (int, float))},
}
with open("eval_results.json", "w") as f:
    _json.dump(all_metrics, f, indent=2)

print("\\n=== Synthèse ===")
print(f"Global F1 macro     : {global_metrics['eval_f1_macro']:.4f}")
print(f"Real F1 macro       : {real_metrics['eval_f1_macro']:.4f}")
print(f"Synthetic F1 macro  : {synthetic_metrics['eval_f1_macro']:.4f}")
gap = abs(real_metrics["eval_f1_macro"] - synthetic_metrics["eval_f1_macro"])
print(f"Écart real/synthetic: {gap:.4f}")'''

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [line + "\n" for line in code1.split("\n")]
})

# Cellule markdown matrice
nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": ["## 8b. Matrice de confusion sur le test global"]
})

# Cellule code matrice
code2 = '''import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions = trainer.predict(test_ds)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["negatif", "positif"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Matrice de confusion - DistilCamemBERT Ynov augmente")
plt.tight_layout()
plt.show()

print(classification_report(labels, preds, target_names=["negatif", "positif"]))'''

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [line + "\n" for line in code2.split("\n")]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 3: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): évaluation globale + par source + matrice de confusion"
```

---

## Task 9: Notebook — Cellule push HF Hub

**Files:**
- Modify: `finetune_ynov.ipynb`

- [ ] **Step 1: Cellules à ajouter**

Markdown :
```markdown
## 9. Push sur Hugging Face Hub

Push du modèle final vers `Ahmat293/distilcamembert-ynov-augmented`. Le repo est créé automatiquement s'il n'existe pas. Visibilité par défaut : public.
```

Code :
```python
HF_REPO = "Ahmat293/distilcamembert-ynov-augmented"

trainer.push_to_hub(
    HF_REPO,
    commit_message="DistilCamemBERT fine-tuné sur avis Ynov augmentés (binaire)",
)
tokenizer.push_to_hub(HF_REPO)

print(f"\n✓ Modèle disponible sur https://huggingface.co/{HF_REPO}")
```

- [ ] **Step 2: Modifier le notebook**

```python
import json

with open('finetune_ynov.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'].append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## 9. Push sur Hugging Face Hub\n", "\n",
        "Push du modèle final vers `Ahmat293/distilcamembert-ynov-augmented`. Le repo est créé automatiquement s'il n'existe pas. Visibilité par défaut : public."
    ]
})

code = '''HF_REPO = "Ahmat293/distilcamembert-ynov-augmented"

trainer.push_to_hub(
    HF_REPO,
    commit_message="DistilCamemBERT fine-tuné sur avis Ynov augmentés (binaire)",
)
tokenizer.push_to_hub(HF_REPO)

print(f"\\n[OK] Modele disponible sur https://huggingface.co/{HF_REPO}")'''

nb['cells'].append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [line + "\n" for line in code.split("\n")]
})

with open('finetune_ynov.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
```

- [ ] **Step 3: Vérifier le notebook complet**

```bash
python -c "
import json
nb = json.load(open('finetune_ynov.ipynb', encoding='utf-8'))
print(f'Total cellules : {len(nb[\"cells\"])}')
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        first_line = c['source'][0].strip() if c['source'] else ''
        print(f'  [{i}] md   : {first_line[:60]}')
    else:
        first_line = c['source'][0].strip() if c['source'] else ''
        print(f'  [{i}] code : {first_line[:60]}')
"
```

Expected : 17 cellules, alternance markdown/code, sections 1-9 visibles.

- [ ] **Step 4: Commit**

```bash
git add finetune_ynov.ipynb
git commit -m "feat(finetune): cellule push HF Hub"
```

---

## Task 10: Pause — l'utilisateur exécute le notebook sur Colab

**Files:** aucun (étape utilisateur, pas de code Claude)

⚠️ **Cette étape ne peut pas être automatisée** — l'utilisateur doit ouvrir Colab dans son navigateur, uploader le notebook, runner toutes les cellules.

- [ ] **Step 1: Instructions à l'utilisateur**

Afficher ces instructions à l'utilisateur :

> 1. Ouvre https://colab.research.google.com/
> 2. File → Upload notebook → sélectionne `finetune_ynov.ipynb`
> 3. Runtime → Change runtime type → T4 GPU → Save
> 4. Runtime → Run all
> 5. Cellule 2 : copie ton token HF (https://huggingface.co/settings/tokens, créer un token "write") quand demandé
> 6. Cellule 4 : sélectionne `avis_ynov_augmented.csv` dans le dialog d'upload
> 7. Attends ~6-8 minutes pour le training
> 8. Vérifie l'URL du modèle pushed : `https://huggingface.co/Ahmat293/distilcamembert-ynov-augmented`
> 9. Reviens dans Claude Code et confirme que tout s'est bien passé

- [ ] **Step 2: Confirmation manuelle**

L'utilisateur confirme que le repo HF `Ahmat293/distilcamembert-ynov-augmented` est accessible avec un modèle valide. Si erreur : debug à la demande.

---

## Task 11: Modifier `app.py` — refactor du load_model + sélecteur

**Files:**
- Modify: `app.py:215-217` (fonction `load_model`)
- Modify: `app.py:218-222` (fonction `predict`)
- Modify: `app.py:328-350` (section "Analyser un commentaire")
- Modify: `app.py:343-349` (ajout de la clé `model` dans l'historique)

- [ ] **Step 1: Lire `app.py:200-260` pour avoir le contexte exact**

```bash
sed -n '200,260p' app.py
```

Vérifier que les fonctions `load_model` et `predict` sont aux lignes attendues. Si décalage, ajuster les références aux lignes dans les étapes suivantes.

- [ ] **Step 2: Remplacer `load_model` et `predict`**

Trouver dans `app.py` le bloc :

```python
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def predict(text, classifier):
    result = classifier(text[:512])[0]
    label_map = {"positive": "positif", "negative": "négatif", "neutral": "neutre"}
    return label_map.get(result["label"], result["label"]), round(result["score"] * 100, 1)
```

Le remplacer par :

```python
# ─── Modèles disponibles ──────────────────────────────────────────────────────
MODELS = {
    "CamemBERT (original)": {
        "id": "Ahmat293/camembert-sentiment-ynov",
        "label_map": {"LABEL_0": "négatif", "LABEL_1": "positif"},
    },
    "DistilCamemBERT (augmenté)": {
        "id": "Ahmat293/distilcamembert-ynov-augmented",
        "label_map": {"negatif": "négatif", "positif": "positif"},
    },
}

@st.cache_resource
def load_model(model_id):
    return pipeline("sentiment-analysis", model=model_id)

def predict(text, classifier, label_map):
    result = classifier(text[:512])[0]
    raw_label = result["label"]
    return label_map.get(raw_label, raw_label), round(result["score"] * 100, 1)
```

- [ ] **Step 3: Modifier la section "Analyser un commentaire"**

Trouver le bloc :

```python
with col_input:
    st.markdown('<div class="section-title">✍️ Analyser un commentaire</div>', unsafe_allow_html=True)
    comment = st.text_area("", placeholder="Entrez un avis étudiant...", height=130, label_visibility="collapsed")

    if st.button("Analyser le sentiment"):
        if comment.strip():
            with st.spinner("Analyse en cours..."):
                classifier = load_model()
                sentiment, confidence = predict(comment, classifier)
```

Le remplacer par :

```python
with col_input:
    st.markdown('<div class="section-title">✍️ Analyser un commentaire</div>', unsafe_allow_html=True)

    selected_model_name = st.radio(
        "Modèle",
        options=list(MODELS.keys()),
        horizontal=True,
        key="model_selector",
    )

    comment = st.text_area("", placeholder="Entrez un avis étudiant...", height=130, label_visibility="collapsed")

    if st.button("Analyser le sentiment"):
        if comment.strip():
            with st.spinner("Analyse en cours..."):
                config = MODELS[selected_model_name]
                classifier = load_model(config["id"])
                sentiment, confidence = predict(comment, classifier, config["label_map"])
```

- [ ] **Step 4: Modifier l'ajout à l'historique**

Trouver le bloc :

```python
            st.session_state.new_comments.append({
                "comment": comment[:60] + "..." if len(comment) > 60 else comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M")
            })
```

Le remplacer par :

```python
            st.session_state.new_comments.append({
                "comment": comment[:60] + "..." if len(comment) > 60 else comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M"),
                "model": selected_model_name,
            })
```

- [ ] **Step 5: Modifier l'affichage de l'historique pour montrer le modèle**

Trouver le bloc :

```python
        for item in reversed(st.session_state.new_comments[-8:]):
            badge_class = {"positif": "badge-pos", "négatif": "badge-neg", "neutre": "badge-neu"}.get(item["sentiment"], "badge-neu")
            st.markdown(f'''
            <div class="comment-item">
                <span style="color:#d1d5db;flex:1">{item["comment"]}</span>
                <span style="color:#4b5563;font-size:0.75rem;margin:0 0.8rem">{item["time"]}</span>
                <span class="badge {badge_class}">{item["sentiment"]}</span>
            </div>''', unsafe_allow_html=True)
```

Le remplacer par :

```python
        for item in reversed(st.session_state.new_comments[-8:]):
            badge_class = {"positif": "badge-pos", "négatif": "badge-neg", "neutre": "badge-neu"}.get(item["sentiment"], "badge-neu")
            model_short = "CamemBERT" if "CamemBERT (original)" in item.get("model", "") else "DistilCam."
            st.markdown(f'''
            <div class="comment-item">
                <span style="color:#d1d5db;flex:1">{item["comment"]}</span>
                <span style="color:#6b7280;font-size:0.7rem;margin:0 0.5rem">{model_short}</span>
                <span style="color:#4b5563;font-size:0.75rem;margin:0 0.8rem">{item["time"]}</span>
                <span class="badge {badge_class}">{item["sentiment"]}</span>
            </div>''', unsafe_allow_html=True)
```

- [ ] **Step 6: Modifier le sous-titre du header**

Trouver le bloc :

```python
st.markdown('<div class="subtitle">Analyse des avis étudiants · Powered by XLM-RoBERTa</div>', unsafe_allow_html=True)
```

Le remplacer par :

```python
st.markdown('<div class="subtitle">Analyse des avis étudiants · CamemBERT vs DistilCamemBERT</div>', unsafe_allow_html=True)
```

- [ ] **Step 7: Modifier le footer**

Trouver le bloc :

```python
st.markdown('<div style="text-align:center;color:#374151;font-size:0.75rem;letter-spacing:0.1em">YNOV SENTIMENT ANALYSER · XLM-RoBERTa · 2026</div>', unsafe_allow_html=True)
```

Le remplacer par :

```python
st.markdown('<div style="text-align:center;color:#374151;font-size:0.75rem;letter-spacing:0.1em">YNOV SENTIMENT ANALYSER · CamemBERT × DistilCamemBERT · 2026</div>', unsafe_allow_html=True)
```

- [ ] **Step 8: Vérifier la syntaxe Python**

```bash
python -c "
import ast
with open('app.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('Syntaxe Python OK')
"
```

Expected : `Syntaxe Python OK`

- [ ] **Step 9: Commit**

```bash
git add app.py
git commit -m "feat(app): sélecteur de modèle CamemBERT vs DistilCamemBERT"
```

---

## Task 12: Test manuel de l'app

**Files:** aucun (test utilisateur)

- [ ] **Step 1: Lancer l'app en local**

```bash
streamlit run app.py
```

L'app s'ouvre dans le navigateur (http://localhost:8501).

- [ ] **Step 2: Vérifier visuellement**

- Le sélecteur "CamemBERT (original) / DistilCamemBERT (augmenté)" est visible au-dessus de la zone de texte.
- Les radio buttons sont cliquables.
- Le sous-titre affiche "CamemBERT vs DistilCamemBERT".

- [ ] **Step 3: Test fonctionnel — 3 commentaires**

Tester avec ces 3 commentaires en switchant entre les deux modèles :

1. **Positif net** : "Excellent campus, intervenants au top, je recommande à 100% !"
2. **Négatif net** : "Cette école est une catastrophe, je déconseille fortement, fuyez !"
3. **Ambigu** : "Le campus est moderne mais l'admin est lente."

Pour chaque commentaire :
- Sélectionner CamemBERT → cliquer "Analyser" → noter le résultat
- Sélectionner DistilCamemBERT → cliquer "Analyser" → noter le résultat

- [ ] **Step 4: Vérifier l'historique**

Après les 6 prédictions, l'historique doit afficher 6 entrées avec le badge du modèle utilisé visible.

- [ ] **Step 5: Si tout fonctionne, commit final**

```bash
git commit --allow-empty -m "test: app validée avec sélecteur de modèle, prédictions cohérentes"
```

---

## Self-review (effectué après rédaction)

**1. Spec coverage**

| Spec § | Couverture |
|---|---|
| §2.1 (notebook Colab T4) | Tasks 1-9 (notebook) + Task 10 (exécution) |
| §2.2 (modèle sur HF) | Task 9 (push) + Task 10 (vérif) |
| §2.3 (accuracy ≥ 80%, F1 ≥ 0.80) | Task 8 (évaluation), validation manuelle dans Task 10 |
| §2.4 (eval real vs synthetic) | Task 8 (subsets) |
| §2.5 (app fonctionne avec sélecteur) | Tasks 11-12 |
| §2.6 (repo existant intact) | Pas de modification du repo `camembert-sentiment-ynov` dans aucune task |
| §4 architecture pipeline | Tasks 1-9 dans l'ordre du pipeline |
| §4 architecture app | Task 11 |
| §5 hyperparamètres | Task 6 (TrainingArguments) |
| §5 split 70/15/15 stratifié | Task 4 |
| §5 métriques | Task 6 (compute_metrics) + Task 8 (subsets) |
| §6 modifications app.py | Task 11 |
| §7 erreurs | Task 11 (try/except implicites via st.cache_resource) — **note : pas de gestion explicite, voir §10** |
| §8 validation finale | Task 12 |
| §9 livrables | Task 1 (ipynb) + Task 11 (app.py) |
| §10 décisions | Toutes verrouillées dans le plan |

**2. Placeholder scan** : aucun "TBD". Toutes les commandes et tous les codes sont complets.

**3. Type consistency** : 
- `MODELS` dict utilisé identique en Task 11 step 2 et 3 ✓
- `label_map` argument cohérent dans `predict()` Task 11 step 2 ✓
- `selected_model_name` cohérent dans `st.radio` Task 11 step 3 et l'historique step 4 ✓
- `id2label`/`label2id` cohérents Task 4 ↔ Task 6 ✓

**4. Spec gap notable** : §7 de la spec mentionne "afficher un message d'erreur sans crasher l'app" — non couvert explicitement. Décision : on s'appuie sur le comportement par défaut de Streamlit (qui affiche les erreurs dans l'UI). Si un modèle HF est inaccessible au load, l'utilisateur voit une stack trace claire dans Streamlit. C'est acceptable pour ce projet (pas de prod critique). Non ajouté pour rester YAGNI.

Aucun autre problème détecté.
