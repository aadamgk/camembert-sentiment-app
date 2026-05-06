# Plan d'implémentation — Persistance Supabase des prédictions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persister chaque prédiction de l'app dans la table Supabase `predictions`, et afficher les 20 dernières dans une nouvelle section "Historique global".

**Architecture:** Modifications uniquement sur `app.py` (nouvelles fonctions + section UI), `requirements.txt` (ajout dep), `.gitignore` (secrets), `.streamlit/secrets.toml.example` (template). Best-effort : l'app continue de fonctionner si Supabase échoue.

**Tech Stack:** `supabase-py`, Streamlit secrets, RLS Supabase (anon key).

**Spec source:** `docs/superpowers/specs/2026-05-06-supabase-persistence-design.md`

---

## File Structure

**Créés :**
- `.streamlit/secrets.toml.example` — template versionné
- `.streamlit/secrets.toml` — secrets réels (gitignoré, créé manuellement par l'utilisateur)

**Modifiés :**
- `requirements.txt` — ajout `supabase`
- `.gitignore` — ajout `.streamlit/secrets.toml`
- `app.py` — nouvelles fonctions `get_supabase_client`, `save_prediction`, `fetch_global_history` + intégration dans le flux + nouvelle section UI

**Non modifiés :**
- `avis_ynov_augmented.csv`, `data_augmentation_report.md`, notebook
- Tables Supabase existantes (`CommentaireBrut`, `CommentaireClean`)

---

## Task 1: Ajouter la dépendance `supabase`

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Lire le fichier actuel**

```bash
cat requirements.txt
```

Attendu :
```
streamlit
transformers
torch
huggingface_hub
sentencepiece
scikit-learn
plotly
```

- [ ] **Step 2: Ajouter `supabase` à la fin**

Ajouter une ligne `supabase` à `requirements.txt`. Le contenu final doit être :

```
streamlit
transformers
torch
huggingface_hub
sentencepiece
scikit-learn
plotly
supabase
```

- [ ] **Step 3: Installer la dépendance**

```bash
pip install supabase
```

Expected : install réussi sans erreur, version >= 2.0.

- [ ] **Step 4: Vérifier l'import**

```bash
python -c "from supabase import create_client; print('supabase import OK')"
```

Expected : `supabase import OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "feat(deps): ajout de la lib supabase pour persister les prédictions"
```

---

## Task 2: Créer le template de secrets et gitignorer le vrai fichier

**Files:**
- Create: `.streamlit/secrets.toml.example`
- Modify: `.gitignore`

- [ ] **Step 1: Vérifier le contenu actuel du `.gitignore`**

```bash
cat .gitignore
```

Attendu (de la spec data augmentation précédente) :
```
intermediate/
batches/
```

- [ ] **Step 2: Ajouter la ligne `.streamlit/secrets.toml`**

Modifier `.gitignore` pour ajouter à la fin :

```
.streamlit/secrets.toml
```

Le `.gitignore` final doit contenir :
```
intermediate/
batches/
.streamlit/secrets.toml
```

- [ ] **Step 3: Créer le template `secrets.toml.example`**

Créer le fichier `.streamlit/secrets.toml.example` avec ce contenu :

```toml
# Copier ce fichier en .streamlit/secrets.toml et remplir avec vos vraies clés.
# Le fichier secrets.toml est gitignoré, ne le committez jamais.

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key-here"
```

- [ ] **Step 4: Vérifier que `.streamlit/secrets.toml` (s'il existe) est bien gitignoré**

```bash
git check-ignore -v .streamlit/secrets.toml
```

Si le fichier existe : la commande doit afficher `.gitignore:3:.streamlit/secrets.toml`. Si le fichier n'existe pas encore : la commande peut afficher rien ou un message vide, ce n'est pas grave (on créera le fichier juste après).

- [ ] **Step 5: Commit du template + gitignore**

```bash
git add .gitignore .streamlit/secrets.toml.example
git commit -m "feat(secrets): template secrets.toml.example + gitignore"
```

---

## Task 3: Créer le fichier secrets.toml local (action utilisateur)

**Files:**
- Create (manuel): `.streamlit/secrets.toml`

⚠️ Cette task ne peut pas être automatisée — l'utilisateur doit créer manuellement le fichier avec ses vraies clés.

- [ ] **Step 1: Copier le template**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

- [ ] **Step 2: Éditer `.streamlit/secrets.toml` avec les vraies valeurs**

Contenu attendu :
```toml
SUPABASE_URL = "https://edpkrgdlxpdtvrwbkahh.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVkcGtyZ2RseHBkdHZyd2JrYWhoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5NTQxNDQsImV4cCI6MjA4MTUzMDE0NH0.X5xZi-NzaV2EqWgqu-iN7S1_Vv2Bar-3ox3-_xPN1tY"
```

L'anon key est celle qui contient `"role":"anon"` dans son payload JWT (la deuxième fournie par l'utilisateur, pas la `service_role`).

- [ ] **Step 3: Vérifier que le fichier n'apparaît pas dans `git status`**

```bash
git status
```

Expected : `.streamlit/secrets.toml` ne doit PAS apparaître (gitignoré).
Si présent dans untracked : revérifier l'étape 2 de Task 2.

---

## Task 4: Ajouter les imports et la fonction client Supabase dans `app.py`

**Files:**
- Modify: `app.py` (zone d'imports en haut + zone après les imports)

- [ ] **Step 1: Repérer la zone des imports**

```bash
sed -n '1,10p' app.py
```

Attendu :
```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from datetime import datetime
```

- [ ] **Step 2: Ajouter l'import `create_client`**

Modifier le bloc d'imports en remplaçant :

```python
from datetime import datetime
```

par :

```python
from datetime import datetime
from supabase import create_client, Client
```

- [ ] **Step 3: Ajouter la fonction `get_supabase_client` juste après le bloc `MODELS`**

Trouver la fin du bloc `MODELS = {...}` (l'accolade fermante `}` après `"hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",`).

Juste après cette accolade fermante, ajouter ce bloc :

```python

# ─── Client Supabase ──────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase_client():
    """Retourne un client Supabase singleton, ou None si les secrets manquent."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        return create_client(url, key)
    except (KeyError, FileNotFoundError):
        return None

def save_prediction(comment, sentiment, confidence, model_name, model_id):
    """Insère une prédiction dans la table predictions de Supabase. Best-effort."""
    client = get_supabase_client()
    if client is None:
        return
    try:
        client.table("predictions").insert({
            "comment": comment,
            "predicted_sentiment": sentiment,
            "confidence": float(confidence),
            "model_name": model_name,
            "model_id": model_id,
        }).execute()
    except Exception as e:
        st.warning(f"⚠️ Persistance Supabase indisponible : {type(e).__name__}", icon="⚠️")

@st.cache_data(ttl=60)
def fetch_global_history(limit=20):
    """Récupère les N dernières prédictions, triées par date desc. Renvoie None si erreur."""
    client = get_supabase_client()
    if client is None:
        return None
    try:
        result = client.table("predictions") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data
    except Exception:
        return None
```

- [ ] **Step 4: Vérifier la syntaxe**

```bash
python -c "
import ast
with open('app.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('OK')
"
```

Expected : `OK`

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat(app): client Supabase + fonctions save_prediction et fetch_global_history"
```

---

## Task 5: Intégrer `save_prediction` dans le flux de prédiction

**Files:**
- Modify: `app.py` (zone après `st.session_state.new_comments.append(...)`)

- [ ] **Step 1: Repérer la zone**

```bash
grep -n "new_comments.append" app.py
```

- [ ] **Step 2: Modifier le bloc d'append**

Trouver dans `app.py` :

```python
            st.session_state.new_comments.append({
                "comment": comment[:60] + "..." if len(comment) > 60 else comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M"),
                "model": selected_model_name,
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

            # Persistance Supabase (best-effort)
            save_prediction(
                comment=comment,
                sentiment=sentiment,
                confidence=confidence,
                model_name=selected_model_name,
                model_id=config["id"],
            )
            # Invalider le cache de l'historique global pour qu'il se rafraîchisse
            fetch_global_history.clear()
```

- [ ] **Step 3: Vérifier la syntaxe**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): persister chaque prédiction sur Supabase + invalider cache historique"
```

---

## Task 6: Ajouter la section "Historique global" dans l'UI

**Files:**
- Modify: `app.py` (zone juste avant le footer)

- [ ] **Step 1: Repérer la zone du footer**

```bash
grep -n "# ─── Footer" app.py
```

- [ ] **Step 2: Insérer la nouvelle section avant le footer**

Trouver dans `app.py` :

```python
# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('''
<div class="footer-links">
```

Insérer juste avant le commentaire `# ─── Footer` ce nouveau bloc :

```python
# ─── Historique global Supabase ────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_title, col_refresh = st.columns([4, 1])
with col_title:
    st.markdown('<div class="section-title">🌐 Historique global · Supabase</div>', unsafe_allow_html=True)
with col_refresh:
    if st.button("↻ Rafraîchir", key="refresh_global", help="Recharger depuis Supabase"):
        fetch_global_history.clear()
        st.rerun()

global_history = fetch_global_history(limit=20)

if global_history is None:
    st.markdown('<div style="color:#6b7280;text-align:center;padding:1.5rem 0;font-size:0.85rem">⚠️ Connexion Supabase indisponible — l\'historique global ne peut pas être chargé.</div>', unsafe_allow_html=True)
elif len(global_history) == 0:
    st.markdown('<div style="color:#4b5563;text-align:center;padding:1.5rem 0;font-size:0.85rem">Aucune prédiction enregistrée pour le moment.</div>', unsafe_allow_html=True)
else:
    for row in global_history:
        sent_raw = row.get("predicted_sentiment", "")
        badge_class = {"positif": "badge-pos", "négatif": "badge-neg", "neutre": "badge-neu"}.get(sent_raw, "badge-neu")
        model = row.get("model_name", "")
        if "augmenté" in model.lower():
            model_class = "comment-item-augmented"
            model_icon = "✨"
        else:
            model_class = "comment-item-original"
            model_icon = "📜"
        comment_full = row.get("comment", "")
        comment_short = (comment_full[:80] + "...") if len(comment_full) > 80 else comment_full
        # Format date created_at en HH:MM (parsing tolérant)
        try:
            dt = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            time_str = dt.strftime("%d/%m %H:%M")
        except Exception:
            time_str = ""
        confidence = row.get("confidence", 0)
        st.markdown(f'''
        <div class="comment-item {model_class}">
            <span class="history-model-icon">{model_icon}</span>
            <span style="color:#d1d5db;flex:1">{comment_short}</span>
            <span style="color:#4b5563;font-size:0.7rem;margin:0 0.5rem">{confidence}%</span>
            <span style="color:#4b5563;font-size:0.75rem;margin:0 0.8rem">{time_str}</span>
            <span class="badge {badge_class}">{sent_raw}</span>
        </div>''', unsafe_allow_html=True)

```

- [ ] **Step 3: Vérifier la syntaxe**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): section Historique global avec lecture Supabase et bouton refresh"
```

---

## Task 7: Test fonctionnel end-to-end

**Files:** aucun (test utilisateur)

- [ ] **Step 1: Lancer l'app**

```bash
streamlit run app.py
```

- [ ] **Step 2: Tester une prédiction et vérifier la persistance**

Dans l'app :
1. Sélectionner "CamemBERT (augmenté)"
2. Saisir : `Excellent campus, je recommande à 100% !`
3. Cliquer "Analyser le sentiment"
4. Vérifier que la prédiction apparaît dans l'UI sans erreur ni warning

Côté Supabase (https://supabase.com/dashboard/project/edpkrgdlxpdtvrwbkahh/editor) :
1. Ouvrir la table `predictions`
2. Vérifier qu'une nouvelle ligne est apparue avec :
   - `comment` = texte complet (pas tronqué)
   - `predicted_sentiment` = "positif"
   - `confidence` = un nombre entre 0 et 100
   - `model_name` = "CamemBERT (augmenté)"
   - `model_id` = "Ahmat293/camembert-ynov-augmented"
   - `created_at` = timestamp récent

- [ ] **Step 3: Tester l'historique global**

Dans l'app :
1. Scroller en bas, voir la section "🌐 Historique global · Supabase"
2. Vérifier qu'au moins une ligne est affichée (la prédiction qu'on vient de faire)
3. Faire une 2e prédiction avec l'autre modèle
4. Cliquer "↻ Rafraîchir"
5. Vérifier que les 2 prédictions apparaissent maintenant dans l'historique global, avec les icônes ✨ / 📜 distinctes

- [ ] **Step 4: Tester la résilience aux erreurs**

Tester le cas "Supabase down" en renommant temporairement les clés dans `secrets.toml` :
1. Ouvrir `.streamlit/secrets.toml`
2. Mettre une URL invalide : `SUPABASE_URL = "https://invalid.supabase.co"`
3. Recharger l'app (Ctrl+R)
4. Faire une prédiction → l'app doit afficher la prédiction normalement, avec un warning Streamlit en plus
5. La section "Historique global" doit afficher "⚠️ Connexion Supabase indisponible..."
6. Restaurer les bonnes clés et recharger → tout doit refonctionner

- [ ] **Step 5: Vérifier que les secrets ne sont pas committés**

```bash
git status
```

Expected : `.streamlit/secrets.toml` ne doit PAS apparaître dans les modifs.

- [ ] **Step 6: Commit final**

```bash
git commit --allow-empty -m "test: persistance Supabase validée end-to-end"
```

---

## Self-review

**1. Spec coverage**

| Spec § | Couverture |
|---|---|
| §2 Critère 1 (insertion DB) | Task 5 (intégration save_prediction) |
| §2 Critère 2 (historique global 20) | Task 6 (section UI + fetch_global_history) |
| §2 Critère 3 (best-effort) | Task 4 (try/except dans save_prediction et fetch) |
| §2 Critère 4 (secrets gitignorés) | Task 2 (gitignore) + Task 3 (création locale) |
| §2 Critère 5 (anon key only) | Task 3 (template + instructions) |
| §2 Critère 6 (RLS) | Spec §5 (SQL documenté) — l'utilisateur a confirmé table créée |
| §3 Inclus tous éléments | Tasks 1-6 |
| §4 Architecture client/save/fetch | Task 4 |
| §4 Section UI | Task 6 |
| §5 Schéma SQL | Spec §5 (référence, table déjà créée) |
| §6 Secrets, dépendance, cache | Task 1 (dep) + Task 2-3 (secrets) + Task 4 (cache) |
| §7 Gestion d'erreurs | Task 4 (try/except) + Task 6 (UI fallback) |
| §8 Validation finale | Task 7 (test end-to-end) |
| §9 Décisions verrouillées | Toutes appliquées |

**2. Placeholder scan** : aucun "TBD". Toutes les commandes shell et tous les blocs Python sont complets.

**3. Type consistency** :
- `save_prediction(comment, sentiment, confidence, model_name, model_id)` — signature identique entre Task 4 (définition) et Task 5 (appel) ✓
- `fetch_global_history(limit=20)` — appel cohérent en Task 6 ✓
- Cache invalidation `fetch_global_history.clear()` cohérent en Task 5 et Task 6 ✓
- Structure du dict d'INSERT cohérente avec le schéma de la spec §5 ✓
- Classes CSS référencées en Task 6 (`comment-item-augmented`, `comment-item-original`, `history-model-icon`, `badge-pos/neg/neu`, `comment-item`) sont toutes définies dans le CSS ajouté lors de la spec UI Polish ✓

**Aucun problème détecté.**
