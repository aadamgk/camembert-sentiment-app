# Persistance Supabase des prédictions de l'app

**Date** : 2026-05-06
**Auteur** : Ahmat (collab. Claude)
**Statut** : Design validé, table créée côté Supabase

## 1. Contexte et motivation

L'app Ynov Sentiment Analyser permet de tester deux modèles CamemBERT en interactif. Actuellement, l'historique des prédictions vit dans `st.session_state` et disparaît au refresh. L'utilisateur veut persister les prédictions dans Supabase (instance personnelle) pour :

- Voir un historique global qui survit entre sessions
- Constituer une trace des comparaisons entre les deux modèles
- Préparer un futur dashboard analytique (hors scope ici)

Le projet Supabase de l'utilisateur contient déjà deux tables (`CommentaireBrut`, `CommentaireClean`) liées aux **données d'entraînement** — on ne les touche pas. On ajoute une nouvelle table `predictions` dédiée aux prédictions générées par l'app.

## 2. Objectif et critères de succès

### Objectif

Étendre `app.py` pour qu'à chaque clic "Analyser le sentiment", la prédiction soit insérée dans la table `predictions` de Supabase. Ajouter une section "Historique global" affichant les 20 dernières prédictions persistantes, rafraîchissable manuellement.

### Critères de succès

1. Une ligne est créée dans `predictions` à chaque prédiction réussie (sentiment + confidence + model + comment + timestamp).
2. La section "Historique global" charge les 20 dernières lignes triées par `created_at desc`.
3. Si Supabase est inaccessible (réseau, clés invalides, table manquante) → l'app continue de fonctionner normalement (best-effort, pas de crash).
4. Les secrets (URL + anon key) ne sont **jamais committés** (`.streamlit/secrets.toml` gitignoré).
5. Aucun appel utilise la `service_role` key — uniquement l'anon key.
6. Politiques RLS configurées : INSERT et SELECT autorisés à l'anon, UPDATE/DELETE refusés.

## 3. Scope

### Inclus

- Ajout de la dépendance `supabase` dans `requirements.txt`
- Création de `.streamlit/secrets.toml.example` (template versionné) + `.streamlit/secrets.toml` (gitignoré, à remplir manuellement par l'utilisateur)
- Mise à jour du `.gitignore` pour `.streamlit/secrets.toml`
- Schéma SQL de la table `predictions` documenté dans la spec (l'utilisateur a confirmé l'avoir créée)
- 3 nouvelles fonctions dans `app.py` : `get_supabase_client()`, `save_prediction()`, `fetch_global_history()`
- Intégration dans le flux de prédiction (insertion best-effort)
- Nouvelle section UI "🌐 Historique global" en bas de la zone d'analyse, avec bouton "Rafraîchir"
- Gestion d'erreur : warning Streamlit discret si la connexion échoue

### Exclus (futurs travaux)

- Auth utilisateur (toutes les prédictions sont anonymes)
- Dashboard analytique (graphiques sur la table `predictions`)
- Pagination / filtre par modèle / recherche dans l'historique global
- Modification des tables `CommentaireBrut` ou `CommentaireClean`
- Export CSV depuis l'app (peut se faire directement via Supabase)

## 4. Architecture

### Flux de prédiction étendu

```
[Click "Analyser le sentiment"]
        │
        ▼
[Modèle HF prédit] ──────────────────┐
        │                            │
        ▼                            │
[Affichage immédiat dans l'UI]       │
        │                            │
        ▼                            │
[Append à st.session_state]          │
        │                            │
        ▼                            │
[save_prediction() → INSERT Supabase]│
        │                            │
        ├─ succès → silencieux       │
        └─ échec → st.warning()      │
                                     │
[Section "Historique global"]        │
        │                            │
        ▼                            │
[fetch_global_history(20)]           │
   - Cache 60 sec (TTL)              │
   - SELECT order by created_at desc │
        │                            │
        ▼                            │
[Affichage liste persistante]        │
```

### Composants

**A. Module client Supabase** (dans `app.py`)
```python
@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)
```

**B. Fonction d'insertion**
```python
def save_prediction(comment, sentiment, confidence, model_name, model_id):
    try:
        client = get_supabase_client()
        client.table("predictions").insert({
            "comment": comment,
            "predicted_sentiment": sentiment,
            "confidence": confidence,
            "model_name": model_name,
            "model_id": model_id,
        }).execute()
    except Exception as e:
        st.warning(f"⚠️ Persistance Supabase indisponible : {type(e).__name__}")
```

**C. Fonction de lecture**
```python
@st.cache_data(ttl=60)
def fetch_global_history(limit=20):
    try:
        client = get_supabase_client()
        result = client.table("predictions") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data
    except Exception:
        return None  # signal d'erreur silencieux
```

**D. Section UI "Historique global"**

Placée juste après le bloc `col_history` (au-dessus du footer). Layout :
```
┌──────────────────────────────────────────────────┐
│ 🌐 Historique global (Supabase)    [↻ Rafraîchir]│
├──────────────────────────────────────────────────┤
│ ✨ "Excellent campus..."   Augmenté · 99% positif│
│ 📜 "Cours médiocres..."    Original  · 87% négatif│
│ ✨ "Bof, sans plus..."     Augmenté · 51% positif│
│ ...                                              │
└──────────────────────────────────────────────────┘
```

Chaque ligne reprend le style `comment-item` existant + l'icône modèle.
Si `fetch_global_history()` retourne `None` → message "Connexion Supabase indisponible".
Si retourne `[]` → message "Aucune prédiction enregistrée pour le moment".

## 5. Schéma SQL de référence

Table créée par l'utilisateur (déjà en place) :

```sql
CREATE TABLE public.predictions (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  comment text NOT NULL,
  predicted_sentiment text NOT NULL,
  confidence numeric(5,2) NOT NULL,
  model_name text NOT NULL,
  model_id text NOT NULL
);

-- RLS attendue
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can insert predictions"
  ON public.predictions FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Anyone can read predictions"
  ON public.predictions FOR SELECT TO anon USING (true);
```

**Note** : si l'utilisateur a créé la table sans RLS configurée, l'app fera des erreurs "permission denied" silencieuses (le warning sera visible). À vérifier au moment du test.

## 6. Spécifications techniques

### Secrets

`.streamlit/secrets.toml` (gitignoré, à créer par l'utilisateur) :
```toml
SUPABASE_URL = "https://edpkrgdlxpdtvrwbkahh.supabase.co"
SUPABASE_ANON_KEY = "<clé anon, jamais le service_role>"
```

`.streamlit/secrets.toml.example` (versionné, template) :
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key-here"
```

`.gitignore` doit contenir :
```
.streamlit/secrets.toml
```

### Dépendance

Ajouter à `requirements.txt` :
```
supabase
```

### Cache

- `get_supabase_client()` : `@st.cache_resource` (un seul client par session)
- `fetch_global_history()` : `@st.cache_data(ttl=60)` (cache 60 sec, rafraîchi par bouton)

### Format des champs

| Champ DB | Source app | Type Python |
|---|---|---|
| `comment` | `comment` (str entier, pas tronqué) | str |
| `predicted_sentiment` | `sentiment` (str : "positif" / "négatif" / "neutre") | str |
| `confidence` | `confidence` (float 0-100, déjà arrondi à 1 décimale) | float |
| `model_name` | `selected_model_name` (str ex: "CamemBERT (augmenté)") | str |
| `model_id` | `MODELS[selected_model_name]["id"]` (str ex: "Ahmat293/camembert-ynov-augmented") | str |

`created_at` est rempli par Supabase via `DEFAULT now()`.
`id` est auto-généré.

## 7. Gestion des erreurs

- **Clés manquantes dans secrets** : `st.secrets["SUPABASE_URL"]` lève `KeyError` → catch dans `get_supabase_client()` → renvoie `None` → `save_prediction` et `fetch_global_history` détectent le `None` et skippent silencieusement.
- **Réseau down** : timeout/exception → `st.warning()` discret, app continue.
- **RLS bloque l'INSERT** : exception 403 → `st.warning()` avec message clair "Vérifiez la politique RLS de la table predictions".
- **Table inexistante** : exception 404 → `st.warning()` avec lien vers le SQL de création.

Aucune de ces erreurs ne doit faire crasher l'app — la prédiction et l'affichage local doivent toujours fonctionner.

## 8. Validation finale

1. `pip install supabase` réussit (locale)
2. `.streamlit/secrets.toml` créé avec les bonnes clés
3. `streamlit run app.py` ouvre l'app sans erreur
4. Cliquer "Analyser" → la prédiction apparaît dans l'UI **et** une nouvelle ligne dans `predictions` côté Supabase (vérifiable via dashboard Supabase)
5. Cliquer "Rafraîchir" sur la section "Historique global" → recharge depuis la base
6. Renommer temporairement la clé anon dans `secrets.toml` → app continue de fonctionner, warning visible
7. `git status` → `secrets.toml` n'apparaît pas dans les modifs
8. Test croisé : prédire avec les deux modèles, vérifier que `model_name` distingue bien les deux dans la base

## 9. Décisions verrouillées

- Hybride session + global (option C du brainstorming)
- Anonyme (option A du brainstorming)
- Streamlit secrets (option A du brainstorming)
- Anon key only (jamais le service_role)
- Lib officielle `supabase-py`
- Nouvelle table dédiée `predictions`, sans modifier les tables existantes
- Limite 20 lignes affichées dans "Historique global" (ordre desc)
- Cache de lecture : 60 sec TTL

## 10. Hors scope explicite

- Pas de page d'admin Supabase intégrée
- Pas d'auth (Supabase Auth, magic links, OAuth)
- Pas de pagination ou recherche
- Pas de migrations gérées par l'app (l'utilisateur applique le SQL manuellement via dashboard)
- Pas d'edge functions Supabase

## 11. Avertissement sécurité

L'utilisateur a partagé sa `service_role` key dans la conversation. **Cette clé doit être rotated immédiatement** (Settings → API → Reset service_role) avant tout déploiement, car elle bypasse RLS et permet un accès admin total à la base. L'app n'utilise que l'anon key.
