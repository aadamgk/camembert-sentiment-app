# Plan d'implémentation — Page Méthodologie + Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un onglet "📖 Méthodologie" à l'app contenant l'architecture du pipeline, les métriques, le stack et les ressources, sans régression sur l'analyse existante.

**Architecture:** Refactor de `app.py` pour wrapper le contenu actuel sous un premier `st.tabs` "Analyse", puis ajouter un second onglet "Méthodologie" avec 7 sections (HTML/CSS custom + Plotly Heatmap pour la matrice de confusion).

**Tech Stack:** Streamlit `st.tabs`, HTML/CSS inline (pattern existant), Plotly (déjà présent).

**Spec source:** `docs/superpowers/specs/2026-05-06-methodology-page-design.md`

---

## File Structure

**Modifié uniquement :** `app.py`

Zones impactées :
1. Bloc CSS — ajout des styles `.arch-*`, `.methodo-*`, `.pipeline-step*`, `.stack-card*`, `.limit-list`
2. Wrapping du contenu existant dans `tab_analyse`
3. Nouveau bloc complet `tab_methodo` avec 7 sections

**Non modifiés :** notebook, dataset, modèles HF, requirements.txt, secrets.

---

## Task 1: Ajouter le CSS de la page Méthodologie

**Files:**
- Modify: `app.py` (bloc `<style>` existant qui contient les classes des cards modèle)

- [ ] **Step 1: Repérer la fin du dernier bloc CSS**

```bash
grep -n "dual-model-label" app.py
```

On va insérer juste après la dernière règle CSS, avant `</style>` qui ferme ce bloc.

- [ ] **Step 2: Insérer le nouveau CSS**

Trouver dans `app.py` :

```css
.dual-model-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
</style>
""", unsafe_allow_html=True)
```

Le remplacer par :

```css
.dual-model-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ─── Page Méthodologie ─── */
.methodo-h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f0f5;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.methodo-intro {
    color: #d1d5db;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

/* ─── Diagramme architecture ─── */
.arch-diagram {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0;
    padding: 1.5rem 1rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
}

.arch-row {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1rem;
    width: 100%;
}

.arch-row-2col { gap: 1.5rem; }

.arch-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #e5e7eb;
    font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif;
    text-align: center;
    min-width: 200px;
    max-width: 320px;
    transition: all 0.2s ease;
    line-height: 1.4;
}

.arch-box:hover {
    border-color: rgba(167,139,250,0.4);
    transform: translateY(-1px);
}

.arch-box-source { border-left: 3px solid #60a5fa; }
.arch-box-llm    { border-left: 3px solid #a78bfa; background: rgba(167,139,250,0.08); }
.arch-box-data   { border-left: 3px solid #34d399; background: rgba(52,211,153,0.08); }
.arch-box-train  { border-left: 3px solid #fbbf24; background: rgba(251,191,36,0.08); }
.arch-box-hf     { border-left: 3px solid #f97316; background: rgba(249,115,22,0.08); }
.arch-box-app    { border-left: 3px solid #ec4899; background: rgba(236,72,153,0.08); }
.arch-box-db     { border-left: 3px solid #14b8a6; background: rgba(20,184,166,0.08); }

.arch-arrow {
    color: #6b7280;
    font-size: 1.4rem;
    font-weight: 300;
    line-height: 0.8;
}

.arch-arrow-h {
    color: #6b7280;
    font-size: 1.2rem;
}

/* ─── Pipeline étapes ─── */
.pipeline-step {
    display: flex;
    gap: 1rem;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #a78bfa;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.pipeline-step-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #a78bfa;
    line-height: 1;
    min-width: 30px;
}

.pipeline-step-content { flex: 1; }

.pipeline-step-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f0f0f5;
    margin-bottom: 0.2rem;
}

.pipeline-step-meta {
    color: #6b7280;
    font-size: 0.72rem;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.pipeline-step-desc {
    color: #d1d5db;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ─── Stack technique ─── */
.stack-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}

.stack-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 0.4rem;
}

.stack-card-tools {
    color: #d1d5db;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ─── Limitations ─── */
.limit-list { padding-left: 0; list-style: none; margin: 0.5rem 0; }
.limit-list li {
    padding: 0.5rem 0 0.5rem 1.5rem;
    position: relative;
    color: #d1d5db;
    font-size: 0.85rem;
    line-height: 1.5;
}
.limit-list li::before {
    content: "⚠";
    position: absolute;
    left: 0;
    color: #fbbf24;
}

/* ─── Ressources ─── */
.resource-link {
    display: block;
    padding: 0.8rem 1.2rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    color: #d1d5db;
    text-decoration: none;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
    font-size: 0.9rem;
}
.resource-link:hover {
    border-color: rgba(167,139,250,0.4);
    background: rgba(167,139,250,0.06);
    color: #f0f0f5;
}

/* ─── st.tabs personnalisation ─── */
[data-baseweb="tab-list"] {
    gap: 0.5rem;
}

button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #9ca3af !important;
    padding: 0.5rem 1.2rem !important;
    font-family: 'Syne', sans-serif !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.10)) !important;
    border-color: rgba(167,139,250,0.5) !important;
    color: #f0f0f5 !important;
}
</style>
""", unsafe_allow_html=True)
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): CSS pour page méthodologie (architecture, pipeline, stack)"
```

---

## Task 2: Wrapper le contenu existant dans `tab_analyse`

**Files:**
- Modify: `app.py` (zone après `# ─── Load data ──` jusqu'à la fin de l'historique)

- [ ] **Step 1: Repérer le début du contenu à wrapper**

```bash
grep -n "# ─── Load data" app.py
grep -n "# ─── Footer" app.py
```

Le contenu à wrapper commence à la ligne `# ─── Load data ──────...` et finit juste avant `# ─── Footer ──────...`.

- [ ] **Step 2: Insérer la création des onglets juste avant `# ─── Load data`**

Trouver dans `app.py` :

```python
st.markdown('<div class="subtitle">Analyse des avis étudiants · CamemBERT original vs augmenté</div>', unsafe_allow_html=True)

# ─── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
```

Le remplacer par :

```python
st.markdown('<div class="subtitle">Analyse des avis étudiants · CamemBERT original vs augmenté</div>', unsafe_allow_html=True)

# ─── Onglets ──────────────────────────────────────────────────────────────────
tab_analyse, tab_methodo = st.tabs(["📊 Analyse", "📖 Méthodologie"])

with tab_analyse:
    pass  # placeholder, contenu inséré dans Task 2 step 3

# ─── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
```

⚠️ Note : on ne peut pas wrapper du code top-level dans un `with` directement à cause de la fonction `@st.cache_data load_data` qui doit être définie au top-level. La stratégie : on place tout le contenu d'affichage dans le `with tab_analyse:` mais on garde les définitions de fonctions cachées (`load_data`) au top-level juste avant les onglets.

Vérifier la structure actuelle du fichier — `load_data()` doit déjà être au top-level.

- [ ] **Step 3: Indenter tout le contenu existant sous `with tab_analyse:`**

C'est un gros bloc d'indentation. Pour éviter les erreurs manuelles, utiliser un script Python qui :
1. Lit le fichier
2. Trouve la ligne du placeholder `with tab_analyse:\n    pass`
3. Trouve la ligne `# ─── Footer ──────...`
4. Prend tout le contenu entre `# ─── Load data ──` et `# ─── Footer`
5. Indente chaque ligne de 4 espaces
6. Remplace le `pass` placeholder par ce contenu indenté

Mais c'est fragile. Approche **plus sûre** : refactor manuel par sections.

**Sous-step 3a** : Lire les lignes précises du contenu à indenter

```bash
grep -n "^# ─── \|^df_source = load_data" app.py
```

Ça va donner les lignes des grandes sections : `# ─── Load data`, `# ─── Analyse`, `# ─── Footer`.

**Sous-step 3b** : Refactor en plaçant le `with tab_analyse:` juste APRÈS `df_source = load_data()` (donc le chargement des données reste au top-level pour bénéficier du cache, mais l'affichage du dataset overview + analyse + historique passe sous l'onglet).

Trouver la ligne :
```python
df_source = load_data()
```

Insérer juste après :
```python
df_source = load_data()

# ─── Onglets ──────────────────────────────────────────────────────────────────
tab_analyse, tab_methodo = st.tabs(["📊 Analyse", "📖 Méthodologie"])
```

Et **supprimer** le placeholder précédemment ajouté (l'autre `tab_analyse, tab_methodo = ...` qu'on avait mis avant `# ─── Load data`).

**Sous-step 3c** : Tout le contenu de `if df_source is not None:` jusqu'à la fin de l'historique (juste avant `# ─── Footer`) est mis sous `with tab_analyse:`.

Concrètement, repérer le bloc suivant :

```python
if df_source is not None:
    try:
        if "sentiment_label" in df_source.columns:
            ...
            (toute la section dataset overview)
            ...
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")

# ─── Analyse ───────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_input, col_history = st.columns([1, 1], gap="large")

def render_model_card(model_name, is_selected):
    ...

with col_input:
    ...

with col_history:
    ...
```

Et le restructurer en :

```python
with tab_analyse:
    if df_source is not None:
        try:
            if "sentiment_label" in df_source.columns:
                ...
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_input, col_history = st.columns([1, 1], gap="large")

    with col_input:
        ...

    with col_history:
        ...

with tab_methodo:
    pass  # rempli dans Task 3-7
```

**Important** : la fonction `render_model_card` reste **au top-level** (pas dans le `with`), pour rester accessible si on l'utilise ailleurs.

Vu la complexité, on procède en édition directe section par section :

**a) Repérer et lire toute la zone à modifier**

```bash
sed -n '/^# ─── Load data/,/^# ─── Footer/p' app.py | head -50
```

**b) Ouvrir `app.py` dans un éditeur Streamlit-aware** (ou modifier via Edit) pour :

1. Supprimer le placeholder `with tab_analyse:\n    pass` ajouté à l'étape 2
2. Juste après `df_source = load_data()`, ajouter :
   ```python

   # ─── Onglets ──────────────────────────────────────────────────────────────────
   tab_analyse, tab_methodo = st.tabs(["📊 Analyse", "📖 Méthodologie"])

   with tab_analyse:
   ```
3. Indenter de 4 espaces toutes les lignes entre `if df_source is not None:` et la fin du bloc `with col_history:` (jusqu'à la fermeture du `if st.session_state.new_comments:` block).
4. Sortir `def render_model_card(...)` du bloc indenté (le placer **avant** le `with tab_analyse:` pour qu'il reste au top-level).
5. Ajouter à la fin (avant `# ─── Footer`) :
   ```python

   with tab_methodo:
       st.markdown("**Page méthodologie en construction.**")  # placeholder
   ```

- [ ] **Step 4: Vérifier que le `with tab_analyse:` ne casse rien**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
streamlit run app.py
```

L'app doit s'ouvrir avec **2 onglets visibles**. L'onglet "📊 Analyse" doit montrer exactement le même contenu qu'avant. L'onglet "📖 Méthodologie" affiche juste le placeholder.

Si KO : revoir l'indentation manuellement.

- [ ] **Step 5: Commit (jalon)**

```bash
git add app.py
git commit -m "refactor(app): wrapper le contenu d'analyse dans st.tabs (onglet 1/2)"
```

---

## Task 3: Section "En bref" + KPI cards (onglet Méthodologie)

**Files:**
- Modify: `app.py` (bloc `with tab_methodo:`)

- [ ] **Step 1: Remplacer le placeholder par la section "En bref"**

Trouver dans `app.py` :

```python
with tab_methodo:
    st.markdown("**Page méthodologie en construction.**")  # placeholder
```

Le remplacer par :

```python
with tab_methodo:
    # ─── Section 1 : En bref ──────────────────────────────────────────────
    st.markdown('<div class="methodo-h2">📌 En bref</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="methodo-intro">'
        'Cette application classe automatiquement les avis étudiants Ynov en <strong>positif</strong> ou <strong>négatif</strong>, '
        'en s\'appuyant sur <strong>CamemBERT</strong>, un modèle d\'IA français open source. À partir de 550 vrais avis Google Maps, '
        'on a entraîné un modèle spécialisé qui atteint <strong>98% de précision</strong> sur le domaine éducatif Ynov.'
        '</div>',
        unsafe_allow_html=True
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(
            '<div class="metric-card"><div class="metric-number total">394</div>'
            '<div class="metric-label">Avis dans le dataset</div></div>',
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            '<div class="metric-card"><div class="metric-number neu">2</div>'
            '<div class="metric-label">Modèles comparés</div></div>',
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            '<div class="metric-card"><div class="metric-number pos">98.3%</div>'
            '<div class="metric-label">Accuracy test set</div></div>',
            unsafe_allow_html=True
        )
```

- [ ] **Step 2: Vérifier syntaxe + lancer l'app pour visualiser**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

Refresh l'app, cliquer "📖 Méthodologie" → la section "En bref" doit apparaître avec les 3 KPI cards.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): méthodologie - section En bref + 3 KPI cards"
```

---

## Task 4: Section "Architecture du projet" (diagramme)

**Files:**
- Modify: `app.py` (bloc `with tab_methodo:`, juste après la section "En bref")

- [ ] **Step 1: Ajouter le bloc architecture**

Trouver dans `app.py` la fin du bloc KPI (juste après le `with kpi3:` et son contenu).

Ajouter juste après ce bloc, toujours à l'intérieur de `with tab_methodo:` :

```python
    # ─── Section 2 : Architecture du projet ──────────────────────────────
    st.markdown('<div class="methodo-h2">🏗️ Architecture du projet</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="arch-diagram">
        <div class="arch-row">
            <div class="arch-box arch-box-source">📊 <strong>Source Google Maps</strong><br><span style="color:#9ca3af;font-size:0.75rem">avis_ynov_All_final.csv · 550 lignes</span></div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-row arch-row-2col">
            <div class="arch-box">🧹 <strong>Filtrage</strong><br><span style="color:#9ca3af;font-size:0.75rem">383 lignes valides</span></div>
            <div class="arch-arrow-h">→</div>
            <div class="arch-box arch-box-llm">🤖 <strong>Augmentation LLM</strong><br><span style="color:#9ca3af;font-size:0.75rem">+100 négatifs synthétiques (Claude)</span></div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-row">
            <div class="arch-box arch-box-data">📦 <strong>Dataset équilibré</strong><br><span style="color:#9ca3af;font-size:0.75rem">avis_ynov_augmented.csv · 197/197 binaire</span></div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-row">
            <div class="arch-box arch-box-train">🎓 <strong>Fine-tuning CamemBERT</strong><br><span style="color:#9ca3af;font-size:0.75rem">Colab T4 · 5 epochs · ~6 min</span></div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-row arch-row-2col">
            <div class="arch-box arch-box-hf">🤗 <strong>HF Hub : original</strong><br><span style="color:#9ca3af;font-size:0.75rem">3 classes (pos/neu/neg)</span></div>
            <div class="arch-box arch-box-hf">🤗 <strong>HF Hub : augmenté</strong><br><span style="color:#9ca3af;font-size:0.75rem">2 classes (binaire)</span></div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-row">
            <div class="arch-box arch-box-app">💻 <strong>App Streamlit</strong><br><span style="color:#9ca3af;font-size:0.75rem">Comparaison live des 2 modèles</span></div>
        </div>
        <div class="arch-arrow">↓</div>

        <div class="arch-row">
            <div class="arch-box arch-box-db">🗄️ <strong>Supabase PostgreSQL</strong><br><span style="color:#9ca3af;font-size:0.75rem">Table predictions · analytics futures</span></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
```

- [ ] **Step 2: Vérifier syntaxe + visualiser**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

Refresh l'app → onglet Méthodologie doit afficher le diagramme avec les boîtes colorées et flèches.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(app): méthodologie - diagramme d'architecture du pipeline"
```

---

## Task 5: Section "Pipeline en 5 étapes"

**Files:**
- Modify: `app.py` (`with tab_methodo:`, après la section architecture)

- [ ] **Step 1: Ajouter le bloc pipeline étapes**

Ajouter à la suite, toujours à l'intérieur de `with tab_methodo:` :

```python
    # ─── Section 3 : Pipeline en 5 étapes ────────────────────────────────
    st.markdown('<div class="methodo-h2">🔄 Pipeline en 5 étapes</div>', unsafe_allow_html=True)

    pipeline_steps = [
        {
            "num": "1",
            "title": "Collecte des avis",
            "meta": "Pré-existant",
            "desc": "550 avis scrapés depuis Google Maps des campus Ynov, livrés au format CSV avec colonnes (author, rating, sentiment_label, date, comment)."
        },
        {
            "num": "2",
            "title": "Augmentation par LLM",
            "meta": "~30 minutes",
            "desc": "Génération de 100 avis négatifs synthétiques par Claude, calibrés sur le style des vrais avis Ynov (11 campus, 10 filières, 8+ angles différents). Combinés avec 97 vrais négatifs et 197 positifs échantillonnés → dataset binaire équilibré 197/197."
        },
        {
            "num": "3",
            "title": "Fine-tuning CamemBERT",
            "meta": "~6 minutes sur Colab T4",
            "desc": "CamemBERT base (110M paramètres), 5 epochs, learning rate 2e-5, batch size 16, weight decay 0.01, warmup ratio 0.1. Split stratifié 70/15/15 sur sentiment×source. Métriques évaluées par epoch sur la validation."
        },
        {
            "num": "4",
            "title": "Déploiement",
            "meta": "~5 minutes",
            "desc": "Push automatique du modèle sur Hugging Face Hub via model.push_to_hub(). App Streamlit déployée sur Streamlit Cloud, charge les modèles à la volée depuis HF Hub. Aucun modèle hébergé dans le repo Git."
        },
        {
            "num": "5",
            "title": "Persistance des prédictions",
            "meta": "Continu",
            "desc": "Chaque prédiction utilisateur est insérée dans Supabase (table predictions) en best-effort, pour analytics futures. La table accumule comment, sentiment prédit, confidence, modèle utilisé et timestamp."
        },
    ]

    for step in pipeline_steps:
        st.markdown(f'''
        <div class="pipeline-step">
            <div class="pipeline-step-num">{step["num"]}</div>
            <div class="pipeline-step-content">
                <div class="pipeline-step-title">{step["title"]}</div>
                <div class="pipeline-step-meta">{step["meta"]}</div>
                <div class="pipeline-step-desc">{step["desc"]}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
```

- [ ] **Step 2: Vérifier + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): méthodologie - 5 étapes du pipeline détaillées"
```

---

## Task 6: Section "Résultats détaillés" + matrice de confusion

**Files:**
- Modify: `app.py` (`with tab_methodo:`, après la section pipeline)

- [ ] **Step 1: Ajouter le tableau métriques**

Ajouter à la suite, à l'intérieur de `with tab_methodo:` :

```python
    # ─── Section 4 : Résultats détaillés ─────────────────────────────────
    st.markdown('<div class="methodo-h2">📈 Résultats détaillés</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="methodo-intro">Métriques mesurées sur le test set (60 avis : 45 réels + 15 synthétiques) après 5 epochs de fine-tuning du modèle augmenté.</div>',
        unsafe_allow_html=True
    )

    metrics_df = pd.DataFrame({
        "Métrique":    ["Accuracy",  "F1 macro",  "F1 négatif", "F1 positif", "n test"],
        "Global":      ["98.3%",     "0.9833",    "0.9831",     "0.9836",     "60"],
        "Real":        ["97.8%",     "0.9746",    "0.9655",     "0.9836",     "45"],
        "Synthetic":   ["100.0%",    "1.0000",    "1.0000",     "—",          "15"],
    })
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="methodo-intro" style="font-size:0.85rem">'
        '<em>L\'écart real / synthetic est de 2,5 points → biais de génération minimal. '
        'Le F1 positif sur le subset synthetic est noté "—" car aucun avis positif synthétique n\'existe '
        '(on a généré uniquement des négatifs).</em>'
        '</div>',
        unsafe_allow_html=True
    )
```

- [ ] **Step 2: Ajouter la matrice de confusion en Plotly Heatmap**

Ajouter juste après le tableau, toujours dans `with tab_methodo:` :

```python
    st.markdown('<div class="methodo-h2" style="font-size:1.1rem;margin-top:1.5rem">Matrice de confusion (test global)</div>', unsafe_allow_html=True)

    cm_values = [[29, 1], [0, 30]]
    cm_labels = ["négatif", "positif"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_values,
        x=cm_labels,
        y=cm_labels,
        text=cm_values,
        texttemplate="%{text}",
        textfont=dict(size=18, color="#f0f0f5"),
        colorscale=[[0, "rgba(167,139,250,0.05)"], [1, "rgba(167,139,250,0.6)"]],
        showscale=False,
        hoverongaps=False,
    ))
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#9ca3af",
        xaxis=dict(title="Prédit", side="bottom"),
        yaxis=dict(title="Réel", autorange="reversed"),
        margin=dict(t=20, b=40, l=60, r=20),
        height=320,
    )
    st.plotly_chart(fig_cm, use_container_width=True)
```

- [ ] **Step 3: Vérifier + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): méthodologie - tableau métriques + matrice de confusion"
```

---

## Task 7: Section "Stack technique"

**Files:**
- Modify: `app.py` (`with tab_methodo:`, après section résultats)

- [ ] **Step 1: Ajouter le bloc stack**

Ajouter à la suite, à l'intérieur de `with tab_methodo:` :

```python
    # ─── Section 5 : Stack technique ─────────────────────────────────────
    st.markdown('<div class="methodo-h2">🛠️ Stack technique</div>', unsafe_allow_html=True)

    stack = [
        ("🤖 ML",       "Transformers · PyTorch · CamemBERT · scikit-learn"),
        ("🎨 Frontend", "Streamlit · Plotly · HTML/CSS custom"),
        ("☁️ Hosting",  "Streamlit Cloud · Hugging Face Hub"),
        ("🗄️ Data",     "Pandas · CSV · Supabase (PostgreSQL)"),
        ("🛠️ Ops",      "Git · GitHub · Google Colab · Claude Code"),
    ]
    for cat, tools in stack:
        st.markdown(f'''
        <div class="stack-card">
            <div class="stack-card-title">{cat}</div>
            <div class="stack-card-tools">{tools}</div>
        </div>
        ''', unsafe_allow_html=True)
```

- [ ] **Step 2: Commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): méthodologie - stack technique en 5 catégories"
```

---

## Task 8: Sections "Limitations" et "Ressources"

**Files:**
- Modify: `app.py` (`with tab_methodo:`, fin de l'onglet)

- [ ] **Step 1: Ajouter les limitations**

Ajouter à la suite, à l'intérieur de `with tab_methodo:` :

```python
    # ─── Section 6 : Limitations honnêtes ────────────────────────────────
    st.markdown('<div class="methodo-h2">⚠️ Limitations honnêtes</div>', unsafe_allow_html=True)
    st.markdown('''
    <ul class="limit-list">
        <li><strong>Labels mécaniques</strong> : sentiment_label est dérivé du rating (1-2 → négatif, 3 → neutre, 4-5 → positif), pas annoté indépendamment du texte.</li>
        <li><strong>Synthetic mono-source</strong> : les 100 négatifs synthétiques sont produits par un seul LLM (Claude), risque de signature stylistique détectable.</li>
        <li><strong>Taille modeste</strong> : 394 avis suffisent pour un POC ; pour la production, viser 1000+ par classe avec scrap réel.</li>
        <li><strong>Validation à 100% dès l'epoch 2</strong> : signal possible d'un val set trop petit (~58 lignes) ou de signaux trop forts dans les commentaires polarisés.</li>
        <li><strong>Pas de classe "neutre"</strong> dans le modèle augmenté : choix volontaire faute de vrais avis neutres dans le dataset original (seulement 2).</li>
    </ul>
    ''', unsafe_allow_html=True)

    # ─── Section 7 : Ressources ──────────────────────────────────────────
    st.markdown('<div class="methodo-h2">🔗 Ressources</div>', unsafe_allow_html=True)
    st.markdown('''
    <a class="resource-link" href="https://huggingface.co/Ahmat293/camembert-sentiment-ynov" target="_blank">
        🤗 <strong>Modèle CamemBERT original</strong> · entraîné sur le dataset brut (3 classes)
    </a>
    <a class="resource-link" href="https://huggingface.co/Ahmat293/camembert-ynov-augmented" target="_blank">
        🤗 <strong>Modèle CamemBERT augmenté</strong> · fine-tuné sur dataset augmenté (binaire)
    </a>
    <a class="resource-link" href="https://github.com/aadamgk/camembert-sentiment-app" target="_blank">
        ⌨ <strong>Code source GitHub</strong> · app + notebook + specs détaillées
    </a>
    ''', unsafe_allow_html=True)
```

- [ ] **Step 2: Vérifier syntaxe + commit final**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): méthodologie - limitations + ressources (page complète)"
```

---

## Task 9: Test fonctionnel manuel

**Files:** aucun (validation utilisateur)

- [ ] **Step 1: Lancer / refresher l'app**

```bash
streamlit run app.py
```

- [ ] **Step 2: Vérifier l'onglet Analyse (non-régression)**

- ☐ 2 onglets visibles : "📊 Analyse" et "📖 Méthodologie"
- ☐ "📊 Analyse" est sélectionné par défaut
- ☐ Donut + bar chart notes affichés
- ☐ 3 cards de sélection de modèle visibles
- ☐ Saisir un commentaire et cliquer "Analyser le sentiment" → fonctionne
- ☐ Mode comparaison fonctionne
- ☐ Historique session s'incrémente

- [ ] **Step 3: Vérifier l'onglet Méthodologie**

Cliquer "📖 Méthodologie", scroller et vérifier :
- ☐ Section **En bref** : texte introductif + 3 KPI cards (394 / 2 / 98.3%)
- ☐ Section **Architecture** : diagramme avec boîtes colorées et flèches verticales/horizontales
- ☐ Section **Pipeline en 5 étapes** : 5 cards numérotées 1→5
- ☐ Section **Résultats** : tableau 5 lignes × 4 colonnes + matrice de confusion 2x2 (29 / 1 / 0 / 30)
- ☐ Section **Stack** : 5 cards (ML, Frontend, Hosting, Data, Ops)
- ☐ Section **Limitations** : 5 bullets avec icône ⚠
- ☐ Section **Ressources** : 3 cards-liens cliquables (HF×2 + GitHub)
- ☐ Cliquer chaque lien : ouvre dans un nouvel onglet

- [ ] **Step 4: Test croisé**

Cliquer "📊 Analyse" → la vue analyse réapparaît identique.
Faire une prédiction → l'historique session de l'onglet 1 s'incrémente.
Re-cliquer "📖 Méthodologie" → la page méthodologie est intacte.

- [ ] **Step 5: Commit final**

```bash
git commit --allow-empty -m "test: page méthodologie validée end-to-end (7 sections + non-régression)"
```

---

## Self-review

**1. Spec coverage**

| Spec § | Couverture |
|---|---|
| §2 Critère 1 (navigation onglets) | Task 2 (st.tabs) |
| §2 Critère 2 (non-régression analyse) | Task 2 (wrapping) + Task 9 step 2 |
| §2 Critère 3 (1 scroll) | Tasks 3-8 (sections compactes) |
| §2 Critère 4 (diagramme HTML/CSS) | Task 1 (CSS) + Task 4 (rendu) |
| §2 Critère 5 (liens nouvel onglet) | Task 8 (target="_blank") |
| §2 Critère 6 (responsive) | Streamlit columns gèrent automatiquement |
| §3 Section En bref | Task 3 |
| §3 Section Architecture | Task 4 |
| §3 Section Pipeline 5 étapes | Task 5 |
| §3 Section Résultats détaillés | Task 6 |
| §3 Section Stack technique | Task 7 |
| §3 Section Limitations | Task 8 (1ère partie) |
| §3 Section Ressources | Task 8 (2e partie) |
| §4 Refactor st.tabs | Task 2 |
| §5 Contenu détaillé sections 1-7 | Tasks 3-8 (textes intégraux) |
| §6 CSS classes | Task 1 (toutes définies) |
| §7 Validation finale | Task 9 |
| §8 Décisions verrouillées | Toutes appliquées |

**2. Placeholder scan** : aucun "TBD". Tous les textes français de chaque section sont fournis intégralement, prêts à coller.

**3. Type consistency** :
- Classes CSS référencées dans Task 3-8 (`methodo-h2`, `arch-*`, `pipeline-step*`, `stack-card*`, `limit-list`, `resource-link`) sont toutes définies dans Task 1 ✓
- `tab_analyse, tab_methodo` cohérents entre Task 2 et Tasks 3-8 ✓
- Le `pipeline_steps` list of dicts en Task 5 et `stack` list of tuples en Task 7 sont définis et utilisés localement, pas de fuite ✓

**Aucun problème détecté.**
