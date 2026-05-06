# Plan d'implémentation — UI Polish (cards comparatives + barre de confiance + historique enrichi)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer le sélecteur radio de modèle par 2 cards cliquables comparatives, ajouter une barre de confiance graphique sous le résultat, distinguer visuellement les modèles dans l'historique, et enrichir le footer.

**Architecture:** Modifications ciblées sur `app.py` uniquement (~6 zones). Pattern Streamlit "card cliquable" via `st.button` transparent en overlay sur du HTML stylé. État de sélection persisté dans `st.session_state.selected_model`.

**Tech Stack:** Streamlit, HTML/CSS inline (déjà le pattern existant dans `app.py`).

**Spec source:** `docs/superpowers/specs/2026-05-06-ui-polish-model-cards-design.md`

---

## File Structure

**Modifié uniquement :** `app.py`

Zones impactées (référencées par numéros de ligne approximatifs basés sur l'état actuel) :
1. `app.py:122-208` — bloc `<style>` : ajouter les nouvelles classes CSS
2. `app.py:215-232` — `MODELS` dict : enrichir avec `emoji`, `base_model`, `num_classes`, `accuracy`, `f1_macro`, `description`, `hf_url`
3. `app.py:340-360` — section "Analyser un commentaire" : remplacer le `st.radio` par 2 cards + boutons transparents
4. `app.py:347-356` — initialisation `st.session_state.selected_model`
5. `app.py:362-365` — bloc résultat : ajouter la barre de confiance
6. `app.py:402-411` — boucle historique : ajouter icône modèle + bordure colorée
7. `app.py:419-421` — footer : ajouter ligne de liens

**Non modifiés :** notebook, dataset, modèles HF, requirements.txt.

---

## Task 1: Enrichir le `MODELS` dict avec les métadonnées

**Files:**
- Modify: `app.py:215-232`

- [ ] **Step 1: Lire la zone actuelle**

```bash
sed -n '210,235p' app.py
```

Confirmer que le bloc `MODELS = {...}` est aux lignes attendues (ajuster si décalage).

- [ ] **Step 2: Remplacer le bloc `MODELS`**

Trouver dans `app.py` :

```python
# ─── Modèles disponibles ──────────────────────────────────────────────────────
MODELS = {
    "CamemBERT (original)": {
        "id": "Ahmat293/camembert-sentiment-ynov",
        "subfolder": "model",            # le modèle est dans un sous-dossier
        "tokenizer_subfolder": "tokenizer",
        # Modèle 3 classes (négatif/neutre/positif), ordre alphabétique sklearn par défaut
        "label_map": {"LABEL_0": "négatif", "LABEL_1": "neutre", "LABEL_2": "positif"},
    },
    "CamemBERT (augmenté)": {
        "id": "Ahmat293/camembert-ynov-augmented",
        "subfolder": None,                # à la racine
        "tokenizer_subfolder": None,
        "label_map": {"negatif": "négatif", "positif": "positif"},
    },
}
```

Le remplacer par :

```python
# ─── Modèles disponibles ──────────────────────────────────────────────────────
MODELS = {
    "CamemBERT (original)": {
        "id": "Ahmat293/camembert-sentiment-ynov",
        "subfolder": "model",
        "tokenizer_subfolder": "tokenizer",
        "label_map": {"LABEL_0": "négatif", "LABEL_1": "neutre", "LABEL_2": "positif"},
        "emoji": "📜",
        "base_model": "camembert-base",
        "num_classes": 3,
        "accuracy": None,
        "f1_macro": None,
        "description": "Modèle initial entraîné sur le dataset original (550 avis, 3 classes pos/neu/neg)",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-sentiment-ynov",
    },
    "CamemBERT (augmenté)": {
        "id": "Ahmat293/camembert-ynov-augmented",
        "subfolder": None,
        "tokenizer_subfolder": None,
        "label_map": {"negatif": "négatif", "positif": "positif"},
        "emoji": "✨",
        "base_model": "camembert-base",
        "num_classes": 2,
        "accuracy": 0.9833,
        "f1_macro": 0.9833,
        "description": "Fine-tuné sur dataset augmenté (394 avis, binaire pos/neg)",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",
    },
}
```

- [ ] **Step 3: Vérifier la syntaxe**

```bash
python -c "
import ast
with open('app.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('OK')
"
```

Expected : `OK`

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): enrichir MODELS avec metadata (emoji, base_model, accuracy, etc.)"
```

---

## Task 2: Ajouter le CSS des cards modèle

**Files:**
- Modify: `app.py` (à l'intérieur du second bloc `<style>` qui commence vers ligne 197)

- [ ] **Step 1: Repérer la fin du second bloc `<style>`**

Le second bloc commence par `st.markdown("""<style>` et finit par `</style>""", unsafe_allow_html=True)`. Trouver l'endroit exact :

```bash
grep -n 'unsafe_allow_html=True)' app.py | head -5
```

- [ ] **Step 2: Insérer un nouveau bloc CSS juste avant le `# ─── Session state` (vers ligne 210)**

Trouver dans `app.py` :

```python
""", unsafe_allow_html=True)

# ─── Session state ──────────────────────────────────────────────────────────────
```

Insérer juste avant cette zone (entre la fin du dernier `unsafe_allow_html=True)` et le commentaire Session state) :

```python
st.markdown("""
<style>
/* ─── Cards de sélection de modèle ─── */
.model-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    position: relative;
    height: 100%;
    min-height: 220px;
}

.model-card:hover {
    border-color: rgba(167,139,250,0.4);
    background: rgba(167,139,250,0.06);
    transform: translateY(-2px);
}

.model-card.selected {
    border-color: rgba(167,139,250,0.7);
    background: linear-gradient(135deg, rgba(167,139,250,0.08), rgba(96,165,250,0.08));
    box-shadow: 0 0 30px rgba(167,139,250,0.15);
}

.model-card.selected::after {
    content: "✓";
    position: absolute;
    top: 12px;
    right: 16px;
    color: #a78bfa;
    font-size: 1.3rem;
    font-weight: 700;
}

.model-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
}

.model-card-emoji {
    font-size: 1.4rem;
}

.model-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f0f5;
}

.model-card-subtitle {
    font-size: 0.78rem;
    color: #9ca3af;
    margin-bottom: 1rem;
}

.model-card-badges {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.model-badge {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    text-align: center;
    flex: 1;
}

.model-badge-label {
    color: #6b7280;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.15rem;
}

.model-badge-value {
    color: #f0f0f5;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
}

.model-card-desc {
    font-size: 0.78rem;
    color: #9ca3af;
    line-height: 1.45;
    margin-bottom: 0.8rem;
}

.model-card-link {
    color: #a78bfa;
    font-size: 0.72rem;
    text-decoration: none;
    border-bottom: 1px dashed rgba(167,139,250,0.4);
    padding-bottom: 1px;
}

.model-card-link:hover {
    color: #c4b5fd;
    border-color: rgba(196,181,253,0.7);
}

/* ─── Bouton de sélection en overlay (transparent) ─── */
.model-selector-row .stButton > button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    width: 100% !important;
    height: 220px !important;
    margin-top: -240px !important;
    padding: 0 !important;
    z-index: 10;
    position: relative;
    cursor: pointer;
    box-shadow: none !important;
}

.model-selector-row .stButton > button:hover {
    background: transparent !important;
}

.model-selector-row .stButton > button:focus {
    background: transparent !important;
    box-shadow: none !important;
}

/* ─── Barre de confiance ─── */
.confidence-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    margin-top: 0.8rem;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s ease;
}

.confidence-bar-label {
    font-size: 0.7rem;
    color: #6b7280;
    text-align: right;
    margin-top: 0.3rem;
}

/* ─── Historique enrichi avec border modèle ─── */
.comment-item-augmented { border-left: 3px solid #a78bfa; }
.comment-item-original { border-left: 3px solid #6b7280; }

.history-model-icon {
    font-size: 0.9rem;
    margin: 0 0.4rem;
}

/* ─── Footer enrichi ─── */
.footer-links {
    text-align: center;
    margin-bottom: 0.5rem;
    font-size: 0.75rem;
}

.footer-links a {
    color: #6b7280;
    text-decoration: none;
    margin: 0 0.4rem;
}

.footer-links a:hover {
    color: #a78bfa;
}
</style>
""", unsafe_allow_html=True)
```

- [ ] **Step 3: Vérifier la syntaxe**

```bash
python -c "
import ast
with open('app.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): CSS pour cards modèle, barre de confiance, historique enrichi"
```

---

## Task 3: Initialiser `selected_model` dans le session_state

**Files:**
- Modify: `app.py` (zone `# ─── Session state`)

- [ ] **Step 1: Trouver la zone**

```bash
grep -n "Session state" app.py
```

- [ ] **Step 2: Modifier le bloc**

Trouver :

```python
# ─── Session state ──────────────────────────────────────────────────────────────
if "new_comments" not in st.session_state:
    st.session_state.new_comments = []
```

Le remplacer par :

```python
# ─── Session state ──────────────────────────────────────────────────────────────
if "new_comments" not in st.session_state:
    st.session_state.new_comments = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "CamemBERT (augmenté)"
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): init session_state.selected_model par défaut sur augmenté"
```

---

## Task 4: Remplacer le `st.radio` par les 2 cards cliquables

**Files:**
- Modify: `app.py` (section "Analyser un commentaire", `with col_input:`)

- [ ] **Step 1: Trouver la zone**

```bash
grep -n "selected_model_name = st.radio" app.py
```

- [ ] **Step 2: Remplacer le bloc complet**

Trouver dans `app.py` :

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
                classifier = load_model(config["id"], config["subfolder"], config["tokenizer_subfolder"])
                sentiment, confidence = predict(comment, classifier, config["label_map"])
```

Le remplacer par :

```python
def render_model_card(model_name, is_selected):
    cfg = MODELS[model_name]
    selected_class = "selected" if is_selected else ""

    acc = f"{cfg['accuracy']*100:.1f}%" if cfg['accuracy'] is not None else "—"
    f1 = f"{cfg['f1_macro']:.3f}" if cfg['f1_macro'] is not None else "—"

    return f"""
    <div class="model-card {selected_class}">
        <div class="model-card-header">
            <span class="model-card-emoji">{cfg['emoji']}</span>
            <span class="model-card-title">{model_name}</span>
        </div>
        <div class="model-card-subtitle">{cfg['base_model']} · {cfg['num_classes']} classes</div>
        <div class="model-card-badges">
            <div class="model-badge">
                <div class="model-badge-label">Accuracy</div>
                <div class="model-badge-value">{acc}</div>
            </div>
            <div class="model-badge">
                <div class="model-badge-label">F1 macro</div>
                <div class="model-badge-value">{f1}</div>
            </div>
        </div>
        <div class="model-card-desc">{cfg['description']}</div>
        <a href="{cfg['hf_url']}" target="_blank" class="model-card-link">→ voir sur Hugging Face</a>
    </div>
    """

with col_input:
    st.markdown('<div class="section-title">✍️ Analyser un commentaire</div>', unsafe_allow_html=True)

    # ─── Sélecteur de modèle en cards ─────────────────────────────────────
    st.markdown('<div class="model-selector-row">', unsafe_allow_html=True)
    card_cols = st.columns(2)
    model_names = list(MODELS.keys())
    for col, name in zip(card_cols, model_names):
        with col:
            is_selected = (st.session_state.selected_model == name)
            st.markdown(render_model_card(name, is_selected), unsafe_allow_html=True)
            if st.button(f"select_{name}", key=f"btn_{name}"):
                st.session_state.selected_model = name
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    selected_model_name = st.session_state.selected_model

    comment = st.text_area("", placeholder="Entrez un avis étudiant...", height=130, label_visibility="collapsed")

    if st.button("Analyser le sentiment", key="analyze_btn"):
        if comment.strip():
            with st.spinner("Analyse en cours..."):
                config = MODELS[selected_model_name]
                classifier = load_model(config["id"], config["subfolder"], config["tokenizer_subfolder"])
                sentiment, confidence = predict(comment, classifier, config["label_map"])
```

- [ ] **Step 3: Vérifier syntaxe**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): remplace st.radio par 2 cards cliquables avec metadata"
```

---

## Task 5: Ajouter la barre de confiance sous le résultat

**Files:**
- Modify: `app.py` (zone d'affichage du résultat de prédiction)

- [ ] **Step 1: Trouver la zone**

```bash
grep -n 'class="result-box' app.py
```

- [ ] **Step 2: Modifier l'affichage du résultat**

Trouver :

```python
            css_class = {"positif": "result-pos", "négatif": "result-neg", "neutre": "result-neu"}.get(sentiment, "result-neu")
            icon = {"positif": "😊", "négatif": "😞", "neutre": "😐"}.get(sentiment, "😐")

            st.markdown(f'<div class="result-box {css_class}">{icon} <span>{sentiment.upper()}</span> <span style="opacity:0.6;font-size:0.9rem;margin-left:auto">{confidence}% confiance</span></div>', unsafe_allow_html=True)
```

Le remplacer par :

```python
            css_class = {"positif": "result-pos", "négatif": "result-neg", "neutre": "result-neu"}.get(sentiment, "result-neu")
            icon = {"positif": "😊", "négatif": "😞", "neutre": "😐"}.get(sentiment, "😐")
            bar_gradient = {
                "positif": "linear-gradient(90deg, #34d399, #10b981)",
                "négatif": "linear-gradient(90deg, #f87171, #ef4444)",
                "neutre":  "linear-gradient(90deg, #60a5fa, #3b82f6)",
            }.get(sentiment, "linear-gradient(90deg, #60a5fa, #3b82f6)")

            st.markdown(f'<div class="result-box {css_class}">{icon} <span>{sentiment.upper()}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:{confidence}%;background:{bar_gradient}"></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar-label">{confidence}% de confiance</div>', unsafe_allow_html=True)
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): barre de confiance graphique sous le résultat"
```

---

## Task 6: Enrichir l'historique avec icône + bordure colorée par modèle

**Files:**
- Modify: `app.py` (boucle d'affichage de `new_comments`)

- [ ] **Step 1: Trouver la zone**

```bash
grep -n "model_short" app.py
```

- [ ] **Step 2: Remplacer la boucle**

Trouver :

```python
        for item in reversed(st.session_state.new_comments[-8:]):
            badge_class = {"positif": "badge-pos", "négatif": "badge-neg", "neutre": "badge-neu"}.get(item["sentiment"], "badge-neu")
            model_short = "CamemBERT" if "CamemBERT (original)" in item.get("model", "") else "Augmenté"
            st.markdown(f'''
            <div class="comment-item">
                <span style="color:#d1d5db;flex:1">{item["comment"]}</span>
                <span style="color:#6b7280;font-size:0.7rem;margin:0 0.5rem">{model_short}</span>
                <span style="color:#4b5563;font-size:0.75rem;margin:0 0.8rem">{item["time"]}</span>
                <span class="badge {badge_class}">{item["sentiment"]}</span>
            </div>''', unsafe_allow_html=True)
```

Le remplacer par :

```python
        for item in reversed(st.session_state.new_comments[-8:]):
            badge_class = {"positif": "badge-pos", "négatif": "badge-neg", "neutre": "badge-neu"}.get(item["sentiment"], "badge-neu")
            item_model = item.get("model", "")
            if "augmenté" in item_model.lower():
                model_class = "comment-item-augmented"
                model_icon = "✨"
            else:
                model_class = "comment-item-original"
                model_icon = "📜"
            st.markdown(f'''
            <div class="comment-item {model_class}">
                <span class="history-model-icon">{model_icon}</span>
                <span style="color:#d1d5db;flex:1">{item["comment"]}</span>
                <span style="color:#4b5563;font-size:0.75rem;margin:0 0.8rem">{item["time"]}</span>
                <span class="badge {badge_class}">{item["sentiment"]}</span>
            </div>''', unsafe_allow_html=True)
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): historique avec icône modèle et bordure colorée gauche"
```

---

## Task 7: Enrichir le footer avec liens HF + source

**Files:**
- Modify: `app.py` (footer en bas du fichier)

- [ ] **Step 1: Trouver la zone**

```bash
grep -n "YNOV SENTIMENT ANALYSER" app.py
```

- [ ] **Step 2: Remplacer le footer**

Trouver :

```python
# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#374151;font-size:0.75rem;letter-spacing:0.1em">YNOV SENTIMENT ANALYSER · CamemBERT original × augmenté · 2026</div>', unsafe_allow_html=True)
```

Le remplacer par :

```python
# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('''
<div class="footer-links">
    <a href="https://huggingface.co/Ahmat293/camembert-sentiment-ynov" target="_blank">📜 Modèle original</a>·
    <a href="https://huggingface.co/Ahmat293/camembert-ynov-augmented" target="_blank">✨ Modèle augmenté</a>·
    <a href="https://github.com/aadamgk/camembert-sentiment-app" target="_blank">⌨ Code source</a>
</div>
<div style="text-align:center;color:#374151;font-size:0.7rem;letter-spacing:0.1em;margin-top:0.5rem">
    YNOV SENTIMENT ANALYSER · CamemBERT original × augmenté · 2026
</div>
''', unsafe_allow_html=True)
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): footer enrichi avec liens vers les modèles HF et le code source"
```

---

## Task 8: Test visuel manuel de l'app

**Files:** aucun (test utilisateur)

- [ ] **Step 1: Lancer l'app**

```bash
streamlit run app.py
```

(Ou `python -m streamlit run app.py` si `streamlit` n'est pas dans le PATH.)

- [ ] **Step 2: Checklist visuelle**

Vérifier dans le navigateur :

- ☐ Les 2 cards modèles s'affichent côte-à-côte
- ☐ Chaque card montre : emoji, nom, sous-titre (base_model · classes), 2 badges (Accuracy/F1), description, lien HF
- ☐ La card par défaut sélectionnée est "CamemBERT (augmenté)" (bordure violet brillante + ✓)
- ☐ Cliquer sur "CamemBERT (original)" change la sélection visuellement (bordure se déplace, ✓ bouge)
- ☐ Hover sur une card non-sélectionnée la fait remonter de quelques pixels
- ☐ Le bouton "Analyser le sentiment" est toujours fonctionnel
- ☐ Après prédiction : la barre de confiance apparaît avec la bonne couleur (vert/rouge/bleu selon sentiment)
- ☐ La largeur de la barre correspond bien au % affiché
- ☐ Dans l'historique : chaque entrée a son icône (📜 ou ✨) et une bordure gauche colorée différente
- ☐ Footer : 3 liens cliquables (modèle original, modèle augmenté, code source) qui ouvrent dans un nouvel onglet

- [ ] **Step 3: Test fonctionnel**

Tester avec le commentaire "Excellent campus, je recommande à 100% !" :
- Sélectionner "CamemBERT (original)" → cliquer Analyser → vérifier que le résultat est cohérent
- Sélectionner "CamemBERT (augmenté)" → cliquer Analyser → vérifier que le résultat est cohérent
- Vérifier que les 2 entrées apparaissent dans l'historique avec leurs icônes respectives

- [ ] **Step 4: Commit final**

```bash
git commit --allow-empty -m "test: validation visuelle UI polish réussie"
```

---

## Self-review

**1. Spec coverage**

| Spec § | Couverture |
|---|---|
| §3 Inclus — cards cliquables | Task 4 |
| §3 Inclus — métadonnées par card | Task 1 (data) + Task 4 (rendu) |
| §3 Inclus — état sélectionné visuel | Task 2 (CSS) + Task 4 (logique) |
| §3 Inclus — état hover | Task 2 (CSS `:hover`) |
| §3 Inclus — barre de confiance | Task 5 |
| §3 Inclus — distinction historique | Task 6 |
| §3 Inclus — footer enrichi | Task 7 |
| §3 Inclus — `MODELS` enrichi | Task 1 |
| §4 Architecture flux | Tasks 3-4 (session_state + columns + button overlay) |
| §5 `MODELS` dict | Task 1 (correspondance exacte) |
| §6 CSS | Task 2 (toutes les classes définies) |
| §7 Barre de confiance | Task 5 (gradients selon sentiment) |
| §8 Historique enrichi | Task 6 (icône + bordure) |
| §9 Footer enrichi | Task 7 |
| §10 Validation finale | Task 8 (checklist) |
| §11 Décisions verrouillées | Implicite dans toutes les tasks |

**2. Placeholder scan** : aucun "TBD". Toutes les commandes et tous les codes sont complets et copiables.

**3. Type consistency** :
- `MODELS` dict utilisé identique en Task 1 et Task 4 ✓
- `st.session_state.selected_model` cohérent en Task 3 et Task 4 ✓
- Classes CSS référencées en Task 4-6 sont toutes définies en Task 2 ✓
- `model_class` (`comment-item-augmented`/`-original`) en Task 6 correspond au CSS de Task 2 ✓

**Aucun problème détecté.**
