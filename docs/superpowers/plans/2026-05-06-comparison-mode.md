# Plan d'implémentation — Mode comparaison directe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter une 3e card "Comparer les deux modèles" qui exécute les 2 modèles et affiche un verdict d'accord/désaccord avec les 2 résultats côte-à-côte.

**Architecture:** Modifications sur `app.py` uniquement. Ajout d'une 3e clé dans `MODELS` avec flag `is_comparison`, deux nouvelles fonctions (`predict_dual`, `render_dual_result`), branchement conditionnel dans le flux de prédiction, libellé du bouton dynamique.

**Tech Stack:** Streamlit (st.columns, st.markdown HTML), CSS inline (déjà le pattern).

**Spec source:** `docs/superpowers/specs/2026-05-06-comparison-mode-design.md`

---

## File Structure

**Modifié uniquement :** `app.py`

Zones impactées :
1. Bloc `MODELS` — ajout d'une 3e clé "Comparer les deux"
2. Bloc CSS — ajout des styles `.comparison-card`, `.verdict-banner`, etc.
3. Fonction `render_model_card` — gérer le rendu spécial de la card comparaison
4. Nouvelle fonction `predict_dual()` — exécute les 2 modèles + calcule verdict
5. Nouvelle fonction `render_dual_result()` — affiche verdict + 2 colonnes résultats
6. Section "Analyser un commentaire" — branchement conditionnel selon `is_comparison`

---

## Task 1: Ajouter le CSS pour la 3e card et le verdict banner

**Files:**
- Modify: `app.py` (bloc CSS existant)

- [ ] **Step 1: Repérer la fin du bloc CSS des cards modèle**

```bash
grep -n "footer-links a:hover" app.py
```

Note la ligne — c'est la dernière règle CSS du bloc à étendre. On va insérer juste après le bloc `</style>` du dernier `st.markdown` qui contient les classes `.model-card`.

- [ ] **Step 2: Insérer les nouvelles règles CSS**

Trouver dans `app.py` (à l'intérieur du bloc `<style>` qui contient `.model-card`, juste avant `</style>` qui ferme ce bloc, soit juste après la règle `.footer-links a:hover { color: #a78bfa; }`) :

```css
.footer-links a:hover {
    color: #a78bfa;
}
</style>
""", unsafe_allow_html=True)
```

Le remplacer par :

```css
.footer-links a:hover {
    color: #a78bfa;
}

/* ─── Card de comparaison (3e card) ─── */
.model-card.comparison-card {
    border: 2px dashed rgba(96,165,250,0.3);
    background: linear-gradient(135deg, rgba(96,165,250,0.05), rgba(52,211,153,0.05));
}

.model-card.comparison-card:hover {
    border-color: rgba(96,165,250,0.5);
    background: linear-gradient(135deg, rgba(96,165,250,0.10), rgba(52,211,153,0.08));
    transform: translateY(-3px);
}

.model-card.comparison-card.selected {
    border: 2px solid #60a5fa;
    background: linear-gradient(135deg, rgba(96,165,250,0.15), rgba(52,211,153,0.10));
    box-shadow: 0 0 0 1px #60a5fa, 0 12px 40px rgba(96,165,250,0.35);
}

.model-card.comparison-card.selected::before {
    background: linear-gradient(90deg, #60a5fa, #34d399, #a78bfa);
}

.model-card.comparison-card.selected::after {
    content: "✓ MODE DUAL";
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    box-shadow: 0 2px 8px rgba(96,165,250,0.5);
}

.comparison-badge-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin: 1rem 0;
    flex: 1;
}

.comparison-vs-icon {
    font-size: 2.2rem;
}

/* ─── Verdict banner ─── */
.verdict-banner {
    border-radius: 12px;
    padding: 0.9rem 1.4rem;
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    line-height: 1.4;
}

.verdict-accord {
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.4);
    color: #34d399;
}

.verdict-partiel {
    background: rgba(251,191,36,0.12);
    border: 1px solid rgba(251,191,36,0.4);
    color: #fbbf24;
}

.verdict-desaccord {
    background: rgba(248,113,113,0.12);
    border: 1px solid rgba(248,113,113,0.4);
    color: #f87171;
}

/* ─── Mini-titre dans chaque colonne du dual ─── */
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

- [ ] **Step 3: Vérifier la syntaxe**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): CSS pour la 3e card de comparaison + verdict banner"
```

---

## Task 2: Ajouter la 3e clé `Comparer les deux` dans `MODELS`

**Files:**
- Modify: `app.py` (bloc `MODELS = {...}`)

- [ ] **Step 1: Repérer la fin du bloc MODELS**

```bash
grep -n '"hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",' app.py
```

- [ ] **Step 2: Ajouter la 3e clé**

Trouver dans `app.py` :

```python
        "description": "Fine-tuné sur dataset augmenté (394 avis, binaire pos/neg)",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",
    },
}
```

Le remplacer par :

```python
        "description": "Fine-tuné sur dataset augmenté (394 avis, binaire pos/neg)",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",
    },
    "Comparer les deux": {
        "is_comparison": True,
        "emoji": "🆚",
        "subtitle": "Mode dual",
        "description": "Voir les 2 verdicts + accord/désaccord en un clic",
    },
}
```

- [ ] **Step 3: Vérifier la syntaxe**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): ajouter 3e clé MODELS pour le mode comparaison"
```

---

## Task 3: Adapter `render_model_card` pour gérer la card de comparaison

**Files:**
- Modify: `app.py` (fonction `render_model_card`)

- [ ] **Step 1: Repérer la fonction**

```bash
grep -n "def render_model_card" app.py
```

- [ ] **Step 2: Remplacer la fonction entière**

Trouver dans `app.py` :

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
```

Le remplacer par :

```python
def render_model_card(model_name, is_selected):
    cfg = MODELS[model_name]
    selected_class = "selected" if is_selected else ""
    is_comparison = cfg.get("is_comparison", False)

    if is_comparison:
        return f"""
        <div class="model-card comparison-card {selected_class}">
            <div class="model-card-header">
                <span class="model-card-emoji">{cfg['emoji']}</span>
                <span class="model-card-title">{model_name}</span>
            </div>
            <div class="model-card-subtitle">{cfg['subtitle']}</div>
            <div class="comparison-badge-row">
                <span class="comparison-vs-icon">⚖️</span>
            </div>
            <div class="model-card-desc">{cfg['description']}</div>
        </div>
        """

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
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): render_model_card gère le mode comparaison (rendu différent)"
```

---

## Task 4: Étendre `st.columns(2)` à 3 colonnes pour afficher la 3e card

**Files:**
- Modify: `app.py` (zone `card_cols = st.columns(2)`)

- [ ] **Step 1: Repérer la zone**

```bash
grep -n "card_cols = st.columns" app.py
```

- [ ] **Step 2: Remplacer**

Trouver dans `app.py` :

```python
    # ─── Sélecteur de modèle en cards ─────────────────────────────────────
    st.markdown('<div class="model-selector-row">', unsafe_allow_html=True)
    card_cols = st.columns(2)
    model_names = list(MODELS.keys())
    for col, name in zip(card_cols, model_names):
```

Le remplacer par :

```python
    # ─── Sélecteur de modèle en cards ─────────────────────────────────────
    st.markdown('<div class="model-selector-row">', unsafe_allow_html=True)
    model_names = list(MODELS.keys())
    card_cols = st.columns(len(model_names))
    for col, name in zip(card_cols, model_names):
```

- [ ] **Step 3: Mise à jour du CSS overlay button (height ajusté pour les 3 cards de hauteur 290px)**

Le CSS du bouton overlay est déjà calé sur `height: 290px`, donc ça reste compatible. Aucune modif nécessaire ici.

- [ ] **Step 4: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): sélecteur étendu à N colonnes (3 cards visibles)"
```

---

## Task 5: Ajouter `predict_dual` et `render_dual_result`

**Files:**
- Modify: `app.py` (juste après la fonction `predict`)

- [ ] **Step 1: Repérer la zone**

```bash
grep -n "^def predict" app.py
```

Repérer la fin de la fonction `predict` (la dernière ligne est `return label_map.get(...) ..., round(...)`).

- [ ] **Step 2: Insérer les 2 nouvelles fonctions juste après `predict`**

Trouver dans `app.py` :

```python
def predict(text, classifier, label_map):
    result = classifier(text[:512])[0]
    raw_label = result["label"]
    return label_map.get(raw_label, raw_label), round(result["score"] * 100, 1)
```

Le remplacer par :

```python
def predict(text, classifier, label_map):
    result = classifier(text[:512])[0]
    raw_label = result["label"]
    return label_map.get(raw_label, raw_label), round(result["score"] * 100, 1)

def predict_dual(text):
    """Exécute les 2 modèles réels et calcule le verdict d'accord/désaccord."""
    real_models = [name for name, cfg in MODELS.items() if not cfg.get("is_comparison", False)]
    results = {}
    for name in real_models:
        cfg = MODELS[name]
        clf = load_model(cfg["id"], cfg["subfolder"], cfg["tokenizer_subfolder"])
        sentiment, confidence = predict(text, clf, cfg["label_map"])
        results[name] = {
            "sentiment": sentiment,
            "confidence": confidence,
            "model_id": cfg["id"],
        }
    s_orig = results["CamemBERT (original)"]["sentiment"]
    s_aug = results["CamemBERT (augmenté)"]["sentiment"]
    if s_orig == s_aug:
        verdict = "accord"
        verdict_text = f"🟢 Les 2 modèles sont d'accord ({s_orig})"
    elif s_orig == "neutre":
        verdict = "partiel"
        verdict_text = f"🟡 L'augmenté tranche ({s_aug}), l'original reste neutre"
    else:
        verdict = "desaccord"
        verdict_text = f"🟡 Désaccord — Original: {s_orig} · Augmenté: {s_aug}"
    return results, verdict, verdict_text

def render_dual_result(results, verdict, verdict_text):
    """Affiche le verdict banner puis les 2 résultats côte-à-côte."""
    st.markdown(
        f'<div class="verdict-banner verdict-{verdict}">{verdict_text}</div>',
        unsafe_allow_html=True
    )
    cols = st.columns(2)
    for col, (model_name, r) in zip(cols, results.items()):
        with col:
            icon = "📜" if "original" in model_name.lower() else "✨"
            short_name = "Original" if "original" in model_name.lower() else "Augmenté"
            sentiment = r["sentiment"]
            confidence = r["confidence"]
            css_class = {"positif": "result-pos", "négatif": "result-neg", "neutre": "result-neu"}.get(sentiment, "result-neu")
            sentiment_icon = {"positif": "😊", "négatif": "😞", "neutre": "😐"}.get(sentiment, "😐")
            bar_gradient = {
                "positif": "linear-gradient(90deg, #34d399, #10b981)",
                "négatif": "linear-gradient(90deg, #f87171, #ef4444)",
                "neutre":  "linear-gradient(90deg, #60a5fa, #3b82f6)",
            }.get(sentiment, "linear-gradient(90deg, #60a5fa, #3b82f6)")
            st.markdown(f'<div class="dual-model-label">{icon} {short_name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box {css_class}">{sentiment_icon} <span>{sentiment.upper()}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:{confidence}%;background:{bar_gradient}"></div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-bar-label">{confidence}% de confiance</div>', unsafe_allow_html=True)
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): predict_dual + render_dual_result pour le mode comparaison"
```

---

## Task 6: Brancher le mode comparaison dans le flux principal

**Files:**
- Modify: `app.py` (zone `if st.button("Analyser le sentiment", key="analyze_btn"):`)

- [ ] **Step 1: Repérer la zone**

```bash
grep -n 'st.button("Analyser le sentiment"' app.py
```

- [ ] **Step 2: Remplacer le bloc complet**

Trouver dans `app.py` :

```python
    selected_model_name = st.session_state.selected_model

    comment = st.text_area("", placeholder="Entrez un avis étudiant...", height=130, label_visibility="collapsed")

    if st.button("Analyser le sentiment", key="analyze_btn"):
        if comment.strip():
            with st.spinner("Analyse en cours..."):
                config = MODELS[selected_model_name]
                classifier = load_model(config["id"], config["subfolder"], config["tokenizer_subfolder"])
                sentiment, confidence = predict(comment, classifier, config["label_map"])

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
        else:
            st.warning("Entrez un commentaire d'abord.")
```

Le remplacer par :

```python
    selected_model_name = st.session_state.selected_model
    selected_config = MODELS[selected_model_name]
    is_comparison = selected_config.get("is_comparison", False)
    button_label = "Comparer les modèles" if is_comparison else "Analyser le sentiment"

    comment = st.text_area("", placeholder="Entrez un avis étudiant...", height=130, label_visibility="collapsed")

    if st.button(button_label, key="analyze_btn"):
        if comment.strip():
            if is_comparison:
                with st.spinner("Comparaison des 2 modèles..."):
                    results, verdict, verdict_text = predict_dual(comment)

                render_dual_result(results, verdict, verdict_text)

                # Persist + history pour les 2 modèles
                for model_name, r in results.items():
                    short_comment = comment[:60] + "..." if len(comment) > 60 else comment
                    st.session_state.new_comments.append({
                        "comment": short_comment,
                        "sentiment": r["sentiment"],
                        "confidence": r["confidence"],
                        "time": datetime.now().strftime("%H:%M"),
                        "model": model_name,
                    })
                    save_prediction(
                        comment=comment,
                        sentiment=r["sentiment"],
                        confidence=r["confidence"],
                        model_name=model_name,
                        model_id=r["model_id"],
                    )
                fetch_global_history.clear()
            else:
                with st.spinner("Analyse en cours..."):
                    classifier = load_model(selected_config["id"], selected_config["subfolder"], selected_config["tokenizer_subfolder"])
                    sentiment, confidence = predict(comment, classifier, selected_config["label_map"])

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

                st.session_state.new_comments.append({
                    "comment": comment[:60] + "..." if len(comment) > 60 else comment,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "time": datetime.now().strftime("%H:%M"),
                    "model": selected_model_name,
                })

                save_prediction(
                    comment=comment,
                    sentiment=sentiment,
                    confidence=confidence,
                    model_name=selected_model_name,
                    model_id=selected_config["id"],
                )
                fetch_global_history.clear()
        else:
            st.warning("Entrez un commentaire d'abord.")
```

- [ ] **Step 3: Vérifier syntaxe + commit**

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
git add app.py
git commit -m "feat(app): branchement mode comparaison (predict_dual + persistance x2)"
```

---

## Task 7: Test fonctionnel manuel

**Files:** aucun (test utilisateur)

- [ ] **Step 1: Lancer/refresher l'app**

```bash
streamlit run app.py
```

(Ou Ctrl+R sur la page si l'app tourne déjà.)

- [ ] **Step 2: Vérifier visuellement**

- ☐ 3 cards alignées : Original / Augmenté / **🆚 Comparer les deux**
- ☐ La 3e card a un design distinct (border dashed cyan/teal au lieu du violet uni)
- ☐ Cliquer la 3e card change son état "selected" (badge "MODE DUAL", gradient tricolore)
- ☐ Le libellé du bouton devient **"Comparer les modèles"**
- ☐ Re-cliquer une autre card revient à "Analyser le sentiment"

- [ ] **Step 3: Test fonctionnel (3 cas)**

Sélectionner "Comparer les deux" puis tester ces 3 commentaires :

**Cas 1 — Accord attendu (positif net)**
Saisir : `Excellent campus, je recommande à 100% !`
Cliquer "Comparer les modèles"
- ☐ Spinner "Comparaison des 2 modèles..."
- ☐ Verdict 🟢 vert "Les 2 modèles sont d'accord (positif)"
- ☐ 2 colonnes avec 2 boîtes vertes positives + barres de confiance
- ☐ Historique de session : 2 nouvelles entrées (📜 + ✨)

**Cas 2 — Désaccord franc (négatif net)**
Saisir : `Cette école est une catastrophe, fuyez !`
Cliquer "Comparer les modèles"
- ☐ Verdict 🟢 vert "Les 2 modèles sont d'accord (négatif)" — les 2 modèles devraient être d'accord ici aussi sur un cas extrême
- ☐ Si désaccord apparait : verdict 🟡

**Cas 3 — Désaccord probable (ambigu)**
Saisir : `Le campus est correct, sans plus.`
Cliquer "Comparer les modèles"
- ☐ Possible verdict 🟡 partiel : original = "neutre" ou "positif", augmenté = pos/nég
- ☐ Le verdict text indique précisément le décalage

- [ ] **Step 4: Vérifier la persistance Supabase**

Aller sur https://supabase.com/dashboard/project/edpkrgdlxpdtvrwbkahh/editor
- ☐ Table `predictions` : 6 nouvelles lignes (3 cas × 2 modèles)
- ☐ Pour chaque cas : 2 lignes différentes par `model_name`
- ☐ Le commentaire est identique entre les 2 lignes du même cas

- [ ] **Step 5: Vérifier l'historique global**

Dans l'app, scroller jusqu'à "🌐 Historique global · Supabase"
- ☐ Cliquer "↻ Rafraîchir"
- ☐ Les 6 nouvelles entrées apparaissent (ou les plus récentes parmi les 20)

- [ ] **Step 6: Vérifier la non-régression du mode unique**

Sélectionner "CamemBERT (augmenté)" (mode classique)
- ☐ Le bouton redevient "Analyser le sentiment"
- ☐ Saisir un commentaire et cliquer
- ☐ Comportement identique à avant : 1 résultat, 1 ligne Supabase, 1 entrée historique

- [ ] **Step 7: Commit final**

```bash
git commit --allow-empty -m "test: mode comparaison validé end-to-end"
```

---

## Self-review

**1. Spec coverage**

| Spec § | Couverture |
|---|---|
| §2 Critère 1 (3e card alignée) | Task 4 (st.columns dynamique) + Task 1 (CSS) |
| §2 Critère 2 (libellé bouton dynamique) | Task 6 (button_label) |
| §2 Critère 3 (les 2 modèles tournent) | Task 5 (predict_dual) |
| §2 Critère 4 (verdict accord/désaccord) | Task 5 (logique) + Task 1 (CSS classes) |
| §2 Critère 5 (2 colonnes équilibrées) | Task 5 (render_dual_result avec st.columns) |
| §2 Critère 6 (2 inserts Supabase) | Task 6 (boucle for + save_prediction) |
| §2 Critère 7 (2 entrées historique) | Task 6 (boucle for + append) |
| §2 Critère 8 (non-régression mode unique) | Task 6 (else branch) + Task 7 step 6 (validation) |
| §4 architecture flux | Task 5 + Task 6 |
| §4 logique du verdict 3 niveaux | Task 5 (compute) |
| §5 spec UI 3e card | Task 3 (rendu spécial) + Task 1 (CSS) |
| §5 verdict banner | Task 1 (CSS) + Task 5 (render) |
| §5 affichage dual | Task 5 (render_dual_result) |
| §5 bouton dynamique | Task 6 (button_label) |
| §6 specs techniques | Tasks 2, 3, 5, 6 |
| §7 CSS | Task 1 |
| §8 validation finale | Task 7 |
| §9 décisions verrouillées | Toutes appliquées |

**2. Placeholder scan** : aucun "TBD". Toutes les commandes shell et tous les blocs Python sont complets.

**3. Type consistency** :
- `predict_dual` retourne `(results, verdict, verdict_text)` — utilisé dans Task 6 avec ce même unpacking ✓
- `results` dict structure cohérente entre Task 5 (création) et Task 6 (consommation) ✓
- Classes CSS référencées dans `render_dual_result` (Task 5) toutes définies dans Task 1 ✓
- `is_comparison` flag : ajouté en Task 2, consommé en Task 3 et Task 6 ✓
- `MODELS["Comparer les deux"]` keys (`emoji`, `subtitle`, `description`) cohérentes entre Task 2 (définition) et Task 3 (consommation) ✓

**Aucun problème détecté.**
