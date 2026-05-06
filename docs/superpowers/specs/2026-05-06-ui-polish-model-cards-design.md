# UI Polish — Cards comparatives pour le sélecteur de modèle

**Date** : 2026-05-06
**Auteur** : Ahmat (collab. Claude)
**Statut** : Design validé, en attente de relecture utilisateur
**Spec source** : `docs/superpowers/specs/2026-05-06-finetuning-distilcamembert-design.md` (le sélecteur radio défini par cette spec est remplacé par des cards)

## 1. Contexte et motivation

Le sélecteur de modèle actuel est un `st.radio` horizontal — fonctionnel mais trop "basique" comparé au reste du design (gradients violet/bleu/vert, glassmorphism dark, polices Syne/DM Sans). L'utilisateur ne voit aucune information sur les modèles (uniquement les noms). Pour comparer les deux modèles efficacement, il manque visiblement les métriques (accuracy, F1), le modèle de base, le nombre de classes, le lien vers le repo HF.

L'objectif est de remplacer ce radio par **deux cards comparatives côte-à-côte** affichant ces métadonnées, et d'ajouter quelques touches finales sur le reste de l'interface (barre de confiance dans la zone de résultat, distinction visuelle dans l'historique).

## 2. Objectif et critères de succès

### Objectif
Remplacer le sélecteur radio par deux cards cliquables avec métadonnées visibles, et apporter des améliorations cohérentes au reste de l'UI.

### Critères de succès
1. Les deux cards sont affichées côte-à-côte, visuellement équilibrées.
2. L'utilisateur comprend en 3 secondes les différences entre les deux modèles (modèle de base, classes, performances).
3. La sélection d'une card est instantanée (clic = sélection visible).
4. Les prédictions, l'historique et les stats du dataset continuent de fonctionner sans régression.
5. Le design est cohérent avec le reste de l'app (mêmes polices, gradients, glassmorphism).
6. `streamlit run app.py` démarre sans erreur, syntaxe Python valide.

## 3. Scope

### Inclus
- Remplacement du `st.radio` par des cards cliquables (deux colonnes Streamlit)
- Affichage par card : nom, modèle de base, nombre de classes, accuracy, F1 macro, lien HF, description courte
- État sélectionné visuellement distinct (bordure violet, gradient subtil, checkmark)
- État hover (`translateY(-2px)` + bordure violet plus claire)
- Barre de confiance graphique sous le résultat de prédiction
- Distinction visuelle dans l'historique (icône + couleur différente par modèle)
- Liens HF + dataset dans le footer
- Mise à jour du `MODELS` dict avec les métadonnées nécessaires

### Exclus (futurs travaux)
- Side-by-side prediction (les deux modèles en même temps) — décidé toggle plutôt
- Dark/light mode switch
- Animations complexes (transitions Lottie, etc.)
- Réorganisation profonde de l'app (sections restent dans cet ordre : header → dataset → analyse → historique → footer)
- Ajout de nouveaux modèles (architecture extensible mais on n'en ajoute pas maintenant)

## 4. Architecture et flux

### Composant : sélecteur de cards

```
┌──────────────────────────────────────────────────────────────────┐
│  ┌────────────────────────────┐  ┌────────────────────────────┐  │
│  │ 🇫🇷 CamemBERT (original) ✓ │  │ 🇫🇷 CamemBERT (augmenté)  │  │
│  │ camembert-base · 3 classes │  │ camembert-base · 2 classes │  │
│  │                            │  │                            │  │
│  │ ┌──────────┐ ┌───────────┐ │  │ ┌──────────┐ ┌───────────┐ │  │
│  │ │ Acc 78%  │ │ F1 0.78   │ │  │ │ Acc 98%  │ │ F1 0.98   │ │  │
│  │ └──────────┘ └───────────┘ │  │ └──────────┘ └───────────┘ │  │
│  │                            │  │                            │  │
│  │ Modèle initial entraîné    │  │ Fine-tuné sur dataset      │  │
│  │ sur dataset original (550) │  │ augmenté binaire (394)     │  │
│  │                            │  │                            │  │
│  │ → repo HF                  │  │ → repo HF                  │  │
│  └────────────────────────────┘  └────────────────────────────┘  │
│        SÉLECTIONNÉ (border violet brillante + glow + ✓)          │
└──────────────────────────────────────────────────────────────────┘
```

### Flux d'interaction

1. Au chargement de l'app : par défaut le modèle "augmenté" est sélectionné (le nouveau qu'on vient d'entraîner).
2. L'utilisateur clique sur une card → `st.session_state.selected_model` est mis à jour → re-render → la card sélectionnée prend le style "selected" → la card non-sélectionnée revient à l'état "default".
3. Au clic sur "Analyser le sentiment" → `MODELS[st.session_state.selected_model]` est utilisé pour la prédiction.

### Implémentation Streamlit

Streamlit n'a pas de "div cliquable" native. Stratégie retenue :

1. **Layout** : `st.columns(2)` pour les deux cards.
2. **Rendu visuel** : `st.markdown(html, unsafe_allow_html=True)` pour le HTML/CSS de chaque card (cohérent avec le reste de l'app qui utilise déjà `unsafe_allow_html=True`).
3. **Click handler** : un `st.button` plein-largeur sous chaque card, stylé en transparent (CSS `opacity: 0` + `position: absolute`) pour qu'il "prenne" le clic sur la card.
4. **État** : `st.session_state.selected_model` (string), initialisé à "CamemBERT (augmenté)" si absent.

Ce pattern (card visuelle + bouton transparent) est un compromis classique en Streamlit pour avoir des cards cliquables.

## 5. Spécifications du `MODELS` dict enrichi

```python
MODELS = {
    "CamemBERT (original)": {
        "id": "Ahmat293/camembert-sentiment-ynov",
        "subfolder": "model",
        "tokenizer_subfolder": "tokenizer",
        "label_map": {"LABEL_0": "négatif", "LABEL_1": "neutre", "LABEL_2": "positif"},
        "emoji": "🇫🇷",
        "base_model": "camembert-base",
        "num_classes": 3,
        "accuracy": None,            # inconnu (modèle entraîné par l'utilisateur 2 mois ago)
        "f1_macro": None,            # idem
        "description": "Modèle initial entraîné sur dataset original (550 avis, 3 classes)",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-sentiment-ynov",
    },
    "CamemBERT (augmenté)": {
        "id": "Ahmat293/camembert-ynov-augmented",
        "subfolder": None,
        "tokenizer_subfolder": None,
        "label_map": {"negatif": "négatif", "positif": "positif"},
        "emoji": "🇫🇷",
        "base_model": "camembert-base",
        "num_classes": 2,
        "accuracy": 0.9833,
        "f1_macro": 0.9833,
        "description": "Fine-tuné sur dataset augmenté (394 avis, binaire pos/neg)",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",
    },
}
```

Note : les valeurs `accuracy` et `f1_macro` du modèle original sont **inconnues** (modèle entraîné il y a 2 mois sans rapport métrique conservé). Sur les cards, on affiche "—" plutôt qu'un nombre inventé. Si l'utilisateur veut les afficher, il devra ré-évaluer le modèle séparément. Pour le modèle augmenté, on a les valeurs exactes du training (98.3% / 0.9833).

## 6. Spécifications CSS

À ajouter dans le bloc `<style>` existant en haut de `app.py` :

### Card de modèle (état par défaut)
```css
.model-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    position: relative;
    cursor: pointer;
    height: 100%;
}

.model-card:hover {
    border-color: rgba(167,139,250,0.4);
    background: rgba(167,139,250,0.06);
    transform: translateY(-2px);
}
```

### Card sélectionnée
```css
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
    font-size: 1.2rem;
    font-weight: 700;
}
```

### Sous-éléments (titre, sous-titre, badges, description, lien)
```css
.model-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f0f0f5;
    margin-bottom: 0.2rem;
}

.model-card-subtitle {
    font-size: 0.8rem;
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
    padding: 0.4rem 0.7rem;
    font-size: 0.75rem;
    text-align: center;
    flex: 1;
}

.model-badge-label {
    color: #6b7280;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.model-badge-value {
    color: #f0f0f5;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
}

.model-card-desc {
    font-size: 0.8rem;
    color: #9ca3af;
    line-height: 1.4;
    margin-bottom: 0.8rem;
}

.model-card-link {
    color: #a78bfa;
    font-size: 0.75rem;
    text-decoration: none;
}

.model-card-link:hover { color: #c4b5fd; }
```

### Bouton transparent overlay
```css
/* Bouton Streamlit dans la zone des cards : transparent et plein écran */
div[data-testid="column"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    padding: 0 !important;
    margin: 0 !important;
    z-index: 10;
    cursor: pointer;
}

div[data-testid="column"] .stButton {
    margin-top: -100%;  /* superpose le bouton sur la card */
    height: 100%;
}
```

## 7. Spécifications barre de confiance

Sous le bloc `result-box` actuel, ajouter :

```html
<div class="confidence-bar-track">
    <div class="confidence-bar-fill" style="width: {confidence}%; background: {gradient_color}"></div>
</div>
<div class="confidence-bar-label">{confidence}% de confiance</div>
```

```css
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
```

Gradient selon sentiment :
- positif → `linear-gradient(90deg, #34d399, #10b981)`
- négatif → `linear-gradient(90deg, #f87171, #ef4444)`
- neutre → `linear-gradient(90deg, #60a5fa, #3b82f6)`

## 8. Spécifications historique enrichi

Modifier le rendu de chaque entrée d'historique pour ajouter :
- Une icône modèle (`📜` pour original, `✨` pour augmenté)
- Une bordure gauche colorée (violet pour augmenté, gris pour original) pour repérer rapidement quel modèle a fait la prédiction

## 9. Footer enrichi

Avant le copyright, ajouter une ligne avec liens :

```html
<div class="footer-links">
    <a href="https://huggingface.co/Ahmat293/camembert-sentiment-ynov">Modèle original</a> ·
    <a href="https://huggingface.co/Ahmat293/camembert-ynov-augmented">Modèle augmenté</a> ·
    <a href="https://github.com/aadamgk/camembert-sentiment-app">Code source</a>
</div>
```

## 10. Validation finale

1. `streamlit run app.py` lance l'app sans erreur
2. Les deux cards apparaissent côte-à-côte
3. Cliquer sur une card change visuellement la sélection (bordure, glow, ✓)
4. Hover sur une card non-sélectionnée la fait remonter (`translateY`)
5. Cliquer sur "Analyser le sentiment" utilise bien le modèle sélectionné
6. La barre de confiance s'affiche sous le résultat avec la bonne couleur et largeur
7. L'historique distingue visuellement les prédictions des deux modèles
8. Footer avec 3 liens cliquables (deux repos HF + code source)
9. Toutes les sections existantes (dataset, donut, bar chart) restent inchangées
10. Mobile responsive (les colonnes Streamlit gèrent automatiquement le wrap sur petits écrans)

## 11. Décisions verrouillées

- Layout : 2 cards en colonnes égales
- Click handler : `st.button` transparent overlay (pas de composant custom JS)
- Source des stats : hardcodées dans `MODELS` dict (pas de fetch dynamique)
- Modèle par défaut : "CamemBERT (augmenté)"
- Cohérence design : Syne/DM Sans, gradients existants, glassmorphism dark
- Pas de side-by-side prediction (validé en spec précédente)

## 12. Hors scope explicite

- Pas de `streamlit-elements` ou autre composant tiers
- Pas de modification des polices ni de la palette principale
- Pas d'ajout de nouvelles dépendances dans `requirements.txt`
- Pas de tests automatisés (validation visuelle uniquement)
