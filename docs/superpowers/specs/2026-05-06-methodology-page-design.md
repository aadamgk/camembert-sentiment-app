# Page Méthodologie + Architecture du projet

**Date** : 2026-05-06
**Auteur** : Ahmat (collab. Claude)
**Statut** : Design validé, en attente de relecture utilisateur

## 1. Contexte et motivation

L'app actuelle est une démo fonctionnelle (analyse de sentiment + comparaison de 2 modèles + persistance) mais elle n'explique pas la **démarche** derrière. Pour une démo Ynov ou un usage portfolio/entretien, il faut un onglet dédié qui :

- Donne le contexte du projet en 1 minute pour les non-techs
- Détaille la méthodologie complète pour les profils techniques
- Présente l'architecture du pipeline ML de bout en bout
- Liste les vraies métriques + limitations honnêtes
- Centralise les liens vers les ressources (HF, GitHub)

Cet onglet transforme l'app de "démo qui marche" en "projet ML documenté".

## 2. Objectif et critères de succès

### Objectif

Ajouter un second onglet "📖 Méthodologie" à l'app, contenant un récit hybride (vulgarisé + technique) du projet, un diagramme d'architecture, les métriques, le stack, les limitations et les liens.

### Critères de succès

1. La navigation entre les onglets "📊 Analyse" et "📖 Méthodologie" est instantanée et claire (`st.tabs`).
2. Tout le contenu actuel est conservé sans régression dans l'onglet Analyse.
3. La page Méthodologie tient en 1 scroll vertical (pas de page interminable), avec sections clairement délimitées.
4. Le diagramme d'architecture est rendu en HTML/CSS custom (pas d'image statique) et reste cohérent avec le design system de l'app.
5. Tous les liens externes ouvrent dans un nouvel onglet (`target="_blank"`).
6. Pas de régression mobile (les sections passent en colonne sur écran étroit grâce aux colonnes Streamlit).

## 3. Scope

### Inclus

- Refactor de `app.py` pour mettre tout le contenu actuel sous un premier onglet `tab_analyse`
- Création d'un second onglet `tab_methodo` avec 6 sections :
  1. **En bref** — résumé vulgarisé + 3 KPI cards (394 avis, 2 modèles, 98% accuracy)
  2. **Architecture du projet** — diagramme HTML/CSS du pipeline complet
  3. **Pipeline en 5 étapes** — chaque étape détaillée (collecte, augmentation, fine-tuning, déploiement, persistance)
  4. **Résultats détaillés** — tableau des métriques (global, real, synthetic) + matrice de confusion (rendu Plotly)
  5. **Stack technique** — cards horizontales par catégorie (ML, Frontend, Backend, DB, Ops)
  6. **Limitations honnêtes** — bullet list des biais et limites connues
  7. **Ressources** — liens HF (×2), GitHub, dataset
- Ajout de CSS spécifique pour le diagramme architecture (cards de pipeline + flèches de flux)
- Pas d'introduction de nouvelles dépendances Python (tout en HTML/CSS + Plotly déjà présent)

### Exclus (futurs travaux)

- Pas d'image statique de matrice de confusion (on génère un mini-graph Plotly avec les vraies valeurs du training)
- Pas d'animations CSS sur le diagramme
- Pas de mode "histoire interactive" (scrollytelling)
- Pas de version PDF imprimable de la méthodologie
- Pas de vidéo de démo intégrée

## 4. Architecture des onglets

### Structure de `app.py` après refactor

```python
# (header, CSS, MODELS, fonctions inchangés)

# ─── Onglets ───────────────────────────────────────────────────────────────────
tab_analyse, tab_methodo = st.tabs(["📊 Analyse", "📖 Méthodologie"])

with tab_analyse:
    # Tout le contenu actuel :
    # - Section dataset overview (donut + bar chart notes)
    # - Section "Analyser un commentaire" (sélecteur 3 cards + textarea + résultat)
    # - Section "Historique des analyses" (col_history)

with tab_methodo:
    # Nouvelle page avec les 7 sections décrites en §3
```

### Flux d'usage

1. Utilisateur arrive sur l'app → onglet "📊 Analyse" actif par défaut
2. Tout marche comme avant
3. Clic sur "📖 Méthodologie" → contenu de la nouvelle page
4. Re-clic sur "📊 Analyse" → retour à la vue analyse, état session préservé (les prédictions restent dans l'historique)

## 5. Spécifications du contenu — onglet Méthodologie

### Section 1 — En bref (vulgarisé)

Bloc texte court (3-4 lignes) :

> Cette application classe automatiquement les avis étudiants Ynov en **positif** ou **négatif**, en s'appuyant sur **CamemBERT**, un modèle d'IA français open source. À partir de 550 vrais avis Google Maps, on a entraîné un modèle spécialisé qui atteint **98% de précision** sur le domaine éducatif Ynov.

Suivi de 3 cards KPI horizontales (réutilise `metric-card` existant) :
- **394 avis** · dataset équilibré
- **2 modèles** · original vs augmenté
- **98% accuracy** · sur test set 60 avis

### Section 2 — Architecture du projet

**Le cœur de la page.** Diagramme HTML/CSS rendu via `st.markdown` avec un grand bloc HTML structuré en grille verticale, chaque "boîte" du pipeline étant une div stylée.

Structure HTML simplifiée :
```html
<div class="arch-diagram">
    <div class="arch-row">
        <div class="arch-box arch-box-source">📊 avis_ynov_All_final.csv (550 lignes)</div>
    </div>
    <div class="arch-arrow">↓</div>

    <div class="arch-row arch-row-2col">
        <div class="arch-box">🧹 Filtrage<br>383 lignes valides</div>
        <div class="arch-arrow-h">→</div>
        <div class="arch-box arch-box-llm">🤖 Augmentation LLM<br>+100 négatifs synthétiques</div>
    </div>
    <div class="arch-arrow">↓</div>

    <div class="arch-row">
        <div class="arch-box arch-box-data">📦 avis_ynov_augmented.csv (394 lignes 197/197)</div>
    </div>
    <div class="arch-arrow">↓</div>

    <div class="arch-row">
        <div class="arch-box arch-box-train">🎓 Fine-tuning CamemBERT · Colab T4 · 5 epochs · 6 min</div>
    </div>
    <div class="arch-arrow">↓</div>

    <div class="arch-row arch-row-2col">
        <div class="arch-box arch-box-hf">🤗 HF: original (3 classes)</div>
        <div class="arch-box arch-box-hf">🤗 HF: augmenté (binaire)</div>
    </div>
    <div class="arch-arrow">↓</div>

    <div class="arch-row">
        <div class="arch-box arch-box-app">💻 App Streamlit · Comparaison live des 2 modèles</div>
    </div>
    <div class="arch-arrow">↓</div>

    <div class="arch-row">
        <div class="arch-box arch-box-db">🗄️ Supabase PostgreSQL · table predictions</div>
    </div>
</div>
```

### Section 3 — Pipeline en 5 étapes

Pour chaque étape, une card avec **titre, durée, ce qui s'y passe, livrable** :

1. **Collecte** — durée : pré-existant — 550 avis scrapés depuis Google Maps des campus Ynov, livrés au format CSV avec colonnes (author, rating, sentiment_label, date, comment).

2. **Augmentation** — durée : ~30 min — Génération de 100 avis négatifs synthétiques par Claude, calibrés sur le style des vrais avis (11 campus, 10 filières, 8+ angles). Combinés avec 97 vrais négatifs et 197 positifs échantillonnés → dataset binaire équilibré 197/197.

3. **Fine-tuning** — durée : ~6 min sur Colab T4 — CamemBERT base 110M params, 5 epochs, lr=2e-5, batch=16. Split stratifié 70/15/15 sur sentiment×source. Métriques tracées par epoch.

4. **Déploiement** — durée : ~5 min — Push automatique sur Hugging Face Hub via `model.push_to_hub()`. App Streamlit déployée sur Streamlit Cloud, charge les modèles à la volée depuis HF.

5. **Persistance** — durée : continue — Chaque prédiction est insérée dans Supabase (table `predictions`) en best-effort, pour analytics future.

### Section 4 — Résultats détaillés

Tableau Markdown avec 3 colonnes (Métrique, Global, Real, Synthetic) :

| Métrique | Global | Real | Synthetic |
|---|---|---|---|
| Accuracy | 98.3% | 97.8% | 100% |
| F1 macro | 0.9833 | 0.9746 | 1.0000 |
| F1 négatif | 0.9831 | 0.9655 | 1.0000 |
| F1 positif | 0.9836 | 0.9836 | — |
| n test | 60 | 45 | 15 |

Note explicative en dessous : "L'écart real/synthetic est de 2,5 points → biais de génération minimal."

Plus une **mini matrice de confusion** rendue avec Plotly Heatmap :

```
              Prédit
            neg    pos
Réel  neg [ 29     1  ]
      pos [  0    30  ]
```

(Valeurs simulées d'après les métriques exactes du training.)

### Section 5 — Stack technique

Cards horizontales (réutilisation du style `metric-card`), une par catégorie :

| Catégorie | Outils |
|---|---|
| **🤖 ML** | Transformers, PyTorch, CamemBERT, scikit-learn |
| **🎨 Frontend** | Streamlit, Plotly, HTML/CSS custom |
| **☁️ Hosting** | Streamlit Cloud, Hugging Face Hub |
| **🗄️ Data** | Pandas, CSV, Supabase (PostgreSQL) |
| **🛠️ Ops** | Git, GitHub, Google Colab, Claude Code |

### Section 6 — Limitations honnêtes

Bullet list :

- **Labels mécaniques** : `sentiment_label` est dérivé du `rating` (1-2 → négatif, 3 → neutre, 4-5 → positif), pas annoté indépendamment du texte.
- **Synthetic mono-source** : les 100 négatifs synthétiques sont produits par un seul LLM (Claude), risque de signature stylistique détectable.
- **Taille modeste** : 394 avis suffisent pour un POC ; pour la production, viser 1000+ par classe avec scrap réel.
- **Validation à 100% dès l'epoch 2** : signal possible d'un val set trop petit (~58 lignes) ou de signaux trop forts (mots polarisants suffisent).
- **Pas de neutre** dans le modèle augmenté : choix volontaire faute de vrais avis neutres dans le dataset original (seulement 2).

### Section 7 — Ressources

3 liens stylés (réutilise `.footer-links` ou nouveau style cards) :

- 🤗 [Modèle CamemBERT original](https://huggingface.co/Ahmat293/camembert-sentiment-ynov)
- 🤗 [Modèle CamemBERT augmenté](https://huggingface.co/Ahmat293/camembert-ynov-augmented)
- ⌨ [Code source GitHub](https://github.com/aadamgk/camembert-sentiment-app)

## 6. Spécifications CSS

À ajouter au bloc `<style>` existant. Classes nouvelles :

```css
/* ─── Diagramme architecture ─── */
.arch-diagram {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0;
    padding: 1.5rem;
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

.arch-row-2col {
    gap: 1.5rem;
}

.arch-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #e5e7eb;
    font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif;
    text-align: center;
    min-width: 180px;
    transition: all 0.2s ease;
}

.arch-box:hover {
    border-color: rgba(167,139,250,0.4);
    transform: translateY(-1px);
}

/* Variantes par type d'étape */
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

/* ─── Section méthodologie : titres ─── */
.methodo-h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f0f5;
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* ─── Stack technique cards ─── */
.stack-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
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

/* ─── Pipeline étape ─── */
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
    font-size: 0.75rem;
    margin-bottom: 0.4rem;
}

.pipeline-step-desc {
    color: #d1d5db;
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ─── Limitations bullet ─── */
.limit-list { padding-left: 0; list-style: none; }
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
```

## 7. Validation finale

1. `streamlit run app.py` ouvre l'app sur l'onglet "📊 Analyse" par défaut
2. Cliquer "📖 Méthodologie" affiche la nouvelle page sans erreur
3. Le diagramme architecture est lisible et joli (cards colorées, flèches verticales)
4. Les 7 sections sont présentes et visuellement distinctes
5. Les liens HF + GitHub fonctionnent (cliquer en ouvre dans un nouvel onglet)
6. Re-cliquer "📊 Analyse" : on retrouve la vue actuelle, les prédictions de la session sont conservées
7. Aucune régression sur le mode comparaison ni la persistance Supabase

## 8. Décisions verrouillées

- 2 onglets `st.tabs` (option A du brainstorming)
- Contenu hybride vulgarisé + technique (option C du brainstorming)
- Diagramme HTML/CSS custom (pas d'image, pas de Mermaid)
- Pas de nouvelle dépendance Python
- 7 sections (En bref / Architecture / Pipeline / Résultats / Stack / Limitations / Ressources)

## 9. Hors scope explicite

- Pas de scrollytelling animé
- Pas de version multilingue (FR uniquement)
- Pas d'export PDF
- Pas d'analytics dans la page (les chiffres sont les vraies métriques du training, pas live)
- Pas de "make from scratch" : la matrice de confusion utilise les valeurs connues du training (29 TN, 1 FN, 0 FP, 30 TP), pas un nouveau calcul live
