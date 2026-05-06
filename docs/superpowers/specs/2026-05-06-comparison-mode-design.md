# Mode comparaison directe — affichage dual des 2 modèles

**Date** : 2026-05-06
**Auteur** : Ahmat (collab. Claude)
**Statut** : Design validé, en attente de relecture utilisateur

## 1. Contexte et motivation

L'app dispose actuellement d'un sélecteur de modèle (cards) permettant de tester un modèle à la fois. Pour une démo Ynov, l'objectif principal du projet — montrer que le fine-tuning sur dataset augmenté améliore les performances — devrait être visible **immédiatement et de manière comparative**. L'utilisateur doit pouvoir saisir un avis et voir les 2 verdicts côte-à-côte, avec un indicateur clair d'accord ou de désaccord entre les modèles.

## 2. Objectif et critères de succès

### Objectif

Ajouter une 3e card "Comparer les deux modèles" au sélecteur. Quand sélectionnée, le bouton d'analyse exécute les deux modèles en parallèle et affiche un verdict d'accord/désaccord suivi des 2 résultats côte-à-côte.

### Critères de succès

1. La 3e card est alignée avec les 2 autres et visuellement reconnaissable (design distinct).
2. La sélectionner change le libellé du bouton de "Analyser le sentiment" à "Comparer les modèles".
3. Au clic, les 2 modèles produisent une prédiction (en série, pas de parallélisme stricte requis pour la démo).
4. Un encart **verdict** s'affiche au-dessus des résultats : 🟢 accord ou 🟡 désaccord.
5. Les 2 résultats sont affichés en 2 colonnes égales avec icône modèle, sentiment, barre de confiance.
6. La persistance Supabase enregistre **2 lignes** (une par modèle).
7. L'historique de session reçoit **2 entrées**.
8. Le mode unique (sélection des 2 cards existantes) continue de fonctionner identiquement.

## 3. Scope

### Inclus
- Ajout d'une 3e clé dans `MODELS` avec flag spécial `is_comparison`
- Rendu d'une card "Comparaison" avec design distinct (pas de badges Acc/F1, pas de lien HF, accent visuel particulier)
- Logique conditionnelle dans le flux de prédiction : si mode comparaison → boucle sur les 2 modèles "réels"
- Affichage du verdict (texte + couleur) basé sur la comparaison des sentiments
- Affichage dual des 2 résultats avec leurs barres de confiance
- 2 inserts Supabase, 2 ajouts à l'historique de session
- Adaptation dynamique du libellé du bouton selon le mode

### Exclus (futurs travaux)
- Cache spécial pour la comparaison (on accepte 2 inférences à chaque clic)
- Statistiques cumulées d'accord/désaccord (sera dans la feature dashboard analytics)
- Animation ou transition complexe entre les 2 résultats
- Mode "side-by-side perpétuel" (toujours visible quel que soit le modèle sélectionné)

## 4. Architecture

### Flux

```
[Click "Comparer les modèles"]
        │
        ▼
[Vérifier que selected_model == "Comparer les deux"]
        │
        ▼
[Pour chaque modèle "réel" (Original, Augmenté)]
        │   - load_model(...)
        │   - predict(comment, ...)
        │
        ▼
[predict_dual()] retourne :
{
   "original": {sentiment, confidence, label_brut},
   "augmente": {sentiment, confidence, label_brut},
   "verdict": "accord" | "desaccord",
   "verdict_text": "Les 2 modèles sont d'accord" / "Désaccord ..."
}
        │
        ▼
[Rendu UI dual]
        │   - encart verdict (couleur selon accord/désaccord)
        │   - 2 colonnes avec result-box + barre de confiance
        │
        ▼
[Sauvegarde Supabase x2]
[Historique session x2]
```

### Logique du verdict

```python
def compute_verdict(s_orig, s_aug):
    if s_orig == s_aug:
        return ("accord", f"🟢 Les 2 modèles sont d'accord ({s_orig})")
    if s_orig == "neutre":
        # cas particulier : original dit neutre, augmenté tranche
        return ("partiel", f"🟡 L'augmenté tranche ({s_aug}), l'original reste neutre")
    return ("desaccord", f"🟡 Désaccord — Original: {s_orig} · Augmenté: {s_aug}")
```

3 niveaux de verdict :
- **accord** (vert) — même sentiment exact
- **partiel** (orange clair) — l'un dit "neutre", l'autre tranche
- **desaccord** (orange) — sentiments franchement opposés

## 5. Spécifications UI

### Card "Comparaison" (la 3e)

Différenciée des cards modèles standards par :
- Pas de section badges (Accuracy / F1 absentes)
- Pas de lien Hugging Face
- Emoji distinctif : `🆚`
- Sous-titre : "Mode dual" au lieu de "camembert-base · X classes"
- Description : "Voir les 2 verdicts + accord/désaccord"
- Border accent : gradient subtil mais distinct (couleur cyan/teal pour démarquer du violet des modèles)

### Encart verdict

```html
<div class="verdict-banner verdict-{accord|partiel|desaccord}">
  <span class="verdict-icon">🟢|🟡</span>
  <span class="verdict-text">{verdict_text}</span>
</div>
```

CSS :
- `verdict-accord` : fond vert translucide, bordure verte
- `verdict-partiel` : fond ambre translucide, bordure ambre
- `verdict-desaccord` : fond orange translucide, bordure orange

### Affichage dual

`st.columns(2)` avec dans chaque colonne :
- Une mini-card avec l'icône modèle (📜 ou ✨), nom court
- La `result-box` réutilisée (style existant)
- La barre de confiance avec son gradient de couleur

### Bouton dynamique

```python
button_label = "Comparer les modèles" if is_comparison_mode else "Analyser le sentiment"
```

## 6. Spécifications techniques

### Nouveau dict `MODELS`

```python
MODELS = {
    "CamemBERT (original)": {...inchangé...},
    "CamemBERT (augmenté)": {...inchangé...},
    "Comparer les deux": {
        "is_comparison": True,
        "emoji": "🆚",
        "subtitle": "Mode dual",
        "description": "Voir les 2 verdicts + accord/désaccord",
    },
}
```

Note : on ajoute `is_comparison` dans tous les modèles existants, default `False`, pour éviter les `KeyError`. (Ou alors on utilise `cfg.get("is_comparison", False)` partout.)

### Nouvelles fonctions

**`predict_dual(text)`** :
```python
def predict_dual(text):
    results = {}
    for name in ["CamemBERT (original)", "CamemBERT (augmenté)"]:
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
```

**`render_dual_result(results, verdict, verdict_text)`** :
- Affiche l'encart verdict (avec classe CSS conditionnelle)
- Affiche 2 `st.columns(2)` avec les résultats (sentiment + barre confiance)

### Adaptation du flux principal

```python
config = MODELS[selected_model_name]
is_comparison = config.get("is_comparison", False)
button_label = "Comparer les modèles" if is_comparison else "Analyser le sentiment"

if st.button(button_label, key="analyze_btn"):
    if comment.strip():
        with st.spinner("Analyse en cours..."):
            if is_comparison:
                results, verdict, verdict_text = predict_dual(comment)
                render_dual_result(results, verdict, verdict_text)
                # Persist + history pour les 2 modèles
                for model_name, r in results.items():
                    save_prediction(comment, r["sentiment"], r["confidence"], model_name, r["model_id"])
                    st.session_state.new_comments.append({...})
                fetch_global_history.clear()
            else:
                # Comportement actuel (1 modèle)
                ...
```

## 7. Spécifications CSS

À ajouter au bloc `<style>` existant :

```css
/* ─── Card de comparaison (3e card) ─── */
.model-card.comparison-card {
    border: 2px dashed rgba(96,165,250,0.3);
    background: linear-gradient(135deg, rgba(96,165,250,0.05), rgba(52,211,153,0.05));
}

.model-card.comparison-card.selected {
    border: 2px solid #60a5fa;
    border-style: solid;
    background: linear-gradient(135deg, rgba(96,165,250,0.15), rgba(52,211,153,0.10));
    box-shadow: 0 0 0 1px #60a5fa, 0 12px 40px rgba(96,165,250,0.35);
}

.model-card.comparison-card.selected::before {
    background: linear-gradient(90deg, #60a5fa, #34d399, #a78bfa);
}

.model-card.comparison-card.selected::after {
    content: "✓ MODE DUAL";
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
}

/* ─── Verdict banner ─── */
.verdict-banner {
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.8rem;
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
    font-size: 0.8rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
```

## 8. Validation finale

1. La 3e card s'affiche correctement, alignée avec les 2 autres
2. Sélectionner la 3e card affiche son état "selected" distinct (verdict cyan/teal vs violet)
3. Le bouton change de libellé en "Comparer les modèles"
4. Saisir un commentaire et cliquer le bouton lance les 2 inférences (visible via spinner)
5. Le verdict s'affiche en haut, dans la bonne couleur (vert / ambre / orange)
6. Les 2 colonnes de résultats sont équilibrées
7. Les barres de confiance ont les bonnes couleurs
8. La table Supabase reçoit bien 2 lignes par clic (vérification dashboard)
9. L'historique de session affiche 2 entrées
10. Test croisé avec un commentaire ambigu pour s'assurer qu'on peut obtenir un désaccord (par ex. "Le campus est correct, sans plus" → original: positif, augmenté: 50/50)

## 9. Décisions verrouillées

- Layout : 3 cards alignées (option A du brainstorming)
- Verdict : 3 niveaux accord / partiel / désaccord (option B du brainstorming)
- Persistance : 2 inserts Supabase + 2 entrées historique
- Bouton dynamique selon le mode
- Inférences séquentielles (pas de parallélisme)

## 10. Hors scope explicite

- Pas de mode "always dual" (vue permanente)
- Pas de winner auto (option C rejetée)
- Pas d'animation de transition entre les résultats
- Pas de stats "X% d'accord depuis le début" (sera dans le dashboard)
- Pas de cache spécial (chaque clic = 2 inférences fresh)
