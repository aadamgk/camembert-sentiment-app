# Plan d'implémentation — Augmentation dataset Ynov

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produire `avis_ynov_augmented.csv` (3000 lignes équilibrées 1000/1000/1000) en mélangeant les vrais avis exploitables et des avis synthétiques générés par Claude dans cette session.

**Architecture:** Pas de script Python applicatif. Le travail est piloté par Claude via 3 outils : `Bash` (one-liners pandas pour filtrer/valider/concat), `Read` (lecture d'avis réels pour calibrage), `Write` (écriture des batches synthétiques au format CSV). Les batches intermédiaires sont stockés dans `batches/` pour permettre la reprise.

**Tech Stack:** pandas (one-liners shell), Claude (génération texte), CSV UTF-8.

**Spec source:** `docs/superpowers/specs/2026-05-06-data-augmentation-ynov-design.md`

---

## File Structure

**Créés :**
- `intermediate/real_filtered.csv` — vrais avis avec `comment` non-null + colonne `source="real"`
- `intermediate/gap_counts.txt` — texte avec le nombre d'avis à générer par classe
- `batches/synthetic_negatif_01.csv` à `synthetic_negatif_NN.csv` — batches négatifs synthétiques
- `batches/synthetic_neutre_01.csv` à `synthetic_neutre_NN.csv` — batches neutres synthétiques
- `batches/synthetic_positif_01.csv` à `synthetic_positif_NN.csv` — batches positifs synthétiques
- `avis_ynov_augmented.csv` — fichier final équilibré
- `data_augmentation_report.md` — rapport stats + exemples

**Non modifiés :**
- `avis_ynov_All_final.csv` — original conservé intact
- `app.py` — hors scope

**Convention CSV pour tous les fichiers** :
- Encodage : UTF-8
- Séparateur : virgule
- Quote : `"` autour des champs contenant `,`, `\n`, ou `"`
- Colonnes : `author,rating,sentiment_label,date,comment,source`

---

## Task 1: Setup — créer les répertoires de travail

**Files:**
- Create dirs: `intermediate/`, `batches/`

- [ ] **Step 1: Créer les répertoires**

```bash
mkdir -p intermediate batches
```

- [ ] **Step 2: Vérifier**

Run: `ls -la intermediate batches`
Expected: les deux dossiers existent et sont vides.

- [ ] **Step 3: Ajouter `intermediate/` et `batches/` au `.gitignore`**

Si `.gitignore` n'existe pas, le créer. Ajouter ces lignes :

```
intermediate/
batches/
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore dossiers intermédiaires data augmentation"
```

---

## Task 2: Filtrer les vrais avis et calculer le gap par classe

**Files:**
- Create: `intermediate/real_filtered.csv`
- Create: `intermediate/gap_counts.txt`

- [ ] **Step 1: Filtrer + ajouter colonne `source`**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('avis_ynov_All_final.csv', on_bad_lines='skip')
df = df.dropna(subset=['comment']).copy()
df['source'] = 'real'
df = df[['author','rating','sentiment_label','date','comment','source']]
df.to_csv('intermediate/real_filtered.csv', index=False, encoding='utf-8')
print('Lignes conservées :', len(df))
print(df['sentiment_label'].value_counts())
"
```

Expected output : ~383 lignes, distribution approximative `positif: 290, negatif: 88, neutre: 5` (les chiffres exacts seront affichés).

- [ ] **Step 2: Calculer le gap par classe (combien à générer)**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('intermediate/real_filtered.csv')
TARGET = 1000
counts = df['sentiment_label'].value_counts()
gaps = {}
for cls in ['positif', 'neutre', 'negatif']:
    gaps[cls] = max(0, TARGET - counts.get(cls, 0))
with open('intermediate/gap_counts.txt', 'w', encoding='utf-8') as f:
    for cls, gap in gaps.items():
        n_batches = (gap + 49) // 50
        line = f'{cls}: real={counts.get(cls,0)}, gap={gap}, batches={n_batches}'
        print(line)
        f.write(line + '\n')
"
```

Expected : trois lignes affichant le gap et le nombre de batches de 50 nécessaires par classe (par ex. `negatif: real=88, gap=912, batches=19`).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-06-data-augmentation-ynov.md
git commit -m "feat(data-aug): plan d'implémentation data augmentation"
```

(Les fichiers `intermediate/` sont gitignorés.)

---

## Task 3: Calibrage — lire un échantillon représentatif des vrais avis

**Files:**
- Read only: `intermediate/real_filtered.csv`

- [ ] **Step 1: Échantillonner 30 avis variés**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('intermediate/real_filtered.csv')
df['len'] = df['comment'].str.len()
samples = []
for cls in ['positif', 'negatif', 'neutre']:
    sub = df[df['sentiment_label'] == cls]
    if len(sub) >= 10:
        for bucket in ['short','med','long']:
            if bucket == 'short': s = sub[sub['len'] < 100]
            elif bucket == 'med': s = sub[(sub['len'] >= 100) & (sub['len'] < 300)]
            else: s = sub[sub['len'] >= 300]
            samples.append(s.sample(min(3, len(s)), random_state=42))
    else:
        samples.append(sub)
out = pd.concat(samples)[['sentiment_label','rating','comment']]
out.to_csv('intermediate/calibration_sample.csv', index=False, encoding='utf-8')
print(f'Calibration sample : {len(out)} avis')
"
```

Expected : ~30 avis sauvegardés.

- [ ] **Step 2: Lire et internaliser le style**

Read `intermediate/calibration_sample.csv` (entier). Objectif : observer **explicitement** :
- Niveau de langage (familier vs soutenu, fautes typiques)
- Longueur médiane par classe
- Sujets récurrents (alternance, intervenants, IA, locaux, etc.)
- Formules d'ouverture et de clôture typiques
- Présence/absence de signatures, prénoms, mentions de campus

Consigner mentalement (pas besoin de fichier) ces observations pour guider la génération.

---

## Task 4: Générer le 1er batch négatif (test pilote)

**Files:**
- Create: `batches/synthetic_negatif_01.csv`

- [ ] **Step 1: Rédiger 50 avis négatifs variés**

Écrire `batches/synthetic_negatif_01.csv` avec ce header exact :
```
author,rating,sentiment_label,date,comment,source
```

Suivi de 50 lignes synthétiques. Contraintes par batch (récap section 6 de la spec) :
- **Sentiment** : tous `negatif`
- **Rating** : 70% rating=1, 30% rating=2 (donc ~35 lignes rating=1, ~15 rating=2)
- **Source** : tous `synthetic`
- **Author** : prénom français courant + initiale (ex: "Marie L.", "Lucas D.")
- **Date** : format Google Maps ("il y a 2 mois", "il y a 1 an", "il y a 3 semaines"...)
- **Comment** : texte français, qualité humaine, pas de pattern uniforme

Distribution longueur dans ce batch :
- 10 très courts (1-15 mots)
- 17 courts (15-50 mots)
- 15 moyens (50-150 mots)
- 6 longs (150-400 mots)
- 2 très longs (400-600 mots)

Diversité minimale (à varier sur les 50 lignes) :
- Au moins 8 campus différents parmi : Paris, Lyon, Bordeaux, Toulouse, Nantes, Lille, Aix, Sophia, Rennes, Strasbourg, Montpellier
- Au moins 6 filières parmi : Informatique, 3D, Audiovisuel, Marketing, Création Digitale, Game Design, Architecture, Tech & Data, Cyber, Business
- Au moins 8 angles différents : pédagogie, intervenants, alternance, frais, locaux, vie étudiante, projets, jury, IA, admin, débouchés
- Tons variés : déçu, sarcastique, factuel critique, mise en garde, regret, colère retenue
- **Aucune phrase répétée à l'identique entre lignes**
- Échapper correctement les `,` et `"` dans les commentaires (entourer de guillemets, doubler les guillemets internes)

- [ ] **Step 2: Valider le batch**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('batches/synthetic_negatif_01.csv')
assert len(df) == 50, f'attendu 50 lignes, obtenu {len(df)}'
assert (df['sentiment_label'] == 'negatif').all(), 'sentiment incohérent'
assert df['rating'].isin([1,2]).all(), 'rating hors 1-2'
assert (df['source'] == 'synthetic').all(), 'source incohérent'
assert df['comment'].duplicated().sum() == 0, 'doublons détectés'
df['len'] = df['comment'].str.split().str.len()
print('OK : batch validé')
print('Distribution rating :', df['rating'].value_counts().to_dict())
print('Longueur (mots) — min/médiane/max :', df['len'].min(), df['len'].median(), df['len'].max())
"
```

Expected : `OK : batch validé`, distribution rating proche de 35/15, longueurs cohérentes.

- [ ] **Step 3: Si validation KO**

- Doublons → réécrire les lignes concernées avec phrases différentes
- Rating incohérent → corriger la colonne `rating`
- Compte ≠ 50 → ajouter/retirer des lignes
- Re-run validation jusqu'à OK.

---

## Task 5: Générer les batches négatifs restants

**Files:**
- Create: `batches/synthetic_negatif_02.csv` à `batches/synthetic_negatif_NN.csv`

`NN` est le nombre total de batches négatifs requis (calculé en Task 2, typiquement 19).

- [ ] **Step 1: Pour chaque batch de 02 à NN**

Pour chaque numéro de batch `n` :

**A. Lire la liste des phrases déjà utilisées dans les batches précédents** (anti-doublon inter-batch) :

```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('batches/synthetic_negatif_*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])
# Top 30 mots/phrases d'ouverture pour vérifier la diversité
opens = df['comment'].str.split('.').str[0].str[:50].value_counts().head(20)
print('Ouvertures déjà utilisées (top 20) :')
print(opens)
print(f'Total commentaires uniques : {df[\"comment\"].nunique()} / {len(df)}')
"
```

**B. Rédiger `batches/synthetic_negatif_<n>.csv`** avec les mêmes contraintes que Task 4 :
- 50 avis négatifs, rating 1 ou 2 (70/30)
- Distribution longueur : 10/17/15/6/2 (TC/C/M/L/TL)
- Diversité campus/filière/angle/ton (varier par rapport aux batches précédents)
- **Aucune phrase d'ouverture identique à un batch précédent**

**C. Valider le batch** (même commande qu'en Task 4 step 2, en remplaçant `_01` par `_<n>`).

- [ ] **Step 2: Vérification globale après le dernier batch négatif**

Run :
```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('batches/synthetic_negatif_*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])
print(f'Total négatifs synthétiques : {len(df)}')
print(f'Doublons : {df[\"comment\"].duplicated().sum()}')
print(f'Distribution rating : {df[\"rating\"].value_counts().to_dict()}')
df['len'] = df['comment'].str.split().str.len()
print(f'Longueur médiane : {df[\"len\"].median():.0f} mots')
"
```

Expected : `Total négatifs synthétiques : ~912` (gap calculé en Task 2), `Doublons : 0`.

- [ ] **Step 3: Commit (sécurité)**

```bash
git add docs/superpowers/plans/
git commit -m "wip(data-aug): batches négatifs générés" --allow-empty
```

(Les batches eux-mêmes sont gitignorés mais on jalonne le commit.)

---

## Task 6: Générer les batches neutres

**Files:**
- Create: `batches/synthetic_neutre_01.csv` à `batches/synthetic_neutre_NN.csv`

`NN` typiquement ~20 batches.

- [ ] **Step 1: Pour chaque batch (mêmes étapes A/B/C que Task 5 step 1)**

Différences pour le sentiment **neutre** :
- **Rating** : tous = 3 (100%)
- **Sentiment_label** : tous `neutre`
- **Distribution longueur** : 10/17/15/6/2 (idem)
- **Particularité ton** : équilibré, mitigé, "des plus et des moins", recommandation conditionnelle, factuel descriptif
- **Pièges à éviter** : ne pas glisser vers le positif ("super école mais...") ou le négatif ("nul mais quand même..."). Vraiment équilibré.

Exemples de tons neutres typiques :
- "Bonne école pour certaines filières, médiocre pour d'autres."
- "Globalement correct, sans plus."
- "Ça dépend vraiment de l'intervenant."
- "Quelques bons points, quelques bémols."

- [ ] **Step 2: Vérification globale neutres**

Run :
```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('batches/synthetic_neutre_*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])
print(f'Total neutres synthétiques : {len(df)}')
print(f'Doublons : {df[\"comment\"].duplicated().sum()}')
assert (df['rating'] == 3).all(), 'rating neutre doit être 3'
print('OK')
"
```

Expected : `Total neutres synthétiques : ~995`, `Doublons : 0`, `OK`.

- [ ] **Step 3: Commit (sécurité)**

```bash
git commit -m "wip(data-aug): batches neutres générés" --allow-empty
```

---

## Task 7: Générer les batches positifs

**Files:**
- Create: `batches/synthetic_positif_01.csv` à `batches/synthetic_positif_NN.csv`

`NN` typiquement ~15 batches.

- [ ] **Step 1: Pour chaque batch (mêmes étapes A/B/C que Task 5 step 1)**

Différences pour le sentiment **positif** :
- **Rating** : 30% rating=4, 70% rating=5 (donc ~15 lignes rating=4, ~35 rating=5)
- **Sentiment_label** : tous `positif`
- **Distribution longueur** : 10/17/15/6/2 (idem)
- **Particularité ton** : enthousiaste, recommandation, satisfait, gratitude, fier, parfois nuancé positif (rating 4 avec petits bémols)

Exemples de tons positifs :
- "Super école, je recommande à 100%."
- "Alternance bien suivie, intervenants de qualité."
- "Une vraie évolution depuis ma première année, je suis content de mon choix."
- "Quelques points à améliorer mais globalement très satisfait." (rating 4)

- [ ] **Step 2: Vérification globale positifs**

Run :
```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('batches/synthetic_positif_*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])
print(f'Total positifs synthétiques : {len(df)}')
print(f'Doublons : {df[\"comment\"].duplicated().sum()}')
print(f'Distribution rating : {df[\"rating\"].value_counts().to_dict()}')
assert df['rating'].isin([4,5]).all(), 'rating positif doit être 4 ou 5'
print('OK')
"
```

Expected : `Total positifs synthétiques : ~710`, distribution rating proche de 30%/70%, `OK`.

- [ ] **Step 3: Commit (sécurité)**

```bash
git commit -m "wip(data-aug): batches positifs générés" --allow-empty
```

---

## Task 8: Concaténation finale et validation

**Files:**
- Create: `avis_ynov_augmented.csv`

- [ ] **Step 1: Concaténer real + tous les batches synthétiques**

Run :
```bash
python -c "
import pandas as pd, glob

real = pd.read_csv('intermediate/real_filtered.csv')
synth_files = sorted(glob.glob('batches/synthetic_*.csv'))
synth = pd.concat([pd.read_csv(f) for f in synth_files], ignore_index=True)

full = pd.concat([real, synth], ignore_index=True)

# Équilibrage strict à 1000 par classe (tronquer si surplus)
balanced_parts = []
for cls in ['positif', 'neutre', 'negatif']:
    sub = full[full['sentiment_label'] == cls]
    if len(sub) > 1000:
        sub = sub.sample(1000, random_state=42)
    balanced_parts.append(sub)
balanced = pd.concat(balanced_parts, ignore_index=True)

# Shuffle final
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
balanced.to_csv('avis_ynov_augmented.csv', index=False, encoding='utf-8')
print(f'Lignes finales : {len(balanced)}')
print(balanced['sentiment_label'].value_counts())
print(balanced['source'].value_counts())
"
```

Expected : 3000 lignes, distribution 1000/1000/1000, mix real/synthetic visible.

- [ ] **Step 2: Validation complète**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('avis_ynov_augmented.csv')

# Critères de succès de la spec
assert len(df) == 3000, f'len={len(df)}'

counts = df['sentiment_label'].value_counts().to_dict()
for cls in ['positif','neutre','negatif']:
    assert abs(counts.get(cls,0) - 1000) <= 5, f'{cls} déséquilibré : {counts.get(cls,0)}'

assert df['comment'].duplicated().sum() == 0, 'doublons exacts'

# Cohérence rating ↔ sentiment
assert ((df['rating'].isin([1,2])) == (df['sentiment_label']=='negatif')).all()
assert ((df['rating']==3) == (df['sentiment_label']=='neutre')).all()
assert ((df['rating'].isin([4,5])) == (df['sentiment_label']=='positif')).all()

# Source valide
assert df['source'].isin(['real','synthetic']).all()

# Toutes colonnes présentes
expected_cols = {'author','rating','sentiment_label','date','comment','source'}
assert set(df.columns) == expected_cols

print('TOUS LES CRITÈRES PASSENT')
print(df['sentiment_label'].value_counts())
print(df['source'].value_counts())
"
```

Expected : `TOUS LES CRITÈRES PASSENT`.

- [ ] **Step 3: Spot-check manuel**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('avis_ynov_augmented.csv')
sample = df.sample(20, random_state=7)
for i, row in sample.iterrows():
    print(f'[{row[\"sentiment_label\"]:8s} | rating={row[\"rating\"]} | {row[\"source\"]:9s}] {row[\"comment\"][:200]}')
    print('---')
"
```

Lire les 20 sorties. Pour chacun, juger : sentiment cohérent avec texte ? Français correct ? Sujet plausible (Ynov) ?

Si > 2/20 sont jugés mauvais, identifier la classe/source problématique et régénérer le ou les batches incriminés.

---

## Task 9: Rapport `data_augmentation_report.md`

**Files:**
- Create: `data_augmentation_report.md`

- [ ] **Step 1: Calculer les stats finales**

Run :
```bash
python -c "
import pandas as pd
df = pd.read_csv('avis_ynov_augmented.csv')
df['len_words'] = df['comment'].str.split().str.len()
df['len_chars'] = df['comment'].str.len()

print('=== Distribution sentiment ===')
print(df['sentiment_label'].value_counts())
print()
print('=== Distribution source ===')
print(df['source'].value_counts())
print()
print('=== Source × Sentiment ===')
print(pd.crosstab(df['source'], df['sentiment_label']))
print()
print('=== Longueur (mots) par sentiment ===')
print(df.groupby('sentiment_label')['len_words'].describe())
print()
print('=== Longueur (mots) par source ===')
print(df.groupby('source')['len_words'].describe())
" > intermediate/final_stats.txt 2>&1
cat intermediate/final_stats.txt
```

- [ ] **Step 2: Rédiger le rapport**

Écrire `data_augmentation_report.md` avec cette structure :

```markdown
# Rapport — Augmentation du dataset Ynov

**Date** : 2026-05-06
**Spec source** : `docs/superpowers/specs/2026-05-06-data-augmentation-ynov-design.md`

## Résumé

- **Dataset original** : 550 lignes (383 exploitables après filtrage des `comment` null)
- **Dataset augmenté** : 3000 lignes équilibrées (1000 / 1000 / 1000)
- **Méthode** : 100% synthétique généré par Claude in-session, calibré sur les vrais avis Ynov

## Distribution finale

[Coller le bloc "Distribution sentiment" depuis intermediate/final_stats.txt]

## Mix réel vs synthétique

[Coller le bloc "Source × Sentiment"]

## Distribution des longueurs

[Coller les blocs "Longueur par sentiment" et "Longueur par source"]

## Méthode

1. Filtrage des vrais avis avec `comment` non-null → ~383 lignes conservées
2. Calibrage sur 30 avis variés pour internaliser le style
3. Génération par batches de 50 avec matrice de diversité (campus × filière × angle × ton × longueur)
4. Validation par batch : unicité, rating cohérent, format
5. Concaténation, équilibrage strict à 1000/classe, shuffle, écriture finale

## Exemples par classe

### Positifs (5 exemples random)

[5 lignes formatées : `[real|synthetic] {comment}`]

### Neutres (5 exemples random)

[idem]

### Négatifs (5 exemples random)

[idem]

## Limites connues

- Labels dérivés mécaniquement du rating (pas d'annotation humaine indépendante du texte)
- Style synthétique potentiellement détectable par un classifieur dédié
- 17,5% des vrais avis dépassent 512 caractères (troncature lors du fine-tuning à prévoir)
- Génération mono-source (Claude) → biais stylistique possible
- `neutre` original sous-représenté (5 vrais → 995 synthétiques)

## Prochaines étapes (hors scope de ce travail)

- Fine-tuning d'un modèle (CamemBERT ou XLM-RoBERTa) sur ce dataset
- Évaluation séparée real/synthetic pour mesurer le biais de génération
- Intégration dans `app.py` (remplacement du modèle Twitter actuel)
```

Pour générer les exemples random :

```bash
python -c "
import pandas as pd
df = pd.read_csv('avis_ynov_augmented.csv')
for cls in ['positif','neutre','negatif']:
    print(f'### {cls}')
    sample = df[df['sentiment_label']==cls].sample(5, random_state=42)
    for _, row in sample.iterrows():
        c = row['comment'].replace('\n',' ')[:200]
        print(f'- [{row[\"source\"]}] {c}')
    print()
"
```

Coller la sortie dans la section "Exemples par classe".

- [ ] **Step 3: Commit final**

```bash
git add avis_ynov_augmented.csv data_augmentation_report.md
git commit -m "feat(data-aug): dataset augmenté 3000 lignes équilibré 1000/1000/1000

- Filtrage des 383 avis réels exploitables
- Génération de ~2617 avis synthétiques via Claude
- Diversité contrôlée (campus, filière, angle, ton, longueur)
- Validation : 0 doublon, distribution rating cohérente, longueurs alignées
- Rapport détaillé dans data_augmentation_report.md
"
```

---

## Self-review (effectué après rédaction)

**Spec coverage** :
- §2 critères 1-5 : couverts par Task 8 step 2 (assertions automatiques) + Task 8 step 3 (spot-check)
- §3 inclus : tous mappés (filtrage Task 2, génération Task 4-7, CSV Task 8, rapport Task 9)
- §4 architecture : Task 1 (setup), Task 2 (composant A), Task 3 (B), Task 4-7 (C+D), Task 8 (E)
- §5 schéma : explicité dans File Structure + Task 4 step 1
- §6 matrice : reproduite dans Task 4 step 1 et Task 5/6/7 step 1
- §7 reprise : Task 5 step 1 A (anti-doublon inter-batch), batches gitignorés mais persistants entre runs
- §8 validation finale : Task 8 step 2-3
- §9 livrables : Task 8 (CSV) + Task 9 (rapport)

**Placeholder scan** : aucun "TBD" ; toutes les commandes sont exactes ; tous les contenus à écrire sont décrits avec leurs contraintes précises.

**Type/path consistency** : noms de fichiers cohérents (`synthetic_<sentiment>_<n>.csv`), colonnes identiques partout, `real_filtered.csv` référencé par les mêmes chemins.

Aucun problème détecté.
