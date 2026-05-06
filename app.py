import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from datetime import datetime
from supabase import create_client, Client

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ynov · Sentiment Analyser",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #f0f0f5;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a0533 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 80%, #001a2e 0%, transparent 60%);
    background-blend-mode: screen;
}

[data-testid="stHeader"] { background: transparent; }

.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #6b7280;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(167,139,250,0.4);
    background: rgba(167,139,250,0.06);
    transform: translateY(-2px);
}

.metric-number {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
}

.pos { color: #34d399; }
.neg { color: #f87171; }
.neu { color: #60a5fa; }
.total { color: #a78bfa; }

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.result-box {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-top: 1rem;
}

.result-pos { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: #34d399; }
.result-neg { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: #f87171; }
.result-neu { background: rgba(96,165,250,0.1); border: 1px solid rgba(96,165,250,0.3); color: #60a5fa; }

textarea, .stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f0f0f5 !important;
    font-family: 'DM Sans', sans-serif !important;
    caret-color: #a78bfa !important;
}

textarea::placeholder, .stTextArea textarea::placeholder {
    color: #4b5563 !important;
}

.stTextArea > div > div {
    background: transparent !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

.comment-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
}

.badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.badge-pos { background: rgba(52,211,153,0.15); color: #34d399; }
.badge-neg { background: rgba(248,113,113,0.15); color: #f87171; }
.badge-neu { background: rgba(96,165,250,0.15); color: #60a5fa; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 2rem 0;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-baseweb="textarea"] textarea {
    background-color: #1e1e2e !important;
    color: #f0f0f5 !important;
    -webkit-text-fill-color: #f0f0f5 !important;
}
div[data-baseweb="base-input"] {
    background-color: #1e1e2e !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ─── Cards de sélection de modèle ─── */
.model-card {
    background: rgba(255,255,255,0.04);
    border: 2px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.3rem 1.1rem 1.3rem;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    position: relative;
    height: 360px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-sizing: border-box;
}

.model-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, transparent);
    transition: background 0.25s ease;
}

.model-card:hover {
    border-color: rgba(167,139,250,0.5);
    background: rgba(167,139,250,0.06);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.model-card.selected {
    border-color: #a78bfa;
    background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(96,165,250,0.12));
    box-shadow: 0 0 0 1px #a78bfa, 0 12px 40px rgba(167,139,250,0.35);
    transform: translateY(-2px);
}

.model-card.selected::before {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
}

.model-card.selected::after {
    content: "✓ SÉLECTIONNÉ";
    position: absolute;
    top: 10px;
    right: 12px;
    color: #ffffff;
    background: linear-gradient(135deg, #a78bfa, #7c3aed);
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.25rem 0.55rem;
    border-radius: 20px;
    box-shadow: 0 2px 8px rgba(167,139,250,0.5);
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
    margin-top: auto;
    align-self: flex-start;
}

.model-card-link:hover {
    color: #c4b5fd;
    border-color: rgba(196,181,253,0.7);
}

/* ─── Bouton de sélection en overlay (transparent et plein-card) ─── */
.model-selector-row [data-testid="column"] {
    position: relative;
}

.model-selector-row [data-testid="column"] .stButton {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 360px;
    margin: 0 !important;
    padding: 0 !important;
}

.model-selector-row [data-testid="column"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    cursor: pointer;
}

.model-selector-row [data-testid="column"] .stButton > button:hover,
.model-selector-row [data-testid="column"] .stButton > button:focus,
.model-selector-row [data-testid="column"] .stButton > button:active {
    background: transparent !important;
    box-shadow: none !important;
    color: transparent !important;
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

/* ─── Historique enrichi ─── */
.comment-item-augmented { border-left: 3px solid #a78bfa; }
.comment-item-original  { border-left: 3px solid #6b7280; }

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

# ─── Session state ──────────────────────────────────────────────────────────────
if "new_comments" not in st.session_state:
    st.session_state.new_comments = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "CamemBERT (augmenté)"

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
        "description": "Entraîné sur les 550 avis bruts (3 classes).",
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
        "description": "Fine-tuné sur 394 avis augmentés (binaire).",
        "hf_url": "https://huggingface.co/Ahmat293/camembert-ynov-augmented",
    },
    "Comparer les deux": {
        "is_comparison": True,
        "emoji": "🆚",
        "subtitle": "Mode dual",
        "description": "Voir les 2 verdicts + accord/désaccord.",
    },
}

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

@st.cache_resource
def load_model(model_id, subfolder, tokenizer_subfolder):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_kwargs = {"subfolder": subfolder} if subfolder else {}
    tok_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

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

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Ynov Sentiment<br>Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse des avis étudiants · CamemBERT original vs augmenté</div>', unsafe_allow_html=True)

# ─── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/aadamgk/camembert-sentiment-app/master/avis_ynov_All_final.csv"
    return pd.read_csv(url, on_bad_lines='skip')

df_source = load_data()

if df_source is not None:
    try:
        if "sentiment_label" in df_source.columns:
            counts = df_source["sentiment_label"].value_counts()
            total = len(df_source)
            pos = counts.get("positif", 0)
            neg = counts.get("negatif", 0)
            neu = counts.get("neutre", 0)

            # ─── Metrics ───────────────────────────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Vue d\'ensemble du dataset</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-number total">{total}</div><div class="metric-label">Total avis</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-number pos">{pos}</div><div class="metric-label">Positifs · {pos/total*100:.0f}%</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-number neg">{neg}</div><div class="metric-label">Négatifs · {neg/total*100:.0f}%</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-number neu">{neu}</div><div class="metric-label">Neutres · {neu/total*100:.0f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ─── Charts ────────────────────────────────────────────────────────
            col_left, col_right = st.columns([1, 1])

            with col_left:
                fig_donut = go.Figure(data=[go.Pie(
                    labels=["Positif", "Négatif", "Neutre"],
                    values=[pos, neg, neu],
                    hole=0.65,
                    marker_colors=["#34d399", "#f87171", "#60a5fa"],
                    textinfo="none",
                    hovertemplate="<b>%{label}</b><br>%{value} avis (%{percent})<extra></extra>"
                )])
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#9ca3af",
                    showlegend=True,
                    legend=dict(font=dict(size=12, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=20, b=20, l=20, r=20),
                    annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10px'>avis</span>",
                                      x=0.5, y=0.5, font_size=20, font_color="#f0f0f5", showarrow=False)]
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_right:
                if "date" in df_source.columns:
                    df_source["date_clean"] = pd.to_datetime(df_source["date"], errors="coerce")
                    df_time = df_source.dropna(subset=["date_clean"]).groupby(
                        [df_source["date_clean"].dt.to_period("M"), "sentiment_label"]
                    ).size().reset_index(name="count")
                    df_time["date_clean"] = df_time["date_clean"].astype(str)

                    fig_bar = px.bar(df_time, x="date_clean", y="count", color="sentiment_label",
                                     color_discrete_map={"positif": "#34d399", "negatif": "#f87171", "neutre": "#60a5fa"},
                                     barmode="stack")
                    fig_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#9ca3af", showlegend=False,
                        xaxis=dict(showgrid=False, color="#4b5563"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4b5563"),
                        margin=dict(t=20, b=20, l=10, r=10)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    fig_h = go.Figure(go.Bar(
                        x=[pos, neg, neu],
                        y=["Positif", "Négatif", "Neutre"],
                        orientation="h",
                        marker_color=["#34d399", "#f87171", "#60a5fa"]
                    ))
                    fig_h.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#9ca3af",
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(showgrid=False),
                        margin=dict(t=20, b=20, l=10, r=10)
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")

# ─── Analyse ───────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_input, col_history = st.columns([1, 1], gap="large")

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

with col_input:
    st.markdown('<div class="section-title">✍️ Analyser un commentaire</div>', unsafe_allow_html=True)

    # ─── Sélecteur de modèle en cards ─────────────────────────────────────
    st.markdown('<div class="model-selector-row">', unsafe_allow_html=True)
    model_names = list(MODELS.keys())
    card_cols = st.columns(len(model_names))
    for col, name in zip(card_cols, model_names):
        with col:
            is_selected = (st.session_state.selected_model == name)
            st.markdown(render_model_card(name, is_selected), unsafe_allow_html=True)
            if st.button(f"select_{name}", key=f"btn_{name}"):
                st.session_state.selected_model = name
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

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
        else:
            st.warning("Entrez un commentaire d'abord.")

with col_history:
    st.markdown('<div class="section-title">🕐 Historique des analyses</div>', unsafe_allow_html=True)

    if st.session_state.new_comments:
        # Mini stats — afficher Neutres uniquement si présent (le modèle augmenté est binaire)
        sentiments = [c["sentiment"] for c in st.session_state.new_comments]
        n_pos = sentiments.count("positif")
        n_neg = sentiments.count("négatif")
        n_neu = sentiments.count("neutre")
        total_new = len(sentiments)

        if n_neu > 0:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-number pos" style="font-size:1.8rem">{n_pos}</div><div class="metric-label">Positifs</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-number neg" style="font-size:1.8rem">{n_neg}</div><div class="metric-label">Négatifs</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-number neu" style="font-size:1.8rem">{n_neu}</div><div class="metric-label">Neutres</div></div>', unsafe_allow_html=True)
        else:
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-number pos" style="font-size:1.8rem">{n_pos}</div><div class="metric-label">Positifs</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-number neg" style="font-size:1.8rem">{n_neg}</div><div class="metric-label">Négatifs</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

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

        if st.button("Effacer l'historique"):
            st.session_state.new_comments = []
            st.rerun()
    else:
        st.markdown('<div style="color:#4b5563;text-align:center;padding:3rem 0;font-size:0.9rem">Aucune analyse encore effectuée</div>', unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('''
<div class="footer-links">
    <a href="https://huggingface.co/Ahmat293/camembert-sentiment-ynov" target="_blank">📜 Modèle original</a> ·
    <a href="https://huggingface.co/Ahmat293/camembert-ynov-augmented" target="_blank">✨ Modèle augmenté</a> ·
    <a href="https://github.com/aadamgk/camembert-sentiment-app" target="_blank">⌨ Code source</a>
</div>
<div style="text-align:center;color:#374151;font-size:0.7rem;letter-spacing:0.1em;margin-top:0.5rem">
    YNOV SENTIMENT ANALYSER · CamemBERT original × augmenté · 2026
</div>
''', unsafe_allow_html=True)