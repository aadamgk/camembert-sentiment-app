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
    padding: 1.2rem 1.1rem 1rem 1.1rem;
    backdrop-filter: blur(10px);
    transition: all 0.25s ease;
    position: relative;
    height: 280px;
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
    border-top-left-radius: 14px;
    border-top-right-radius: 14px;
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
    height: 280px;
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
    gap: 0.6rem !important;
    justify-content: center !important;
    margin-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    margin-bottom: 1.5rem !important;
}

button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #9ca3af !important;
    padding: 0.7rem 1.8rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
}

button[data-baseweb="tab"]:hover {
    border-color: rgba(167,139,250,0.3) !important;
    color: #d1d5db !important;
    transform: translateY(-1px) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(96,165,250,0.12)) !important;
    border-color: rgba(167,139,250,0.6) !important;
    color: #f0f0f5 !important;
    box-shadow: 0 4px 16px rgba(167,139,250,0.2) !important;
}

[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] {
    display: none !important;
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

# ─── Onglets ──────────────────────────────────────────────────────────────────
tab_analyse, tab_methodo = st.tabs(["📊 Analyse", "📖 Méthodologie"])

with tab_analyse:

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
                    # Distribution des notes (1 à 5 étoiles) — granularité plus fine que le sentiment
                    if "rating" in df_source.columns:
                        rating_counts = df_source["rating"].value_counts().sort_index()
                        # Couleur selon le mapping rating → sentiment (1-2 rouge, 3 bleu, 4-5 vert)
                        rating_colors = {1: "#f87171", 2: "#fb923c", 3: "#60a5fa", 4: "#86efac", 5: "#34d399"}
                        colors = [rating_colors.get(int(r), "#9ca3af") for r in rating_counts.index]
                        labels = [f"{r}★" for r in rating_counts.index]

                        fig_rating = go.Figure(go.Bar(
                            x=labels,
                            y=rating_counts.values,
                            marker_color=colors,
                            text=rating_counts.values,
                            textposition="outside",
                            textfont=dict(color="#f0f0f5", size=12),
                        ))
                        fig_rating.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#9ca3af",
                            showlegend=False,
                            xaxis=dict(showgrid=False, color="#9ca3af"),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#4b5563"),
                            margin=dict(t=40, b=20, l=10, r=10),
                            title=dict(text="Distribution des notes (1-5 ★)", font=dict(size=13, color="#9ca3af"), x=0, xanchor="left")
                        )
                        st.plotly_chart(fig_rating, use_container_width=True)

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
                    <span class="model-card-title">Comparer</span>
                </div>
                <div class="model-card-subtitle">{cfg['subtitle']}</div>
                <div class="comparison-badge-row">
                    <span class="comparison-vs-icon">⚖️</span>
                </div>
                <div class="model-card-desc">{cfg['description']}</div>
            </div>
            """

        acc = f"{cfg['accuracy']*100:.0f}%" if cfg['accuracy'] is not None else "—"
        f1 = f"{cfg['f1_macro']:.2f}" if cfg['f1_macro'] is not None else "—"
        short_title = "Original" if "original" in model_name.lower() else "Augmenté"

        return f"""
        <div class="model-card {selected_class}">
            <div class="model-card-header">
                <span class="model-card-emoji">{cfg['emoji']}</span>
                <span class="model-card-title">{short_title}</span>
            </div>
            <div class="model-card-subtitle">{cfg['num_classes']} classes</div>
            <div class="model-card-badges">
                <div class="model-badge">
                    <div class="model-badge-label">Acc</div>
                    <div class="model-badge-value">{acc}</div>
                </div>
                <div class="model-badge">
                    <div class="model-badge-label">F1</div>
                    <div class="model-badge-value">{f1}</div>
                </div>
            </div>
            <div class="model-card-desc">{cfg['description']}</div>
            <a href="{cfg['hf_url']}" target="_blank" class="model-card-link">↗ Hugging Face</a>
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

    # ─── Section 2 : Architecture du projet ──────────────────────────────
    st.markdown('<div class="methodo-h2">🏗️ Architecture du projet</div>', unsafe_allow_html=True)
    st.markdown(
'<div class="arch-diagram">'
'<div class="arch-row"><div class="arch-box arch-box-source">📊 <strong>Source Google Maps</strong><br><span style="color:#9ca3af;font-size:0.75rem">avis_ynov_All_final.csv · 550 lignes</span></div></div>'
'<div class="arch-arrow">↓</div>'
'<div class="arch-row arch-row-2col">'
'<div class="arch-box">🧹 <strong>Filtrage</strong><br><span style="color:#9ca3af;font-size:0.75rem">383 lignes valides</span></div>'
'<div class="arch-arrow-h">→</div>'
'<div class="arch-box arch-box-llm">🤖 <strong>Augmentation LLM</strong><br><span style="color:#9ca3af;font-size:0.75rem">+100 négatifs synthétiques (Claude)</span></div>'
'</div>'
'<div class="arch-arrow">↓</div>'
'<div class="arch-row"><div class="arch-box arch-box-data">📦 <strong>Dataset équilibré</strong><br><span style="color:#9ca3af;font-size:0.75rem">avis_ynov_augmented.csv · 197/197 binaire</span></div></div>'
'<div class="arch-arrow">↓</div>'
'<div class="arch-row"><div class="arch-box arch-box-train">🎓 <strong>Fine-tuning CamemBERT</strong><br><span style="color:#9ca3af;font-size:0.75rem">Colab T4 · 5 epochs · ~6 min</span></div></div>'
'<div class="arch-arrow">↓</div>'
'<div class="arch-row arch-row-2col">'
'<div class="arch-box arch-box-hf">🤗 <strong>HF Hub : original</strong><br><span style="color:#9ca3af;font-size:0.75rem">3 classes (pos/neu/neg)</span></div>'
'<div class="arch-box arch-box-hf">🤗 <strong>HF Hub : augmenté</strong><br><span style="color:#9ca3af;font-size:0.75rem">2 classes (binaire)</span></div>'
'</div>'
'<div class="arch-arrow">↓</div>'
'<div class="arch-row"><div class="arch-box arch-box-app">💻 <strong>App Streamlit</strong><br><span style="color:#9ca3af;font-size:0.75rem">Comparaison live des 2 modèles</span></div></div>'
'<div class="arch-arrow">↓</div>'
'<div class="arch-row"><div class="arch-box arch-box-db">🗄️ <strong>Supabase PostgreSQL</strong><br><span style="color:#9ca3af;font-size:0.75rem">Table predictions · analytics futures</span></div></div>'
'</div>',
        unsafe_allow_html=True
    )

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
        st.markdown(
            f'<div class="pipeline-step">'
            f'<div class="pipeline-step-num">{step["num"]}</div>'
            f'<div class="pipeline-step-content">'
            f'<div class="pipeline-step-title">{step["title"]}</div>'
            f'<div class="pipeline-step-meta">{step["meta"]}</div>'
            f'<div class="pipeline-step-desc">{step["desc"]}</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # ─── Section 4 : Résultats détaillés ─────────────────────────────────
    st.markdown('<div class="methodo-h2">📈 Résultats détaillés</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="methodo-intro">Métriques mesurées sur le test set (60 avis : 45 réels + 15 synthétiques) après 5 epochs de fine-tuning du modèle augmenté.</div>',
        unsafe_allow_html=True
    )

    metrics_df = pd.DataFrame({
        "Métrique":  ["Accuracy", "F1 macro", "F1 négatif", "F1 positif", "n test"],
        "Global":    ["98.3%",    "0.9833",   "0.9831",     "0.9836",     "60"],
        "Real":      ["97.8%",    "0.9746",   "0.9655",     "0.9836",     "45"],
        "Synthetic": ["100.0%",   "1.0000",   "1.0000",     "—",          "15"],
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
        st.markdown(
            f'<div class="stack-card">'
            f'<div class="stack-card-title">{cat}</div>'
            f'<div class="stack-card-tools">{tools}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ─── Section 6 : Limitations honnêtes ────────────────────────────────
    st.markdown('<div class="methodo-h2">⚠️ Limitations honnêtes</div>', unsafe_allow_html=True)
    st.markdown(
        '<ul class="limit-list">'
        '<li><strong>Labels mécaniques</strong> : sentiment_label est dérivé du rating (1-2 → négatif, 3 → neutre, 4-5 → positif), pas annoté indépendamment du texte.</li>'
        '<li><strong>Synthetic mono-source</strong> : les 100 négatifs synthétiques sont produits par un seul LLM (Claude), risque de signature stylistique détectable.</li>'
        '<li><strong>Taille modeste</strong> : 394 avis suffisent pour un POC ; pour la production, viser 1000+ par classe avec scrap réel.</li>'
        '<li><strong>Validation à 100% dès l\'epoch 2</strong> : signal possible d\'un val set trop petit (~58 lignes) ou de signaux trop forts dans les commentaires polarisés.</li>'
        '<li><strong>Pas de classe "neutre"</strong> dans le modèle augmenté : choix volontaire faute de vrais avis neutres dans le dataset original (seulement 2).</li>'
        '</ul>',
        unsafe_allow_html=True
    )

    # ─── Section 7 : Ressources ──────────────────────────────────────────
    st.markdown('<div class="methodo-h2">🔗 Ressources</div>', unsafe_allow_html=True)
    st.markdown(
        '<a class="resource-link" href="https://huggingface.co/Ahmat293/camembert-sentiment-ynov" target="_blank">🤗 <strong>Modèle CamemBERT original</strong> · entraîné sur le dataset brut (3 classes)</a>'
        '<a class="resource-link" href="https://huggingface.co/Ahmat293/camembert-ynov-augmented" target="_blank">🤗 <strong>Modèle CamemBERT augmenté</strong> · fine-tuné sur dataset augmenté (binaire)</a>'
        '<a class="resource-link" href="https://github.com/aadamgk/camembert-sentiment-app" target="_blank">⌨ <strong>Code source GitHub</strong> · app + notebook + specs détaillées</a>',
        unsafe_allow_html=True
    )

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