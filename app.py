import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
from datetime import datetime
import re

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

# ─── Session state ──────────────────────────────────────────────────────────────
if "new_comments" not in st.session_state:
    st.session_state.new_comments = []

# ─── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def predict(text, classifier):
    result = classifier(text, truncation=True, max_length=512)[0]
    label_map = {"positive": "positif", "negative": "négatif", "neutral": "neutre"}
    return label_map.get(result["label"], result["label"]), round(result["score"] * 100, 1)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Ynov Sentiment<br>Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse des avis étudiants · Powered by XLM-RoBERTa</div>', unsafe_allow_html=True)

def parse_french_relative_date(date_str):
    if not isinstance(date_str, str):
        return pd.NaT

    date_str = str(date_str).lower().strip()

    match = re.search(r'(?:il y a)\s+(un|une|\d+)\s+(minute|minutes|heure|heures|jour|jours|semaine|semaines|mois|an|ans)', date_str)

    if not match:
        return pd.to_datetime(date_str, errors="coerce")

    qty_str = match.group(1)
    unit_str = match.group(2)

    if qty_str in ['un', 'une']:
        qty = 1
    else:
        qty = int(qty_str)

    now = pd.Timestamp.now()

    if 'minute' in unit_str:
        return now - pd.DateOffset(minutes=qty)
    elif 'heure' in unit_str:
        return now - pd.DateOffset(hours=qty)
    elif 'jour' in unit_str:
        return now - pd.DateOffset(days=qty)
    elif 'semaine' in unit_str:
        return now - pd.DateOffset(weeks=qty)
    elif 'mois' in unit_str:
        return now - pd.DateOffset(months=qty)
    elif 'an' in unit_str:
        return now - pd.DateOffset(years=qty)

    return pd.NaT

# ─── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
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
                    df_source["date_clean"] = df_source["date"].apply(parse_french_relative_date)
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

with col_input:
    st.markdown('<div class="section-title">✍️ Analyser un commentaire</div>', unsafe_allow_html=True)
    comment = st.text_area("Commentaire", placeholder="Entrez un avis étudiant...", height=130, label_visibility="collapsed")

    if st.button("Analyser le sentiment"):
        if comment.strip():
            with st.spinner("Analyse en cours..."):
                classifier = load_model()
                sentiment, confidence = predict(comment, classifier)

            css_class = {"positif": "result-pos", "négatif": "result-neg", "neutre": "result-neu"}.get(sentiment, "result-neu")
            icon = {"positif": "😊", "négatif": "😞", "neutre": "😐"}.get(sentiment, "😐")

            st.markdown(f'<div class="result-box {css_class}">{icon} <span>{sentiment.upper()}</span> <span style="opacity:0.6;font-size:0.9rem;margin-left:auto">{confidence}% confiance</span></div>', unsafe_allow_html=True)

            st.session_state.new_comments.append({
                "comment": comment[:60] + "..." if len(comment) > 60 else comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M")
            })
        else:
            st.warning("Entrez un commentaire d'abord.")

with col_history:
    st.markdown('<div class="section-title">🕐 Historique des analyses</div>', unsafe_allow_html=True)

    if st.session_state.new_comments:
        # Mini stats
        sentiments = [c["sentiment"] for c in st.session_state.new_comments]
        n_pos = sentiments.count("positif")
        n_neg = sentiments.count("négatif")
        n_neu = sentiments.count("neutre")
        total_new = len(sentiments)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-number pos" style="font-size:1.8rem">{n_pos}</div><div class="metric-label">Positifs</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-number neg" style="font-size:1.8rem">{n_neg}</div><div class="metric-label">Négatifs</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-number neu" style="font-size:1.8rem">{n_neu}</div><div class="metric-label">Neutres</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for item in reversed(st.session_state.new_comments[-8:]):
            badge_class = {"positif": "badge-pos", "négatif": "badge-neg", "neutre": "badge-neu"}.get(item["sentiment"], "badge-neu")
            st.markdown(f'''
            <div class="comment-item">
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
st.markdown('<div style="text-align:center;color:#374151;font-size:0.75rem;letter-spacing:0.1em">YNOV SENTIMENT ANALYSER · XLM-RoBERTa · 2026</div>', unsafe_allow_html=True)