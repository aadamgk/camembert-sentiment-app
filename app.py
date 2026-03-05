import streamlit as st
import pickle
from huggingface_hub import hf_hub_download

st.title("Analyse de sentiment - Ynov")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="Ahmat293/camembert-sentiment-ynov", filename="sentiment_model.pkl")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_model()

comment = st.text_area("Entrez un commentaire :")

if st.button("Analyser"):
    if comment:
        sentiment = pipeline.predict([comment])[0]
        
        if sentiment == "positif":
            st.success(f"Sentiment : {sentiment} 😊")
        elif sentiment == "negatif":
            st.error(f"Sentiment : {sentiment} 😞")
        else:
            st.warning(f"Sentiment : {sentiment} 😐")