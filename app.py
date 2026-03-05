import streamlit as st
import torch
import pickle
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Analyse de sentiment - CamemBERT")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Ahmat293/camembert-sentiment-ynov", subfolder="tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("Ahmat293/camembert-sentiment-ynov", subfolder="model")
    le_path = hf_hub_download(repo_id="Ahmat293/camembert-sentiment-ynov", filename="label_encoder.pkl")
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    model.eval()
    return tokenizer, model, le

tokenizer, model, le = load_model()

comment = st.text_area("Entrez un commentaire :")

if st.button("Analyser"):
    if comment:
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            pred = model(**inputs).logits.argmax(dim=1).item()
        sentiment = le.inverse_transform([pred])[0]
        print(sentiment)
        if sentiment == "positif":
            st.success(f"Sentiment : {sentiment} 😊")
        elif sentiment == "negatif":
            st.error(f"Sentiment : {sentiment} 😞")
        else:
            st.warning(f"Sentiment : {sentiment} 😐")