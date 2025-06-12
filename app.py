import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- Text Cleaning Function ----------
def clean_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv").sample(1000)
    true = pd.read_csv("True.csv").sample(1000)

    fake['label'] = 0
    true['label'] = 1

    data = pd.concat([fake, true], axis=0)
    data['text'] = data['text'].apply(clean_text)
    data = data.sample(frac=1).reset_index(drop=True)
    return data

# ---------- Model Training ----------
@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer

# ---------- Prediction ----------
def predict_news(news_text, model, vectorizer):
    transformed = vectorizer.transform([news_text])
    prediction = model.predict(transformed)
    return "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Built using Logistic Regression and TF-IDF.")

st.info("Loading and training models...")
data = load_data()
model, vectorizer = train_model(data)
st.success("Model trained successfully!")

# ---------- User Input ----------
user_input = st.text_area("Enter news text to verify:")

if st.button("Detect"):
    if user_input.strip():
        result = predict_news(user_input, model, vectorizer)
        st.subheader("Prediction:")
        st.write(result)
    else:
        st.warning("Please enter some news text.")
