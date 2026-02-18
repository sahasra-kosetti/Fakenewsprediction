import streamlit as st
import pickle
import base64
from textblob import TextBlob
from PIL import Image

# ---------------- Page Config ---------------- #
st.set_page_config(page_title="Fake News Detection", layout="centered")

# ---------------- Styling ---------------- #
st.markdown(
    """
    <style>
    .stTextArea>div>div>textarea {
        font-size: 16px;
        color: white;
        background-color: black;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load ML Model ---------------- #
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- Intelligence Functions ---------------- #

trusted_sources = {
    "bbc.com": 0.95,
    "ndtv.com": 0.90,
    "thehindu.com": 0.95,
    "indiatimes.com": 0.85
}

def emotion_score(text):
    pol = TextBlob(text).sentiment.polarity
    subj = TextBlob(text).sentiment.subjectivity
    if pol < -0.4 and subj > 0.6:
        return 0.8
    return 0.2

def linguistic_score(text):
    caps = sum(1 for c in text if c.isupper()) / max(len(text),1)
    ex = text.count("!")
    score = 0
    if caps > 0.15: score += 0.4
    if ex > 3: score += 0.4
    return min(score, 1.0)

def source_score(url):
    if not url:
        return 0.5
    for s in trusted_sources:
        if s in url:
            return trusted_sources[s]
    return 0.4

def image_score(image):
    if image is None:
        return 0.5
    img = Image.open(image)
    w, h = img.size
    if w < 300 or h < 300:
        return 0.3
    return 0.7

def fact_score(text):
    words = ["confirmed", "official", "verified", "government", "report"]
    return min(sum(1 for w in words if w in text.lower()) / 4, 1)

# ---------------- UI ---------------- #
st.title("📰 Fake News Intelligence System")
st.write("Multi-signal AI to detect misinformation")

# ---------------- Inputs ---------------- #
image = st.file_uploader("Upload related image (optional)")
url = st.text_input("Source URL (optional)")
author = st.text_input("Author (optional)")
news = st.text_area("Enter News Text", height=220)
analyze = st.button("Analyze")

st.markdown("---")

# ---------------- Prediction ---------------- #
if analyze:

    if news.strip() == "":
        st.warning("Please enter news text")
    else:
        data = vectorizer.transform([news])
        prob = model.predict_proba(data).max()

        emo = emotion_score(news)
        ling = linguistic_score(news)
        src = source_score(url)
        img = image_score(image)
        fact = fact_score(news)

        final = (0.4*prob + 0.15*src + 0.15*fact +
                 0.15*(1-emo) + 0.15*(1-ling)) * 100

        if final > 60:
            st.success(f"🟢 Likely REAL — Truth Score: {final:.2f}%")
        else:
            st.error(f"🔴 Likely FAKE — Truth Score: {final:.2f}%")

        st.markdown("### 🔍 Why?")
        st.write(f"Text ML: {prob*100:.1f}%")
        st.write(f"Source: {src*100:.1f}%")
        st.write(f"Fact: {fact*100:.1f}%")
        st.write(f"Emotion Manipulation: {emo*100:.1f}%")
        st.write(f"Linguistic Manipulation: {ling*100:.1f}%")
        st.write(f"Image Reliability: {img*100:.1f}%")

        if final < 60:
            st.warning("⚠ Possible misinformation detected")
            st.write("Check trusted sources:")
            st.write("https://www.bbc.com")
            st.write("https://www.thehindu.com")
