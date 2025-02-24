import streamlit as st
import joblib
import re
from textblob import TextBlob
import pandas as pd
import scipy.sparse as sp

# Load the saved model and vectorizer
model = joblib.load('sarcasm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define stopwords for Hindi and English
stop_words_english = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
    "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

stop_words_hindi = set([
    "hai", "ho", "hain", "se", "mein", "ko", "ka", "ki", "ke", "ne", "par", "kya", "nahi", "tha", "thi", "the", "hum", "tum", "apne", "liye", "aur", "bhi", "ye", "toh"
])

# Combine English and Hindi stopwords
stop_words = stop_words_english.union(stop_words_hindi)

# Text preprocessing functions
def clean_text(text):
    """
    Clean text by removing URLs, mentions, hashtags, special characters, and extra spaces.
    """
    # Combine all regex operations into a single pass
    text = re.sub(r"http\S+|www\S+|https\S+|@\w+|#\w+|[^a-zA-Z\s]", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def remove_stopwords(text):
    """
    Remove stopwords from a text.
    """
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def preprocess_text(text):
    """
    Full preprocessing pipeline: cleaning and stopword removal.
    """
    cleaned_text = clean_text(text)  # Step 1: Clean text
    filtered_text = remove_stopwords(cleaned_text)  # Step 2: Remove stopwords
    return filtered_text

# Streamlit App
st.set_page_config(page_title="Sarcasm Detection", page_icon="😏", layout="centered")

# Custom CSS for colorful design
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMarkdown {
        color: #333333;
    }
    .stHeader {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("😏 **Multilingual Sarcasm Detection**")
st.markdown("Detect sarcasm in Hinglish (Hindi + English) social media posts!")

# Input Text Box
user_input = st.text_area("Enter your text here:", placeholder="Type a Hinglish sentence...")

# Predict Button
if st.button("Detect Sarcasm"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)

        # Vectorize the processed text
        X_input = vectorizer.transform([processed_text])

        # Make prediction
        prediction = model.predict(X_input)
        prediction_proba = model.predict_proba(X_input)[0][1]  # Probability of sarcasm

        # Display result
        if prediction[0] == 1:
            st.success("**Prediction:** Sarcastic 😏")
        else:
            st.success("**Prediction:** Not Sarcastic 😐")

        # Display prediction probability
        st.write(f"**Confidence:** {prediction_proba * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by [Quantum Coders]")