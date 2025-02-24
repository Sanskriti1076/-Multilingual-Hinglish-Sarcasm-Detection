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

# Custom CSS for light and dark mode
st.markdown(
    """
    <style>
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stTextInput>div>div>input {
        background-color: var(--input-bg-color);
        color: var(--input-text-color);
    }
    .stButton>button {
        background-color: var(--button-bg-color);
        color: var(--button-text-color);
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: var(--button-hover-bg-color);
    }
    .stMarkdown {
        color: var(--text-color);
    }
    .stHeader {
        color: var(--header-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Detect the current theme using JavaScript
st.markdown(
    """
    <script>
    function getTheme() {
        const theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        window.parent.postMessage({theme: theme}, '*');
    }
    getTheme();
    </script>
    """,
    unsafe_allow_html=True
)

# Listen for the theme message from JavaScript
theme = st.session_state.get("theme", "light")

# Update the theme based on the message
if "theme" in st.session_state:
    theme = st.session_state["theme"]

# Set CSS variables based on the theme
if theme == "light":
    st.markdown(
        """
        <style>
        :root {
            --background-color: #f0f2f6;
            --text-color: #333333;
            --input-bg-color: #ffffff;
            --input-text-color: #000000;
            --button-bg-color: #4CAF50;
            --button-text-color: #ffffff;
            --button-hover-bg-color: #45a049;
            --header-color: #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        :root {
            --background-color: #0E1117;
            --text-color: #ffffff;
            --input-bg-color: #262730;
            --input-text-color: #ffffff;
            --button-bg-color: #4CAF50;
            --button-text-color: #ffffff;
            --button-hover-bg-color: #45a049;
            --header-color: #4CAF50;
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
