
# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Set the NLTK data path and download resources
nltk.data.path.append("C:/Users/ASUSD/AppData/Roaming/nltk_data")
nltk.download('punkt')
nltk.download('stopwords')

# Define stopwords for Hindi and English
stop_words_english = set(stopwords.words('english'))
stop_words_hindi = set([
    "hai", "ho", "hain", "se", "mein", "ko", "ka", "ki", "ke", "ne", "par", "kya", "nahi", "tha", "thi", "the", "hum", "tum", "apne", "liye", "aur", "bhi", "ye", "toh"
])
stop_words = stop_words_english.union(stop_words_hindi)

# Initialize the TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

# Text preprocessing functions
def clean_text(text):
    """
    Clean text by removing URLs, mentions, hashtags, special characters, and extra spaces.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove mentions and hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

def tokenize_text(text):
    """
    Tokenize text into words using TreebankWordTokenizer.
    """
    return tokenizer.tokenize(text)

def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.
    """
    return [word for word in tokens if word.lower() not in stop_words]

def preprocess_text(text):
    """
    Full preprocessing pipeline: cleaning, tokenization, and stopword removal.
    """
    cleaned_text = clean_text(text)  # Step 1: Clean text
    tokens = tokenize_text(cleaned_text)  # Step 2: Tokenize text
    filtered_tokens = remove_stopwords(tokens)  # Step 3: Remove stopwords
    return " ".join(filtered_tokens)  # Step 4: Join tokens into a sentence

# Load the dataset
train_df = pd.read_csv(r"C:\Users\ASUSD\Downloads\New folder (3)\train .csv")
test_df = pd.read_csv(r"C:\Users\ASUSD\Downloads\New folder (3)\test.csv")

# Drop rows with missing target values
train_df = train_df.dropna(subset=['Label'])
test_df = test_df.dropna(subset=['Label'])

# Verify no missing values remain
print("Missing values in train_df['Label'] after handling:", train_df['Label'].isna().sum())
print("Missing values in test_df['Label'] after handling:", test_df['Label'].isna().sum())

# Convert labels to binary format
train_df['Label'] = train_df['Label'].map({'yes': 1, 'no': 0})
test_df['Label'] = test_df['Label'].map({'yes': 1, 'no': 0})

# Verify the conversion
print("Unique values in train_df['Label'] after conversion:", train_df['Label'].unique())
print("Unique values in test_df['Label'] after conversion:", test_df['Label'].unique())

# Parallelize preprocessing for faster execution
train_df['processed_text'] = Parallel(n_jobs=-1)(delayed(preprocess_text)(text) for text in train_df['Tweet'])
test_df['processed_text'] = Parallel(n_jobs=-1)(delayed(preprocess_text)(text) for text in test_df['Tweet'])

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Reduced features for faster processing
X_train = vectorizer.fit_transform(train_df['processed_text'])
y_train = train_df['Label']
X_test = vectorizer.transform(test_df['processed_text'])
y_test = test_df['Label']

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'text_classification_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot ROC Curve (for binary classification)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Hyperparameter tuning using GridSearchCV (optional)
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, n_jobs=-1)  # Parallelize grid search
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Save the best model from GridSearchCV
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_text_classification_model.pkl')