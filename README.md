# Multilingual (Hinglish) Sarcasm Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.12.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning model to detect sarcasm in **Hinglish** (Hindi + English) social media posts. The model is trained on code-mixed text and deployed as a **Streamlit web application**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Sarcasm detection in multilingual text (especially code-mixed languages like Hinglish) is a challenging task due to the informal nature of social media posts. This project aims to build a **machine learning model** to classify text as sarcastic or non-sarcastic. The model is trained on a dataset of Hinglish social media posts and deployed as a **Streamlit web application**.

---

## Features

- **Preprocessing**:
  - Cleans text by removing URLs, mentions, hashtags, and special characters.
  - Handles code-mixed text (Hindi + English) written in Roman script.
  - Removes stopwords in both Hindi and English.

- **Feature Extraction**:
  - Uses **TF-IDF** for text vectorization.
  - Extracts **sentiment features** (polarity and subjectivity) using TextBlob.

- **Model**:
  - Trains a **Multinomial Naive Bayes** classifier for sarcasm detection.
  - Evaluates the model using accuracy, precision, recall, F1-score, and confusion matrix.

- **Deployment**:
  - Deploys the model as a **Streamlit web application**.
  - Allows users to input text and get real-time predictions.

---

## Dataset

The dataset consists of Hinglish social media posts labeled as **sarcastic** or **non-sarcastic**. The dataset is preprocessed to handle code-mixed text and extract relevant features.

### Example Data

| **Text**                                      | **Label**       |
|-----------------------------------------------|-----------------|
| "Haan bilkul, ye toh bahut hi useful feature hai. 😏" | Sarcastic       |
| "Mujhe ye movie bahut pasand aayi. 😊"        | Non-Sarcastic   |
| "Oh sure, because waking up at 5 AM is my favorite hobby. 🙄" | Sarcastic       |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hinglish-sarcasm-detection.git
   cd hinglish-sarcasm-detection


   pip install -r requirements.txt


 ##Acknowledgements
 
Scikit-learn for machine learning tools.

TextBlob for sentiment analysis.

Streamlit for building the web app.

Kaggle for providing datasets.
