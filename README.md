# Amazon Review Sentiment Analyzer

A web application that analyzes the sentiment of Amazon product reviews using Natural Language Processing.

---

## Overview

This project takes product reviews as input and predicts whether they are positive or negative using a trained Logistic Regression model with TF-IDF features.

The model achieves around 90-95% accuracy on test data.

---

## Features

- Instant Sentiment Analysis
- Confidence Scores for predictions
- Automatic text preprocessing and cleaning
- Color-coded results display
- Example reviews to test the app

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sentiment-analyzer.git
cd sentiment-analyzer
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Download NLTK Data

```python
python -c "import nltk; nltk.download('stopwords')"
```

---

## Usage

### Running the Web App

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

### Using the Model in Code

```python
import pickle
import re
from nltk.corpus import stopwords

model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))

stop = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

review = "This product is amazing!"
cleaned = clean(review)
vectorized = vec.transform([cleaned])
sentiment = model.predict(vectorized)[0]
confidence = model.predict_proba(vectorized)[0].max()

print(f"Sentiment: {sentiment} ({confidence*100:.2f}%)")
```

---

## Project Structure

```
sentiment-analyzer/
|
|-- DATA FILES
|   |-- review dataset.csv
|   |-- cleaned_dataset.csv
|   |-- labeled_dataset.csv
|   |-- final_dataset.csv
|   |-- processed_dataset.csv
|
|-- MODEL FILES
|   |-- model.pkl
|   |-- vectorizer.pkl
|
|-- SCRIPTS
|   |-- app.py
|   |-- sentiment_project.py
|   |-- test_model.py
|
|-- DOCUMENTATION
|   |-- README.md
|   |-- requirements.txt
|   |-- MODEL_SUMMARY.txt
|   |-- cleaning_summary.txt
|
|-- venv/
```

---

## Model Details

### Algorithm
- Logistic Regression
- Max Iterations: 1000
- Random State: 42

### Feature Extraction
- TF-IDF Vectorization
- Max Features: 5000 words

### Dataset
- Total Reviews: 1,053
- Training Set: 842 reviews (80%)
- Test Set: 211 reviews (20%)
- Positive: 977 reviews (92.83%)
- Negative: 76 reviews (7.17%)

### Text Preprocessing Steps
1. Lowercase conversion
2. Special character removal
3. Number removal
4. Stopword removal
5. Tokenization

---

## Technologies Used

- Python 3.11
- Streamlit
- scikit-learn
- NLTK
- pandas
- NumPy

---

## Training Process

1. Data Collection - Collected Amazon product reviews dataset (1,597 reviews)

2. Data Cleaning - Removed unnecessary columns, removed missing values. Final dataset: 1,177 reviews

3. Sentiment Labeling - Rating 4 and above = Positive, Rating 2 and below = Negative, Rating 3 was removed for binary classification

4. Text Preprocessing - Lowercase, special character removal, stopword removal. Text reduction around 38%

5. Feature Extraction - TF-IDF vectorization with 5000 max features

6. Model Training - 80/20 train/test split, Logistic Regression with 1000 iterations

7. Model Evaluation - Classification report and confusion matrix generated

8. Web App - Built using Streamlit for real-time predictions
