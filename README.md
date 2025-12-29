# 🛍️ Amazon Review Sentiment Analyzer

A machine learning-powered web application that analyzes the sentiment of Amazon product reviews using Natural Language Processing (NLP).

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🎯 Overview

This project analyzes product reviews and predicts whether they are **positive** or **negative** using a trained Logistic Regression model with TF-IDF features.

**Key Highlights:**
- ✅ 90-95% accuracy on test data
- ✅ Real-time sentiment prediction
- ✅ Beautiful, user-friendly web interface
- ✅ Confidence score display
- ✅ Fast and lightweight

---

## ✨ Features

- **Instant Sentiment Analysis**: Get immediate feedback on review sentiment
- **Confidence Scores**: See how confident the model is about its prediction
- **Text Preprocessing**: Automatic cleaning and normalization of input text
- **Visual Feedback**: Color-coded results with emojis (😊 for positive, 😠 for negative)
- **Example Reviews**: Pre-loaded examples to test the app
- **Responsive Design**: Works on desktop and mobile browsers

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

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

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
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

## 💻 Usage

### Running the Web App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Using the Model Programmatically

```python
import pickle
import re
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
stop = set(stopwords.words('english'))
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

# Predict
review = "This product is amazing!"
cleaned = clean(review)
vectorized = vec.transform([cleaned])
sentiment = model.predict(vectorized)[0]
confidence = model.predict_proba(vectorized)[0].max()

print(f"Sentiment: {sentiment} ({confidence*100:.2f}%)")
```

---

## 📁 Project Structure

```
sentiment-analyzer/
│
├── 📊 DATA FILES
│   ├── review dataset.csv              # Original dataset (18.3 MB)
│   ├── cleaned_dataset.csv             # After column selection
│   ├── labeled_dataset.csv             # With sentiment labels
│   ├── final_dataset.csv               # Binary classification
│   └── processed_dataset.csv           # With cleaned text
│
├── 🤖 MODEL FILES
│   ├── model.pkl                       # Trained Logistic Regression
│   └── vectorizer.pkl                  # TF-IDF Vectorizer
│
├── 📝 SCRIPTS
│   ├── app.py                          # Streamlit web app
│   ├── sentiment_project.py            # Model training script
│   └── test_model.py                   # Model testing script
│
├── 📄 DOCUMENTATION
│   ├── README.md                       # This file
│   ├── requirements.txt                # Python dependencies
│   ├── MODEL_SUMMARY.txt               # Model documentation
│   └── cleaning_summary.txt            # Data cleaning report
│
└── 📦 venv/                            # Virtual environment
```

---

## 🧠 Model Details

### Algorithm
- **Model**: Logistic Regression
- **Max Iterations**: 1000
- **Random State**: 42

### Feature Extraction
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features**: 5000 words
- **Vocabulary Size**: ~5000 unique words

### Dataset
- **Total Reviews**: 1,053
- **Training Set**: 842 reviews (80%)
- **Test Set**: 211 reviews (20%)
- **Classes**:
  - Positive: 977 reviews (92.83%)
  - Negative: 76 reviews (7.17%)

### Performance
- **Accuracy**: ~90-95%
- **Precision**: High for positive class
- **Recall**: High for positive class

### Text Preprocessing
1. Lowercase conversion
2. Special character removal
3. Number removal
4. Stopword removal (179 English stopwords)
5. Tokenization

---

## 📸 Screenshots

### Main Interface
![Main Interface](screenshots/main.png)

### Positive Sentiment Result
![Positive Result](screenshots/positive.png)

### Negative Sentiment Result
![Negative Result](screenshots/negative.png)

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and `app.py`
6. Click "Deploy"

### Deploy to Heroku

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Deploy to AWS/GCP

Refer to the respective cloud provider's documentation for deploying Streamlit apps.

---

## 🛠️ Technologies Used

- **Python 3.11**: Programming language
- **Streamlit**: Web framework
- **scikit-learn**: Machine learning library
- **NLTK**: Natural language processing
- **pandas**: Data manipulation
- **NumPy**: Numerical computing

---

## 📊 Training Process

### Step 1: Data Collection
- Collected Amazon product reviews dataset (1,597 reviews)

### Step 2: Data Cleaning
- Removed unnecessary columns (kept only text and rating)
- Removed missing values
- Final dataset: 1,177 reviews

### Step 3: Sentiment Labeling
- Rating ≥ 4 → Positive
- Rating ≤ 2 → Negative
- Rating = 3 → Neutral (removed for binary classification)

### Step 4: Text Preprocessing
- Lowercase conversion
- Special character removal
- Stopword removal
- Text reduction: ~38%

### Step 5: Feature Extraction
- TF-IDF vectorization
- 5000 max features
- Created numerical feature matrix

### Step 6: Model Training
- Train/test split (80/20)
- Logistic Regression with 1000 iterations
- Stratified sampling for balanced sets

### Step 7: Model Evaluation
- Classification report generated
- Confusion matrix analyzed
- Model saved as `model.pkl`

### Step 8: Web App Development
- Streamlit interface created
- Real-time predictions enabled
- Deployed for public use

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**AFSAL AHMED**

- GitHub: [@afsalahmed](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Amazon for the reviews dataset
- scikit-learn for the ML library
- Streamlit for the web framework
- NLTK for NLP tools

---

## 📧 Contact

For questions or feedback, please reach out at: your.email@example.com

---

**⭐ If you found this project helpful, please give it a star!**

---

**Made with ❤️ using Python, Streamlit, and Machine Learning**
