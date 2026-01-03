import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

nltk.download('stopwords')
stop = set(stopwords.words('english'))

negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none'}
stop = stop - negation_words

df = pd.read_csv("final_dataset.csv")

print("="*60)
print("SENTIMENT ANALYSIS MODEL TRAINING")
print("="*60)
print(f"Dataset loaded: {df.shape[0]} reviews, {df.shape[1]} columns")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
print("="*60)

def clean(text):
    text = text.lower()
    text = text.replace("n't", " not")
    text = text.replace("'t", " not")
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

print("\nCleaning text data...")
df['clean_text'] = df['review'].astype(str).apply(clean)
print("Text cleaning complete!")

X = df['clean_text']
y = df['sentiment']

print("\nTF-IDF Vectorization with bigrams...")
vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_vec = vec.fit_transform(X)
print(f"Vectorization complete! Shape: {X_vec.shape}")
print(f"Vocabulary size: {len(vec.vocabulary_)} unique terms")

print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced', C=0.5)
model.fit(X_train, y_train)
print("Model training complete!")

print("\nMaking predictions on test set...")
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"\n{'='*60}")
print("CLASSIFICATION REPORT:")
print(f"{'='*60}")
print(classification_report(y_test, pred))

print(f"{'='*60}")
print("CONFUSION MATRIX:")
print(f"{'='*60}")
cm = confusion_matrix(y_test, pred)
print(cm)
print("\n[Rows: Actual, Columns: Predicted]")

print(f"\n{'='*60}")
print("Saving model and vectorizer...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vec, open("vectorizer.pkl", "wb"))
print("Model saved as: model.pkl")
print("Vectorizer saved as: vectorizer.pkl")
print(f"{'='*60}")

print("\nTESTING WITH SAMPLE REVIEWS:")
print("="*60)

test_samples = [
    ("This product is absolutely amazing! I love it so much!", "should be POSITIVE"),
    ("Terrible quality. Waste of money. Very disappointed.", "should be NEGATIVE"),
    ("Great purchase, works perfectly and arrived on time!", "should be POSITIVE"),
    ("Worst product ever. Do not buy this garbage.", "should be NEGATIVE"),
    ("Horrible experience. Broken on arrival.", "should be NEGATIVE"),
    ("Excellent product! Highly recommend!", "should be POSITIVE"),
    ("The product was really bad and not satisfactory", "should be NEGATIVE"),
    ("Not good at all, very poor quality", "should be NEGATIVE"),
    ("I do not recommend this product", "should be NEGATIVE"),
    ("The product wasn't really good", "should be NEGATIVE"),
    ("I didn't like this at all", "should be NEGATIVE"),
    ("This isn't what I expected, very bad", "should be NEGATIVE"),
    ("bad product", "should be NEGATIVE"),
    ("not worth the money", "should be NEGATIVE")
]

correct = 0
for i, (sample, expected) in enumerate(test_samples, 1):
    cleaned = clean(sample)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    is_correct = (expected == "should be POSITIVE" and prediction == "positive") or (expected == "should be NEGATIVE" and prediction == "negative")
    if is_correct:
        correct += 1
    status = "OK" if is_correct else "WRONG"
    
    print(f"\n[{status}] Sample {i} ({expected}):")
    print(f"Review: {sample}")
    print(f"Cleaned: {cleaned}")
    print(f"Prediction: {prediction.upper()} ({max(probability)*100:.1f}%)")

print(f"\n{'='*60}")
print(f"Test accuracy: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.0f}%)")
print("="*60)
