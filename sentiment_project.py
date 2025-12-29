import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Download stopwords
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Load the final dataset
df = pd.read_csv("final_dataset.csv")

print("="*60)
print("SENTIMENT ANALYSIS MODEL TRAINING - BALANCED")
print("="*60)
print(f"Dataset loaded: {df.shape[0]} reviews, {df.shape[1]} columns")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
print("="*60)

# Define text cleaning function
def clean(text):
    text = text.lower()                          # Convert to lowercase
    text = re.sub(r'[^a-z ]', '', text)          # Remove special chars & numbers
    words = [w for w in text.split() if w not in stop]  # Remove stopwords
    return " ".join(words)

# Apply cleaning to review column
print("\n🧹 Cleaning text data...")
df['clean_text'] = df['review'].astype(str).apply(clean)
print("✅ Text cleaning complete!")

# Prepare features and target
X = df['clean_text']
y = df['sentiment']

print("\n📊 TF-IDF Vectorization...")
vec = TfidfVectorizer(max_features=5000)
X_vec = vec.fit_transform(X)
print(f"✅ Vectorization complete! Shape: {X_vec.shape}")
print(f"   Vocabulary size: {len(vec.vocabulary_)} unique words")

# Train-test split
print("\n✂️ Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"✅ Test set:  {X_test.shape[0]} samples")

# Train the model WITH CLASS_WEIGHT='balanced' to handle imbalance
print("\n🤖 Training Logistic Regression model (BALANCED)...")
print("   ⚖️ Using class_weight='balanced' to handle class imbalance")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Model training complete!")

# Make predictions
print("\n🔮 Making predictions on test set...")
pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, pred)
print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"\n{'='*60}")
print("CLASSIFICATION REPORT:")
print(f"{'='*60}")
print(classification_report(y_test, pred))

# Confusion Matrix
print(f"{'='*60}")
print("CONFUSION MATRIX:")
print(f"{'='*60}")
cm = confusion_matrix(y_test, pred)
print(cm)
print("\n[Rows: Actual, Columns: Predicted]")
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Save model and vectorizer
print(f"\n{'='*60}")
print("💾 Saving model and vectorizer...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vec, open("vectorizer.pkl", "wb"))
print("✅ Model saved as: model.pkl")
print("✅ Vectorizer saved as: vectorizer.pkl")
print(f"{'='*60}")

# Test with sample predictions
print("\n🧪 TESTING WITH SAMPLE REVIEWS:")
print("="*60)

test_samples = [
    ("This product is absolutely amazing! I love it so much!", "should be POSITIVE"),
    ("Terrible quality. Waste of money. Very disappointed.", "should be NEGATIVE"),
    ("Great purchase, works perfectly and arrived on time!", "should be POSITIVE"),
    ("Worst product ever. Do not buy this garbage.", "should be NEGATIVE"),
    ("Horrible experience. Broken on arrival.", "should be NEGATIVE"),
    ("Excellent product! Highly recommend!", "should be POSITIVE")
]

for i, (sample, expected) in enumerate(test_samples, 1):
    cleaned = clean(sample)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    print(f"\nSample {i} ({expected}):")
    print(f"Review: {sample}")
    print(f"Prediction: {prediction.upper()}")
    print(f"Confidence: {max(probability)*100:.2f}%")
    print(f"Probabilities: Negative={probability[0]*100:.1f}%, Positive={probability[1]*100:.1f}%")

print("\n" + "="*60)
print("✅ BALANCED MODEL TRAINING COMPLETE!")
print("="*60)
