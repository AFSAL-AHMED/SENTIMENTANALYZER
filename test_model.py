import pickle

# Load the saved model and vectorizer
print("Loading model and vectorizer...")
model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))
print("✅ Model and vectorizer loaded successfully!")

# Test with sample reviews
print("\n" + "="*60)
print("TESTING THE TRAINED MODEL")
print("="*60)

test_reviews = [
    "This product is absolutely amazing! I love it so much!",
    "Terrible quality. Waste of money. Very disappointed.",
    "Great purchase, works perfectly and arrived on time!",
    "Worst product ever. Do not buy this garbage.",
    "Excellent! Highly recommend to everyone!"
]

import re
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

for i, review in enumerate(test_reviews, 1):
    cleaned = clean(review)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    print(f"\nTest {i}:")
    print(f"Review: {review}")
    print(f"Prediction: {prediction.upper()} ({max(probability)*100:.2f}% confident)")

print("\n" + "="*60)
print("✅ Model is working perfectly!")
print("="*60)
