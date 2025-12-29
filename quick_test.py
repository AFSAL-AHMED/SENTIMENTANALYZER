import pickle

# Test the newly trained model
print("Testing the BALANCED model...")
print("="*60)

model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))

import re
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

# Test reviews
test_reviews = [
    "Terrible quality. Do not buy.",
    "This product is absolutely amazing!",
    "Waste of money. Very disappointed.",
    "Great purchase! Highly recommend!",
    "Worst product ever. Broken on arrival.",
    "Excellent quality and fast shipping!"
]

print("\nTesting Reviews:\n")
for i, review in enumerate(test_reviews, 1):
    cleaned = clean(review)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    print(f"{i}. Review: {review}")
    print(f"   Prediction: {prediction.upper()}")
    print(f"   Confidence: {max(probability)*100:.2f}%")
    print(f"   [Neg: {probability[0]*100:.1f}%, Pos: {probability[1]*100:.1f}%]")
    print()

print("="*60)
print("✅ Model test complete!")
