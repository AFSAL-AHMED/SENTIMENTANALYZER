import pickle
import re
from nltk.corpus import stopwords

# Load model
model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))

print("="*60)
print("MODEL VERIFICATION")
print("="*60)

# Check if model has class_weight parameter
print(f"\nModel type: {type(model).__name__}")
print(f"Model params: {model.get_params()}")
print(f"\nClass weight: {model.class_weight}")

# Test with negative reviews
print("\n" + "="*60)
print("TESTING NEGATIVE REVIEWS:")
print("="*60)

stop = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

negative_tests = [
    "Terrible quality. Do not buy.",
    "Waste of money. Very disappointed.",
    "Worst product ever.",
    "Horrible experience. Broken on arrival.",
    "Complete garbage. Save your money.",
]

print("\nNegative Review Tests:")
for i, review in enumerate(negative_tests, 1):
    cleaned = clean(review)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    
    print(f"\n{i}. '{review}'")
    print(f"   Cleaned: '{cleaned}'")
    print(f"   Prediction: {prediction.upper()}")
    print(f"   Probabilities: [Neg: {proba[0]:.2%}, Pos: {proba[1]:.2%}]")
    print(f"   Status: {'✅ CORRECT' if prediction == 'negative' else '❌ WRONG - STILL BIASED!'}")

print("\n" + "="*60)
print("TESTING POSITIVE REVIEWS:")
print("="*60)

positive_tests = [
    "This product is amazing!",
    "Great quality. Highly recommend!",
    "Excellent purchase!",
]

print("\nPositive Review Tests:")
for i, review in enumerate(positive_tests, 1):
    cleaned = clean(review)
    vectorized = vec.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    
    print(f"\n{i}. '{review}'")
    print(f"   Prediction: {prediction.upper()}")
    print(f"   Probabilities: [Neg: {proba[0]:.2%}, Pos: {proba[1]:.2%}]")
    print(f"   Status: {'✅ CORRECT' if prediction == 'positive' else '❌ WRONG'}")

print("\n" + "="*60)
