import pickle
import re
from nltk.corpus import stopwords

stop = set(stopwords.words('english')) - {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none'}
model = pickle.load(open('model.pkl', 'rb'))
vec = pickle.load(open('vectorizer.pkl', 'rb'))

def clean(text):
    text = text.lower()
    text = text.replace("n't", " not")
    text = text.replace("'t", " not")
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

test_reviews = [
    "the product wasn't really good",
    "the product was really bad and not satisfactory",
    "I didn't like this at all",
    "This product is amazing!",
    "Terrible quality, waste of money",
    "bad product",
    "not good",
    "poor quality",
    "excellent service"
]

print("Testing model predictions:")
print("=" * 60)
for review in test_reviews:
    cleaned = clean(review)
    v = vec.transform([cleaned])
    pred = model.predict(v)[0]
    prob = model.predict_proba(v)[0]
    print(f"Review: {review}")
    print(f"Cleaned: {cleaned}")
    print(f"Prediction: {pred.upper()} (Neg: {prob[0]*100:.1f}%, Pos: {prob[1]*100:.1f}%)")
    print("-" * 60)
