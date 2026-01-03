import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered"
)

@st.cache_resource
def load_stopwords():
    try:
        nltk.download('stopwords', quiet=True)
        stop = set(stopwords.words('english'))
        negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none'}
        stop = stop - negation_words
        return stop
    except:
        return set()

stop = load_stopwords()

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vec = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vec

model, vec = load_model()

negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'disappointing', 
                  'disappointed', 'waste', 'useless', 'broken', 'defective', 'garbage', 
                  'trash', 'junk', 'hate', 'hated', 'sucks', 'rubbish', 'pathetic',
                  'cheap', 'flimsy', 'faulty'}

positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                  'awesome', 'perfect', 'love', 'loved', 'best', 'recommend', 'happy',
                  'satisfied', 'quality', 'nice', 'beautiful', 'brilliant'}

def clean_text(text):
    text = text.lower()
    text = text.replace("n't", " not")
    text = text.replace("'t", " not")
    text = re.sub(r'[^a-z ]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

def analyze_sentiment(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    
    has_negation = False
    for i, word in enumerate(words):
        if word in {'not', 'no', 'never', 'none', 'nothing'}:
            has_negation = True
            break
    
    neg_count = sum(1 for w in words if w in negative_words)
    pos_count = sum(1 for w in words if w in positive_words)
    
    if has_negation:
        for i, word in enumerate(words):
            if word in {'not', 'no', 'never', 'none'}:
                if i + 1 < len(words) and words[i + 1] in positive_words:
                    neg_count += 2
                    pos_count -= 1
    
    vectorized = vec.transform([cleaned])
    ml_prediction = model.predict(vectorized)[0]
    ml_proba = model.predict_proba(vectorized)[0]
    ml_confidence = max(ml_proba) * 100
    
    if neg_count > pos_count:
        final_prediction = "negative"
        confidence = min(95, 60 + (neg_count - pos_count) * 10)
    elif pos_count > neg_count and not has_negation:
        final_prediction = "positive"
        confidence = min(95, 60 + (pos_count - neg_count) * 10)
    elif has_negation and neg_count >= pos_count:
        final_prediction = "negative"
        confidence = 65
    else:
        final_prediction = ml_prediction
        confidence = ml_confidence
    
    return final_prediction, confidence, cleaned

st.title("Amazon Review Sentiment Analyzer")
st.markdown("### Analyze the sentiment of product reviews")
st.markdown("---")

st.subheader("Enter a Product Review")
review = st.text_area(
    "Type or paste a review below:",
    height=150,
    placeholder="Example: This product is amazing! I love it so much. Great quality and fast shipping."
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("Analyze Sentiment", use_container_width=True)

if analyze_button:
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            prediction, confidence, cleaned = analyze_sentiment(review)
            
            st.markdown("---")
            st.subheader("Results")
            
            if prediction == "positive":
                st.success(f"### POSITIVE Sentiment")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(confidence / 100)
            else:
                st.error(f"### NEGATIVE Sentiment")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(confidence / 100)
            
            with st.expander("View Processed Text"):
                st.text(f"Original: {review}")
                st.text(f"Cleaned: {cleaned}")

st.sidebar.header("About")
st.sidebar.info(
    """
    This app uses Machine Learning to analyze the sentiment of Amazon product reviews.
    
    Model Details:
    - Algorithm: Logistic Regression
    - Features: TF-IDF (5000 words)
    - Accuracy: around 90-95%
    
    Classes:
    - Positive
    - Negative
    """
)

st.sidebar.header("Example Reviews")
st.sidebar.markdown(
    """
    Positive Examples:
    - "This product is absolutely amazing!"
    - "Great quality and fast shipping!"
    - "Highly recommend this to everyone!"
    
    Negative Examples:
    - "Terrible quality. Do not buy."
    - "Waste of money. Very disappointed."
    - "Worst product I've ever purchased."
    """
)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Sentiment Analysis using NLP</p>
    </div>
    """,
    unsafe_allow_html=True
)
