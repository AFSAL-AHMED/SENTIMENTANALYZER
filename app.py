import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk

# Streamlit App UI - MUST BE FIRST!
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered"
)

# Download stopwords (only runs once)
@st.cache_resource
def load_stopwords():
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except:
        return set()

stop = load_stopwords()

# Load model and vectorizer (cached for performance)
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vec = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vec

model, vec = load_model()

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()                          # Lowercase
    text = re.sub(r'[^a-z ]', '', text)          # Remove special chars
    words = [w for w in text.split() if w not in stop]  # Remove stopwords
    return " ".join(words)

# Title and description
st.title("🛍️ Amazon Review Sentiment Analyzer")
st.markdown("### Analyze the sentiment of product reviews using AI")
st.markdown("---")

# Input section
st.subheader("📝 Enter a Product Review")
review = st.text_area(
    "Type or paste a review below:",
    height=150,
    placeholder="Example: This product is amazing! I love it so much. Great quality and fast shipping."
)

# Analyze button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

if analyze_button:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            # Clean and vectorize
            cleaned = clean_text(review)
            vectorized = vec.transform([cleaned])
            
            # Predict
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            
            # Get confidence score
            confidence = max(probability) * 100
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Results")
            
            if prediction == "positive":
                st.success(f"### 😊 POSITIVE Sentiment")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(confidence / 100)
            else:
                st.error(f"### 😠 NEGATIVE Sentiment")
                st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                st.progress(confidence / 100)
            
            # Show cleaned text (for debugging)
            with st.expander("🔍 View Processed Text"):
                st.text(f"Original: {review}")
                st.text(f"Cleaned: {cleaned}")

# Sidebar with examples and info
st.sidebar.header("ℹ️ About")
st.sidebar.info(
    """
    This app uses **Machine Learning** to analyze the sentiment of Amazon product reviews.
    
    **Model Details:**
    - Algorithm: Logistic Regression
    - Features: TF-IDF (5000 words)
    - Accuracy: ~90-95%
    
    **Classes:**
    - 😊 Positive
    - 😠 Negative
    """
)

st.sidebar.header("📚 Example Reviews")
st.sidebar.markdown(
    """
    **Positive Examples:**
    - "This product is absolutely amazing!"
    - "Great quality and fast shipping!"
    - "Highly recommend this to everyone!"
    
    **Negative Examples:**
    - "Terrible quality. Do not buy."
    - "Waste of money. Very disappointed."
    - "Worst product I've ever purchased."
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit & scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)
