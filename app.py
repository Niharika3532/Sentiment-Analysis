import streamlit as st
st.set_page_config(
    page_title="Restaurant Review Sentiment",
    page_icon="🍽️",
    layout="centered"
)
import streamlit as st
import joblib
import re
import contractions
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
stop_words = stop_words - {'not', 'no', 'nor'}
lemmatizer = WordNetLemmatizer()
tfidf = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')
def clean_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = ' '.join(
        word for word in text.split()
        if word not in stop_words
    )

    text = ' '.join(
        lemmatizer.lemmatize(word)
        for word in text.split()
    )

    return text
st.set_page_config(
    page_title="Restaurant Review Sentiment",
    page_icon="🍽️",
    layout="centered"
)
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #17121F, #302033);
    color: white;
}

/* Headings */
h1, h2, h3 {
    font-family: Arial, sans-serif;
}

/* Review label */
.stTextArea label {
    color: #FFC72C !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

/* Review box */
.stTextArea textarea {
    background-color: #292536 !important;
    color: white !important;
    border: 2px solid #FFC72C !important;
    border-radius: 15px !important;
    font-size: 16px !important;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(90deg, #FFC72C, #FF9800) !important;
    color: #241500 !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    transition: 0.3s ease;
}

/* Button hover */
div.stButton > button:hover {
    background: #FFE082 !important;
    color: #241500 !important;
    transform: scale(1.02);
}

/* Remove focus outline */
div.stButton > button:focus {
    box-shadow: 0 0 0 2px #FFC72C !important;
}

/* Result spacing */
div[data-testid="stMarkdownContainer"] {
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
    <h1 style='text-align: center; color: #FFC72C;'>
        🍽️ Restaurant Review
    </h1>
    <h1 style='text-align: center; color: white;'>
        Sentiment Analysis
    </h1>
    <p style='text-align: center; color: #D3D3D3;'>
        Good Food. Good Mood! 😋
    </p>
""", unsafe_allow_html=True)
st.markdown("### 📝 Share Your Food Experience")

review = st.text_area(
    "Enter your restaurant review:",
    placeholder="E.g. The food was delicious and the service was amazing!",
    height=120
)
predict_button = st.button(
    "🎯  Predict Sentiment",
    use_container_width=True,
    type="primary"
)
if predict_button:
    if review.strip():

        cleaned_review = clean_text(review)

        review_vector = tfidf.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            st.markdown("""
                <div style="
                    background-color: #123D25;
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #00C853;
                    text-align: center;
                ">
                    <h2 style="color: #00E676;">
                        😍 Positive Sentiment!
                    </h2>
                    <h3 style="color: white;">
                        Great Food, Great Experience! 👍
                    </h3>
                    <p style="color: white;">
                        This review expresses a positive opinion.
                    </p>
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div style="
                    background-color: #451C25;
                    padding: 25px;
                    border-radius: 15px;
                    border: 2px solid #FF5252;
                    text-align: center;
                ">
                    <h2 style="color: #FF5252;">
                        😞 Negative Sentiment!
                    </h2>
                    <h3 style="color: white;">
                        Oh no! Not a great experience. 👎
                    </h3>
                    <p style="color: white;">
                        This review expresses a negative opinion.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("👀 Please enter a review first!")