# Sentiment Analysis Using Naive Bayes

## Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify text reviews as **Positive** or **Negative**.

The project uses TF-IDF Vectorization to convert text into numerical features and a Naive Bayes classifier to predict sentiment.

An interactive web application is built using Streamlit, allowing users to enter a review and get its predicted sentiment.

## Live Demo

Try the Live Application
https://naive-bayes-sentiment.streamlit.app/

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF Vectorization
- Naive Bayes
- Joblib
- Streamlit

## Project Workflow

1. Load and explore the sentiment dataset.
2. Clean and preprocess the text data using NLP techniques.
3. Convert text into numerical features using TF-IDF.
4. Train a Naive Bayes classification model.
5. Save the trained model and TF-IDF vectorizer using Joblib.
6. Build an interactive application using Streamlit.
7. Deploy the application using Streamlit Community Cloud.

## Application Features

- Enter a text review.
- Predict whether the review is Positive or Negative.
- View the prediction through an interactive interface.

## Project Structure

Sentiment_Analysis/
│
├── app.py
├── Naive Bayes on review classification.ipynb
├── requirements.txt
├── sentiment_data.csv
├── sentiment_model.pkl
└── tfidf_vectorizer.pkl


## Installation and Setup

### 1. Clone the repository

git clone https://github.com/Niharika3532/Sentiment-Analysis.git

### 2. Navigate to the project folder

cd Sentiment-Analysis

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the application

streamlit run app.py


## Model Information

| **Component** | **Description** |
|:---|:---|
| Algorithm | Naive Bayes Classifier |
| Feature Extraction | TF-IDF Vectorization |
| Input | Text Review |
| Output | Positive / Negative Sentiment |
| Model File | sentiment_model.pkl |
| Vectorizer File | tfidf_vectorizer.pkl |

## Learning Outcomes

- Understanding text preprocessing techniques.
- Learning TF-IDF feature extraction.
- Training a Naive Bayes classification model.
- Saving and loading machine learning models.
- Deploying an ML application using Streamlit.

## Author

**Niharika**
