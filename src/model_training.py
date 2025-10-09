# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Import the cleaning function from your other file
from data_preprocessing import preprocess_text

print("🚀 Starting model training script...")

# Load Data
df = pd.read_csv('data/reviews.csv')

# Preprocess Data
print("🧹 Cleaning and preprocessing text data...")
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Feature Engineering (TF-IDF) & Data Splitting
tfidf = TfidfVectorizer(max_features=5000)
X = df['cleaned_review']
y = df['sentiment']
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Naive Bayes Model
print("🤖 Training the model...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# Save the Model and Vectorizer
os.makedirs('models', exist_ok=True) # Create models directory if it doesn't exist
joblib.dump(model, 'models/sentiment_model.joblib')
joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')

print("💾 Model and TF-IDF Vectorizer saved successfully in the 'models' folder.")