# predict.py

import joblib
from .data_preprocessing import preprocess_text

# Load the trained model and vectorizer
print("🧠 Loading model and vectorizer...")
model = joblib.load('models/sentiment_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
print("✅ Model and vectorizer loaded.")

def predict_sentiment(text: str) -> str:
    """Predicts sentiment for a given text string."""
    # 1. Clean the input text using the same function
    cleaned_text = preprocess_text(text)
    
    # 2. Transform the cleaned text using the loaded vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    
    # 3. Predict using the loaded model
    prediction = model.predict(text_vector)
    
    return prediction[0]

# # --- Example Usage ---
# if __name__ == "__main__":
#     # Test with some new sentences
#     review1 = "This movie was absolutely fantastic, I loved every second of it!"
#     review2 = "It was a complete waste of time, the acting was terrible."

#     prediction1 = predict_sentiment(review1)
#     prediction2 = predict_sentiment(review2)

#     print("\n--- Predictions ---")
#     print(f"Review: '{review1}'\nSentiment: {prediction1.upper()}")
#     print("-" * 20)
#     print(f"Review: '{review2}'\nSentiment: {prediction2.upper()}")