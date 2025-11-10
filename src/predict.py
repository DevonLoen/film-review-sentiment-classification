import joblib
from .data_preprocessing import preprocess_text # (Asumsi file ini ada)

# Load SEMUA komponen
print("🧠 Loading model, vectorizer, and selector...")
model = joblib.load('models/sentiment_model.joblib')
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
selector = joblib.load('models/feature_selector.joblib') # <-- TAMBAHKAN INI
print("✅ All components loaded.")

def predict_sentiment(text: str) -> str:
    """Prediksi sentimen untuk sebuah string teks."""
    
    # 1. Bersihkan teks
    cleaned_text = preprocess_text(text)
    
    # 2. Ubah teks bersih dengan vectorizer (TF-IDF)
    #    Input harus berupa list, mis: [cleaned_text]
    text_vector = vectorizer.transform([cleaned_text])
    
    # 3. Pilih fitur menggunakan selector <-- TAMBAHKAN INI
    text_selected = selector.transform(text_vector)
    
    # 4. Prediksi menggunakan model
    #    Prediksi harus dilakukan pada 'text_selected', bukan 'text_vector'
    prediction = model.predict(text_selected)
    
    return prediction[0]