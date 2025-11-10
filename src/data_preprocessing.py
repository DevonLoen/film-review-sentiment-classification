# --- INI FUNGSI YANG BENAR UNTUK PREDIKSI ---
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Pastikan resource NLTK ada (penting untuk server/deployment)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Inisialisasi satu kali saja
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """
    Fungsi untuk membersihkan dan memproses teks review:
    INI HARUS SAMA PERSIS DENGAN FUNGSI TRAINING.
    """
    # 1. Menghapus tag HTML
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Lowercasing
    text = text.lower()

    # 3. Menghapus karakter selain huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 4. Tokenisasi
    words = text.split()

    # 5. Lemmatization dan penghapusan stopwords
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return ' '.join(words)