# data_preprocessing.py

import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text: str) -> str:
    """Cleans raw text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)