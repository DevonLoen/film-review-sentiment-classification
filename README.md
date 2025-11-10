# 🎬 Movie Review Sentiment Analysis Model Training

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Library-Scikit--learn-orange.svg)

This repository contains the development and training process of a **machine learning model for classifying movie review sentiments**.
The final trained model artifact is designed to be integrated into a **backend service** for real-world prediction.

---

## 📘 Table of Contents

- [🎯 Project Goal](#-project-goal)
- [📁 Project Structure](#-project-structure)
- [🧠 Technology Stack](#-technology-stack)
- [⚙️ Workflow: Model Generation Steps](#️-workflow-model-generation-steps)
- [💻 Installation](#-installation)
- [🚀 Usage](#-usage)
- [🔮 Future Improvements](#-future-improvements)
- [🧾 License](#-license)
- [📬 Contact](#-contact)

---

## 🎯 Project Goal

The primary goal of this project is to **experiment with various preprocessing techniques and classification algorithms** to produce a robust model capable of accurately predicting whether a given movie review expresses a **positive** or **negative** sentiment.

---

## 📁 Project Structure

```
.
├── data/               # Raw and processed datasets (e.g., CSV files) (excluded via .gitignore)
├── models/             # Trained model artifacts (.joblib, .pkl) (excluded via .gitignore)
├── notebooks/          # Jupyter notebooks for EDA and experimentation
│   └── sentiment_classification.ipynb
├── src/                # Python scripts for preprocessing and training
│   ├── __init__.py     # Marks this folder as a Python package
│   ├── data_preprocessing.py    # Handles text preprocessing (cleaning, delete stopwords, etc.)
│   └── model_training.py        # Script for training and saving the sentiment model
│   └── predict.py               # Script for making sentiment predictions using the trained model
├── venv/               # Virtual environment (excluded via .gitignore)
├── .gitignore          # Files and folders to be ignored by Git
├── main.py             # Entry Point
├── README.md           # Project documentation (you are here)
├── requirements.txt    # List of dependencies
└── setup.py            # Project setup configuration for packaging/distribution
```

---

## 🧠 Technology Stack

| Category            | Tools / Libraries         |
| ------------------- | ------------------------- |
| **Language**        | Python 3.9+               |
| **ML Framework**    | Scikit-learn              |
| **Data Handling**   | Pandas                    |
| **Text Processing** | NLTK / SpaCy _(optional)_ |
| **Visualization**   | Matplotlib, Seaborn       |
| **Development**     | Jupyter Notebook          |

---

## ⚙️ Workflow: Model Generation Steps

The workflow to train and generate the sentiment analysis model is documented in
`notebooks/sentiment_classification.ipynb`.

## 🚀 Sentiment Classification Pipeline

### 1. Data Loading & Exploration

- **Pemuatan Data:** Memuat *dataset* ulasan film (e.g., `data/reviews.csv`). Jika file tidak ditemukan, *script* akan menggunakan data sampel untuk pengembangan.
- **Inspeksi Data:** Memeriksa struktur, ukuran, dan distribusi sentimen (`positive` vs. `negative`).
- **Penanganan Data:** Menangani **missing values** (nilai yang hilang) dan **duplicates** (data ganda) untuk memastikan kualitas data training.

---

### 2. Exploratory Data Analysis (EDA)

- **Word Clouds:** Menghasilkan visualisasi *word cloud* untuk setiap kategori sentimen guna mengidentifikasi kata-kata kunci teratas.
- **Analisis Panjang Review:** Menganalisis distribusi panjang teks untuk mengidentifikasi pola atau anomali data.
- **Analisis N-gram:** Menganalisis frasa (bigram/trigram) teratas untuk menangkap konteks penting seperti negasi (**"tidak bagus"**).

---

### 3. Text Preprocessing & Cleaning (Langkah Wajib)

Langkah ini memastikan teks diubah ke format standar yang dipahami model:

- **Lowercasing & Pembersihan:** Mengubah teks menjadi huruf kecil, menghapus angka, tanda baca, dan tag HTML.
- **Stop Word Removal:** Menghapus kata-kata umum yang tidak menambah nilai sentimen (e.g., *"the", "is", "was"*).
- **Lemmatization:** Mengubah kata-kata kembali ke bentuk dasarnya yang bermakna (e.g., *"running"* menjadi *"run"*), yang **harus** dikonsistenkan antara *training* dan *prediction*.

---

### 4. Feature Engineering (Vectorization)

- **TF-IDF Vectorization:** Mengubah teks bersih menjadi representasi numerik menggunakan **Term Frequency–Inverse Document Frequency (TF-IDF)**.
- **N-gram Inclusion:** Menggunakan `ngram_range=(1, 2)` dalam TF-IDF untuk menangkap *unigram* (kata tunggal) dan *bigram* (dua kata berurutan), meningkatkan akurasi kontekstual.
- **Feature Selection:** Menggunakan **Chi-Square (`chi2`)** dan `SelectKBest` untuk memilih N fitur terbaik, mengurangi *noise* dan kompleksitas komputasi.

---

### 5. Model Training & Evaluation

- **Data Splitting:** Membagi *dataset* menjadi **Training Set** (untuk melatih model) dan **Test Set** (untuk evaluasi).
- **Algoritma Utama:**
    - **Optimized:** **Logistic Regression** (efektif untuk klasifikasi teks, dikenal karena kinerjanya yang kuat).
- **Hyperparameter Tuning:** Menggunakan **GridSearchCV** untuk menemukan kombinasi *hyperparameter* terbaik (`C`, `solver`, dll.) pada Logistic Regression.
- **Metrik Evaluasi:**
    - **Accuracy:** Proporsi prediksi yang benar secara keseluruhan.
    - **Precision, Recall, F1-Score:** Metrik yang lebih fokus pada performa model pada setiap kelas sentimen. **F1-Score** adalah metrik utama untuk keseimbangan.
    - **Confusion Matrix:** Visualisasi yang jelas tentang hasil True Positive, False Positive, True Negative, dan False Negative.

---

### 6. Model Serialization & Deployment Preparation

- **Model Saving:** Menyimpan **TIGA** artefak penting menggunakan `joblib`:
    1.  Model Terbaik (`best_model`).
    2.  Fitted Vectorizer (`tfidf`).
    3.  Fitted Feature Selector (`selector`).
- **Storage:** Menyimpan artefak akhir di folder `models/` untuk memastikan kesiapan deployment (digunakan oleh *script* prediksi).
---

## 💻 Installation

Follow these steps to set up your environment:

### 1. Clone the Repository

```bash
git clone https://github.com/[DevonLoen]/[film-review-sentiment-classification].git
cd [film-review-sentiment-classification]
```

### 2. Create and Activate Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1 — Run via Jupyter Notebook

1. Start Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open the notebook:
   `notebooks/sentiment-classification.ipynb`
3. Run all cells sequentially to reproduce the full workflow.

## 🔮 Future Improvements

- Experiment with **SVM**, **Gradient Boosting**, or **Neural Networks**.
- Use **pre-trained word embeddings** (Word2Vec, GloVe, or BERT).
- Apply **cross-validation** for better generalization.
- Integrate **hyperparameter tuning** using GridSearchCV or Optuna.

---

## 🧾 License

Distributed under the **MIT License**.
See the [LICENSE](./LICENSE) file for more details.

---

## 📬 Contact

**[Devon Marvellous Loen]**
📧 [[devonloen99@gmail.com](mailto:devonloen99@gmail.com)]
🔗 Project Link: [https://github.com/DevonLoen/film-review-sentiment-classification/](https://github.com/DevonLoen/film-review-sentiment-classification/)
