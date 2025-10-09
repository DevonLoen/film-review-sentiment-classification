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

### 1. Data Loading & Exploration

- Load dataset (e.g., `data/reviews.csv`).
- Inspect structure, size, and sentiment distribution (positive vs. negative).

### 2. Exploratory Data Analysis (EDA)

- Analyze review length distribution.
- Generate **word clouds** for each sentiment.
- Handle **missing values** and **duplicates**.

### 3. Text Preprocessing & Cleaning

- Convert text to lowercase.
- Remove punctuation, numbers, and special characters.
- Remove stop words (e.g., _“the”, “a”, “is”_).
- _(Optional)_ Apply stemming or lemmatization.

### 4. Feature Engineering (Vectorization)

- Convert text into numerical format using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

### 5. Model Training & Evaluation

- Split dataset into **train/test** sets.
- Train using algorithms **Naive Bayes**.
- Evaluate with **Accuracy**, **Precision**, **Recall**, and **Confusion Matrix**.

### 6. Model Serialization

- Save the trained model (including vectorizer) using `joblib`.
- Store the final artifact in `models/` for later deployment.

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

### Option 2 — Run via Script

Execute the finalized training script:

```bash
python src/train.py
```

This will train the model and save it to the `models/` directory.

---

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
🔗 Project Link: [https://github.com/[DevonLoen]/[]](https://github.com/DevonLoen/film-review-sentiment-classification/)
