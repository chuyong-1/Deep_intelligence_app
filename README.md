# 📰 AI News Credibility Checker

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![DistilBERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A Second year project that uses **DistilBERT embeddings**, **stylistic signal analysis**, and **Explainable AI (LIME)** to detect fake news articles. Paste text or provide a URL — the system returns a credibility score with a full breakdown of why it made that decision.

> ⚠️ This tool assists credibility assessment and does not replace human fact-checking.

---

## 📸 Demo

> _Add a screenshot of the app here after running it_
> `screenshots/demo.png`

---

## ✨ Features

- **DistilBERT + Stylistic features** — combines transformer embeddings (768 dims) with 10 hand-crafted fake-news signals for superior accuracy
- **Stylistic signal breakdown** — visual bars showing ALL CAPS ratio, exclamation marks, source citations, alarm words, and more
- **LIME explainability** — highlights which words pushed the model towards Real or Fake
- **URL scraping** — paste any news article URL and the app extracts the text automatically
- **Model metrics dashboard** — accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
- **WELFake + ISOT training** — trained on 70,000+ articles for robust generalisation
- **Dark editorial UI** — clean Streamlit interface with Playfair Display typography

---

## 🧠 How It Works

```
Article Text / URL
       │
       ▼
┌─────────────────────────────┐
│  DistilBERT CLS Embeddings  │  (768 dimensions)
│  +                          │
│  Stylistic Features (×10)   │  ALL CAPS, exclamations, sources...
└─────────────────────────────┘
       │
       ▼
  Logistic Regression Classifier
       │
       ▼
  Credibility Score + LIME Explanation
```

### Stylistic features extracted:
| Feature | Fake signal? |
|---|---|
| ALL CAPS word ratio | High = suspicious |
| Exclamation mark ratio | High = sensationalist |
| Alarm words (BREAKING, deep state...) | High = suspicious |
| Source citation score (according to...) | High = credible |
| Article length | Short = suspicious |
| Numeric facts ratio | High = credible |
| Quoted phrases | High = credible |

---

## 📁 Project Structure

```
news-credibility-checker/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training pipeline
├── requirements.txt        # Dependencies
├── README.md
│
├── data/                   # (not committed — see setup below)
│   ├── Fake.csv
│   ├── True.csv
│   └── WELFake_Dataset.csv
│
└── model/                  # (generated after training)
    ├── model.pkl
    ├── scaler.pkl
    ├── bert_model_name.pkl
    └── metrics.json
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/chuyong-1/news-credibility-checker.git
cd news-credibility-checker
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
```

**Windows PowerShell:**
```bash
.\venv\Scripts\Activate.ps1
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

This project uses two datasets. Download and place them in the `data/` folder.

**ISOT Fake News Dataset**
- Download: [Kaggle — ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Place as: `data/Fake.csv` and `data/True.csv`

**WELFake Dataset** (recommended — 72,000 articles)
- Download: [Kaggle — WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- Place as: `data/WELFake_Dataset.csv`

---

## 🏋️ Training the Model

```bash
python train_model.py
```

This will:
1. Load and merge ISOT + WELFake datasets
2. Extract DistilBERT CLS embeddings for all articles
3. Extract 10 stylistic features per article
4. Train a Logistic Regression classifier on the combined features
5. Save `model/model.pkl`, `model/scaler.pkl`, and `model/metrics.json`

Training time: ~20–40 minutes on CPU depending on dataset size.

To use TF-IDF instead of BERT (faster, less accurate), set `USE_BERT = False` in `train_model.py`.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Model Performance

Results on 20% held-out test set after training on WELFake + ISOT:

| Metric | Score |
|---|---|
| Accuracy | ~97–98% |
| Precision | ~97% |
| Recall | ~98% |
| F1 Score | ~97% |
| ROC-AUC | ~99% |

> Exact scores are saved to `model/metrics.json` after training and displayed live in the app's Metrics tab.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web framework | Streamlit |
| NLP embeddings | DistilBERT (HuggingFace Transformers) |
| Classifier | Scikit-learn Logistic Regression |
| Explainability | LIME |
| URL scraping | newspaper3k |
| Data processing | Pandas, NumPy, NLTK |

---

## 🗺️ Roadmap

- [ ] Domain reputation blacklist check
- [ ] Separate headline/short-text classifier
- [ ] Wikipedia entity verification
- [ ] Analysis history (SQLite)
- [ ] PDF report export
- [ ] Model comparison dashboard (TF-IDF vs BERT vs BERT+Style)

---

## 👤 Author

**Kwok Wai Taifa**
Second Year Project — Machine Learning
GitHub: [@chuyong-1](https://github.com/chuyong-1)

---

## 📄 License

This project is licensed under the MIT License.