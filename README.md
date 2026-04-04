# 📰 Verity — AI News Credibility Checker

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56+-red?style=flat-square&logo=streamlit)
![DistilBERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A second-year machine learning project that uses **DistilBERT embeddings**, **16 stylistic signal features**, **Wikipedia entity verification**, and **Explainable AI (LIME)** to assess the credibility of news articles. Paste text or provide a URL — the system returns a credibility score with a full breakdown of why it made that decision.

> ⚠️ This tool assists credibility assessment and does not replace human fact-checking.

---

## 📸 Screenshots

![App Screenshot](screenshots/demo.png)

---

## ✨ Features

- **DistilBERT + 16 Stylistic Features** — combines transformer embeddings (768 dims) with hand-crafted fake-news signals for superior accuracy
- **Animated SVG gauge** — colour-coded credibility meter with smooth needle animation
- **LIME explainability** — highlights which words pushed the model towards Real or Fake, cached per article
- **Wikipedia entity verification** — checks named people, organisations, and places against Wikipedia and Wikidata; contributes 30% to the final score
- **Sentence-level suspicion scoring** — each sentence scored 0–1 for suspiciousness based on signal patterns
- **URL scraping** — paste any news article URL and the app extracts text automatically via `newspaper3k`
- **Batch mode** — analyse multiple articles at once separated by `---`
- **Domain reputation check** — flags known low-credibility and satire domains
- **Model metrics dashboard** — accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix, and CV results
- **Analysis history** — sidebar tracks your last 20 analyses during the session
- **GPU-accelerated training** — CUDA support for RTX cards, with checkpoint/resume on crash
- **Editorial dark UI** — redesigned interface with DM Serif Display + Syne typography

---

## 🧠 How It Works

```
Article Text / URL
       │
       ▼
┌──────────────────────────────────┐
│  DistilBERT CLS Embeddings       │  768 dimensions
│  +                               │
│  16 Stylistic Features           │  CAPS, exclamations, sources, hedging...
└──────────────────────────────────┘
       │
       ▼
  Logistic Regression (CalibratedClassifierCV)
       │
       ├──── ML Credibility Score (70% weight)
       │
       ├──── Wikipedia Entity Score (30% weight)
       │
       ▼
  Final Score + LIME Word Explanation + Sentence Breakdown
```

### 16 Stylistic Features

| # | Feature | Signal |
|---|---|---|
| 1 | ALL CAPS ratio | High = suspicious |
| 2 | Title Case ratio | Slightly elevated in fake |
| 3 | Exclamation ratio | High = sensationalist |
| 4 | Question mark count | Neutral |
| 5 | Ellipsis count | Neutral |
| 6 | Source citations | High = credible |
| 7 | Alarm words (BREAKING, deep state…) | High = suspicious |
| 8 | Article length (log) | Longer = credible |
| 9 | Quoted phrases | High = credible |
| 10 | Numeric facts ratio | High = credible |
| 11 | Avg sentence length | Longer = credible |
| 12 | Passive voice ratio | Higher = journalistic |
| 13 | Hedging language | High = cautious/credible |
| 14 | Sentiment polarity | Extreme adjectives = suspicious |
| 15 | Punctuation abuse (!!!/???) | High = sensationalist |
| 16 | First-person ratio | High = editorialising |

---

## 📁 Project Structure

```
news-credibility-checker/
│
├── app.py                  # Streamlit web application (v3)
├── train_model.py          # Model training pipeline (v3, GPU + checkpoint)
├── features.py             # 16 stylistic feature extraction (shared)
├── entity_checker.py       # Wikipedia + Wikidata entity verification (v2)
├── Sentence_scorer.py      # Per-sentence suspicion scoring
├── requirements.txt        # Python dependencies
├── README.md
│
├── data/                   # (not committed — download separately)
│   ├── Fake.csv
│   ├── True.csv
│   └── WELFake_Dataset.csv
│
├── model/                  # (generated after training)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── bert_model_name.pkl
│   ├── metrics.json
│   └── checkpoints/        # auto-saved training checkpoints
│
└── screenshots/
    └── demo.png
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/chuyong-1/news-credibility-checker.git
cd news-credibility-checker
```

### 2. Install Python 3.11
PyTorch does not yet support Python 3.13. Use Python 3.11 or 3.12.

### 3. Install PyTorch with CUDA (for GPU training)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
> For CPU-only: `pip install torch torchvision torchaudio`

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 5. Install spaCy model (for entity verification)
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

---

## 📦 Dataset Setup

Download and place in the `data/` folder:

**ISOT Fake News Dataset**
- [Kaggle — ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Files: `data/Fake.csv` and `data/True.csv`

**WELFake Dataset** (recommended — 72,000 articles)
- [Kaggle — WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- File: `data/WELFake_Dataset.csv`

---

## 🏋️ Training the Model

```bash
python train_model.py
```

This will:
1. Load and balance ISOT + WELFake datasets
2. Extract 16 stylistic features per article
3. Extract DistilBERT CLS embeddings (GPU-accelerated if available)
4. Run 5-fold cross-validation
5. Train a calibrated Logistic Regression classifier
6. Save model, scaler, and metrics to `model/`

**If training crashes**, just re-run — checkpoints are saved every 100 batches in `model/checkpoints/` so it resumes automatically.

### Training config (top of `train_model.py`)
```python
BATCH_SIZE       = 64       # reduce to 32 if CUDA out-of-memory
MAX_ROWS_ISOT    = 10_000   # reduce for faster training
MAX_ROWS_WELFAKE = 50_000   # reduce for faster training
```

### Estimated training time
| Hardware | Time |
|---|---|
| CPU only | ~40–60 min |
| RTX 4050 (batch 64) | ~2–4 min |
| Google Colab T4 | ~5–8 min |

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Model Performance

Results on 20% held-out test set (WELFake + ISOT, balanced):

| Metric | Score |
|---|---|
| Accuracy | ~97–98% |
| Precision | ~97% |
| Recall | ~98% |
| F1 Score | ~97% |
| ROC-AUC | ~99% |
| Brier Score | ~0.03 |

> Exact scores saved to `model/metrics.json` and shown live in the app's Metrics tab.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Web framework | Streamlit 1.56 |
| NLP embeddings | DistilBERT (HuggingFace Transformers) |
| Deep learning | PyTorch 2.5 + CUDA 12.1 |
| Classifier | Scikit-learn Logistic Regression (CalibratedClassifierCV) |
| Explainability | LIME |
| Entity verification | Wikipedia REST API + Wikidata API + spaCy |
| Sentence scoring | Custom rule-based scorer (Sentence_scorer.py) |
| URL scraping | newspaper3k |
| Data processing | Pandas, NumPy, NLTK |
| Visualisation | Plotly (radar chart), SVG (gauge) |

---

## 🗺️ Roadmap

- [x] DistilBERT + stylistic feature hybrid model
- [x] LIME word-level explainability
- [x] Wikipedia entity verification
- [x] Sentence-level suspicion scoring
- [x] URL article scraping
- [x] Batch mode
- [x] Domain reputation blacklist
- [x] GPU-accelerated training with checkpoint/resume
- [x] Analysis history sidebar
- [x] Animated SVG gauge + radar chart
- [ ] PDF report export
- [ ] SQLite analysis history (persistent across sessions)
- [ ] Model comparison dashboard (TF-IDF vs BERT vs BERT+Style)
- [ ] Separate headline/short-text classifier

---

## 👤 Author

**Kwok Wai Taifa**
Second Year Project — Machine Learning
GitHub: [@chuyong-1](https://github.com/chuyong-1)

---

## 📄 License

This project is licensed under the MIT License.