import os
import json
import re
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

nltk.download("stopwords", quiet=True)

# --------------------------------------------------
# Config
# --------------------------------------------------
USE_BERT         = True
BERT_MODEL       = "distilbert-base-uncased"
BATCH_SIZE       = 32
MAX_SEQ_LEN      = 256
RANDOM_STATE     = 42
MAX_ROWS_ISOT    = 10000   # per class from ISOT
MAX_ROWS_WELFAKE = 50000   # total from WELFake

os.makedirs("model", exist_ok=True)
stop_words = set(stopwords.words("english"))

# --------------------------------------------------
# Stylistic features
# --------------------------------------------------
SOURCE_PHRASES = [
    "according to", "said in a statement", "told reporters",
    "confirmed by", "spokesperson said", "officials said",
    "research shows", "study found", "published in",
    "percent", "per cent", "%",
]
ALARM_WORDS = [
    "breaking", "exclusive", "shocking", "bombshell", "exposed",
    "conspiracy", "deep state", "globalist", "whistleblower",
    "they don't want", "mainstream media", "hoax",
    "secret", "banned", "censored", "suppressed", "wake up",
    "share before deleted", "must read", "urgent",
]

def stylistic_features(text: str) -> np.ndarray:
    words       = text.split()
    num_words   = max(len(words), 1)
    lower       = text.lower()
    alpha_words = [w for w in words if w.isalpha()]

    caps_ratio     = sum(1 for w in alpha_words if w.isupper() and len(w) > 1) / max(len(alpha_words), 1)
    title_ratio    = sum(1 for w in alpha_words if w.istitle()) / max(len(alpha_words), 1)
    exclaim_ratio  = text.count("!") / num_words
    question_count = text.count("?")
    ellipsis_count = text.count("...")
    source_score   = sum(1 for p in SOURCE_PHRASES if p in lower)
    alarm_score    = sum(1 for w in ALARM_WORDS if w in lower)
    log_length     = np.log1p(num_words)
    quote_count    = text.count('"') // 2
    number_ratio   = len(re.findall(r'\b\d+\.?\d*\b', text)) / num_words

    return np.array([
        caps_ratio, title_ratio, exclaim_ratio, question_count,
        ellipsis_count, source_score, alarm_score, log_length,
        quote_count, number_ratio,
    ], dtype=np.float32)

STYLE_DIM = 10

def batch_stylistic(texts: list) -> np.ndarray:
    return np.vstack([stylistic_features(t) for t in texts])

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return " ".join(w for w in text.split() if w not in stop_words)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
print("📂 Loading datasets...")
frames = []

for path, label in [("data/Fake.csv", 1), ("data/True.csv", 0)]:
    if os.path.exists(path):
        df_tmp = pd.read_csv(path, nrows=MAX_ROWS_ISOT)
        df_tmp["label"] = label
        frames.append(df_tmp)
        print(f"  ✅ ISOT {path}: {len(df_tmp):,} rows")
    else:
        print(f"  ⚠️  {path} not found — skipping.")

welfake_path = "data/WELFake_Dataset.csv"
if os.path.exists(welfake_path):
    df_wel = pd.read_csv(welfake_path, nrows=MAX_ROWS_WELFAKE)
    df_wel.columns = [c.strip().lower() for c in df_wel.columns]
    if "label" in df_wel.columns:
        frames.append(df_wel)
        print(f"  ✅ WELFake: {len(df_wel):,} rows")
    else:
        print("  ⚠️  WELFake: no 'label' column — skipping.")
else:
    print(
        "  ⚠️  WELFake not found at data/WELFake_Dataset.csv\n"
        "     Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
        "     Continuing with ISOT only."
    )

if not frames:
    raise RuntimeError("No datasets loaded. Check your data/ folder.")

df = pd.concat(frames, ignore_index=True)

if "title" in df.columns and "text" in df.columns:
    df["content"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
elif "text" in df.columns:
    df["content"] = df["text"].fillna("").astype(str)
else:
    raise RuntimeError("Dataset must have a 'text' column.")

df = df.dropna(subset=["content", "label"])
df = df[df["content"].str.strip().str.len() > 20]

before = len(df)
df = df.drop_duplicates(subset=["content"])
print(f"\n📊 {len(df):,} articles after dedup (removed {before - len(df):,})")
print(f"   Fake: {(df['label']==1).sum():,}  |  Real: {(df['label']==0).sum():,}")

# Balance classes
min_class = min((df["label"] == 0).sum(), (df["label"] == 1).sum())
df = pd.concat([
    df[df["label"] == 0].sample(min_class, random_state=RANDOM_STATE),
    df[df["label"] == 1].sample(min_class, random_state=RANDOM_STATE),
]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"   Balanced: {len(df):,} total ({min_class:,} per class)")

X_content = df["content"].tolist()
y = df["label"].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_content, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

# --------------------------------------------------
# Stylistic features
# --------------------------------------------------
print("\n🎨 Extracting stylistic features...")
X_train_style = batch_stylistic(X_train_raw)
X_test_style  = batch_stylistic(X_test_raw)

scaler = StandardScaler()
X_train_style = scaler.fit_transform(X_train_style)
X_test_style  = scaler.transform(X_test_style)
joblib.dump(scaler, "model/scaler.pkl")

# --------------------------------------------------
# BERT embeddings
# --------------------------------------------------
feature_type = "tfidf+style"

if USE_BERT:
    try:
        from transformers import AutoTokenizer, DistilBertModel, logging as hf_logging
        import torch

        hf_logging.set_verbosity_error()
        print(f"\n🤖 Loading {BERT_MODEL}...")
        tokenizer  = AutoTokenizer.from_pretrained(BERT_MODEL)
        bert_model = DistilBertModel.from_pretrained(BERT_MODEL)
        bert_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model = bert_model.to(device)
        print(f"   Device: {device}")

        def get_bert_embeddings(texts: list, label: str) -> np.ndarray:
            all_emb = []
            total = len(texts)
            for i in range(0, total, BATCH_SIZE):
                batch = texts[i: i + BATCH_SIZE]
                enc = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    out = bert_model(**enc)
                all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
                done = min(i + BATCH_SIZE, total)
                if done % 2000 < BATCH_SIZE or done == total:
                    print(f"   [{label}] {done:,}/{total:,}")
            return np.vstack(all_emb)

        print("\n⚙️  Train embeddings...")
        X_train_bert = get_bert_embeddings(X_train_raw, "train")
        print("⚙️  Test embeddings...")
        X_test_bert  = get_bert_embeddings(X_test_raw, "test")

        # Combine: BERT (768 dims) + stylistic (10 dims)
        X_train = np.hstack([X_train_bert, X_train_style])
        X_test  = np.hstack([X_test_bert,  X_test_style])

        joblib.dump(BERT_MODEL, "model/bert_model_name.pkl")
        feature_type = "bert+style"
        print(f"✅ Feature matrix: {X_train.shape}  (BERT 768 + style {STYLE_DIM})")

    except ImportError:
        print("⚠️  transformers/torch not installed — using TF-IDF + style.")
        USE_BERT = False

if not USE_BERT:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix

    print("\n📊 TF-IDF + stylistic features...")
    vectorizer = TfidfVectorizer(max_features=10000, min_df=3, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf  = vectorizer.transform(X_test_raw)
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_style)])
    X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_style)])
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print(f"   Feature matrix: {X_train.shape}")

# --------------------------------------------------
# Train
# --------------------------------------------------
print("\n🏋️  Training Logistic Regression...")
clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE, n_jobs=-1)
clf.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate
# --------------------------------------------------
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy":         round(accuracy_score(y_test, y_pred) * 100, 2),
    "precision":        round(precision_score(y_test, y_pred) * 100, 2),
    "recall":           round(recall_score(y_test, y_pred) * 100, 2),
    "f1_score":         round(f1_score(y_test, y_pred) * 100, 2),
    "roc_auc":          round(roc_auc_score(y_test, y_proba) * 100, 2),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "feature_type":     feature_type,
    "train_size":       len(X_train_raw),
    "test_size":        len(X_test_raw),
    "model":            "LogisticRegression",
    "bert_model":       BERT_MODEL if USE_BERT else None,
    "style_features":   STYLE_DIM,
}

print("\n📊 Evaluation Results:")
for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
    print(f"  {k:<12}: {metrics[k]}%")

# --------------------------------------------------
# Save
# --------------------------------------------------
joblib.dump(clf, "model/model.pkl")
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Saved: model.pkl, scaler.pkl, metrics.json")