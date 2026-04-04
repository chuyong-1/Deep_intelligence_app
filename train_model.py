"""
train_model.py  v3

Improvements over v2:
  - Checkpoint / resume logic — never restart from scratch after a crash
  - BATCH_SIZE = 64 (optimised for RTX 4050 GPU)
  - torch.no_grad() + .cpu() immediately to reduce peak memory
  - gc.collect() after every checkpoint save
  - Clear progress reporting with estimated time remaining
  - All v2 features preserved: cross-validation, CalibratedClassifierCV,
    feature importance, 16-dim stylistic features
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    brier_score_loss,
)

from features import stylistic_features, batch_stylistic, clean_text, STYLE_DIM, STYLE_LABELS

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

try:
    from tqdm import tqdm
    _tqdm = tqdm
except ImportError:
    _tqdm = None

# --------------------------------------------------
# Config
# --------------------------------------------------
USE_BERT         = True
BERT_MODEL       = "distilbert-base-uncased"
BATCH_SIZE       = 64       # optimised for RTX 4050 — reduce to 32 if CUDA OOM
MAX_SEQ_LEN      = 256
RANDOM_STATE     = 42
CV_FOLDS         = 5
MAX_ROWS_ISOT    = 10_000
MAX_ROWS_WELFAKE = 50_000

CHECKPOINT_DIR   = "model/checkpoints"
os.makedirs("model", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
        print("  ⚠️  WELFake missing 'label' column — skipping.")
else:
    print(
        "  ⚠️  WELFake not found.\n"
        "     Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification"
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
print(f"\n📊 {len(df):,} articles (removed {before - len(df):,} duplicates)")
print(f"   Fake: {(df['label']==1).sum():,}  |  Real: {(df['label']==0).sum():,}")

# Balance classes
min_class = min((df["label"] == 0).sum(), (df["label"] == 1).sum())
df = pd.concat([
    df[df["label"] == 0].sample(min_class, random_state=RANDOM_STATE),
    df[df["label"] == 1].sample(min_class, random_state=RANDOM_STATE),
]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"   Balanced: {len(df):,} ({min_class:,} per class)")

X_content = df["content"].tolist()
y         = df["label"].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_content, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"   Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

# --------------------------------------------------
# Stylistic features  (16-dim from features.py v2)
# --------------------------------------------------
print(f"\n🎨 Extracting {STYLE_DIM} stylistic features...")
X_train_style = batch_stylistic(X_train_raw)
X_test_style  = batch_stylistic(X_test_raw)

scaler        = StandardScaler()
X_train_style = scaler.fit_transform(X_train_style)
X_test_style  = scaler.transform(X_test_style)
joblib.dump(scaler, "model/scaler.pkl")

# --------------------------------------------------
# BERT embeddings  (with checkpoint / resume)
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
        device     = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model = bert_model.to(device)
        print(f"   Device: {device}")

        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️  No GPU detected — running on CPU (slow).")
            print("      Verify with: python -c \"import torch; print(torch.cuda.is_available())\"")

        def get_bert_embeddings(texts: list, label: str) -> np.ndarray:
            """
            Extract CLS embeddings with checkpoint/resume support.
            Saves a checkpoint every 100 batches so a crash never
            forces you to restart from scratch.
            """
            chk_emb = os.path.join(CHECKPOINT_DIR, f"emb_{label}.npy")
            chk_idx = os.path.join(CHECKPOINT_DIR, f"emb_{label}_idx.json")

            all_emb = []
            start_i = 0

            # --- resume from checkpoint if available ---
            if os.path.exists(chk_emb) and os.path.exists(chk_idx):
                try:
                    saved   = np.load(chk_emb, allow_pickle=True)
                    all_emb = list(saved)
                    with open(chk_idx) as f:
                        start_i = json.load(f)["next_index"]
                    print(f"   ▶  Resuming [{label}] from index {start_i} "
                          f"({len(all_emb)} batches already done)")
                except Exception as e:
                    print(f"   ⚠️  Could not load checkpoint ({e}), starting fresh.")
                    all_emb = []
                    start_i = 0

            total   = len(texts)
            batches = list(range(start_i, total, BATCH_SIZE))

            if not batches:
                print(f"   ✅ [{label}] already complete.")
                return np.vstack(all_emb)

            iterator = (
                _tqdm(batches, desc=f"  [{label}]", unit="batch")
                if _tqdm else batches
            )

            t0 = time.time()
            for step, i in enumerate(iterator):
                batch = texts[i: i + BATCH_SIZE]
                enc   = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=MAX_SEQ_LEN, return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    out = bert_model(**enc)

                # move to CPU immediately to free GPU VRAM
                emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                all_emb.append(emb)

                # plain-text progress when tqdm not available
                if not _tqdm and step % 50 == 0:
                    done    = i + len(batch)
                    elapsed = time.time() - t0
                    rate    = (step + 1) / max(elapsed, 1)
                    remain  = (len(batches) - step - 1) / max(rate, 1e-6)
                    print(f"   [{label}] {done:,}/{total:,}  "
                          f"~{remain/60:.1f} min remaining", end="\r")

                # --- save checkpoint every 100 batches ---
                if (step + 1) % 100 == 0:
                    np.save(chk_emb, np.array(all_emb, dtype=object))
                    with open(chk_idx, "w") as f:
                        json.dump({"next_index": i + BATCH_SIZE}, f)
                    gc.collect()

            if not _tqdm:
                print()

            # final checkpoint
            np.save(chk_emb, np.array(all_emb, dtype=object))
            with open(chk_idx, "w") as f:
                json.dump({"next_index": total}, f)

            result = np.vstack(all_emb)
            print(f"   ✅ [{label}] done — shape {result.shape}")
            return result

        print("\n⚙️  Extracting train embeddings...")
        X_train_bert = get_bert_embeddings(X_train_raw, "train")
        print("⚙️  Extracting test embeddings...")
        X_test_bert  = get_bert_embeddings(X_test_raw, "test")

        X_train = np.hstack([X_train_bert, X_train_style])
        X_test  = np.hstack([X_test_bert,  X_test_style])

        joblib.dump(BERT_MODEL, "model/bert_model_name.pkl")
        feature_type = "bert+style"
        print(f"✅ Feature matrix: {X_train.shape}")

    except ImportError:
        print("⚠️  transformers/torch not installed — falling back to TF-IDF + style.")
        USE_BERT = False

if not USE_BERT:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import hstack, csr_matrix

    print("\n📊 TF-IDF + stylistic features...")
    vectorizer    = TfidfVectorizer(max_features=10_000, min_df=3, ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform([clean_text(t, STOP_WORDS) for t in X_train_raw])
    X_test_tfidf  = vectorizer.transform([clean_text(t, STOP_WORDS) for t in X_test_raw])
    X_train = hstack([X_train_tfidf, csr_matrix(X_train_style)])
    X_test  = hstack([X_test_tfidf,  csr_matrix(X_test_style)])
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print(f"   Feature matrix: {X_train.shape}")

# --------------------------------------------------
# Cross-validation  (on train set only)
# --------------------------------------------------
print(f"\n🔄 {CV_FOLDS}-fold cross-validation on training set...")
base_clf = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE, n_jobs=-1)
cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cv_scores = cross_val_score(base_clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
print(f"   CV F1: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# --------------------------------------------------
# Train with probability calibration
# --------------------------------------------------
print("\n🏋️  Training with CalibratedClassifierCV (isotonic regression)...")
clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
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
    "brier_score":      round(brier_score_loss(y_test, y_proba), 4),
    "cv_f1_mean":       round(cv_scores.mean() * 100, 2),
    "cv_f1_std":        round(cv_scores.std() * 100, 2),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "feature_type":     feature_type,
    "train_size":       len(X_train_raw),
    "test_size":        len(X_test_raw),
    "model":            "LogisticRegression+CalibratedCV",
    "bert_model":       BERT_MODEL if USE_BERT else None,
    "style_features":   STYLE_DIM,
}

print("\n📊 Evaluation Results:")
for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "brier_score"]:
    print(f"  {k:<14}: {metrics[k]}")
print(f"  {'cv_f1':<14}: {metrics['cv_f1_mean']}% ± {metrics['cv_f1_std']}%")

# --------------------------------------------------
# Stylistic feature importance  (TF-IDF mode only)
# --------------------------------------------------
if not USE_BERT:
    try:
        inner_clf   = clf.calibrated_classifiers_[0].estimator
        style_coefs = inner_clf.coef_[0][-STYLE_DIM:]
        importance  = {
            STYLE_LABELS[i][0]: round(float(style_coefs[i]), 4)
            for i in range(STYLE_DIM)
        }
        metrics["style_feature_importance"] = importance
        print("\n🎯 Stylistic feature importance (LR coef):")
        for name, val in sorted(importance.items(), key=lambda x: -abs(x[1])):
            bar  = "▓" * int(abs(val) * 20)
            sign = "+" if val > 0 else "-"
            print(f"  {sign} {name:<26} {bar}")
    except Exception:
        pass

# --------------------------------------------------
# Save
# --------------------------------------------------
joblib.dump(clf, "model/model.pkl")
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Saved: model/model.pkl, model/scaler.pkl, model/metrics.json")
print("\n💡 To resume after a crash, just run this script again.")
print("   Checkpoints are in:", CHECKPOINT_DIR)