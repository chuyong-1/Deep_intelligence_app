import json
import os
import re
import numpy as np
import joblib
import nltk
import streamlit as st

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="News Credibility Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e3da;
}
.main { background-color: #0d0d0d; }
h1, h2, h3 { font-family: 'Playfair Display', serif; color: #f5f0e8; }

.stButton > button {
    background: #c9a84c; color: #0d0d0d;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    font-size: 15px; border: none; border-radius: 4px;
    padding: 0.6rem 2rem; width: 100%;
}
.stButton > button:hover { background: #e0ba62; }
.stTextArea textarea {
    background: #1a1a1a; color: #e8e3da;
    border: 1px solid #333; border-radius: 4px;
    font-family: 'DM Sans', sans-serif; font-size: 14px;
}
.stTextInput input {
    background: #1a1a1a; color: #e8e3da;
    border: 1px solid #333; border-radius: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif; font-size: 14px;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: #999; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom: 2px solid #c9a84c !important;
}
.metric-card {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 6px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-card .label {
    font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: #888; margin-bottom: 6px;
}
.metric-card .value { font-family: 'Playfair Display', serif; font-size: 2rem; color: #c9a84c; }
.word-tag {
    display: inline-block; padding: 4px 10px; border-radius: 3px;
    font-size: 13px; font-weight: 500; margin: 3px;
    font-family: 'DM Sans', sans-serif;
}
.word-real { background: #1a3a2a; color: #5dba85; border: 1px solid #2a5a3a; }
.word-fake { background: #3a1a1a; color: #e06060; border: 1px solid #5a2a2a; }
.word-neutral { background: #2a2a1a; color: #c9a84c; border: 1px solid #4a4a2a; }
.verdict-real {
    background: #1a3a2a; border-left: 4px solid #5dba85;
    padding: 1rem 1.5rem; border-radius: 0 6px 6px 0;
    font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #5dba85;
}
.verdict-uncertain {
    background: #3a2e1a; border-left: 4px solid #c9a84c;
    padding: 1rem 1.5rem; border-radius: 0 6px 6px 0;
    font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #c9a84c;
}
.verdict-fake {
    background: #3a1a1a; border-left: 4px solid #e06060;
    padding: 1rem 1.5rem; border-radius: 0 6px 6px 0;
    font-family: 'Playfair Display', serif; font-size: 1.3rem; color: #e06060;
}
.style-bar-wrap { background: #1a1a1a; border-radius: 4px; height: 8px; margin: 4px 0 12px; }
.style-bar { height: 8px; border-radius: 4px; }
.divider { border-top: 1px solid #222; margin: 1.5rem 0; }
.sidebar-label {
    font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: #666; margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Constants (must match train_model.py exactly)
# --------------------------------------------------
MIN_WORDS = 30
STOP_WORDS = set(stopwords.words("english"))

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

STYLE_LABELS = [
    ("ALL CAPS ratio",        "fake",    "High = suspicious"),
    ("Title Case ratio",      "neutral", "Slightly elevated in fake"),
    ("Exclamation ratio",     "fake",    "High = sensationalist"),
    ("Question marks",        "neutral", "Count"),
    ("Ellipsis count",        "neutral", "Count"),
    ("Source citations",      "real",    "High = credible"),
    ("Alarm words",           "fake",    "High = suspicious"),
    ("Article length (log)",  "real",    "Longer = more credible"),
    ("Quoted phrases",        "real",    "High = credible"),
    ("Numeric facts ratio",   "real",    "High = credible"),
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def stylistic_features(text: str) -> np.ndarray:
    words       = text.split()
    num_words   = max(len(words), 1)
    lower       = text.lower()
    alpha_words = [w for w in words if w.isalpha()]

    caps_ratio     = sum(1 for w in alpha_words if w.isupper() and len(w) > 1) / max(len(alpha_words), 1)
    title_ratio    = sum(1 for w in alpha_words if w.istitle()) / max(len(alpha_words), 1)
    exclaim_ratio  = text.count("!") / num_words
    question_count = float(text.count("?"))
    ellipsis_count = float(text.count("..."))
    source_score   = float(sum(1 for p in SOURCE_PHRASES if p in lower))
    alarm_score    = float(sum(1 for w in ALARM_WORDS if w in lower))
    log_length     = float(np.log1p(num_words))
    quote_count    = float(text.count('"') // 2)
    number_ratio   = len(re.findall(r'\b\d+\.?\d*\b', text)) / num_words

    return np.array([
        caps_ratio, title_ratio, exclaim_ratio, question_count,
        ellipsis_count, source_score, alarm_score, log_length,
        quote_count, number_ratio,
    ], dtype=np.float32)


def scrape_url(url: str) -> str:
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        if not article.text.strip():
            return ""
        return f"{article.title or ''}\n\n{article.text}"
    except Exception as e:
        st.error(f"Could not scrape URL: {e}")
        return ""


# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model  = joblib.load("model/model.pkl")
    scaler = joblib.load("model/scaler.pkl") if os.path.exists("model/scaler.pkl") else None

    feature_type = "tfidf"
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f:
            feature_type = json.load(f).get("feature_type", "tfidf")

    vectorizer = bert_tokenizer = bert_model_obj = None

    if "bert" in feature_type:
        try:
            from transformers import AutoTokenizer, DistilBertModel, logging as hf_logging
            import torch
            hf_logging.set_verbosity_error()
            bert_name      = joblib.load("model/bert_model_name.pkl")
            bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
            bert_model_obj = DistilBertModel.from_pretrained(bert_name)
            bert_model_obj.eval()
        except Exception:
            st.warning("BERT model not found — falling back to TF-IDF.")
            feature_type = "tfidf+style"

    if "tfidf" in feature_type and os.path.exists("model/vectorizer.pkl"):
        vectorizer = joblib.load("model/vectorizer.pkl")

    return model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type


@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f:
            return json.load(f)
    return None


def predict(text: str, model, scaler, vectorizer, bert_tokenizer,
            bert_model_obj, feature_type):
    """Full prediction pipeline — returns (real_prob, fake_prob, style_vec_raw)."""
    style_raw = stylistic_features(text)
    style_vec = scaler.transform(style_raw.reshape(1, -1)) if scaler is not None else style_raw.reshape(1, -1)

    if "bert" in feature_type:
        import torch
        from transformers import AutoTokenizer
        bert_name  = joblib.load("model/bert_model_name.pkl")
        tokenizer  = bert_tokenizer
        enc = tokenizer(
            [clean_text(text)], padding=True, truncation=True,
            max_length=256, return_tensors="pt"
        )
        with torch.no_grad():
            out = bert_model_obj(**enc)
        emb = out.last_hidden_state[:, 0, :].numpy()
        X = np.hstack([emb, style_vec])
    else:
        from scipy.sparse import hstack, csr_matrix
        tfidf_vec = vectorizer.transform([clean_text(text)])
        X = hstack([tfidf_vec, csr_matrix(style_vec)])

    proba = model.predict_proba(X)[0]
    return proba[0] * 100, proba[1] * 100, style_raw


def lime_predict_fn(texts, model, scaler, vectorizer,
                    bert_tokenizer, bert_model_obj, feature_type):
    results = []
    for t in texts:
        r, f, _ = predict(t, model, scaler, vectorizer,
                          bert_tokenizer, bert_model_obj, feature_type)
        results.append([r / 100, f / 100])
    return np.array(results)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🔍 News Credibility")
    st.markdown("---")

    metrics_data = load_metrics()
    if metrics_data:
        st.markdown('<p class="sidebar-label">Model Performance</p>', unsafe_allow_html=True)
        st.markdown(f"**Accuracy** — `{metrics_data['accuracy']}%`")
        st.markdown(f"**F1 Score** — `{metrics_data['f1_score']}%`")
        st.markdown(f"**ROC-AUC** — `{metrics_data['roc_auc']}%`")
        st.markdown(f"**Features** — `{metrics_data['feature_type'].upper()}`")
        st.markdown(f"**Train size** — `{metrics_data['train_size']:,}`")

    st.markdown("---")
    num_features = st.slider("LIME features to show", 5, 20, 10)
    show_style   = st.toggle("Show stylistic breakdown", value=True)
    st.markdown("---")
    st.caption("⚠️ Does not replace human fact-checking.")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'>News Credibility Checker</h1>"
    "<p style='color:#888;font-size:14px;letter-spacing:0.05em;margin-top:4px'>"
    "DISTILBERT · STYLISTIC SIGNALS · EXPLAINABLE AI</p>",
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type = load_model()

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=["Real", "Fake"])

# --------------------------------------------------
# Input tabs
# --------------------------------------------------
tab_text, tab_url, tab_metrics = st.tabs(["✍️  Paste Text", "🌐  From URL", "📊  Model Metrics"])

with tab_text:
    st.markdown("")
    news_text    = st.text_area("Paste article", height=220,
                                placeholder="Paste a news article to analyse...",
                                label_visibility="collapsed")
    analyse_text = st.button("Analyse Article", key="btn_text")

with tab_url:
    st.markdown("")
    url_input   = st.text_input("URL", placeholder="https://example.com/article",
                                label_visibility="collapsed")
    analyse_url = st.button("Fetch & Analyse", key="btn_url")

with tab_metrics:
    if metrics_data:
        st.markdown("### Model Evaluation")
        cols = st.columns(5)
        for col, label, key in zip(cols,
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            ["accuracy", "precision", "recall", "f1_score", "roc_auc"]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{metrics_data[key]}%</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown("#### Confusion Matrix")
        import pandas as pd
        cm_df = pd.DataFrame(
            metrics_data["confusion_matrix"],
            index=["Actual Real", "Actual Fake"],
            columns=["Predicted Real", "Predicted Fake"],
        )
        st.dataframe(cm_df, use_container_width=True)
        st.markdown("#### Training Details")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Train samples:** {metrics_data['train_size']:,}")
        c2.info(f"**Test samples:** {metrics_data['test_size']:,}")
        c3.info(f"**Features:** {metrics_data['feature_type'].upper()}")
    else:
        st.info("No metrics found. Run `train_model.py` first.")

# --------------------------------------------------
# Determine what to analyse
# --------------------------------------------------
news_to_analyse = None

if analyse_text and news_text.strip():
    news_to_analyse = news_text.strip()
elif analyse_text:
    st.warning("Please paste some article text first.")

if analyse_url and url_input.strip():
    with st.spinner("Fetching article..."):
        scraped = scrape_url(url_input.strip())
    if scraped:
        st.text_area("Scraped preview", scraped[:600] + "...", height=100)
        news_to_analyse = scraped
    else:
        st.error("Could not extract text. Try pasting directly.")
elif analyse_url:
    st.warning("Please enter a URL first.")

# --------------------------------------------------
# Analysis
# --------------------------------------------------
if news_to_analyse:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Analysis Results")

    word_count = len(news_to_analyse.split())

    if word_count < MIN_WORDS:
        st.warning(
            f"⚠️ Only {word_count} words — the model was trained on full articles "
            f"(100+ words). Results may be unreliable for short text."
        )

    with st.spinner("Analysing..."):
        real_prob, fake_prob, style_raw = predict(
            news_to_analyse, model, scaler, vectorizer,
            bert_tokenizer, bert_model_obj, feature_type
        )
        credibility = round(real_prob, 2)

    # Verdict
    if credibility >= 70:
        st.markdown(f'<div class="verdict-real">✅ Likely Real — {credibility}% credibility</div>', unsafe_allow_html=True)
    elif credibility >= 40:
        st.markdown(f'<div class="verdict-uncertain">⚠️ Uncertain — {credibility}% credibility</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-fake">❌ Likely Fake — {credibility}% credibility</div>', unsafe_allow_html=True)

    st.markdown("")

    # Metric cards
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="label">Credibility Score</div><div class="value">{credibility}%</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="label">Real Probability</div><div class="value" style="color:#5dba85">{real_prob:.1f}%</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="label">Fake Probability</div><div class="value" style="color:#e06060">{fake_prob:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.progress(credibility / 100)

    # --------------------------------------------------
    # Stylistic breakdown
    # --------------------------------------------------
    if show_style:
        st.markdown("")
        st.markdown("#### 🎨 Stylistic Signal Breakdown")
        st.caption("These hand-crafted features are combined with BERT embeddings to catch writing-style signals of fake news.")

        # Normalise raw values to 0-1 for display bars
        style_display_max = [0.3, 1.0, 0.05, 5, 5, 8, 5, 8, 10, 0.1]
        s1, s2 = st.columns(2)

        for i, (label, signal_type, hint) in enumerate(STYLE_LABELS):
            raw_val   = float(style_raw[i])
            max_val   = style_display_max[i]
            bar_pct   = min(raw_val / max(max_val, 1e-9), 1.0)
            bar_color = {"fake": "#e06060", "real": "#5dba85", "neutral": "#c9a84c"}[signal_type]
            display   = f"{raw_val:.3f}" if raw_val < 1 else f"{raw_val:.1f}"

            col = s1 if i % 2 == 0 else s2
            col.markdown(
                f"<div style='margin-bottom:2px'>"
                f"<span style='font-size:12px;color:#aaa'>{label}</span>"
                f"<span style='float:right;font-size:12px;color:{bar_color}'>{display}</span>"
                f"</div>"
                f"<div class='style-bar-wrap'>"
                f"<div class='style-bar' style='width:{bar_pct*100:.1f}%;background:{bar_color}'></div>"
                f"</div>"
                f"<div style='font-size:11px;color:#555;margin-bottom:8px'>{hint}</div>",
                unsafe_allow_html=True,
            )

    # --------------------------------------------------
    # LIME explainability
    # --------------------------------------------------
    st.markdown("")
    st.markdown("#### 🧠 Why did the model decide this?")

    with st.spinner("Generating LIME explanation..."):
        def _lime_fn(texts):
            return lime_predict_fn(
                texts, model, scaler, vectorizer,
                bert_tokenizer, bert_model_obj, feature_type
            )
        explanation = explainer.explain_instance(
            news_to_analyse, _lime_fn, num_features=num_features
        )

    exp_list   = explanation.as_list()
    real_words = [(w, round(s, 4)) for w, s in exp_list if s > 0]
    fake_words = [(w, round(abs(s), 4)) for w, s in exp_list if s < 0]

    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("**Words → Real**")
        if real_words:
            st.markdown("".join(
                f'<span class="word-tag word-real">{w} <small>+{s}</small></span>'
                for w, s in sorted(real_words, key=lambda x: -x[1])
            ), unsafe_allow_html=True)
        else:
            st.caption("None found.")

    with lc2:
        st.markdown("**Words → Fake**")
        if fake_words:
            st.markdown("".join(
                f'<span class="word-tag word-fake">{w} <small>-{s}</small></span>'
                for w, s in sorted(fake_words, key=lambda x: -x[1])
            ), unsafe_allow_html=True)
        else:
            st.caption("None found.")

    st.markdown("")
    st.caption("LIME perturbs the input to estimate each word's contribution to the verdict.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(f"Article: {word_count} words · Features: {feature_type.upper()}")