"""
app.py  v3

UI improvements over v2:
  - Full-width hero header with animated gradient underline
  - Redesigned score gauge: larger, cleaner, colour-coded zones
  - Verdict banner with animated left border pulse
  - Metric cards with hover lift effect
  - Stylistic bars now have gradient fills and cleaner labels
  - Sidebar redesigned: collapsible sections, better typography
  - Tab bar with pill-style active indicator
  - Smoother overall spacing and visual hierarchy
  - Entity cards replace inline tags for cleaner layout
  - Batch results summary uses a mini sparkline bar
"""

import hashlib
import json
import math
import os
import numpy as np
import joblib
import nltk
import streamlit as st

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from features import (
    stylistic_features, clean_text,
    STYLE_LABELS, STYLE_DISPLAY_MAX,
)
from entity_checker import (
    check_entities,
    SCORE_VERIFIED, SCORE_NOT_FOUND, SCORE_AMBIGUOUS,
)

STOP_WORDS = set(stopwords.words("english"))

DISREPUTABLE_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "theonion.com",
    "empirenews.net", "abcnews.com.co", "worldnewsdailyreport.com",
    "huzlers.com", "nationalreport.net", "stuppid.com",
}
SATIRE_DOMAINS = {"theonion.com", "babylonbee.com", "thebeaverton.com"}

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Verity — News Credibility",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom CSS — editorial dark theme
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080b10;
    color: #d4cfc7;
}
.main { background-color: #080b10; padding-top: 0 !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif;
    color: #f0ebe2;
    letter-spacing: -0.01em;
}

/* ── Hero header ── */
.hero-wrap {
    padding: 2.2rem 0 1.4rem;
    border-bottom: 1px solid #1c2030;
    margin-bottom: 1.8rem;
    position: relative;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #f0ebe2;
    letter-spacing: -0.02em;
    line-height: 1;
    margin: 0;
}
.hero-title span { color: #7eb8f7; font-style: italic; }
.hero-sub {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a5068;
    margin-top: 10px;
    display: flex;
    gap: 18px;
}
.hero-sub span::before { content: "◆ "; color: #2a3050; }
.hero-dot { width: 8px; height: 8px; border-radius: 50%; background: #7eb8f7;
            display: inline-block; margin-right: 6px; animation: pulse-dot 2s infinite; }
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0;
    border-bottom: 1px solid #1c2030;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a4060;
    padding: 10px 22px;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: #8090b0; }
.stTabs [aria-selected="true"] {
    color: #7eb8f7 !important;
    border-bottom: 2px solid #7eb8f7 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* ── Inputs ── */
.stTextArea textarea {
    background: #0d1018 !important;
    color: #d4cfc7 !important;
    border: 1px solid #1c2030 !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus { border-color: #7eb8f7 !important; }
.stTextInput input {
    background: #0d1018 !important;
    color: #d4cfc7 !important;
    border: 1px solid #1c2030 !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
}
.stTextInput input:focus { border-color: #7eb8f7 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #7eb8f7;
    color: #080b10;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border: none;
    border-radius: 5px;
    padding: 0.65rem 2.2rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #a8d0ff;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(126,184,247,0.25);
}

/* ── Verdict banners ── */
.verdict {
    padding: 1.1rem 1.6rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 1rem 0;
    animation: slide-in 0.35s ease;
}
@keyframes slide-in {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.verdict-real  { background: #071812; border: 1px solid #1a4a30; border-left: 4px solid #4ade80; }
.verdict-uncertain { background: #12100a; border: 1px solid #3a2e10; border-left: 4px solid #fbbf24; }
.verdict-fake  { background: #120a0a; border: 1px solid #3a1010; border-left: 4px solid #f87171; }
.verdict-icon { font-size: 1.8rem; line-height: 1; }
.verdict-text { font-family: 'DM Serif Display', serif; font-size: 1.25rem; }
.verdict-real  .verdict-text { color: #4ade80; }
.verdict-uncertain .verdict-text { color: #fbbf24; }
.verdict-fake  .verdict-text { color: #f87171; }
.verdict-sub { font-size: 12px; color: #4a5068; margin-top: 3px; }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 1rem 0; }
.mcard {
    background: #0d1018;
    border: 1px solid #1c2030;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.mcard:hover { transform: translateY(-2px); border-color: #2a3555; }
.mcard-label { font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase; color: #3a4060; margin-bottom: 8px; }
.mcard-value { font-family: 'DM Serif Display', serif; font-size: 1.9rem; color: #7eb8f7; line-height: 1; }
.mcard-value.green { color: #4ade80; }
.mcard-value.red   { color: #f87171; }
.mcard-value.amber { color: #fbbf24; }
.mcard-value.small { font-size: 1.3rem; margin-top: 4px; }
.mcard-sub { font-size: 10px; color: #2a3050; margin-top: 5px; font-family: 'JetBrains Mono', monospace; }

/* ── Style bars ── */
.sbar-wrap { margin-bottom: 12px; }
.sbar-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }
.sbar-label { font-size: 12px; color: #8090b0; }
.sbar-val { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.sbar-track { background: #0d1018; border-radius: 3px; height: 5px; border: 1px solid #1c2030; overflow: hidden; }
.sbar-fill { height: 5px; border-radius: 3px; transition: width 1s cubic-bezier(0.4,0,0.2,1); }
.sbar-hint { font-size: 10px; color: #2a3555; margin-top: 3px; }

/* ── Word tags ── */
.tag-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.wtag {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 11px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    transition: transform 0.15s;
}
.wtag:hover { transform: scale(1.06); }
.wtag-real  { background: #071812; color: #4ade80; border: 1px solid #1a4a30; }
.wtag-fake  { background: #120a0a; color: #f87171; border: 1px solid #3a1010; }
.wtag-score { font-size: 10px; opacity: 0.6; }

/* ── Entity cards ── */
.ent-card {
    background: #0d1018;
    border: 1px solid #1c2030;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    margin-bottom: 7px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.ent-badge {
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    padding: 3px 7px; border-radius: 3px; white-space: nowrap; margin-top: 2px;
}
.ent-verified { background: #071812; color: #4ade80; border: 1px solid #1a4a30; }
.ent-notfound { background: #120a0a; color: #f87171; border: 1px solid #3a1010; }
.ent-ambiguous{ background: #12100a; color: #fbbf24; border: 1px solid #3a2e10; }
.ent-name { font-size: 14px; font-weight: 600; color: #f0ebe2; }
.ent-name a { color: #7eb8f7; text-decoration: none; }
.ent-name a:hover { text-decoration: underline; }
.ent-type { font-size: 11px; color: #3a4060; font-family: 'JetBrains Mono', monospace; }
.ent-summary { font-size: 12px; color: #4a5068; margin-top: 4px; line-height: 1.5; }
.ent-impact { font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-left: auto; white-space: nowrap; }

/* ── Domain warning ── */
.domain-warn {
    background: #100d00;
    border: 1px solid #3a2800;
    border-left: 3px solid #f59e0b;
    padding: 0.6rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 13px; color: #f59e0b;
    margin-bottom: 0.8rem;
}

/* ── Section dividers ── */
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #2a3555;
    margin: 1.8rem 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-title::after { content: ''; flex: 1; height: 1px; background: #1c2030; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #06080e;
    border-right: 1px solid #1c2030;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
.sb-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f0ebe2;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}
.sb-brand span { color: #7eb8f7; font-style: italic; }
.sb-tagline { font-size: 10px; color: #2a3555; letter-spacing: 0.15em; text-transform: uppercase; }
.sb-section { font-size: 9px; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: #2a3555; margin: 1.2rem 0 0.5rem; }
.sb-stat { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #0d1018; }
.sb-stat-label { font-size: 12px; color: #4a5068; }
.sb-stat-val { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #7eb8f7; }
.hist-row {
    background: #0d1018; border: 1px solid #1c2030; border-radius: 5px;
    padding: 7px 10px; margin-bottom: 5px; cursor: default;
    display: flex; align-items: center; gap: 8px;
}
.hist-score { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; min-width: 38px; }
.hist-text { font-size: 11px; color: #3a4060; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }

/* ── Misc ── */
.divider { border: none; border-top: 1px solid #1c2030; margin: 1.5rem 0; }
.note { font-size: 11px; color: #2a3555; line-height: 1.5; }
.stAlert { border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

MIN_WORDS = 30

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "lime_cache" not in st.session_state:
    st.session_state.lime_cache = {}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def reading_time(text):
    words = len(text.split())
    mins  = max(1, round(words / 200))
    return f"{mins} min"

def text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def check_domain(url):
    import re
    m = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if not m:
        return None, False
    domain = m.group(1).lower()
    return domain, domain in DISREPUTABLE_DOMAINS

def confidence_interval(score, n_samples=100):
    p  = score / 100
    z  = 1.645
    lo = max(0, (p - z * (p * (1-p) / n_samples) ** 0.5) * 100)
    hi = min(100, (p + z * (p * (1-p) / n_samples) ** 0.5) * 100)
    return round(lo, 1), round(hi, 1)

def scrape_url(url):
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
@st.cache_resource(show_spinner="Loading model…")
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


@st.cache_resource(show_spinner=False)
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        return None


@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f:
            return json.load(f)
    return None


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(text, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type):
    style_raw = stylistic_features(text)
    style_vec = (
        scaler.transform(style_raw.reshape(1, -1))
        if scaler is not None else style_raw.reshape(1, -1)
    )

    if "bert" in feature_type:
        import torch
        enc = bert_tokenizer(
            [clean_text(text, STOP_WORDS)], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        with torch.no_grad():
            out = bert_model_obj(**enc)
        emb = out.last_hidden_state[:, 0, :].numpy()
        X   = np.hstack([emb, style_vec])
    else:
        from scipy.sparse import hstack, csr_matrix
        tfidf_vec = vectorizer.transform([clean_text(text, STOP_WORDS)])
        X         = hstack([tfidf_vec, csr_matrix(style_vec)])

    proba = model.predict_proba(X)[0]
    return proba[0] * 100, proba[1] * 100, style_raw


def lime_predict_fn(texts, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type):
    results = []
    for t in texts:
        r, f, _ = predict(t, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type)
        results.append([r / 100, f / 100])
    return np.array(results)


# --------------------------------------------------
# SVG Gauge (redesigned)
# --------------------------------------------------
def render_gauge(score: float):
    pct       = score / 100
    sweep     = 240
    start_deg = 210
    rad       = math.pi / 180
    angle     = start_deg - pct * sweep

    # Needle tip
    nx = 100 + 58 * math.cos(angle * rad)
    ny = 95  - 58 * math.sin(angle * rad)

    # Filled arc end
    aex = 100 + 75 * math.cos(angle * rad)
    aey = 95  - 75 * math.sin(angle * rad)
    asx = 100 + 75 * math.cos(start_deg * rad)
    asy = 95  - 75 * math.sin(start_deg * rad)
    large = 1 if pct * sweep > 180 else 0

    # Colour zones
    if pct < 0.4:
        col = "#f87171"
    elif pct < 0.7:
        col = "#fbbf24"
    else:
        col = "#4ade80"

    end_x = 100 + 75 * math.cos((start_deg - sweep) * rad)
    end_y = 95  - 75 * math.sin((start_deg - sweep) * rad)

    # Zone markers at 40 and 70
    m40x1 = 100 + 68 * math.cos((start_deg - 0.4*sweep) * rad)
    m40y1 = 95  - 68 * math.sin((start_deg - 0.4*sweep) * rad)
    m40x2 = 100 + 82 * math.cos((start_deg - 0.4*sweep) * rad)
    m40y2 = 95  - 82 * math.sin((start_deg - 0.4*sweep) * rad)
    m70x1 = 100 + 68 * math.cos((start_deg - 0.7*sweep) * rad)
    m70y1 = 95  - 68 * math.sin((start_deg - 0.7*sweep) * rad)
    m70x2 = 100 + 82 * math.cos((start_deg - 0.7*sweep) * rad)
    m70y2 = 95  - 82 * math.sin((start_deg - 0.7*sweep) * rad)

    html = f"""
<div style="display:flex;justify-content:center;margin:0.5rem 0 1.2rem">
<svg viewBox="0 0 200 115" width="300" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="g"><feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="trackGrad" gradientUnits="userSpaceOnUse"
      x1="{asx:.1f}" y1="{asy:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}">
      <stop offset="0%" stop-color="#f87171" stop-opacity="0.25"/>
      <stop offset="45%" stop-color="#fbbf24" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#4ade80" stop-opacity="0.25"/>
    </linearGradient>
  </defs>

  <!-- Faint background track -->
  <path d="M {asx:.2f} {asy:.2f} A 75 75 0 1 1 {end_x:.2f} {end_y:.2f}"
        fill="none" stroke="url(#trackGrad)" stroke-width="12" stroke-linecap="round"/>

  <!-- Dark track -->
  <path d="M {asx:.2f} {asy:.2f} A 75 75 0 1 1 {end_x:.2f} {end_y:.2f}"
        fill="none" stroke="#0d1018" stroke-width="9" stroke-linecap="round"/>

  <!-- Filled arc -->
  <path d="M {asx:.2f} {asy:.2f} A 75 75 0 {large} 1 {aex:.2f} {aey:.2f}"
        fill="none" stroke="{col}" stroke-width="9" stroke-linecap="round"
        filter="url(#g)" opacity="0.9">
    <animate attributeName="stroke-dasharray"
             from="0 472" to="{pct*472:.1f} 472"
             dur="1s" fill="freeze" calcMode="spline" keySplines="0.4 0 0.2 1"/>
  </path>

  <!-- Zone ticks -->
  <line x1="{m40x1:.2f}" y1="{m40y1:.2f}" x2="{m40x2:.2f}" y2="{m40y2:.2f}"
        stroke="#fbbf24" stroke-width="1.5" opacity="0.4"/>
  <line x1="{m70x1:.2f}" y1="{m70y1:.2f}" x2="{m70x2:.2f}" y2="{m70y2:.2f}"
        stroke="#4ade80" stroke-width="1.5" opacity="0.4"/>

  <!-- Zone labels -->
  <text x="28" y="108" text-anchor="middle" font-family="Syne,sans-serif"
        font-size="8" fill="#f87171" opacity="0.5">FAKE</text>
  <text x="172" y="108" text-anchor="middle" font-family="Syne,sans-serif"
        font-size="8" fill="#4ade80" opacity="0.5">REAL</text>

  <!-- Needle -->
  <line x1="100" y1="95" x2="{nx:.2f}" y2="{ny:.2f}"
        stroke="{col}" stroke-width="2" stroke-linecap="round" filter="url(#g)">
    <animateTransform attributeName="transform" type="rotate"
      from="{start_deg} 100 95" to="{start_deg - pct*sweep:.2f} 100 95"
      dur="1s" fill="freeze" calcMode="spline" keySplines="0.4 0 0.2 1"/>
  </line>
  <circle cx="100" cy="95" r="5" fill="{col}" filter="url(#g)"/>
  <circle cx="100" cy="95" r="2.5" fill="#080b10"/>

  <!-- Score -->
  <text x="100" y="78" text-anchor="middle"
        font-family="DM Serif Display,serif" font-size="24" font-weight="700" fill="{col}">
    {score:.0f}%
  </text>
</svg>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------
# Radar chart
# --------------------------------------------------
def render_radar(style_raw):
    try:
        import plotly.graph_objects as go
        labels = [lbl for lbl, _, _ in STYLE_LABELS]
        maxes  = STYLE_DISPLAY_MAX
        vals   = [min(float(style_raw[i]) / max(maxes[i], 1e-9), 1.0) for i in range(len(labels))]
        vals  += vals[:1]
        theta  = labels + labels[:1]

        fig = go.Figure(go.Scatterpolar(
            r=vals, theta=theta,
            fill="toself",
            fillcolor="rgba(126,184,247,0.08)",
            line=dict(color="#7eb8f7", width=1.5),
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="#0d1018",
                angularaxis=dict(color="#2a3555", gridcolor="#1c2030",
                                 tickfont=dict(size=9, color="#4a5068")),
                radialaxis=dict(color="#1c2030", gridcolor="#1c2030",
                                range=[0, 1], visible=True,
                                tickfont=dict(size=7, color="#2a3555")),
            ),
            paper_bgcolor="#080b10", plot_bgcolor="#080b10",
            margin=dict(l=40, r=40, t=10, b=10),
            height=320, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Install plotly for radar chart: `pip install plotly`")


# --------------------------------------------------
# Main analysis function
# --------------------------------------------------
def analyse_article(
    text, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type,
    nlp, num_features, show_style, show_entity, show_radar, source_url="",
):
    word_count = len(text.split())
    rt         = reading_time(text)

    if word_count < MIN_WORDS:
        st.warning(f"Only {word_count} words — results may be unreliable (model trained on 100+ word articles).")

    if source_url:
        domain, is_bad = check_domain(source_url)
        if domain and is_bad:
            label = "Satire site" if domain in SATIRE_DOMAINS else "Known low-credibility domain"
            st.markdown(
                f'<div class="domain-warn">⚠ <strong>{label}:</strong> {domain}</div>',
                unsafe_allow_html=True,
            )

    with st.spinner("Analysing…"):
        real_prob, fake_prob, style_raw = predict(
            text, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type,
        )
        credibility = round(real_prob, 2)

    entity_report = None
    if show_entity and nlp is not None:
        with st.spinner("Checking entities on Wikipedia…"):
            entity_report = check_entities(text, nlp)

    if entity_report and not entity_report.error and entity_report.total_checked > 0:
        final_score = round(0.70 * credibility + 0.30 * entity_report.entity_score, 1)
        score_note  = "ML + entity (blended)"
    else:
        final_score = credibility
        score_note  = "ML model"

    ci_lo, ci_hi = confidence_interval(final_score)

    # --- Verdict ---
    if final_score >= 70:
        vcls, vicon, vtext = "verdict-real", "✅", f"Likely Credible — {final_score}% confidence"
    elif final_score >= 40:
        vcls, vicon, vtext = "verdict-uncertain", "⚠️", f"Uncertain — {final_score}% confidence"
    else:
        vcls, vicon, vtext = "verdict-fake", "❌", f"Likely Fabricated — {final_score}% confidence"

    st.markdown(
        f'<div class="verdict {vcls}">'
        f'<div class="verdict-icon">{vicon}</div>'
        f'<div><div class="verdict-text">{vtext}</div>'
        f'<div class="verdict-sub">{score_note} · 90% CI [{ci_lo}–{ci_hi}%] · {word_count} words · {rt} read</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # --- Gauge ---
    render_gauge(final_score)

    # --- Metric cards ---
    st.markdown(
        f'<div class="metric-grid">'
        f'<div class="mcard"><div class="mcard-label">Final Score</div>'
        f'<div class="mcard-value">{final_score}%</div>'
        f'<div class="mcard-sub">CI {ci_lo}–{ci_hi}</div></div>'

        f'<div class="mcard"><div class="mcard-label">ML Score</div>'
        f'<div class="mcard-value">{credibility}%</div></div>'

        f'<div class="mcard"><div class="mcard-label">Real Prob.</div>'
        f'<div class="mcard-value green">{real_prob:.1f}%</div></div>'

        f'<div class="mcard"><div class="mcard-label">Fake Prob.</div>'
        f'<div class="mcard-value red">{fake_prob:.1f}%</div></div>'

        f'<div class="mcard"><div class="mcard-label">Read Time</div>'
        f'<div class="mcard-value small amber">{rt}</div>'
        f'<div class="mcard-sub">{word_count} words</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Stylistic breakdown ---
    if show_style:
        st.markdown('<div class="sec-title">Stylistic Signals</div>', unsafe_allow_html=True)
        if show_radar:
            render_radar(style_raw)

        bar_colors = {"fake": "#f87171", "real": "#4ade80", "neutral": "#fbbf24"}
        cols = st.columns(2)
        for i, (label, stype, hint) in enumerate(STYLE_LABELS):
            raw_val   = float(style_raw[i])
            max_val   = STYLE_DISPLAY_MAX[i]
            bar_pct   = min(raw_val / max(max_val, 1e-9), 1.0) * 100
            bar_color = bar_colors[stype]
            display   = f"{raw_val:.3f}" if raw_val < 1 else f"{raw_val:.1f}"
            col       = cols[i % 2]
            col.markdown(
                f'<div class="sbar-wrap">'
                f'<div class="sbar-header">'
                f'<span class="sbar-label">{label}</span>'
                f'<span class="sbar-val" style="color:{bar_color}">{display}</span>'
                f'</div>'
                f'<div class="sbar-track">'
                f'<div class="sbar-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div>'
                f'</div>'
                f'<div class="sbar-hint">{hint}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- LIME ---
    st.markdown('<div class="sec-title">Why did the model decide this?</div>', unsafe_allow_html=True)
    if "bert" in feature_type:
        st.markdown('<p class="note">⏱ LIME with BERT runs ~100 passes and may take 30–120 s. Results are cached per article.</p>', unsafe_allow_html=True)

    t_hash = text_hash(text)
    if t_hash in st.session_state.lime_cache:
        exp_list = st.session_state.lime_cache[t_hash]
        st.caption("Using cached LIME explanation.")
    else:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["Real", "Fake"])

        def _lime_fn(texts):
            return lime_predict_fn(texts, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type)

        with st.spinner("Generating explanation…"):
            explanation = explainer.explain_instance(text, _lime_fn, num_features=num_features)
        exp_list = explanation.as_list()
        st.session_state.lime_cache[t_hash] = exp_list

    real_words = sorted([(w, round(s, 4)) for w, s in exp_list if s > 0], key=lambda x: -x[1])
    fake_words = sorted([(w, round(abs(s), 4)) for w, s in exp_list if s < 0], key=lambda x: -x[1])

    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown("**Signals → Real**")
        tags = "".join(
            f'<span class="wtag wtag-real">{w} <span class="wtag-score">+{s}</span></span>'
            for w, s in real_words
        ) or "<span class='note'>None found</span>"
        st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)

    with lc2:
        st.markdown("**Signals → Fake**")
        tags = "".join(
            f'<span class="wtag wtag-fake">{w} <span class="wtag-score">-{s}</span></span>'
            for w, s in fake_words
        ) or "<span class='note'>None found</span>"
        st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)

    st.markdown('<p class="note" style="margin-top:8px">LIME perturbs the text ~5000 times to estimate each word\'s contribution to the verdict.</p>', unsafe_allow_html=True)

    # --- Entity check ---
    if entity_report:
        st.markdown('<div class="sec-title">Wikipedia Entity Verification</div>', unsafe_allow_html=True)
        if entity_report.error:
            st.warning(f"Entity check unavailable: {entity_report.error}")
        elif entity_report.total_checked == 0:
            st.info("No named entities found.")
        else:
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.markdown(f'<div class="mcard"><div class="mcard-label">Entity Score</div><div class="mcard-value">{entity_report.entity_score}</div></div>', unsafe_allow_html=True)
            ec2.markdown(f'<div class="mcard"><div class="mcard-label">Verified</div><div class="mcard-value green">{entity_report.verified_count}</div></div>', unsafe_allow_html=True)
            ec3.markdown(f'<div class="mcard"><div class="mcard-label">Not Found</div><div class="mcard-value red">{entity_report.not_found_count}</div></div>', unsafe_allow_html=True)
            ec4.markdown(f'<div class="mcard"><div class="mcard-label">Ambiguous</div><div class="mcard-value amber">{entity_report.ambiguous_count}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            for ent in entity_report.entities:
                if ent.status == "verified":
                    badge_cls, badge_txt = "ent-verified", "VERIFIED"
                elif ent.status == "not_found":
                    badge_cls, badge_txt = "ent-notfound", "NOT FOUND"
                else:
                    badge_cls, badge_txt = "ent-ambiguous", "AMBIGUOUS"

                impact_str = f"+{ent.score_impact:.0f}" if ent.score_impact >= 0 else f"{ent.score_impact:.0f}"
                freq_txt   = f" ×{ent.frequency}" if ent.frequency > 1 else ""
                wd_txt     = " · Wikidata" if ent.wikidata_found else ""

                name_html = (
                    f'<a href="{ent.wiki_url}" target="_blank">{ent.name}</a>'
                    if ent.wiki_url and ent.status == "verified"
                    else ent.name
                )
                summary_html = (
                    f'<div class="ent-summary">{(ent.summary or "")[:150]}{"…" if len(ent.summary or "") > 150 else ""}</div>'
                    if ent.summary else ""
                )

                st.markdown(
                    f'<div class="ent-card">'
                    f'<span class="ent-badge {badge_cls}">{badge_txt}</span>'
                    f'<div style="flex:1">'
                    f'<div class="ent-name">{name_html}{freq_txt}</div>'
                    f'<div class="ent-type">{ent.entity_type}{wd_txt}</div>'
                    f'{summary_html}'
                    f'</div>'
                    f'<div class="ent-impact" style="color:{"#4ade80" if ent.score_impact >= 0 else "#f87171"}">{impact_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<p class="note">Scoring: verified {SCORE_VERIFIED:+d} · not found {SCORE_NOT_FOUND:+d} · ambiguous {SCORE_AMBIGUOUS:+d} — weighted by entity type & frequency. Contributes 30% to final score.</p>',
                unsafe_allow_html=True,
            )
    elif show_entity and nlp is None:
        st.info("Install spaCy for entity checking: `pip install spacy && python -m spacy download en_core_web_sm`")

    # --- Save history ---
    snippet = text[:80].replace("\n", " ") + ("…" if len(text) > 80 else "")
    st.session_state.history.insert(0, {
        "snippet": snippet,
        "score":   final_score,
        "verdict": "Real" if final_score >= 70 else ("Fake" if final_score < 40 else "Uncertain"),
        "words":   word_count,
    })
    st.session_state.history = st.session_state.history[:20]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="note">{word_count} words · {rt} read · {feature_type.upper()}</p>', unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sb-brand">Ve<span>rity</span></div>'
        '<div class="sb-tagline">News Credibility Engine</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    metrics_data = load_metrics()
    if metrics_data:
        st.markdown('<div class="sb-section">Model Performance</div>', unsafe_allow_html=True)
        for label, key in [("Accuracy", "accuracy"), ("F1 Score", "f1_score"),
                            ("ROC-AUC", "roc_auc"), ("Brier", "brier_score")]:
            st.markdown(
                f'<div class="sb-stat"><span class="sb-stat-label">{label}</span>'
                f'<span class="sb-stat-val">{metrics_data.get(key, "—")}</span></div>',
                unsafe_allow_html=True,
            )
        if "cv_f1_mean" in metrics_data:
            st.markdown(
                f'<div class="sb-stat"><span class="sb-stat-label">CV F1</span>'
                f'<span class="sb-stat-val">{metrics_data["cv_f1_mean"]}±{metrics_data["cv_f1_std"]}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Options</div>', unsafe_allow_html=True)
    num_features = st.slider("LIME features", 5, 20, 10)
    show_style   = st.toggle("Stylistic breakdown", value=True)
    show_radar   = st.toggle("Radar chart", value=True)
    show_entity  = st.toggle("Entity verification", value=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown('<div class="sb-section">Recent Analyses</div>', unsafe_allow_html=True)
        for item in st.session_state.history[:8]:
            color = "#4ade80" if item["verdict"] == "Real" else (
                "#f87171" if item["verdict"] == "Fake" else "#fbbf24"
            )
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-score" style="color:{color}">{item["score"]}%</span>'
                f'<span class="hist-text">{item["snippet"][:48]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="note">⚠ Does not replace human fact-checking.</p>', unsafe_allow_html=True)


# --------------------------------------------------
# Hero header
# --------------------------------------------------
st.markdown(
    '<div class="hero-wrap">'
    '<h1 class="hero-title"><span class="hero-dot"></span>News Credibility <span>Checker</span></h1>'
    '<div class="hero-sub">'
    '<span>DistilBERT</span>'
    '<span>16 Stylistic Signals</span>'
    '<span>LIME Explainability</span>'
    '<span>Entity Verification</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type = load_model()
nlp = load_spacy()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_text, tab_url, tab_batch, tab_metrics = st.tabs([
    "✍  Paste Text", "🌐  From URL", "📋  Batch Mode", "📊  Metrics",
])

with tab_text:
    news_text    = st.text_area("", height=200, placeholder="Paste a news article to analyse…", label_visibility="collapsed")
    analyse_text = st.button("Analyse Article", key="btn_text")

with tab_url:
    url_input   = st.text_input("", placeholder="https://example.com/article", label_visibility="collapsed")
    analyse_url = st.button("Fetch & Analyse", key="btn_url")

with tab_batch:
    st.markdown('<p class="note">Separate multiple articles with a line containing only <code>---</code></p>', unsafe_allow_html=True)
    batch_text    = st.text_area("", height=280, placeholder="Article one…\n---\nArticle two…", label_visibility="collapsed")
    analyse_batch = st.button("Analyse All", key="btn_batch")

with tab_metrics:
    if metrics_data:
        st.markdown("#### Model Evaluation")
        cols = st.columns(5)
        for col, label, key in zip(cols,
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            ["accuracy", "precision", "recall", "f1_score", "roc_auc"]):
            col.markdown(
                f'<div class="mcard"><div class="mcard-label">{label}</div>'
                f'<div class="mcard-value">{metrics_data[key]}%</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        if "cv_f1_mean" in metrics_data:
            st.info(f"Cross-validation F1: {metrics_data['cv_f1_mean']}% ± {metrics_data['cv_f1_std']}%  —  Brier: {metrics_data.get('brier_score', 'N/A')}")
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
        c1.info(f"**Train:** {metrics_data['train_size']:,}")
        c2.info(f"**Test:** {metrics_data['test_size']:,}")
        c3.info(f"**Features:** {metrics_data['feature_type'].upper()}")
    else:
        st.info("No metrics found. Run `train_model.py` first.")


# --------------------------------------------------
# Dispatch
# --------------------------------------------------
st.markdown('<hr class="divider">', unsafe_allow_html=True)

common_kwargs = dict(
    model=model, scaler=scaler, vectorizer=vectorizer,
    bert_tokenizer=bert_tokenizer, bert_model_obj=bert_model_obj,
    feature_type=feature_type, nlp=nlp,
    num_features=num_features, show_style=show_style,
    show_entity=show_entity, show_radar=show_radar,
)

if analyse_text:
    if news_text.strip():
        st.markdown("### Analysis")
        analyse_article(news_text.strip(), **common_kwargs)
    else:
        st.warning("Paste some article text first.")

if analyse_url:
    if url_input.strip():
        with st.spinner("Fetching article…"):
            scraped = scrape_url(url_input.strip())
        if scraped:
            with st.expander("Scraped text preview"):
                st.text(scraped[:600] + "…")
            st.markdown("### Analysis")
            analyse_article(scraped, source_url=url_input.strip(), **common_kwargs)
        else:
            st.error("Could not extract text. Try pasting directly.")
    else:
        st.warning("Enter a URL first.")

if analyse_batch:
    if batch_text.strip():
        articles = [a.strip() for a in batch_text.split("\n---\n") if a.strip()]
        if not articles:
            st.warning("No articles found. Separate with `---`.")
        else:
            st.markdown(f"### Batch — {len(articles)} article(s)")
            scores = []
            for idx, article in enumerate(articles, 1):
                with st.expander(f"Article {idx} — {article[:60].replace(chr(10), ' ')}…", expanded=(idx == 1)):
                    analyse_article(article, **common_kwargs)
                    if st.session_state.history:
                        scores.append(st.session_state.history[0]["score"])
            if scores:
                st.markdown("---")
                st.markdown("#### Batch Summary")
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("Average", f"{np.mean(scores):.1f}%")
                bc3.metric("Lowest", f"{min(scores):.1f}%")
    else:
        st.warning("Paste at least one article.")