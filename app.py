"""
app.py  v9  —  Verity Intelligence Command

v9 changes (Relational Intelligence layer)
──────────────────────────────────────────
1. Neo4j Integration
   • @st.cache_resource loads a Neo4jGraph instance from st.secrets on startup.
   • Every text scan (analyse_article) and image scan now call
     graph.upsert_scan_result() alongside the existing SQLite insert_record().
   • Entities extracted by entity_checker are forwarded to the graph so
     (Article)-[:MENTIONS]->(Entity) edges are created automatically.

2. Proactive Threat Hunter staging support
   • The Analyst Review tab now also surfaces JSON articles written by
     threat_hunter.py (*.json files in staging_data/**/).
   • Analysts can promote/discard threat-hunter articles just like images.

3. Intelligence Graph tab  (🕸️)
   • New 4th top-level tab added between Analyst Review and Model Health.
   • Renders an interactive graph with streamlit-agraph (falls back to a
     plain table when agraph isn't installed).
   • "Narrative Clusters" panel shows entities that appear in ≥2
     Fake/Uncertain articles across ≥2 different domains — the key
     disinformation signal.
   • Domain Risk Profile table shows per-domain verdict breakdown.

4. requirements.txt
   • neo4j, langchain-community, streamlit-agraph added (see requirements.txt).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import numpy as np
import joblib
import nltk
import streamlit as st
from pathlib import Path

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
from claim_verifier import verify_claims
from Sentence_scorer import score_sentences, suspicion_label
from image_forensics import (
    load_image_model, detect_ai_image,
    extract_exif, generate_ela_heatmap,
    blend_ela_overlay,
    consensus_meter,
    extract_faces, pil_to_bytes,
    load_deepfake_model, detect_deepfake,
    generate_forensic_reasoning,
    biological_consistency_check,
)
from database import init_db, insert_record, get_recent_records, clear_history, get_stats
from report_generator import generate_pdf_report

# ── Boot-time dirs ────────────────────────────────────────────────────────────
_STAGING_DIR   = Path("staging_data")
_TRAINING_REAL = Path("training_data/real")
_TRAINING_FAKE = Path("training_data/fake")
_LOG_FILE      = Path("training.log")

for _d in (_STAGING_DIR, _TRAINING_REAL, _TRAINING_FAKE):
    _d.mkdir(parents=True, exist_ok=True)

init_db()
STOP_WORDS = set(stopwords.words("english"))

DISREPUTABLE_DOMAINS = {
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "theonion.com", "empirenews.net", "abcnews.com.co",
    "worldnewsdailyreport.com", "huzlers.com",
    "nationalreport.net", "stuppid.com",
}
SATIRE_DOMAINS = {"theonion.com", "babylonbee.com", "thebeaverton.com"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verity Intelligence Command",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Intelligence Command aesthetic (unchanged from v8)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg:       #05070d;
  --surface:  #0b0f1a;
  --border:   #141c30;
  --border2:  #1e2840;
  --muted:    #2a3555;
  --text-dim: #4a5878;
  --text:     #c8d4e8;
  --text-hi:  #eef2fa;
  --accent:   #4f8fff;
  --accent2:  #7eb8f7;
  --green:    #3dd68c;
  --amber:    #f5a623;
  --red:      #f06060;
  --teal:     #2ab8a0;
  --purple:   #a78bfa;
}

html, body, [class*="css"] {
  font-family: 'Syne', sans-serif;
  background-color: var(--bg);
  color: var(--text);
}
.main { background: var(--bg); padding-top: 0 !important; }
.block-container { padding-top: 1.2rem !important; max-width: 1280px; }
h1,h2,h3,h4 { font-family:'DM Serif Display',serif; color:var(--text-hi); letter-spacing:-0.01em; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 4px;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-dim);
  padding: 8px 22px;
  border-radius: 7px;
  border: none;
  transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: var(--border); }
.stTabs [aria-selected="true"] {
  color: var(--bg) !important;
  background: var(--accent) !important;
  font-weight: 800 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.6rem; }

.stTextArea textarea {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 14px !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; }
.stTextInput input {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
}

.stButton>button {
  background: var(--accent);
  color: var(--bg);
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border: none;
  border-radius: 7px;
  padding: 0.65rem 2rem;
  width: 100%;
  transition: all 0.18s;
}
.stButton>button:hover {
  background: var(--accent2);
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(79,143,255,0.3);
}
button[data-testid="baseButton-secondary"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
}
.sync-btn > button {
  background: linear-gradient(135deg, #1a3a6e 0%, #0f2040 100%) !important;
  color: var(--accent2) !important;
  border: 1px solid var(--accent) !important;
  font-size: 14px !important;
  padding: 0.9rem 2rem !important;
  letter-spacing: 0.12em !important;
  text-shadow: 0 0 20px rgba(79,143,255,0.6);
  box-shadow: 0 0 30px rgba(79,143,255,0.15), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
.sync-btn > button:hover {
  box-shadow: 0 0 40px rgba(79,143,255,0.35), inset 0 1px 0 rgba(255,255,255,0.05) !important;
  transform: translateY(-2px) !important;
}

.hero-wrap {
  padding: 2rem 0 1.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.6rem;
  display: flex;
  align-items: baseline;
  gap: 24px;
}
.hero-title {
  font-family: 'DM Serif Display', serif;
  font-size: 2.6rem;
  color: var(--text-hi);
  letter-spacing: -0.025em;
  line-height: 1;
  margin: 0;
}
.hero-title em { color: var(--accent); font-style: italic; }
.hero-tag {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
  border: 1px solid var(--border2);
  padding: 4px 10px;
  border-radius: 4px;
}
.hero-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--green); display: inline-block; margin-right: 10px;
  animation: pulse-dot 2.4s infinite;
}
@keyframes pulse-dot {
  0%,100%{opacity:1;transform:scale(1)}
  50%{opacity:0.3;transform:scale(0.6)}
}

.metric-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:1rem 0; }
.mcard {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  text-align: center;
  transition: transform 0.18s, border-color 0.18s;
}
.mcard:hover { transform:translateY(-2px); border-color:var(--border2); }
.mcard-label {
  font-size: 9px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
}
.mcard-value {
  font-family:'DM Serif Display',serif; font-size:1.9rem;
  color: var(--accent2); line-height: 1;
}
.mcard-value.green { color:var(--green); }
.mcard-value.red   { color:var(--red); }
.mcard-value.amber { color:var(--amber); }
.mcard-value.small { font-size:1.2rem; margin-top:4px; }
.mcard-sub { font-size:9px; color:var(--muted); margin-top:5px; font-family:'JetBrains Mono',monospace; }

.verdict {
  padding:1.1rem 1.6rem; border-radius:10px;
  display:flex; align-items:center; gap:14px;
  margin:1rem 0; animation:slide-in 0.3s ease;
}
@keyframes slide-in { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
.verdict-real      { background:#071812; border:1px solid #1a4a30; border-left:4px solid var(--green); }
.verdict-uncertain { background:#12100a; border:1px solid #3a2e10; border-left:4px solid var(--amber); }
.verdict-fake      { background:#140808; border:1px solid #3a1010; border-left:4px solid var(--red); }
.verdict-icon { font-size:1.8rem; line-height:1; }
.verdict-text { font-family:'DM Serif Display',serif; font-size:1.2rem; }
.verdict-real .verdict-text      { color:var(--green); }
.verdict-uncertain .verdict-text { color:var(--amber); }
.verdict-fake .verdict-text      { color:var(--red); }
.verdict-sub { font-size:11px; color:var(--text-dim); margin-top:3px; }

.sec-title {
  font-family:'Syne',sans-serif; font-size:9px; font-weight:800;
  letter-spacing:0.22em; text-transform:uppercase; color:var(--muted);
  margin:1.8rem 0 0.9rem; display:flex; align-items:center; gap:10px;
}
.sec-title::after { content:''; flex:1; height:1px; background:var(--border); }

.sbar-wrap { margin-bottom:12px; }
.sbar-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:5px; }
.sbar-label { font-size:12px; color:var(--text-dim); }
.sbar-val { font-family:'JetBrains Mono',monospace; font-size:11px; }
.sbar-track { background:var(--surface); border-radius:3px; height:4px; border:1px solid var(--border); overflow:hidden; }
.sbar-fill { height:4px; border-radius:3px; }
.sbar-hint { font-size:10px; color:var(--muted); margin-top:3px; }

.tag-wrap { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.wtag {
  display:inline-flex; align-items:center; gap:5px; padding:4px 10px;
  border-radius:20px; font-size:11px; font-weight:700;
  font-family:'JetBrains Mono',monospace; transition:transform 0.15s;
}
.wtag:hover { transform:scale(1.06); }
.wtag-real { background:#071812; color:var(--green); border:1px solid #1a4a30; }
.wtag-fake { background:#140808; color:var(--red); border:1px solid #3a1010; }
.wtag-score { font-size:9px; opacity:0.55; }

.ent-card,.claim-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:7px; padding:0.7rem 1rem; margin-bottom:7px;
  display:flex; align-items:flex-start; gap:12px;
}
.ent-badge { font-size:9px; font-weight:800; letter-spacing:0.12em; padding:3px 7px; border-radius:3px; white-space:nowrap; margin-top:2px; }
.ent-verified  { background:#071812; color:var(--green); border:1px solid #1a4a30; }
.ent-notfound  { background:#140808; color:var(--red); border:1px solid #3a1010; }
.ent-ambiguous { background:#12100a; color:var(--amber); border:1px solid #3a2e10; }
.ent-name { font-size:14px; font-weight:600; color:var(--text-hi); }
.ent-name a { color:var(--accent); text-decoration:none; }
.ent-name a:hover { text-decoration:underline; }
.ent-type { font-size:10px; color:var(--muted); font-family:'JetBrains Mono',monospace; }
.ent-summary { font-size:12px; color:var(--text-dim); margin-top:4px; line-height:1.5; }
.ent-impact { font-family:'JetBrains Mono',monospace; font-size:11px; margin-left:auto; white-space:nowrap; }
.claim-text { font-size:13px; color:var(--text); line-height:1.5; margin-bottom:4px; }
.claim-source { font-size:11px; color:var(--text-dim); }
.claim-source a { color:var(--accent); text-decoration:none; }
.claim-body { font-size:11px; color:var(--muted); margin-top:3px; line-height:1.4; }

.sent-row {
  padding:5px 10px; border-radius:5px; margin-bottom:4px;
  font-size:13px; line-height:1.6; border-left:3px solid transparent;
}
.sent-high   { background:rgba(240,96,96,0.08); border-left-color:var(--red); color:var(--text); }
.sent-medium { background:rgba(245,166,35,0.07); border-left-color:var(--amber); color:var(--text); }
.sent-low    { background:rgba(61,214,140,0.04); border-left-color:#3dd68c33; color:var(--text-dim); }
.sent-score  { font-family:'JetBrains Mono',monospace; font-size:10px; opacity:0.45; margin-left:8px; }

.domain-warn {
  background:#100d00; border:1px solid #3a2800;
  border-left:3px solid var(--amber); padding:0.6rem 1rem;
  border-radius:0 7px 7px 0; font-size:13px; color:var(--amber); margin-bottom:0.8rem;
}

.img-panel { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1rem; }
.img-panel-title { font-size:9px; font-weight:800; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:0.6rem; }
.ela-legend { display:flex; gap:18px; margin-top:6px; font-size:10px; color:var(--muted); font-family:'JetBrains Mono',monospace; }
.ela-legend span::before { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; content:''; }
.ela-low::before  { background:#00008b; }
.ela-mid::before  { background:#00aa88; }
.ela-high::before { background:#ff2200; }
.exif-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px 20px; margin-top:6px; }
.exif-row { display:flex; flex-direction:column; padding:5px 0; border-bottom:1px solid #10141e; }
.exif-key { font-size:9px; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); font-family:'JetBrains Mono',monospace; }
.exif-val { font-size:13px; color:var(--text-dim); margin-top:2px; word-break:break-all; }
.forensics-score-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:1rem 0; }

.face-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:0.7rem 0.6rem; text-align:center; transition:border-color 0.2s; }
.face-card.fake   { border-color:#3a1010; }
.face-card.real   { border-color:#1a4a30; }
.face-card.unsure { border-color:#3a2e10; }
.face-card-idx { font-size:9px; letter-spacing:0.15em; text-transform:uppercase; color:var(--muted); margin-bottom:5px; font-family:'JetBrains Mono',monospace; }
.face-prob     { font-family:'DM Serif Display',serif; font-size:1.5rem; line-height:1; margin:4px 0 2px; }
.face-verdict  { font-size:9px; font-weight:800; letter-spacing:0.12em; text-transform:uppercase; padding:3px 8px; border-radius:3px; display:inline-block; margin-top:4px; }
.face-verdict.fake   { background:#140808; color:var(--red); border:1px solid #3a1010; }
.face-verdict.real   { background:#071812; color:var(--green); border:1px solid #1a4a30; }
.face-verdict.unsure { background:#12100a; color:var(--amber); border:1px solid #3a2e10; }
.face-conf { font-size:9px; color:var(--muted); margin-top:3px; font-family:'JetBrains Mono',monospace; }

.reasoning-log {
  background:#07100c; border:1px solid #1a3028;
  border-radius:10px; padding:1rem 1.2rem; margin:0.6rem 0 1rem;
}
.reasoning-log-header {
  font-family:'Syne',sans-serif; font-size:9px; font-weight:800;
  letter-spacing:0.24em; text-transform:uppercase; color:#2a5040;
  margin-bottom:0.7rem; display:flex; align-items:center; gap:8px;
}
.reasoning-log-header::after { content:''; flex:1; height:1px; background:#1a3028; }
.reasoning-bullet {
  display:flex; gap:10px; align-items:flex-start; padding:7px 0;
  border-bottom:1px solid #0d1e18; font-size:12px; line-height:1.6;
  color:#7ab8a0; animation:slide-in 0.3s ease;
}
.reasoning-bullet:last-child { border-bottom:none; }
.reasoning-bullet-idx { font-family:'JetBrains Mono',monospace; font-size:9px; color:#2a5040; min-width:18px; margin-top:3px; letter-spacing:0.1em; }
.reasoning-bullet-text { flex:1; }

/* ─── NEW ─────────────────────────────────────────────────────────────────── */
.consensus-box {
  border-radius:10px; padding:1.1rem 1.4rem; margin:1.2rem 0;
  display:flex; align-items:flex-start; gap:14px;
  border:1px solid var(--border2);
  box-shadow:0 2px 12px rgba(0,0,0,0.18);
  animation:slide-in 0.3s ease;
  position:relative;
}
.consensus-box::before {
  content:'⬡ CONSENSUS METER';
  position:absolute; top:-9px; left:14px;
  font-family:'Syne',sans-serif; font-size:8px; font-weight:800;
  letter-spacing:0.18em; text-transform:uppercase;
  color:var(--muted); background:var(--bg); padding:0 6px;
}
.consensus-icon { font-size:1.6rem; line-height:1; margin-top:2px; }
.consensus-label { font-family:'DM Serif Display',serif; font-size:1.1rem; }
.consensus-detail { font-size:12px; margin-top:5px; line-height:1.6; opacity:0.8; }

/* Forensic overlay panel: distinct teal accent border */
.img-panel--overlay {
  border:1px solid #1a3028 !important;
  background:linear-gradient(180deg,#07100c 0%,var(--surface) 100%) !important;
}
.img-panel--overlay .img-panel-title {
  color:var(--teal) !important;
  letter-spacing:0.14em;
}

.review-header { font-family:'DM Serif Display',serif; font-size:2rem; color:var(--text-hi); margin-bottom:0.3rem; }
.review-sub { font-size:10px; letter-spacing:0.18em; text-transform:uppercase; color:var(--muted); margin-bottom:1.8rem; }
.review-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.4rem; margin-bottom:1.2rem; }
.review-filename { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text-dim); margin-bottom:10px; word-break:break-all; }
.review-scores { display:flex; gap:12px; margin:10px 0 14px; }
.review-score-pill { font-family:'JetBrains Mono',monospace; font-size:11px; padding:4px 10px; border-radius:20px; border:1px solid var(--border2); color:var(--amber); background:#12100a; }
.review-empty { text-align:center; padding:3rem 0; color:var(--muted); font-size:13px; }
.review-counter { font-family:'JetBrains Mono',monospace; font-size:10px; color:var(--muted); text-align:right; margin-bottom:6px; }

.train-log {
  background:#070b14; border:1px solid #0e1828;
  border-radius:10px; padding:1rem; font-family:'JetBrains Mono',monospace;
  font-size:11px; color:var(--text-dim); max-height:320px;
  overflow-y:auto; margin-top:0.6rem;
}
.train-log-line { padding:3px 0; border-bottom:1px solid #0d1420; line-height:1.5; }
.train-log-line:last-child { border-bottom:none; }
.tl-epoch  { color:var(--accent2); }
.tl-best   { color:var(--green); font-weight:700; }
.tl-error  { color:var(--red); }
.tl-done   { color:var(--teal); font-weight:700; }
.tl-muted  { color:var(--muted); }

.health-card {
  background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:1.4rem; margin-bottom:1rem;
}
.health-card-title { font-size:9px; font-weight:800; letter-spacing:0.2em; text-transform:uppercase; color:var(--muted); margin-bottom:1rem; }

section[data-testid="stSidebar"] { background:#060810; border-right:1px solid var(--border); }
section[data-testid="stSidebar"] .block-container { padding-top:1rem !important; }
.sb-brand { font-family:'DM Serif Display',serif; font-size:1.4rem; color:var(--text-hi); }
.sb-brand em { color:var(--accent); font-style:italic; }
.sb-tagline { font-size:9px; color:var(--muted); letter-spacing:0.18em; text-transform:uppercase; }
.sb-section { font-size:9px; font-weight:800; letter-spacing:0.22em; text-transform:uppercase; color:var(--muted); margin:1.2rem 0 0.5rem; }
.sb-stat { display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid var(--surface); }
.sb-stat-label { font-size:12px; color:var(--text-dim); }
.sb-stat-val   { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--accent2); }
.hist-row { background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:7px 10px; margin-bottom:5px; display:flex; align-items:center; gap:8px; }
.hist-score { font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:700; min-width:38px; }
.hist-text  { font-size:11px; color:var(--muted); overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }
.divider { border:none; border-top:1px solid var(--border); margin:1.2rem 0; }
.note    { font-size:11px; color:var(--muted); line-height:1.5; }
.staging-badge { background:#12100a; border:1px solid #3a2e10; border-radius:6px; padding:6px 10px; font-size:11px; color:var(--amber); font-family:'JetBrains Mono',monospace; text-align:center; margin-bottom:8px; }

/* ── Graph tab ───────────────────────────────────────────────── */
.graph-panel {
  background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:1.2rem; margin-bottom:1rem;
}
.cluster-row {
  background:var(--surface); border:1px solid var(--border);
  border-radius:7px; padding:0.7rem 1rem; margin-bottom:7px;
  display:grid; grid-template-columns:2fr 1fr 1fr; gap:12px; align-items:center;
}
.cluster-entity { font-size:14px; font-weight:600; color:var(--text-hi); }
.cluster-type   { font-size:10px; color:var(--muted); font-family:'JetBrains Mono',monospace; }
.cluster-badge  { font-size:10px; padding:3px 8px; border-radius:20px; background:#140808; color:var(--red); border:1px solid #3a1010; text-align:center; font-family:'JetBrains Mono',monospace; }
.domain-risk-row { display:flex; align-items:center; gap:12px; padding:7px 0; border-bottom:1px solid var(--border); font-size:12px; }
.domain-risk-name { flex:2; color:var(--text-dim); font-family:'JetBrains Mono',monospace; }
.domain-risk-fake { color:var(--red); font-weight:700; min-width:40px; text-align:right; }
.domain-risk-total { color:var(--muted); min-width:50px; text-align:right; }
</style>
""", unsafe_allow_html=True)

MIN_WORDS = 30

# ─── NEW ───────────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "lime_cache":         {},
    "review_idx":         0,
    "train_proc":         None,
    # Image forensics — persist across tab navigation
    "_last_img_report":   None,
    "_ela_img_bytes":     None,   # PNG bytes of ELA heatmap (PIL not picklable)
    "_ela_source_hash":   None,   # md5 of original image bytes → cache busting
    # Analyst Review — toast feedback
    "_review_toast":      None,   # ("success"|"error", message) cleared after render
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ──────────────────────────────────────────────────────────────────────────────
# Neo4j — lazy singleton via st.cache_resource
# ──────────────────────────────────────────────────────────────────────────────

# Module-level variable: stores Neo4j failure reason without calling
# st.warning() inside @cache_resource (which re-executes on every
# interaction when cached value is None, flooding the UI with warnings).
_neo4j_warning: str = ""

@st.cache_resource(show_spinner=False)
def _get_graph():
    """
    Load credentials from st.secrets[neo4j] and return a Neo4jGraph instance.
    Returns None gracefully when neo4j is not configured.

    FIX (v9.1): st.warning() calls removed from inside this function.
    @cache_resource re-runs the function body on every Streamlit interaction
    when the cached return value is None, causing the warning to flash
    repeatedly. The failure reason is stored in _neo4j_warning and displayed
    once in the sidebar instead.
    """
    global _neo4j_warning
    try:
        from neo4j_graph import Neo4jGraph
        uri  = st.secrets["neo4j"]["NEO4J_URI"]
        user = st.secrets["neo4j"]["NEO4J_USER"]
        pwd  = st.secrets["neo4j"]["NEO4J_PASSWORD"]
        g = Neo4jGraph(uri, user, pwd)
        if g.ping():
            _neo4j_warning = ""
            return g
        _neo4j_warning = "Neo4j ping failed — graph features disabled."
    except KeyError:
        _neo4j_warning = ""   # secrets not configured — silently skip
    except Exception as exc:
        _neo4j_warning = f"Neo4j unavailable: {exc}"
    return None
def _graph_upsert(
    snippet: str,
    verdict: str,
    final_score: float,
    domain: str = "unknown",
    entity_report=None,
    modality: str = "Text",
) -> None:
    """
    Fire-and-forget graph write. Errors are logged but never crash the app.
    Forwards entity names + types from entity_checker so graph edges are rich.
    """
    graph = _get_graph()
    if graph is None:
        return
    try:
        entities = []
        if entity_report and not entity_report.error:
            entities = [
                (e.name, e.entity_type, e.frequency)
                for e in entity_report.entities
            ]
        graph.upsert_scan_result(
            snippet     = snippet,
            verdict     = verdict,
            final_score = final_score,
            domain      = domain,
            entities    = entities,
            modality    = modality,
        )
    except Exception as exc:
        # Never surface graph errors to the analyst
        print(f"[neo4j] upsert failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty Logger
# ──────────────────────────────────────────────────────────────────────────────

def flag_for_review(image, ai_score: float, deepfake_score: float) -> Optional[str]:
    if not (50.0 <= ai_score <= 84.0 or 45.0 <= deepfake_score <= 79.0):
        return None
    ts    = int(time.time())
    fname = f"flagged_ai_{int(round(ai_score))}_df_{int(round(deepfake_score))}_{ts}.png"
    dest  = _STAGING_DIR / fname
    try:
        image.save(str(dest), format="PNG")
        return str(dest)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def reading_time(text):
    return f"{max(1, round(len(text.split()) / 200))} min"

def text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def check_domain(url):
    import re
    m = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if not m: return None, False
    domain = m.group(1).lower()
    return domain, domain in DISREPUTABLE_DOMAINS

def confidence_interval(score, n=100):
    p=score/100; z=1.645
    lo=max(0,(p-z*(p*(1-p)/n)**0.5)*100)
    hi=min(100,(p+z*(p*(1-p)/n)**0.5)*100)
    return round(lo,1), round(hi,1)

def scrape_url(url):
    try:
        from newspaper import Article
        a=Article(url); a.download(); a.parse()
        return f"{a.title or ''}\n\n{a.text}" if a.text.strip() else ""
    except Exception as e:
        st.error(f"Could not scrape URL: {e}"); return ""

def render_reasoning_log(bullets: List[str], title: str = "Forensic Reasoning Log") -> None:
    if not bullets:
        return
    items_html = "".join(
        f'<div class="reasoning-bullet">'
        f'<span class="reasoning-bullet-idx">{str(i+1).zfill(2)}</span>'
        f'<span class="reasoning-bullet-text">{b}</span>'
        f'</div>'
        for i, b in enumerate(bullets)
    )
    st.markdown(
        f'<div class="reasoning-log">'
        f'<div class="reasoning-log-header">⬡ {title}</div>'
        f'{items_html}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

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
        import spacy; return spacy.load("en_core_web_sm")
    except (OSError, ImportError): return None

@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json") as f: return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict(text, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type):
    style_raw = stylistic_features(text)
    style_vec = scaler.transform(style_raw.reshape(1,-1)) if scaler else style_raw.reshape(1,-1)
    if "bert" in feature_type:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bert_model_obj.to(device)
        enc = bert_tokenizer(
            [clean_text(text, STOP_WORDS)], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = bert_model_obj(**enc)
        emb = out.last_hidden_state[:,0,:].cpu().numpy()
        X   = np.hstack([emb, style_vec])
    else:
        from scipy.sparse import hstack, csr_matrix
        X = hstack([vectorizer.transform([clean_text(text, STOP_WORDS)]),
                    csr_matrix(style_vec)])
    proba = model.predict_proba(X)[0]
    return proba[0]*100, proba[1]*100, style_raw

def lime_predict_fn(texts, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type):
    return np.array([
        [r/100, f/100]
        for r,f,_ in [predict(t,model,scaler,vectorizer,bert_tokenizer,bert_model_obj,feature_type)
                      for t in texts]
    ])


# ──────────────────────────────────────────────────────────────────────────────
# SVG Gauge
# ──────────────────────────────────────────────────────────────────────────────

def render_gauge(score: float):
    pct=score/100; sweep=240; start_deg=210; rad=math.pi/180
    angle=start_deg-pct*sweep
    nx=100+58*math.cos(angle*rad); ny=95-58*math.sin(angle*rad)
    aex=100+75*math.cos(angle*rad); aey=95-75*math.sin(angle*rad)
    asx=100+75*math.cos(start_deg*rad); asy=95-75*math.sin(start_deg*rad)
    large=1 if pct*sweep>180 else 0
    end_x=100+75*math.cos((start_deg-sweep)*rad); end_y=95-75*math.sin((start_deg-sweep)*rad)
    col="  #f06060" if pct<0.4 else ("#f5a623" if pct<0.7 else "#3dd68c")
    st.markdown(f"""
<div style="display:flex;justify-content:center;margin:0.5rem 0 1.2rem">
<svg viewBox="0 0 200 115" width="280" xmlns="http://www.w3.org/2000/svg">
  <defs><filter id="g"><feGaussianBlur stdDeviation="3" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
  <path d="M {asx:.2f} {asy:.2f} A 75 75 0 1 1 {end_x:.2f} {end_y:.2f}"
        fill="none" stroke="#0b0f1a" stroke-width="9" stroke-linecap="round"/>
  <path d="M {asx:.2f} {asy:.2f} A 75 75 0 {large} 1 {aex:.2f} {aey:.2f}"
        fill="none" stroke="{col}" stroke-width="9" stroke-linecap="round" filter="url(#g)" opacity="0.9">
    <animate attributeName="stroke-dasharray" from="0 472" to="{pct*472:.1f} 472"
             dur="1s" fill="freeze" calcMode="spline" keySplines="0.4 0 0.2 1"/></path>
  <text x="28"  y="108" text-anchor="middle" font-family="Syne,sans-serif" font-size="8" fill="#f06060" opacity="0.45">FAKE</text>
  <text x="172" y="108" text-anchor="middle" font-family="Syne,sans-serif" font-size="8" fill="#3dd68c" opacity="0.45">REAL</text>
  <line x1="100" y1="95" x2="{nx:.2f}" y2="{ny:.2f}" stroke="{col}" stroke-width="2" stroke-linecap="round" filter="url(#g)">
    <animateTransform attributeName="transform" type="rotate"
      from="{start_deg} 100 95" to="{start_deg-pct*sweep:.2f} 100 95"
      dur="1s" fill="freeze" calcMode="spline" keySplines="0.4 0 0.2 1"/></line>
  <circle cx="100" cy="95" r="5"   fill="{col}" filter="url(#g)"/>
  <circle cx="100" cy="95" r="2.5" fill="#05070d"/>
  <text x="100" y="78" text-anchor="middle" font-family="DM Serif Display,serif"
        font-size="24" font-weight="700" fill="{col}">{score:.0f}%</text>
</svg></div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Radar chart
# ──────────────────────────────────────────────────────────────────────────────

def render_radar(style_raw):
    try:
        import plotly.graph_objects as go
        labels=[l for l,_,_ in STYLE_LABELS]
        vals=[min(float(style_raw[i])/max(STYLE_DISPLAY_MAX[i],1e-9),1.0) for i in range(len(labels))]
        vals+=vals[:1]; theta=labels+labels[:1]
        fig=go.Figure(go.Scatterpolar(r=vals,theta=theta,fill="toself",
            fillcolor="rgba(79,143,255,0.06)",line=dict(color="#4f8fff",width=1.5)))
        fig.update_layout(
            polar=dict(bgcolor="#0b0f1a",
                angularaxis=dict(color="#1e2840",gridcolor="#141c30",tickfont=dict(size=9,color="#4a5878")),
                radialaxis=dict(color="#141c30",gridcolor="#141c30",range=[0,1],tickfont=dict(size=7,color="#2a3555"))),
            paper_bgcolor="#05070d",plot_bgcolor="#05070d",
            margin=dict(l=40,r=40,t=10,b=10),height=300,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.caption("Install plotly for radar chart.")


# ──────────────────────────────────────────────────────────────────────────────
# Article analysis  (v9: adds graph upsert)
# ──────────────────────────────────────────────────────────────────────────────

def analyse_article(
    text, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type,
    nlp, num_features, show_style, show_entity, show_radar,
    show_claims, show_sentences, source_url="",
):
    word_count = len(text.split())
    rt         = reading_time(text)

    if word_count < MIN_WORDS:
        st.warning(f"Only {word_count} words — results may be unreliable.")

    domain = "unknown"
    if source_url:
        domain, is_bad = check_domain(source_url)
        domain = domain or "unknown"
        if is_bad:
            label = "Satire site" if domain in SATIRE_DOMAINS else "Known low-credibility domain"
            st.markdown(f'<div class="domain-warn">⚠ <strong>{label}:</strong> {domain}</div>', unsafe_allow_html=True)

    with st.spinner("Analysing…"):
        real_prob, fake_prob, style_raw = predict(
            text, model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type)
        ml_score = round(real_prob, 2)

    entity_report = None
    if show_entity and nlp is not None:
        with st.spinner("Checking entities…"):
            entity_report = check_entities(text, nlp)

    claim_report = None
    if show_claims and nlp is not None:
        with st.spinner("Verifying claims…"):
            claim_report = verify_claims(text, nlp)

    sent_scores = score_sentences(text) if show_sentences else []

    entity_score = entity_report.entity_score if (entity_report and not entity_report.error and entity_report.total_checked > 0) else 50.0
    claim_score  = claim_report.claim_score   if (claim_report  and not claim_report.error  and claim_report.claims)            else 50.0
    sent_score   = round((1-sum(s.score for s in sent_scores)/len(sent_scores))*100, 1) if sent_scores else 50.0

    final_score  = round(ml_score*0.50 + entity_score*0.20 + claim_score*0.20 + sent_score*0.10, 1)
    ci_lo, ci_hi = confidence_interval(final_score)

    if final_score >= 70:   vcls,vicon,vtext = "verdict-real",      "✅", f"Likely Credible — {final_score}%"
    elif final_score >= 40: vcls,vicon,vtext = "verdict-uncertain", "⚠️", f"Uncertain — {final_score}%"
    else:                   vcls,vicon,vtext = "verdict-fake",      "❌", f"Likely Fabricated — {final_score}%"

    active     = [k for k,v in {"Entity":show_entity,"Claims":show_claims,"Sentences":show_sentences}.items() if v]
    score_note = f"ML + {' + '.join(active)}" if active else "ML model"

    st.markdown(
        f'<div class="verdict {vcls}"><div class="verdict-icon">{vicon}</div>'
        f'<div><div class="verdict-text">{vtext} confidence</div>'
        f'<div class="verdict-sub">{score_note} · 90% CI [{ci_lo}–{ci_hi}%] · {word_count} words · {rt} read</div>'
        f'</div></div>', unsafe_allow_html=True)

    render_gauge(final_score)

    esc="green" if entity_score>=70 else ("red" if entity_score<40 else "amber")
    csc="green" if claim_score>=70  else ("red" if claim_score<40  else "amber")
    st.markdown(
        f'<div class="metric-grid">'
        f'<div class="mcard"><div class="mcard-label">Final</div><div class="mcard-value">{final_score}%</div><div class="mcard-sub">CI {ci_lo}–{ci_hi}</div></div>'
        f'<div class="mcard"><div class="mcard-label">ML Score</div><div class="mcard-value">{ml_score}%</div></div>'
        f'<div class="mcard"><div class="mcard-label">Entity</div><div class="mcard-value {esc}">{entity_score}%</div></div>'
        f'<div class="mcard"><div class="mcard-label">Claims</div><div class="mcard-value {csc}">{claim_score}%</div></div>'
        f'<div class="mcard"><div class="mcard-label">Read Time</div><div class="mcard-value small amber">{rt}</div><div class="mcard-sub">{word_count} words</div></div>'
        f'</div>', unsafe_allow_html=True)

    if show_style:
        st.markdown('<div class="sec-title">Stylistic Signals</div>', unsafe_allow_html=True)
        if show_radar: render_radar(style_raw)
        bar_colors={"fake":"#f06060","real":"#3dd68c","neutral":"#f5a623"}
        cols=st.columns(2)
        for i,(label,stype,hint) in enumerate(STYLE_LABELS):
            raw_val=float(style_raw[i])
            bar_pct=min(raw_val/max(STYLE_DISPLAY_MAX[i],1e-9),1.0)*100
            bar_color=bar_colors[stype]
            display=f"{raw_val:.3f}" if raw_val<1 else f"{raw_val:.1f}"
            cols[i%2].markdown(
                f'<div class="sbar-wrap"><div class="sbar-header">'
                f'<span class="sbar-label">{label}</span>'
                f'<span class="sbar-val" style="color:{bar_color}">{display}</span></div>'
                f'<div class="sbar-track"><div class="sbar-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div></div>'
                f'<div class="sbar-hint">{hint}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">LIME Explainability</div>', unsafe_allow_html=True)
    t_hash = text_hash(text)
    if t_hash in st.session_state["lime_cache"]:
        exp_list = st.session_state["lime_cache"][t_hash]
        st.caption("Using cached LIME explanation.")
    else:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["Real","Fake"])
        def _lime_fn(texts):
            return lime_predict_fn(texts,model,scaler,vectorizer,bert_tokenizer,bert_model_obj,feature_type)
        with st.spinner("Generating LIME explanation…"):
            explanation = explainer.explain_instance(text, _lime_fn, num_features=num_features)
        exp_list = explanation.as_list()
        st.session_state["lime_cache"][t_hash] = exp_list

    real_words=sorted([(w,round(s,4)) for w,s in exp_list if s>0],key=lambda x:-x[1])
    fake_words=sorted([(w,round(abs(s),4)) for w,s in exp_list if s<0],key=lambda x:-x[1])
    lc1,lc2=st.columns(2)
    with lc1:
        st.markdown("**Signals → Real**")
        tags="".join(f'<span class="wtag wtag-real">{w} <span class="wtag-score">+{s}</span></span>' for w,s in real_words) or "<span class='note'>None found</span>"
        st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)
    with lc2:
        st.markdown("**Signals → Fake**")
        tags="".join(f'<span class="wtag wtag-fake">{w} <span class="wtag-score">-{s}</span></span>' for w,s in fake_words) or "<span class='note'>None found</span>"
        st.markdown(f'<div class="tag-wrap">{tags}</div>', unsafe_allow_html=True)

    if claim_report:
        st.markdown('<div class="sec-title">Claim Verification</div>', unsafe_allow_html=True)
        if claim_report.error:
            st.warning(f"Claim check unavailable: {claim_report.error}")
        elif not claim_report.claims:
            st.info("No verifiable claims found.")
        else:
            cc1,cc2,cc3,cc4=st.columns(4)
            cc1.markdown(f'<div class="mcard"><div class="mcard-label">Claim Score</div><div class="mcard-value">{claim_report.claim_score}</div></div>', unsafe_allow_html=True)
            cc2.markdown(f'<div class="mcard"><div class="mcard-label">Supported</div><div class="mcard-value green">{claim_report.supported_count}</div></div>', unsafe_allow_html=True)
            cc3.markdown(f'<div class="mcard"><div class="mcard-label">Contradicted</div><div class="mcard-value red">{claim_report.contradicted_count}</div></div>', unsafe_allow_html=True)
            cc4.markdown(f'<div class="mcard"><div class="mcard-label">Unverified</div><div class="mcard-value amber">{claim_report.unverified_count}</div></div>', unsafe_allow_html=True)
            for cr in claim_report.claims:
                if cr.verdict=="supported":      col,icon,bcls="#3dd68c","✓","ent-verified"
                elif cr.verdict=="contradicted": col,icon,bcls="#f06060","✗","ent-notfound"
                else:                            col,icon,bcls="#f5a623","?","ent-ambiguous"
                src_html=(f'<a href="{cr.source_url}" target="_blank">{(cr.source_title or "")[:70]}</a>'
                          if cr.source_url else '<span style="color:var(--muted)">No source found</span>')
                body_html=f'<div class="claim-body">{(cr.source_body or "")[:160]}…</div>' if cr.source_body else ""
                nli_detail=(f'ENT {cr.entailment_score:.2f} · CON {cr.contradiction_score:.2f}'
                            if hasattr(cr,"entailment_score") else f'{cr.similarity_score:.2f}')
                st.markdown(
                    f'<div class="claim-card"><span class="ent-badge {bcls}">{icon} {cr.verdict.upper()}</span>'
                    f'<div style="flex:1"><div class="claim-text">{cr.claim}</div>'
                    f'<div class="claim-source">{src_html}</div>{body_html}</div>'
                    f'<div class="ent-impact" style="color:{col}">{nli_detail}</div></div>',
                    unsafe_allow_html=True)

    if entity_report:
        st.markdown('<div class="sec-title">Wikipedia Entity Verification</div>', unsafe_allow_html=True)
        if entity_report.error:
            st.warning(f"Entity check unavailable: {entity_report.error}")
        elif entity_report.total_checked == 0:
            st.info("No named entities found.")
        else:
            ec1,ec2,ec3,ec4=st.columns(4)
            ec1.markdown(f'<div class="mcard"><div class="mcard-label">Entity Score</div><div class="mcard-value">{entity_report.entity_score}</div></div>', unsafe_allow_html=True)
            ec2.markdown(f'<div class="mcard"><div class="mcard-label">Verified</div><div class="mcard-value green">{entity_report.verified_count}</div></div>', unsafe_allow_html=True)
            ec3.markdown(f'<div class="mcard"><div class="mcard-label">Not Found</div><div class="mcard-value red">{entity_report.not_found_count}</div></div>', unsafe_allow_html=True)
            ec4.markdown(f'<div class="mcard"><div class="mcard-label">Ambiguous</div><div class="mcard-value amber">{entity_report.ambiguous_count}</div></div>', unsafe_allow_html=True)
            for ent in entity_report.entities:
                if ent.status=="verified":    bcls,btxt="ent-verified","VERIFIED"
                elif ent.status=="not_found": bcls,btxt="ent-notfound","NOT FOUND"
                else:                         bcls,btxt="ent-ambiguous","AMBIGUOUS"
                impact_str=f"+{ent.score_impact:.0f}" if ent.score_impact>=0 else f"{ent.score_impact:.0f}"
                freq_txt=f" ×{ent.frequency}" if ent.frequency>1 else ""
                name_html=(f'<a href="{ent.wiki_url}" target="_blank">{ent.name}</a>'
                           if ent.wiki_url and ent.status=="verified" else ent.name)
                sum_html=(f'<div class="ent-summary">{(ent.summary or "")[:150]}{"…" if len(ent.summary or "")>150 else ""}</div>'
                          if ent.summary else "")
                st.markdown(
                    f'<div class="ent-card"><span class="ent-badge {bcls}">{btxt}</span>'
                    f'<div style="flex:1"><div class="ent-name">{name_html}{freq_txt}</div>'
                    f'<div class="ent-type">{ent.entity_type}</div>{sum_html}</div>'
                    f'<div class="ent-impact" style="color:{"#3dd68c" if ent.score_impact>=0 else "#f06060"}">{impact_str}</div></div>',
                    unsafe_allow_html=True)

    if show_sentences and sent_scores:
        st.markdown('<div class="sec-title">Sentence Suspicion Heatmap</div>', unsafe_allow_html=True)
        for ss in sent_scores:
            level=suspicion_label(ss.score)
            sigs=f" · {', '.join(ss.signals[:2])}" if ss.signals else ""
            st.markdown(
                f'<div class="sent-row sent-{level}">{ss.text}'
                f'<span class="sent-score">{ss.score:.2f}{sigs}</span></div>',
                unsafe_allow_html=True)

    snippet=text[:80].replace("\n"," ")+("…" if len(text)>80 else "")
    verdict_str="Real" if final_score>=70 else ("Fake" if final_score<40 else "Uncertain")

    # ── Dual write: SQLite + Neo4j ────────────────────────────────────────────
    insert_record(modality="Text", snippet=snippet, final_score=final_score, verdict=verdict_str)
    _graph_upsert(
        snippet       = snippet,
        verdict       = verdict_str,
        final_score   = final_score,
        domain        = domain,
        entity_report = entity_report,
        modality      = "Text",
    )

    stylistic_signals=[
        (label, f"{float(style_raw[i]):.3f}" if float(style_raw[i])<1 else f"{float(style_raw[i]):.1f}", stype)
        for i,(label,stype,_) in enumerate(STYLE_LABELS)
    ] if show_style else []

    return {
        "modality":"Text","final_score":final_score,"verdict":verdict_str,
        "snippet":snippet,"word_count":word_count,"reading_time":rt,
        "feature_type":feature_type,"ml_score":ml_score,
        "entity_score":entity_score,"claim_score":claim_score,
        "stylistic_signals":stylistic_signals,"exif_warnings":[],
        "forensic_reasoning":[],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training log renderer
# ──────────────────────────────────────────────────────────────────────────────

def _render_train_log_lines(lines: List[str]) -> None:
    html_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            ev  = rec.get("event", "")
            if ev == "epoch":
                html_lines.append(
                    f'<div class="train-log-line tl-epoch">'
                    f'Epoch {rec["epoch"]}/{rec.get("max_epochs","?")} &nbsp;│&nbsp; '
                    f'loss <strong>{rec["loss"]}</strong> &nbsp;│&nbsp; '
                    f'val_loss <strong>{rec["val_loss"]}</strong> &nbsp;│&nbsp; '
                    f'val_acc <strong>{rec["val_acc"]}%</strong>'
                    f'</div>'
                )
            elif ev == "best":
                html_lines.append(
                    f'<div class="train-log-line tl-best">'
                    f'★ New best model — epoch {rec["epoch"]}, val_acc {rec["val_acc"]}%'
                    f'</div>'
                )
            elif ev == "eval":
                html_lines.append(
                    f'<div class="train-log-line tl-best">'
                    f'✓ Final eval — acc {rec["accuracy"]}% · f1 {rec["f1"]}% · '
                    f'prec {rec["precision"]}% · rec {rec["recall"]}%'
                    f'</div>'
                )
            elif ev == "done":
                html_lines.append(
                    f'<div class="train-log-line tl-done">'
                    f'🏁 {rec.get("message","Done")}'
                    f'</div>'
                )
            elif ev == "error":
                html_lines.append(
                    f'<div class="train-log-line tl-error">'
                    f'✗ {rec.get("message","Error")}'
                    f'</div>'
                )
            elif ev in ("start", "device", "freeze", "params", "dataset", "loading_model"):
                html_lines.append(
                    f'<div class="train-log-line tl-muted">'
                    f'{ev.upper()} &nbsp;{" &nbsp;│&nbsp; ".join(f"{k} {v}" for k,v in rec.items() if k not in ("event","ts"))}'
                    f'</div>'
                )
            elif ev in ("onnx", "onnx_begin", "onnx_start"):
                html_lines.append(
                    f'<div class="train-log-line" style="color:var(--teal)">'
                    f'⬡ ONNX {rec.get("path","quantizing…")}'
                    f'</div>'
                )
        except Exception:
            html_lines.append(
                f'<div class="train-log-line tl-muted">{line}</div>'
            )
    if html_lines:
        st.markdown(
            f'<div class="train-log">{"".join(html_lines)}</div>',
            unsafe_allow_html=True,
        )


def _is_training_running() -> bool:
    proc = st.session_state.get("train_proc")
    if proc is None:
        return False
    return proc.poll() is None


def _training_counts() -> Tuple[int,int,int]:
    # Count images + JSON articles in staging
    staged_n = len([
        p for p in _STAGING_DIR.rglob("*")
        if p.suffix.lower() in {".png",".jpg",".jpeg",".webp",".json"} and p.is_file()
    ])
    real_n   = len(list(_TRAINING_REAL.glob("*.*")))
    fake_n   = len(list(_TRAINING_FAKE.glob("*.*")))
    return staged_n, real_n, fake_n


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sb-brand">Ve<em>rity</em></div>'
        '<div class="sb-tagline">Intelligence Command · v9</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    metrics_data = load_metrics()
    if metrics_data:
        st.markdown('<div class="sb-section">Model Performance</div>', unsafe_allow_html=True)
        for label, key in [("Accuracy","accuracy"),("F1","f1_score"),("ROC-AUC","roc_auc"),("Brier","brier_score")]:
            st.markdown(
                f'<div class="sb-stat"><span class="sb-stat-label">{label}</span>'
                f'<span class="sb-stat-val">{metrics_data.get(key,"—")}</span></div>',
                unsafe_allow_html=True)

    # Neo4j status indicator
    graph = _get_graph()
    neo4j_online = graph is not None
    neo4j_color  = "var(--green)" if neo4j_online else "var(--muted)"
    neo4j_label  = "Connected" if neo4j_online else "Not configured"
    st.markdown(
        f'<div class="sb-stat"><span class="sb-stat-label">🕸️ Neo4j</span>'
        f'<span class="sb-stat-val" style="color:{neo4j_color}">{neo4j_label}</span></div>',
        unsafe_allow_html=True,
    )
    # FIX (v9.1): display Neo4j failure reason here (once) instead of inside
    # _get_graph() where @cache_resource causes it to repeat on every rerun.
    if not neo4j_online and _neo4j_warning:
        st.warning(_neo4j_warning, icon="🕸️")

    if neo4j_online:
        g_stats = graph.get_stats()
        if g_stats.get("articles"):
            st.markdown(
                f'<div class="sb-stat"><span class="sb-stat-label">Graph articles</span>'
                f'<span class="sb-stat-val">{g_stats["articles"]}</span></div>'
                f'<div class="sb-stat"><span class="sb-stat-label">Entities</span>'
                f'<span class="sb-stat-val">{g_stats.get("entities",0)}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Analysis Modules</div>', unsafe_allow_html=True)
    num_features   = st.slider("LIME features", 5, 20, 10)
    show_style     = st.toggle("Stylistic breakdown", value=True)
    show_radar     = st.toggle("Radar chart",         value=True)
    show_entity    = st.toggle("Entity verification", value=True)
    show_claims    = st.toggle("Claim verification",  value=False)
    show_sentences = st.toggle("Sentence heatmap",    value=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    staged_n, real_n, fake_n = _training_counts()
    if staged_n:
        st.markdown(
            f'<div class="staging-badge">⏳ {staged_n} item(s) awaiting review</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sb-section">Training Data</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sb-stat"><span class="sb-stat-label">Pending</span><span class="sb-stat-val" style="color:var(--amber)">{staged_n}</span></div>'
        f'<div class="sb-stat"><span class="sb-stat-label">Real labels</span><span class="sb-stat-val" style="color:var(--green)">{real_n}</span></div>'
        f'<div class="sb-stat"><span class="sb-stat-label">Fake labels</span><span class="sb-stat-val" style="color:var(--red)">{fake_n}</span></div>',
        unsafe_allow_html=True,
    )
    if _is_training_running():
        st.markdown(
            '<div class="staging-badge" style="color:var(--teal);border-color:var(--teal)">🔄 Intelligence Sync running…</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    recent = get_recent_records(limit=8)
    if recent:
        stats = get_stats()
        st.markdown('<div class="sb-section">Recent Analyses</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;gap:10px;margin-bottom:8px">'
            f'<div class="sb-stat" style="border:none;padding:0;flex:1"><span class="sb-stat-label">Total</span> <span class="sb-stat-val">{stats["total"]}</span></div>'
            f'<div class="sb-stat" style="border:none;padding:0;flex:1"><span class="sb-stat-label">Avg</span> <span class="sb-stat-val">{stats["avg_score"]}%</span></div>'
            f'</div>', unsafe_allow_html=True)
        for item in recent:
            color = "#3dd68c" if item["verdict"]=="Real" else ("#f06060" if item["verdict"]=="Fake" else "#f5a623")
            badge = "🖼" if item["modality"]=="Image" else "✍"
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-score" style="color:{color}">{item["final_score"]:.0f}%</span>'
                f'<span class="hist-text">{badge} {item["snippet"][:44]}</span>'
                f'</div>', unsafe_allow_html=True)
        if st.button("Clear history", key="clear_hist"):
            clear_history(); st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="note">⚠ Does not replace human fact-checking.</p>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero-wrap">'
    '<h1 class="hero-title"><span class="hero-dot"></span>Verity <em>Intelligence</em> Command</h1>'
    '<span class="hero-tag">DistilBERT · SigLIP-2 · LIME · NLI · Deepfake · Neo4j · MLOps</span>'
    '</div>',
    unsafe_allow_html=True,
)

model, scaler, vectorizer, bert_tokenizer, bert_model_obj, feature_type = load_model()
nlp = load_spacy()

common_kwargs = dict(
    model=model, scaler=scaler, vectorizer=vectorizer,
    bert_tokenizer=bert_tokenizer, bert_model_obj=bert_model_obj,
    feature_type=feature_type, nlp=nlp,
    num_features=num_features, show_style=show_style,
    show_entity=show_entity, show_radar=show_radar,
    show_claims=show_claims, show_sentences=show_sentences,
)


# ──────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL TABS  (v9: adds Intelligence Graph)
# ──────────────────────────────────────────────────────────────────────────────

tab_scanner, tab_review, tab_graph, tab_health = st.tabs([
    "🔬  Live Scanner",
    "🗂️  Analyst Review",
    "🕸️  Intelligence Graph",
    "⚕️  Model Health",
])


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SCANNER  (unchanged from v8)
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_scanner:
    scan_text, scan_url, scan_batch, scan_image = st.tabs([
        "✍ Paste Text", "🌐 From URL", "📋 Batch", "🖼️ Image Forensics"
    ])

    with scan_text:
        news_text    = st.text_area("", height=180, placeholder="Paste a news article…", label_visibility="collapsed")
        analyse_text = st.button("Analyse Article", key="btn_text")
        if analyse_text:
            if news_text.strip():
                report_data = analyse_article(news_text.strip(), **common_kwargs)
                if report_data:
                    try:
                        pdf_bytes = generate_pdf_report(report_data)
                        st.download_button("⬇ Download PDF", data=pdf_bytes,
                                           file_name="verity_text_report.pdf",
                                           mime="application/pdf", key="dl_text")
                    except Exception: pass
            else:
                st.warning("Paste article text first.")

    with scan_url:
        url_input   = st.text_input("", placeholder="https://example.com/article", label_visibility="collapsed")
        analyse_url = st.button("Fetch & Analyse", key="btn_url")
        if analyse_url:
            if url_input.strip():
                with st.spinner("Fetching…"):
                    scraped = scrape_url(url_input.strip())
                if scraped:
                    with st.expander("Scraped text preview"):
                        st.text(scraped[:600]+"…")
                    report_data = analyse_article(scraped, source_url=url_input.strip(), **common_kwargs)
                    if report_data:
                        try:
                            pdf_bytes = generate_pdf_report(report_data)
                            st.download_button("⬇ Download PDF", data=pdf_bytes,
                                               file_name="verity_url_report.pdf",
                                               mime="application/pdf", key="dl_url")
                        except Exception: pass
                else:
                    st.error("Could not extract text.")
            else:
                st.warning("Enter a URL first.")

    with scan_batch:
        st.markdown('<p class="note">Separate articles with a line containing only <code>---</code></p>', unsafe_allow_html=True)
        batch_text    = st.text_area("", height=240, placeholder="Article one…\n---\nArticle two…", label_visibility="collapsed")
        analyse_batch = st.button("Analyse All", key="btn_batch")
        if analyse_batch:
            if batch_text.strip():
                articles = [a.strip() for a in batch_text.split("\n---\n") if a.strip()]
                if articles:
                    scores = []
                    for idx, article in enumerate(articles, 1):
                        with st.expander(f"Article {idx} — {article[:60].replace(chr(10),' ')}…", expanded=(idx==1)):
                            r = analyse_article(article, **common_kwargs)
                            if r: scores.append(r["final_score"])
                    if scores:
                        st.markdown("#### Batch Summary")
                        bc1,bc2,bc3=st.columns(3)
                        bc1.metric("Average",f"{np.mean(scores):.1f}%")
                        bc2.metric("Highest",f"{max(scores):.1f}%")
                        bc3.metric("Lowest", f"{min(scores):.1f}%")
            else:
                st.warning("Paste at least one article.")

    with scan_image:
        st.markdown(
            '<p class="note">Upload a photo to run AI detection, ELA forensic overlay, '
            'deepfake analysis, biological consistency check, and EXIF provenance. '
            'Uncertain images are automatically staged for analyst review.</p>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "", type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
            help="JPG · PNG · WebP · max ~20 MB",
        )
        analyse_image = st.button("Analyse Image", key="btn_image")

        if uploaded_file is not None and analyse_image:
            from PIL import Image as PILImage
            try:
                pil_img = PILImage.open(uploaded_file).convert("RGB")
            except Exception as exc:
                st.error(f"Could not open image: {exc}"); st.stop()

            _ai_result        = None
            _deepfake_results = None
            _exif_result      = None
            _bio_results      = None
            _ela_img          = None
            _peak_deepfake_prob = 0.0

            # ─── NEW ───────────────────────────────────────────────────────────────────
            st.markdown('<div class="sec-title">Image Overview</div>', unsafe_allow_html=True)

            # ── Cache ELA heatmap by image hash so the slider never triggers
            # ── a recompute — only a new "Analyse Image" click does.
            _img_bytes_for_hash = pil_to_bytes(pil_img, fmt="PNG")
            _img_hash = hashlib.md5(_img_bytes_for_hash).hexdigest()

            if st.session_state["_ela_source_hash"] != _img_hash:
                # New image — recompute and cache as PNG bytes
                try:
                    _ela_pil = generate_ela_heatmap(pil_img)
                    st.session_state["_ela_img_bytes"]   = pil_to_bytes(_ela_pil, fmt="PNG")
                    st.session_state["_ela_source_hash"] = _img_hash
                except Exception:
                    st.session_state["_ela_img_bytes"]   = None
                    st.session_state["_ela_source_hash"] = _img_hash

            # Reconstruct PIL from cached bytes (avoids PIL not pickling across reruns)
            _ela_img = None
            if st.session_state["_ela_img_bytes"]:
                try:
                    from PIL import Image as PILImage  # already imported above
                    import io as _io
                    _ela_img = PILImage.open(_io.BytesIO(st.session_state["_ela_img_bytes"])).convert("RGB")
                except Exception:
                    _ela_img = None

            # Slider is outside the heavy analysis block — changing it only
            # re-runs this lightweight display section.
            alpha_pct = st.slider(
                "🔬 Forensic Overlay Opacity",
                min_value=0, max_value=100, value=40, step=5,
                key="ela_alpha",
            )
            alpha = alpha_pct / 100.0

            col_orig, col_overlay = st.columns(2)
            with col_orig:
                st.markdown(
                    '<div class="img-panel">'
                    '<div class="img-panel-title">Original</div>',
                    unsafe_allow_html=True,
                )
                st.image(pil_img, use_container_width=True)
                w, h = pil_img.size
                st.markdown(
                    f'<div style="font-size:10px;color:var(--muted);margin-top:4px;'
                    f'font-family:JetBrains Mono,monospace">{w}×{h}px · {pil_img.mode}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col_overlay:
                st.markdown(
                    '<div class="img-panel img-panel--overlay">'
                    '<div class="img-panel-title">Forensic Overlay (ELA blend)</div>',
                    unsafe_allow_html=True,
                )
                if _ela_img is not None:
                    try:
                        blended = blend_ela_overlay(pil_img, _ela_img, alpha=alpha)
                        st.image(blended, use_container_width=True)
                    except Exception as blend_err:
                        st.image(_ela_img, use_container_width=True)
                        st.caption(f"Blend error: {blend_err} — showing raw ELA")
                    st.markdown(
                        '<div class="ela-legend">'
                        '<span class="ela-low">Low error</span>'
                        '<span class="ela-mid">Medium</span>'
                        '<span class="ela-high">High / edited</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("ELA unavailable. Install opencv-python-headless.")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec-title">AI Generation Detection  ·  SigLIP-2</div>', unsafe_allow_html=True)

            @st.cache_resource(show_spinner="Loading AI image detector (~330 MB)…")
            def _cached_img_model():
                return load_image_model()

            with st.spinner("Running SigLIP-2 AI detector…"):
                try:
                    img_model  = _cached_img_model()
                    _ai_result = detect_ai_image(pil_img, img_model)
                    ai_pct     = _ai_result["ai_probability"]
                    real_pct   = _ai_result["real_probability"]
                    ai_label   = _ai_result["label"]

                    ai_colour   = "#f06060" if ai_pct >= 85 else ("#f5a623" if ai_pct >= 75 else "#3dd68c")
                    real_colour = "#3dd68c" if real_pct >= 25 else ("#f5a623" if real_pct >= 15 else "#f06060")

                    st.markdown(
                        f'<div class="forensics-score-grid">'
                        f'<div class="mcard"><div class="mcard-label">AI-Generated Prob.</div><div class="mcard-value" style="color:{ai_colour}">{ai_pct:.1f}%</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Real Photo Prob.</div><div class="mcard-value" style="color:{real_colour}">{real_pct:.1f}%</div></div>'
                        f'<div class="mcard"><div class="mcard-label">Verdict</div><div class="mcard-value small" style="color:{ai_colour}">{ai_label}</div></div>'
                        f'</div>', unsafe_allow_html=True)

                    if ai_pct >= 85:
                        st.markdown(f'<div class="domain-warn">⚠ <strong>High AI probability ({ai_pct:.1f}%).</strong> This image is likely AI-generated.</div>', unsafe_allow_html=True)
                    elif ai_pct >= 75:
                        st.markdown(
                            f'<div class="domain-warn" style="border-left-color:#f5a623;color:#f5a623;background:#1a170a">'
                            f'ℹ <strong>Moderate probability ({ai_pct:.1f}%).</strong> May be heavily edited or contain AI elements.</div>',
                            unsafe_allow_html=True)

                    img_verdict = "Fake" if ai_pct >= 85 else ("Uncertain" if ai_pct >= 75 else "Real")
                    insert_record(
                        modality="Image",
                        snippet=getattr(uploaded_file, "name", "uploaded_image"),
                        final_score=round(100-ai_pct, 2),
                        verdict=img_verdict,
                    )
                    # Write image scan to Neo4j
                    _graph_upsert(
                        snippet     = getattr(uploaded_file, "name", "uploaded_image"),
                        verdict     = img_verdict,
                        final_score = round(100-ai_pct, 2),
                        modality    = "Image",
                    )
                    st.session_state["_last_img_report"] = {
                        "modality":       "Image",
                        "final_score":    round(100-ai_pct, 2),
                        "verdict":        img_verdict,
                        "snippet":        getattr(uploaded_file, "name", "uploaded_image"),
                        "image_filename": getattr(uploaded_file, "name", "uploaded_image"),
                        "ai_probability": ai_pct,
                        "forensic_reasoning": [],
                    }
                except Exception as ai_err:
                    st.warning(f"AI detector unavailable: {ai_err}")

            st.markdown('<div class="sec-title">EXIF Metadata</div>', unsafe_allow_html=True)
            try:
                uploaded_file.seek(0)
                exif_pil     = PILImage.open(uploaded_file)
                _exif_result = extract_exif(exif_pil)
                if _exif_result["warning"]:
                    st.markdown(f'<div class="domain-warn">⚠ <strong>Editing software detected:</strong> {_exif_result["warning"]}</div>', unsafe_allow_html=True)
                if _exif_result["gps_present"]:
                    st.markdown('<div class="domain-warn">📍 <strong>GPS data embedded.</strong> This image contains location metadata.</div>', unsafe_allow_html=True)
                if _exif_result["has_exif"] and _exif_result["data"]:
                    items = list(_exif_result["data"].items()); half = (len(items)+1)//2
                    ec1, ec2 = st.columns(2)
                    def _exif_col(col, rows):
                        col.markdown('<div class="exif-grid">', unsafe_allow_html=True)
                        for k, v in rows:
                            col.markdown(f'<div class="exif-row"><span class="exif-key">{k}</span><span class="exif-val">{v}</span></div>', unsafe_allow_html=True)
                        col.markdown('</div>', unsafe_allow_html=True)
                    _exif_col(ec1, items[:half]); _exif_col(ec2, items[half:])
                    if "_last_img_report" in st.session_state:
                        st.session_state["_last_img_report"]["exif_data"]     = _exif_result["data"]
                        st.session_state["_last_img_report"]["exif_warnings"] = [_exif_result["warning"]] if _exif_result["warning"] else []
                else:
                    st.markdown('<p style="font-size:12px;color:var(--muted);font-style:italic;margin-top:8px">No EXIF metadata found.</p>', unsafe_allow_html=True)
            except Exception as exif_err:
                st.warning(f"EXIF extraction failed: {exif_err}")

            st.markdown('<div class="sec-title">Face Detection &amp; Deepfake Analysis</div>', unsafe_allow_html=True)
            try:
                with st.spinner("Detecting faces…"):
                    faces = extract_faces(pil_img)
                if not faces:
                    st.markdown('<p class="note">No faces detected.</p>', unsafe_allow_html=True)
                    _deepfake_results = []; _bio_results = []
                else:
                    st.markdown(
                        f'<div class="mcard" style="display:inline-block;margin-bottom:14px;">'
                        f'<div class="mcard-label">Faces found</div>'
                        f'<div class="mcard-value">{len(faces)}</div></div>',
                        unsafe_allow_html=True)

                    with st.spinner("Running biological consistency check…"):
                        try:
                            _bio_results = biological_consistency_check(faces)
                        except Exception as bio_err:
                            st.warning(f"Bio check failed: {bio_err}")
                            _bio_results = []

                    if _bio_results:
                        flagged_zones = [z for r in _bio_results for z in r.suspicious_zones]
                        if flagged_zones:
                            zone_str = ", ".join(dict.fromkeys(flagged_zones))
                            st.markdown(f'<div class="domain-warn">⚠ <strong>Biological inconsistencies: {zone_str}.</strong></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="color:#3dd68c;font-size:12px;margin-bottom:8px">✓ Biological consistency check passed</div>', unsafe_allow_html=True)

                    @st.cache_resource(show_spinner="Loading deepfake classifier (~330 MB)…")
                    def _cached_df_model():
                        return load_deepfake_model()

                    _deepfake_results = []
                    try:
                        with st.spinner(f"Running deepfake classifier on {len(faces)} face(s)…"):
                            df_model          = _cached_df_model()
                            _deepfake_results = detect_deepfake(faces, df_model)
                    except Exception as df_err:
                        st.warning(f"Deepfake classifier unavailable: {df_err}")

                    flagged = [r for r in _deepfake_results if r["deepfake_probability"] >= 80]
                    if flagged:
                        worst = max(flagged, key=lambda r: r["deepfake_probability"])
                        st.markdown(
                            f'<div class="domain-warn">⚠ <strong>{len(flagged)} face(s) flagged as likely deepfake.</strong> Peak: {worst["deepfake_probability"]:.1f}%.</div>',
                            unsafe_allow_html=True)

                    n_cols = min(len(faces[:8]), 4)
                    face_cols = st.columns(n_cols)
                    for i, face_img in enumerate(faces[:8]):
                        col = face_cols[i % n_cols]
                        if i < len(_deepfake_results):
                            dr = _deepfake_results[i]
                            df_pct = dr["deepfake_probability"]; verdict = dr["label"]; conf = dr["confidence"]
                        else:
                            df_pct, verdict, conf = None, "Unknown", "N/A"
                        if df_pct is not None:
                            _peak_deepfake_prob = max(_peak_deepfake_prob, df_pct)
                        if df_pct is None:   card_cls="unsure"; prob_colour="var(--text-dim)"; verdict_cls="unsure"
                        elif df_pct >= 80:   card_cls="fake";   prob_colour="var(--red)";      verdict_cls="fake"
                        elif df_pct >= 60:   card_cls="unsure"; prob_colour="var(--amber)";    verdict_cls="unsure"
                        else:                card_cls="real";   prob_colour="var(--green)";    verdict_cls="real"
                        col.image(face_img, use_container_width=True)
                        col.markdown(
                            f'<div class="face-card {card_cls}">'
                            f'<div class="face-card-idx">Face {i+1}</div>'
                            f'<div class="face-prob" style="color:{prob_colour}">{f"{df_pct:.1f}%" if df_pct is not None else "—"}</div>'
                            f'<div style="font-size:9px;color:var(--muted)">deepfake prob.</div>'
                            f'<div><span class="face-verdict {verdict_cls}">{verdict}</span></div>'
                            f'<div class="face-conf">Confidence: {conf}</div></div>',
                            unsafe_allow_html=True)

                    if "_last_img_report" in st.session_state:
                        st.session_state["_last_img_report"]["face_count"]       = len(faces)
                        st.session_state["_last_img_report"]["deepfake_results"] = _deepfake_results

            except Exception as face_err:
                st.warning(f"Face detection failed: {face_err}")
                _deepfake_results = []; _bio_results = []

            if _ai_result is not None:
                st.markdown('<div class="sec-title">Consensus Meter</div>', unsafe_allow_html=True)
                cm = consensus_meter(_ai_result, _bio_results)
                bg_map = {
                    "Consensus":                  ("rgba(61,214,140,0.07)",  "1px solid #1a4a30"),
                    "Minor Discordance":           ("rgba(245,166,35,0.07)",  "1px solid #3a2e10"),
                    "High-Complexity Edge Case":   ("rgba(245,166,35,0.10)",  "1px solid #4a3010"),
                }
                bg_style, border_style = bg_map.get(cm["label"], ("rgba(79,143,255,0.06)", "1px solid #1e2840"))
                st.markdown(
                    f'<div class="consensus-box" style="background:{bg_style};border:{border_style};">'
                    f'<div class="consensus-icon">{cm["icon"]}</div>'
                    f'<div>'
                    f'<div class="consensus-label" style="color:{cm["color"]}">{cm["label"]}</div>'
                    f'<div class="consensus-detail" style="color:{cm["color"]}">{cm["detail"]}</div>'
                    f'<div style="margin-top:8px;font-family:JetBrains Mono,monospace;font-size:10px;color:var(--muted)">'
                    f'SigLIP-2: {cm["ai_score"]:.1f}% · Bio flags: {cm["bio_flags"]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            _current_ai_pct = _ai_result["ai_probability"] if _ai_result else 0.0
            _staged_path = flag_for_review(pil_img, _current_ai_pct, _peak_deepfake_prob)
            if _staged_path:
                st.markdown(
                    f'<div class="domain-warn" style="border-left-color:var(--accent);color:var(--accent2);background:#08101a">'
                    f'🔬 <strong>Staged for analyst review</strong> — AI {_current_ai_pct:.1f}% / '
                    f'Deepfake {_peak_deepfake_prob:.1f}% — uncertainty zone. '
                    f'Saved: <code>{Path(_staged_path).name}</code></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="sec-title">Forensic Reasoning Log</div>', unsafe_allow_html=True)
            try:
                reasoning_bullets = generate_forensic_reasoning(
                    ai_result=_ai_result, deepfake_results=_deepfake_results,
                    exif_data=_exif_result, bio_results=_bio_results,
                )
                render_reasoning_log(reasoning_bullets)
                if "_last_img_report" in st.session_state:
                    st.session_state["_last_img_report"]["forensic_reasoning"] = reasoning_bullets
            except Exception as reason_err:
                st.warning(f"Reasoning log unavailable: {reason_err}")

            if "_last_img_report" in st.session_state:
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                try:
                    pdf_bytes = generate_pdf_report(st.session_state["_last_img_report"])
                    fname = (st.session_state["_last_img_report"].get("image_filename") or "image").rsplit(".", 1)[0]
                    st.download_button("⬇ Download PDF Report", data=pdf_bytes,
                                       file_name=f"verity_{fname}_report.pdf",
                                       mime="application/pdf", key="dl_image")
                except Exception as pdf_err:
                    st.caption(f"PDF generation unavailable: {pdf_err}")

        elif uploaded_file is not None:
            from PIL import Image as PILImage
            try:
                st.image(PILImage.open(uploaded_file), caption="Preview — click Analyse Image to run forensics", width=380)
            except Exception:
                pass


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYST REVIEW  (v9: also shows threat_hunter JSON articles)
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_review:
    st.markdown(
        '<div class="review-header">🗂️ Analyst Review Queue</div>'
        '<div class="review-sub">Human-in-the-Loop · Active Learning Pipeline · Threat Hunter</div>',
        unsafe_allow_html=True,
    )

    # Collect both image files and threat-hunter JSON articles
    staged_images = sorted([
        p for p in _STAGING_DIR.iterdir()
        if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"}
    ])
    staged_articles = sorted([
        p for p in _STAGING_DIR.rglob("*.json")
    ])

    sync_col, stat_col = st.columns([2, 1])
    with sync_col:
        real_n = len(list(_TRAINING_REAL.glob("*.*")))
        fake_n = len(list(_TRAINING_FAKE.glob("*.*")))
        total_labelled = real_n + fake_n
        st.markdown(
            f'<p class="note" style="margin-bottom:6px">'
            f'Labelled training images: <strong style="color:var(--accent2)">{total_labelled}</strong> '
            f'({real_n} real · {fake_n} fake). '
            f'Minimum 10 required to train.</p>',
            unsafe_allow_html=True,
        )

    already_running = _is_training_running()

    st.markdown('<div class="sync-btn">', unsafe_allow_html=True)
    sync_clicked = st.button(
        "🚀  Execute Intelligence Sync",
        key="sync_btn",
        disabled=already_running or total_labelled < 10,
        help="Fine-tunes SigLIP-2 on your labelled images, then re-quantizes to ONNX INT8.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if sync_clicked and not already_running:
        try:
            _LOG_FILE.write_text("", encoding="utf-8")
            proc = subprocess.Popen(
                [os.sys.executable, "auto_train.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            st.session_state["train_proc"] = proc
            st.rerun()
        except Exception as proc_err:
            st.error(f"Could not launch training: {proc_err}")

    if total_labelled < 10:
        st.markdown(
            f'<p class="note" style="color:var(--amber)">Label at least {10 - total_labelled} more image(s) before syncing.</p>',
            unsafe_allow_html=True,
        )

    # ─── NEW ───────────────────────────────────────────────────────────────────
    _log_has_content = False
    try:
        _log_has_content = _LOG_FILE.exists() and _LOG_FILE.stat().st_size > 0
    except OSError:
        pass  # stat() can fail on NFS/network drives — treat as empty

    if already_running or _log_has_content:
        _status_label = (
            "🔄 Intelligence Sync in progress…" if already_running
            else "✅ Last training run log"
        )
        with st.status(_status_label, expanded=True):
            try:
                if _LOG_FILE.exists():
                    log_lines = _LOG_FILE.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    _render_train_log_lines(log_lines)
            except (PermissionError, OSError) as _log_err:
                st.caption(
                    f"⚠ Log temporarily unavailable ({type(_log_err).__name__}). "
                    "Refresh to retry."
                )
            if already_running:
                st.markdown(
                    '<p class="note">Refresh for updated progress.</p>',
                    unsafe_allow_html=True,
                )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Threat Hunter articles ────────────────────────────────────────────────
    if staged_articles:
        st.markdown('<div class="sec-title">Threat Hunter Articles</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="note">{len(staged_articles)} article(s) discovered by the Threat Hunter and staged for review.</p>',
            unsafe_allow_html=True,
        )
        for jpath in staged_articles[:10]:   # cap display at 10
            try:
                rec = json.loads(jpath.read_text(encoding="utf-8"))
            except Exception:
                continue
            score   = rec.get("score", 0)
            verdict = rec.get("verdict", "?")
            title   = rec.get("title", jpath.name)[:80]
            url     = rec.get("url", "")
            query   = rec.get("query", "")
            col_map = {"Fake":"#f06060","Uncertain":"#f5a623","Real":"#3dd68c"}
            col     = col_map.get(verdict, "#94a3b8")
            st.markdown(
                f'<div class="review-card">'
                f'<div class="review-filename">🕵️ Threat Hunter · {jpath.parent.name}/{jpath.name}</div>'
                f'<div class="review-scores">'
                f'<span class="review-score-pill" style="color:{col}">{verdict} · {score:.1f}%</span>'
                f'<span class="review-score-pill">Query: {query[:40]}</span>'
                f'</div>'
                f'<div style="font-size:13px;color:var(--text);margin-bottom:8px">{title}</div>'
                f'{"<a href=" + repr(url) + " target=_blank style=font-size:11px;color:var(--accent)>" + url[:70] + "…</a>" if url else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
            btn_key = hashlib.md5(str(jpath).encode()).hexdigest()[:8]
            dc1, dc2, dc3 = st.columns([2, 2, 1])
            with dc1:
                if st.button("✅ Confirm Real", key=f"art_real_{btn_key}"):
                    jpath.unlink(missing_ok=True); st.rerun()
            with dc2:
                if st.button("❌ Confirm Fake", key=f"art_fake_{btn_key}"):
                    jpath.unlink(missing_ok=True); st.rerun()
            with dc3:
                if st.button("🗑 Skip", key=f"art_skip_{btn_key}"):
                    st.rerun()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Image queue (unchanged) ───────────────────────────────────────────────
    if not staged_images:
        st.markdown(
            '<div class="review-empty">'
            '✅ No images pending review.<br>'
            '<span style="font-size:11px;color:var(--muted)">Uncertain image scans are staged here automatically.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        idx = min(st.session_state["review_idx"], len(staged_images) - 1)
        st.session_state["review_idx"] = idx
        current_file = staged_images[idx]

        ctr_col, nav_col = st.columns([3, 1])
        with ctr_col:
            st.markdown(f'<div class="review-counter">Image {idx+1} of {len(staged_images)}</div>', unsafe_allow_html=True)
        with nav_col:
            nc1, nc2 = st.columns(2)
            if nc1.button("← Prev", key="rev_prev", disabled=(idx==0)):
                st.session_state["review_idx"] = max(0, idx-1); st.rerun()
            if nc2.button("Next →", key="rev_next", disabled=(idx>=len(staged_images)-1)):
                st.session_state["review_idx"] = min(len(staged_images)-1, idx+1); st.rerun()

        fname = current_file.name
        ai_d = df_d = "?"
        try:
            parts = fname.rsplit(".", 1)[0].split("_")
            if len(parts) >= 5 and parts[0] == "flagged":
                ai_d = parts[2]; df_d = parts[4]
        except Exception:
            pass

        st.markdown('<div class="review-card">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="review-filename">📁 {fname}</div>'
            f'<div class="review-scores">'
            f'<span class="review-score-pill">AI score: {ai_d}%</span>'
            f'<span class="review-score-pill">Deepfake score: {df_d}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        from PIL import Image as PILImage
        try:
            img_preview = PILImage.open(str(current_file))
            img_col, _ = st.columns([2, 1])
            with img_col:
                st.image(img_preview, use_container_width=True)
        except Exception as e:
            st.warning(f"Cannot display: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="note" style="margin-bottom:8px">Assign a ground-truth label — the image moves to the appropriate training folder.</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="note" style="margin-bottom:8px">'
            'Assign a ground-truth label — the image moves to the appropriate training folder.</p>',
            unsafe_allow_html=True,
        )

        def _safe_move_to_label(src: Path, dest_dir: Path, label: str) -> Tuple[bool, str]:
            """
            Atomically move src → dest_dir/src.name.
            Returns (success, message).
            Guards against: missing source, locked file, duplicate destination.
            """
            dest = dest_dir / src.name
            try:
                if not src.exists():
                    return False, f"Source file no longer exists: {src.name}"
                if dest.exists():
                    # Deduplicate by appending timestamp suffix
                    stem, suf = src.stem, src.suffix
                    dest = dest_dir / f"{stem}_{int(time.time())}{suf}"
                shutil.move(str(src), str(dest))
                return True, f"Moved to {label}: {dest.name}"
            except PermissionError:
                return False, f"File is locked — try again in a moment."
            except Exception as exc:
                return False, f"Move failed: {exc}"

        b1, b2, b3 = st.columns([2, 2, 1])
        with b1:
            if st.button("✅  Confirm Real", key="confirm_real"):
                ok, msg = _safe_move_to_label(current_file, _TRAINING_REAL, "Real")
                if ok:
                    st.session_state["_review_toast"] = ("success", msg)
                    if idx >= len(staged_images) - 1:
                        st.session_state["review_idx"] = max(0, idx - 1)
                else:
                    st.session_state["_review_toast"] = ("error", msg)
                st.rerun()
        with b2:
            if st.button("❌  Confirm Fake", key="confirm_fake"):
                ok, msg = _safe_move_to_label(current_file, _TRAINING_FAKE, "Fake")
                if ok:
                    st.session_state["_review_toast"] = ("success", msg)
                    if idx >= len(staged_images) - 1:
                        st.session_state["review_idx"] = max(0, idx - 1)
                else:
                    st.session_state["_review_toast"] = ("error", msg)
                st.rerun()
        with b3:
            if st.button("🗑 Skip", key="skip_img"):
                if idx < len(staged_images) - 1:
                    st.session_state["review_idx"] = idx + 1
                st.rerun()

        # ── Render and immediately clear toast (shown once after rerun) ──────
        _toast = st.session_state.get("_review_toast")
        if _toast:
            toast_kind, toast_msg = _toast
            st.session_state["_review_toast"] = None  # consume immediately
            if toast_kind == "success":
                st.success(f"✅ {toast_msg}", icon="✅")
            else:
                st.error(f"⚠️ {toast_msg}")

    s_n, r_n, f_n = _training_counts()
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="mcard"><div class="mcard-label">Pending Review</div><div class="mcard-value amber">{s_n}</div><div class="mcard-sub">staging_data/</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mcard"><div class="mcard-label">Labelled Real</div><div class="mcard-value green">{r_n}</div><div class="mcard-sub">training_data/real/</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="mcard"><div class="mcard-label">Labelled Fake</div><div class="mcard-value red">{f_n}</div><div class="mcard-sub">training_data/fake/</div></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INTELLIGENCE GRAPH  (new in v9)
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.markdown("### 🕸️ Intelligence Graph")
    st.markdown(
        '<p class="note">Relational map of articles, entities, authors, and domains. '
        'Requires Neo4j credentials in <code>.streamlit/secrets.toml</code>.</p>',
        unsafe_allow_html=True,
    )

    graph = _get_graph()

    if graph is None:
        st.info(
            "Neo4j is not configured. Add credentials to `.streamlit/secrets.toml`:\n\n"
            "```toml\n[neo4j]\nNEO4J_URI = \"bolt://localhost:7687\"\n"
            "NEO4J_USER = \"neo4j\"\nNEO4J_PASSWORD = \"your-password\"\n```",
            icon="🕸️",
        )
    else:
        # ── Refresh controls ──────────────────────────────────────────────────
        ref_col, _empty = st.columns([1, 3])
        with ref_col:
            refresh = st.button("🔄 Refresh Graph", key="refresh_graph")

        if refresh or "graph_data" not in st.session_state:
            with st.spinner("Querying Neo4j…"):
                try:
                    st.session_state["graph_data"]    = graph.get_graph_for_viz()
                    st.session_state["clusters"]      = graph.get_narrative_clusters()
                    st.session_state["domain_risk"]   = graph.get_domain_risk_profile()
                    st.session_state["graph_stats"]   = graph.get_stats()
                except Exception as gex:
                    st.error(f"Graph query failed: {gex}")
                    st.session_state.setdefault("graph_data", {"nodes":[],"edges":[]})
                    st.session_state.setdefault("clusters", [])
                    st.session_state.setdefault("domain_risk", [])
                    st.session_state.setdefault("graph_stats", {})

        gdata  = st.session_state.get("graph_data",  {"nodes":[],"edges":[]})
        g_stats = st.session_state.get("graph_stats", {})

        # ── Summary metrics ───────────────────────────────────────────────────
        if g_stats:
            st.markdown(
                f'<div class="metric-grid" style="grid-template-columns:repeat(4,1fr)">'
                f'<div class="mcard"><div class="mcard-label">Articles</div><div class="mcard-value">{g_stats.get("articles",0)}</div></div>'
                f'<div class="mcard"><div class="mcard-label">Entities</div><div class="mcard-value">{g_stats.get("entities",0)}</div></div>'
                f'<div class="mcard"><div class="mcard-label">Domains</div><div class="mcard-value">{g_stats.get("domains",0)}</div></div>'
                f'<div class="mcard"><div class="mcard-label">Authors</div><div class="mcard-value">{g_stats.get("authors",0)}</div></div>'
                f'</div>', unsafe_allow_html=True,
            )

        # ── Interactive Graph ─────────────────────────────────────────────────
        nodes = gdata.get("nodes", [])
        edges = gdata.get("edges", [])

        if not nodes:
            st.info("No graph data yet. Run some article scans to populate the graph.")
        else:
            try:
                from streamlit_agraph import agraph, Node, Edge, Config

                agraph_nodes = [
                    Node(
                        id    = n["id"],
                        label = n["label"][:28],
                        size  = n.get("size", 14),
                        color = n.get("color", "#94a3b8"),
                        title = n.get("title", ""),
                    )
                    for n in nodes
                ]
                agraph_edges = [
                    Edge(
                        source = e["source"],
                        target = e["target"],
                        label  = e.get("label", ""),
                    )
                    for e in edges
                ]
                config = Config(
                    width      = "100%",
                    height     = 560,
                    directed   = True,
                    physics    = True,
                    hierarchical = False,
                    nodeHighlightBehavior = True,
                    highlightColor = "#4f8fff",
                    collapsible = True,
                    node       = {"labelProperty": "label"},
                    link       = {"labelProperty": "label", "renderLabel": True},
                )
                st.markdown('<div class="graph-panel">', unsafe_allow_html=True)
                agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
                st.markdown('</div>', unsafe_allow_html=True)

                # Legend
                st.markdown(
                    '<div style="display:flex;gap:20px;font-size:11px;color:var(--muted);margin-top:6px;font-family:JetBrains Mono,monospace">'
                    '<span>🔴 Fake article</span>'
                    '<span>🟡 Uncertain article</span>'
                    '<span>🟢 Real article</span>'
                    '<span style="color:#6366f1">⬡ Entity</span>'
                    '<span style="color:#0ea5e9">◉ Domain</span>'
                    '<span style="color:#ec4899">♦ Author</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            except ImportError:
                # Fallback: plain table when streamlit-agraph isn't installed
                st.warning(
                    "Install `streamlit-agraph` for the interactive graph:  "
                    "`pip install streamlit-agraph`",
                    icon="⚠️",
                )
                import pandas as pd
                node_df = pd.DataFrame([
                    {"id": n["id"], "label": n["label"], "group": n["group"], "color": n.get("color","")}
                    for n in nodes[:50]
                ])
                edge_df = pd.DataFrame(edges[:100])
                st.markdown("**Nodes (first 50)**")
                st.dataframe(node_df, use_container_width=True)
                st.markdown("**Edges (first 100)**")
                st.dataframe(edge_df, use_container_width=True)

        # ── Narrative Clusters ─────────────────────────────────────────────────
        st.markdown('<div class="sec-title">Narrative Clusters — Coordinated Disinformation Signal</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="note">Entities appearing in ≥2 Fake/Uncertain articles across ≥2 different domains. '
            'High cluster density is a strong indicator of coordinated disinformation.</p>',
            unsafe_allow_html=True,
        )

        clusters = st.session_state.get("clusters", [])
        if not clusters:
            st.info("No narrative clusters detected yet. More scans needed.")
        else:
            for c in clusters[:20]:
                domains_str = ", ".join(c.get("domains", [])[:3])
                if len(c.get("domains", [])) > 3:
                    domains_str += f" +{len(c['domains'])-3} more"
                st.markdown(
                    f'<div class="cluster-row">'
                    f'<div>'
                    f'<div class="cluster-entity">{c["entity"]}</div>'
                    f'<div class="cluster-type">{c["entity_type"]} · across: {domains_str}</div>'
                    f'</div>'
                    f'<div class="cluster-badge">{c["article_count"]} articles</div>'
                    f'<div class="cluster-badge" style="background:#07100c;color:var(--teal);border-color:#1a3028">{c["domain_count"]} domains</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Domain Risk Profile ───────────────────────────────────────────────
        st.markdown('<div class="sec-title">Domain Risk Profile</div>', unsafe_allow_html=True)
        domain_risk = st.session_state.get("domain_risk", [])
        if not domain_risk:
            st.info("No domain data yet.")
        else:
            st.markdown('<div class="graph-panel">', unsafe_allow_html=True)
            for dr in domain_risk[:15]:
                fake_pct  = round(dr["fake_count"] / max(dr["total"], 1) * 100)
                bar_width = fake_pct
                bar_color = "#f06060" if fake_pct > 50 else ("#f5a623" if fake_pct > 20 else "#3dd68c")
                st.markdown(
                    f'<div class="domain-risk-row">'
                    f'<div class="domain-risk-name">{dr["domain"]}</div>'
                    f'<div style="flex:3">'
                    f'<div class="sbar-track"><div class="sbar-fill" style="width:{bar_width}%;background:{bar_color}"></div></div>'
                    f'</div>'
                    f'<div class="domain-risk-fake">{dr["fake_count"]}F</div>'
                    f'<div class="domain-risk-total">{dr["total"]} total</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:var(--muted);min-width:60px;text-align:right">avg {dr["avg_score"]}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL HEALTH  (unchanged from v8)
# ╚══════════════════════════════════════════════════════════════════════════════

with tab_health:
    st.markdown("### ⚕️ Model Health Dashboard")
    st.markdown('<p class="note">Live view of both the NLP credibility model and the SigLIP-2 vision fine-tune.</p>', unsafe_allow_html=True)

    nlp_metrics = load_metrics()

    st.markdown('<div class="sec-title">NLP Credibility Model  (DistilBERT + Logistic Regression)</div>', unsafe_allow_html=True)
    if nlp_metrics:
        m_cols = st.columns(5)
        for col, label, key in zip(m_cols,
            ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
            ["accuracy","precision","recall","f1_score","roc_auc"]):
            col.markdown(
                f'<div class="mcard"><div class="mcard-label">{label}</div>'
                f'<div class="mcard-value">{nlp_metrics.get(key,"—")}%</div></div>',
                unsafe_allow_html=True)
        st.markdown("")
        if "cv_f1_mean" in nlp_metrics:
            st.info(f"CV F1: {nlp_metrics['cv_f1_mean']}% ± {nlp_metrics['cv_f1_std']}%  ·  Brier: {nlp_metrics.get('brier_score','—')}")

        st.markdown('<div class="sec-title">Confusion Matrix</div>', unsafe_allow_html=True)
        import pandas as pd
        st.dataframe(
            pd.DataFrame(
                nlp_metrics["confusion_matrix"],
                index=["Actual Real","Actual Fake"],
                columns=["Predicted Real","Predicted Fake"],
            ),
            use_container_width=True,
        )
        t1, t2, t3 = st.columns(3)
        t1.info(f"**Train:** {nlp_metrics.get('train_size','—'):,}")
        t2.info(f"**Test:** {nlp_metrics.get('test_size','—'):,}")
        t3.info(f"**Features:** {nlp_metrics.get('feature_type','—').upper()}")
    else:
        st.info("No NLP model metrics found. Run `train_model.py` first.")

    st.markdown('<div class="sec-title">Vision Model  (SigLIP-2 Fine-tune via Intelligence Sync)</div>', unsafe_allow_html=True)
    auto_metrics_path = Path("model/auto_train_metrics.json")

    if auto_metrics_path.exists():
        with open(auto_metrics_path) as f:
            am = json.load(f)
        import datetime
        ts_str = datetime.datetime.fromtimestamp(am.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M UTC") if am.get("timestamp") else "—"
        st.markdown(
            f'<div class="health-card">'
            f'<div class="health-card-title">Last sync — {ts_str}</div>'
            f'<div class="metric-grid" style="grid-template-columns:repeat(4,1fr)">'
            f'<div class="mcard"><div class="mcard-label">Accuracy</div><div class="mcard-value green">{am.get("accuracy","—")}%</div></div>'
            f'<div class="mcard"><div class="mcard-label">F1 Score</div><div class="mcard-value">{am.get("f1","—")}%</div></div>'
            f'<div class="mcard"><div class="mcard-label">Precision</div><div class="mcard-value amber">{am.get("precision","—")}%</div></div>'
            f'<div class="mcard"><div class="mcard-label">Recall</div><div class="mcard-value">{am.get("recall","—")}%</div></div>'
            f'</div>'
            f'<div style="margin-top:12px;font-size:11px;color:var(--muted);font-family:JetBrains Mono,monospace">'
            f'backbone {am.get("backbone","—")} · '
            f'train images {am.get("train_images","—")} ({am.get("real_images","?")}R/{am.get("fake_images","?")}F) · '
            f'best epoch {am.get("best_epoch","—")} · lr {am.get("lr","—")}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if _LOG_FILE.exists() and _LOG_FILE.stat().st_size > 0:
            with st.expander("View last Intelligence Sync log", expanded=False):
                log_lines = _LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
                _render_train_log_lines(log_lines)
    else:
        st.info("No vision model sync has run yet. Label images in Analyst Review and click '🚀 Execute Intelligence Sync'.")

    st.markdown('<div class="sec-title">ONNX Artefact Status</div>', unsafe_allow_html=True)
    onnx_paths = [
        ("AI Detector (SigLIP-2)", Path("model/onnx/ai_detector/model_quantized.onnx")),
        ("Deepfake Classifier",    Path("model/onnx/deepfake_detector/model_quantized.onnx")),
    ]
    oc1, oc2 = st.columns(2)
    for col, (name, path) in zip([oc1, oc2], onnx_paths):
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            col.markdown(
                f'<div class="mcard"><div class="mcard-label">{name}</div>'
                f'<div class="mcard-value small green">✓ INT8</div>'
                f'<div class="mcard-sub">{size_mb:.1f} MB</div></div>',
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f'<div class="mcard"><div class="mcard-label">{name}</div>'
                f'<div class="mcard-value small" style="color:var(--muted)">Not cached</div>'
                f'<div class="mcard-sub">Downloads on first use</div></div>',
                unsafe_allow_html=True,
            )