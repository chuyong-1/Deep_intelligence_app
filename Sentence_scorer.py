"""
Sentence_scorer.py

Sentence-level suspiciousness scoring for the Verity News Credibility Dashboard.

Each sentence is scored 0.0 (very credible) → 1.0 (very suspicious)
using the same signal lists already defined in features.py.

Signals that RAISE suspicion (fake indicators):
  - Contains alarm words
  - ALL CAPS words
  - Exclamation marks / repeated punctuation
  - Extreme sentiment adjectives
  - First-person editorialising
  - Very short sentence (< 6 words) — punchy, assertive style

Signals that LOWER suspicion (credibility indicators):
  - Contains source citation phrases
  - Contains hedging language
  - Contains passive voice construction
  - Contains numeric facts
  - Quotation marks (quoting a source)

The score is a weighted sum clamped to [0, 1].
"""

import re
from dataclasses import dataclass, field
from typing import List

from features import (
    ALARM_WORDS, SOURCE_PHRASES, HEDGING_PHRASES,
    _PASSIVE_RE, _FIRST_PERSON, _STRONG_POS, _STRONG_NEG,
)

# --------------------------------------------------
# Per-signal weights  (positive = more suspicious)
# --------------------------------------------------
W_ALARM      =  0.35
W_CAPS       =  0.20
W_EXCLAIM    =  0.15
W_PUNC_ABUSE =  0.15
W_POLARITY   =  0.15
W_FIRST_PERS =  0.10
W_SHORT      =  0.10

W_SOURCE     = -0.25
W_HEDGING    = -0.20
W_PASSIVE    = -0.10
W_NUMERIC    = -0.10
W_QUOTED     = -0.10

SHORT_SENT_THRESHOLD = 6   # words


@dataclass
class SentenceScore:
    text: str
    score: float                        # 0.0 = credible, 1.0 = suspicious
    signals: List[str] = field(default_factory=list)   # human-readable triggers


def _score_sentence(sentence: str) -> SentenceScore:
    text    = sentence.strip()
    lower   = text.lower()
    words   = text.split()
    n       = max(len(words), 1)
    alpha   = [w for w in words if w.isalpha()]
    lalpha  = [w.lower() for w in alpha]
    n_alpha = max(len(alpha), 1)

    score   = 0.0
    signals = []

    # --- suspicious signals ---
    alarm_hits = [w for w in ALARM_WORDS if w in lower]
    if alarm_hits:
        score += W_ALARM
        signals.append(f"alarm word: {alarm_hits[0]!r}")

    caps_ratio = sum(1 for w in alpha if w.isupper() and len(w) > 1) / n_alpha
    if caps_ratio > 0.15:
        score += W_CAPS
        signals.append("excessive ALL CAPS")

    if text.count("!") > 0:
        score += W_EXCLAIM * min(text.count("!"), 3)
        signals.append("exclamation mark(s)")

    punc_abuse = len(re.findall(r'[!?]{2,}', text))
    if punc_abuse:
        score += W_PUNC_ABUSE
        signals.append("repeated punctuation")

    polar_hits = [w for w in lalpha if w in _STRONG_POS or w in _STRONG_NEG]
    if polar_hits:
        score += W_POLARITY
        signals.append(f"extreme adjective: {polar_hits[0]!r}")

    fp_hits = [w for w in lalpha if w in _FIRST_PERSON]
    if fp_hits:
        score += W_FIRST_PERS
        signals.append("first-person voice")

    if n < SHORT_SENT_THRESHOLD and n > 1:
        score += W_SHORT
        signals.append("very short sentence")

    # --- credibility signals ---
    src_hits = [p for p in SOURCE_PHRASES if p in lower]
    if src_hits:
        score += W_SOURCE
        signals.append(f"source citation: {src_hits[0]!r}")

    hedge_hits = [p for p in HEDGING_PHRASES if p in lower]
    if hedge_hits:
        score += W_HEDGING
        signals.append(f"hedging language: {hedge_hits[0]!r}")

    passive_hits = sum(len(pat.findall(lower)) for pat in _PASSIVE_RE)
    if passive_hits:
        score += W_PASSIVE
        signals.append("passive construction")

    numeric_count = len(re.findall(r'\b\d+\.?\d*\b', text))
    if numeric_count >= 1:
        score += W_NUMERIC * min(numeric_count, 3)
        signals.append(f"{numeric_count} numeric fact(s)")

    if text.count('"') >= 2:
        score += W_QUOTED
        signals.append("quoted source")

    return SentenceScore(
        text    = text,
        score   = round(max(0.0, min(1.0, score)), 3),
        signals = signals,
    )


def score_sentences(text: str) -> List[SentenceScore]:
    """
    Split article into sentences and score each one.
    Returns list in original order. Skips blank / single-word lines.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    results = []
    for sent in raw:
        sent = sent.strip()
        if len(sent.split()) < 2:
            continue
        results.append(_score_sentence(sent))
    return results


def suspicion_label(score: float) -> str:
    if score >= 0.5:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"