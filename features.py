"""
features.py  v2

Shared stylistic feature extraction — single source of truth for
train_model.py and app.py.

Expanded from 10 → 16 features:
  11. avg_sentence_length   — fake news tends to be shorter / choppier
  12. passive_voice_ratio   — credible journalism uses more passive constructions
  13. hedging_score         — credible sources hedge claims ("allegedly", "reportedly")
  14. polarity_ratio        — fake news uses more extreme adjectives
  15. punc_abuse            — !!!/??? patterns signal sensationalism
  16. first_person_ratio    — editorialising ("I think", "we must") = less credible
"""

import re
import numpy as np

# --------------------------------------------------
# Signal word lists
# --------------------------------------------------
SOURCE_PHRASES = [
    "according to", "said in a statement", "told reporters",
    "confirmed by", "spokesperson said", "officials said",
    "research shows", "study found", "published in",
    "percent", "per cent", "%", "sources say", "reported by",
    "cited by", "as reported", "data shows", "evidence suggests",
]

ALARM_WORDS = [
    "breaking", "exclusive", "shocking", "bombshell", "exposed",
    "conspiracy", "deep state", "globalist", "whistleblower",
    "they don't want", "mainstream media", "hoax",
    "secret", "banned", "censored", "suppressed", "wake up",
    "share before deleted", "must read", "urgent",
    "what they're not telling", "truth about", "the real reason",
    "cover up", "cover-up", "false flag",
]

HEDGING_PHRASES = [
    "it appears", "it seems", "may be", "might be", "could be",
    "possibly", "probably", "allegedly", "reportedly", "claimed",
    "is said to", "is believed to", "according to", "suggests that",
    "some experts", "it is thought", "unconfirmed", "sources say",
]

_PASSIVE_RE = [
    re.compile(r'\b(was|were|is|are|been|be)\s+\w+ed\b'),
    re.compile(r'\b(was|were|is|are)\s+\w+en\b'),
]

_FIRST_PERSON = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)

_STRONG_POS = frozenset({
    "amazing", "incredible", "wonderful", "fantastic", "brilliant",
    "excellent", "outstanding", "extraordinary", "perfect", "greatest",
})
_STRONG_NEG = frozenset({
    "terrible", "horrible", "awful", "disgusting", "pathetic",
    "disastrous", "catastrophic", "atrocious", "dreadful", "appalling",
})

STYLE_DIM = 16

# Labels for UI display (app.py uses this)
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
    ("Avg sentence length",   "real",    "Longer sentences = more credible"),
    ("Passive voice ratio",   "real",    "Higher = more journalistic"),
    ("Hedging language",      "real",    "High = cautious / credible"),
    ("Sentiment polarity",    "fake",    "Extreme adjectives = suspicious"),
    ("Punctuation abuse",     "fake",    "!!!/??? = sensationalist"),
    ("First-person ratio",    "fake",    "High = editorialising"),
]

STYLE_DISPLAY_MAX = [
    0.3, 1.0, 0.05, 5.0, 5.0,
    8.0, 5.0, 8.0, 10.0, 0.1,
    40.0, 0.3, 6.0, 0.1, 0.05, 0.15,
]


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def stylistic_features(text: str) -> np.ndarray:
    words       = text.split()
    num_words   = max(len(words), 1)
    lower       = text.lower()
    alpha_words = [w for w in words if w.isalpha()]
    lower_alpha = [w.lower() for w in alpha_words]
    num_alpha   = max(len(alpha_words), 1)

    # 1–10: original features
    caps_ratio     = sum(1 for w in alpha_words if w.isupper() and len(w) > 1) / num_alpha
    title_ratio    = sum(1 for w in alpha_words if w.istitle()) / num_alpha
    exclaim_ratio  = text.count("!") / num_words
    question_count = float(text.count("?"))
    ellipsis_count = float(text.count("..."))
    source_score   = float(sum(1 for p in SOURCE_PHRASES if p in lower))
    alarm_score    = float(sum(1 for w in ALARM_WORDS if w in lower))
    log_length     = float(np.log1p(num_words))
    quote_count    = float(text.count('"') // 2)
    number_ratio   = len(re.findall(r'\b\d+\.?\d*\b', text)) / num_words

    # 11: avg sentence length
    sentences    = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.split()) > 1]
    avg_sent_len = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0

    # 12: passive voice ratio  (matches per sentence)
    passive_hits  = sum(len(pat.findall(lower)) for pat in _PASSIVE_RE)
    passive_ratio = passive_hits / max(len(sentences), 1)

    # 13: hedging language
    hedging_score = float(sum(1 for p in HEDGING_PHRASES if p in lower))

    # 14: sentiment polarity (extreme adj ratio)
    polarity_ratio = sum(
        1 for w in lower_alpha if w in _STRONG_POS or w in _STRONG_NEG
    ) / num_alpha

    # 15: punctuation abuse (repeated ! or ?)
    punc_abuse = (
        len(re.findall(r'!{2,}', text)) + len(re.findall(r'\?{2,}', text))
    ) / num_words

    # 16: first-person ratio
    fp_ratio = sum(1 for w in lower_alpha if w in _FIRST_PERSON) / num_alpha

    return np.array([
        caps_ratio, title_ratio, exclaim_ratio, question_count,
        ellipsis_count, source_score, alarm_score, log_length,
        quote_count, number_ratio,
        avg_sent_len, passive_ratio, hedging_score, polarity_ratio,
        punc_abuse, fp_ratio,
    ], dtype=np.float32)


def batch_stylistic(texts: list) -> np.ndarray:
    return np.vstack([stylistic_features(t) for t in texts])


def clean_text(text: str, stop_words: set) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    return " ".join(w for w in text.split() if w not in stop_words)