"""
claim_verifier.py  v3

Claim-level fact verification pipeline.

What changed from v2 → v3
──────────────────────────
  1. Cross-Encoder NLI replaces simple cosine similarity scoring.

     v1/v2 used a bi-encoder (MiniLM) to encode claim and evidence snippet
     independently and computed cosine similarity.  Cosine similarity measures
     topical relatedness — it cannot detect logical contradiction because two
     sentences about the same topic but with opposite polarity (e.g.
     "Biden won the 2020 election" vs "Biden lost the 2020 election") can
     have very high cosine similarity.

     v3 uses cross_encoders.CrossEncoder on a dedicated 3-class NLI model
     (cross-encoder/nli-deberta-v3-small, ~300 MB):
       • Entailment  — evidence SUPPORTS the claim
       • Neutral     — evidence discusses the topic but neither supports
                       nor contradicts
       • Contradiction — evidence OPPOSES the claim (logical contradiction)

     The cross-encoder jointly encodes the claim-evidence pair, enabling it
     to attend to token-level interactions that reveal polarity reversal,
     negation, and factual contradiction — none of which are detectable by
     independent bi-encoder embeddings.

  2. ClaimResult gets entailment_score and contradiction_score float fields
     alongside the old similarity_score (now set to the entailment score for
     backward compatibility with app.py rendering).

  3. Verdict thresholds:
       entailment_score   >= 0.40  -> "supported"
       contradiction_score >= 0.40  -> "contradicted"
       otherwise                   -> "unverified"

     Contradiction takes priority over entailment when both scores exceed 0.40
     (conservative approach — flag contradictions loudly).

  4. Fallback: if sentence_transformers / cross_encoders are not installed
     the pipeline logs the missing dependency and sets error on ClaimReport,
     preserving the same interface as v1/v2 for the app.py error-handling path.

  5. verify_claims() retains the same synchronous interface as v2 — no
     async changes required in app.py.

Dependencies:
    pip install sentence-transformers
    (cross-encoder is part of the sentence-transformers package)

Optionally for faster search:
    pip install duckduckgo-search
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MAX_CLAIMS    = 5      # max claims to extract per article
MAX_RESULTS   = 3      # DuckDuckGo results per claim
SEARCH_DELAY  = 0.8    # seconds between DDG requests

# NLI model — 3-class cross-encoder trained on MNLI + SNLI + FEVER
# Size ~300 MB.  Replaced nli-deberta-v3-base (600 MB) for faster CPU inference
# while retaining strong contradiction detection.
NLI_MODEL_ID  = "cross-encoder/nli-deberta-v3-small"

# ── NLI verdict thresholds ────────────────────────────────────────────────────
# Entailment threshold: score >= this -> evidence supports the claim
NLI_ENTAIL_THRESH      = 0.40
# Contradiction threshold: score >= this -> evidence contradicts the claim
# Contradiction takes priority over entailment when both exceed their threshold.
NLI_CONTRADICT_THRESH  = 0.40

# ── Lazy-loaded cross-encoder (cached after first call) ───────────────────────
_nli_model = None


def _get_nli_model():
    """
    Lazy-load the cross-encoder NLI model.

    CrossEncoder("cross-encoder/nli-deberta-v3-small") downloads the model
    on first call (~300 MB) and caches it in HuggingFace's local model cache.
    Subsequent calls reuse the cached model without re-downloading.

    Returns a CrossEncoder instance or raises ImportError if
    sentence_transformers is not installed.
    """
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(NLI_MODEL_ID)
    return _nli_model


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ClaimResult:
    claim:  str
    verdict: str                    # supported | contradicted | unverified

    # NLI scores (v3 primary signals)
    entailment_score:    float = 0.0   # P(evidence entails claim)       0-1
    contradiction_score: float = 0.0   # P(evidence contradicts claim)   0-1
    neutral_score:       float = 0.0   # P(evidence is neutral to claim) 0-1

    # Backward-compat field (app.py renders this as a fallback)
    # Set equal to entailment_score so older rendering paths still work.
    similarity_score: float = 0.0

    source_title: Optional[str] = None
    source_url:   Optional[str] = None
    source_body:  Optional[str] = None
    search_ran:   bool = False

    # NLI model-level fields
    nli_model:    str  = NLI_MODEL_ID
    best_snippet: str  = ""          # the evidence snippet that produced these scores


@dataclass
class ClaimReport:
    claims: list = field(default_factory=list)   # List[ClaimResult]
    supported_count:    int   = 0
    contradicted_count: int   = 0
    unverified_count:   int   = 0
    claim_score:        float = 50.0
    summary:            str   = ""
    error:              Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Claim extraction  (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────────────

def extract_claims(text: str, nlp, max_claims: int = MAX_CLAIMS) -> list[str]:
    """
    Extract sentences that are likely factual claims:
      - contain at least one named entity  (PERSON, ORG, GPE, DATE, NORP, FAC)
      - contain at least one verb
      - are between 8 and 45 words long
    """
    ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "NORP", "FAC", "LOC"}
    doc    = nlp(text[:4000])
    claims: list[str] = []

    for sent in doc.sents:
        has_entity = any(ent.label_ in ENTITY_TYPES for ent in sent.ents)
        has_verb   = any(tok.pos_ == "VERB" for tok in sent)
        word_count = len(sent.text.split())

        if has_entity and has_verb and 8 <= word_count <= 45:
            clean = sent.text.strip()
            if re.fullmatch(r'[\d\s,\.\-\(\)]+', clean):
                continue
            claims.append(clean)

        if len(claims) >= max_claims:
            break

    return claims


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Evidence search  (DuckDuckGo, unchanged from v2)
# ──────────────────────────────────────────────────────────────────────────────

def search_evidence(claim: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """
    Search DuckDuckGo for evidence snippets related to the claim.
    Returns list of {title, body, url} or empty list on any error.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(claim, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "body":  r.get("body",  ""),
                "url":   r.get("href",  ""),
            }
            for r in results
            if r.get("body")
        ]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Cross-Encoder NLI scoring  [v3 — replaces cosine similarity]
# ──────────────────────────────────────────────────────────────────────────────

def _nli_score_pair(claim: str, snippet: str) -> dict[str, float]:
    """
    Score one (claim, evidence-snippet) pair using the cross-encoder NLI model.

    The cross-encoder receives both strings jointly as a sequence pair
    "[CLS] claim [SEP] snippet [SEP]" and produces logits for three classes:
        label 0 — Contradiction
        label 1 — Entailment
        label 2 — Neutral
    (label order matches the MNLI convention used by deberta-v3 NLI models)

    We apply softmax over the raw logits to obtain calibrated probabilities.

    Returns:
        dict with keys "entailment", "contradiction", "neutral" (float 0-1).
    """
    import numpy as np

    nli = _get_nli_model()

    # CrossEncoder.predict() returns raw logits for a list of pairs
    logits = nli.predict([(claim, snippet)])   # shape (1, 3)
    logit  = logits[0]                          # shape (3,)

    # Numerically stable softmax
    logit_stable = logit - logit.max()
    exp_logits   = np.exp(logit_stable)
    probs        = exp_logits / exp_logits.sum()

    # MNLI label order: contradiction=0, entailment=1, neutral=2
    # Some cross-encoders may differ; we inspect the model's label2id if
    # available to resolve ordering, otherwise fall back to MNLI convention.
    try:
        label2id = nli.model.config.label2id
        # normalise keys (strip whitespace, lower case)
        label2id = {k.strip().lower(): v for k, v in label2id.items()}
        contradiction_idx = label2id.get("contradiction", 0)
        entailment_idx    = label2id.get("entailment",    1)
        neutral_idx       = label2id.get("neutral",       2)
    except Exception:
        contradiction_idx, entailment_idx, neutral_idx = 0, 1, 2

    return {
        "entailment":    round(float(probs[entailment_idx]),    4),
        "contradiction": round(float(probs[contradiction_idx]), 4),
        "neutral":       round(float(probs[neutral_idx]),       4),
    }


def score_claim_against_evidence(claim: str, evidence: list[dict]) -> ClaimResult:
    """
    Run the Cross-Encoder NLI model on each (claim, evidence-snippet) pair
    and return the ClaimResult for the best-scoring piece of evidence.

    "Best" is defined as the snippet with the highest max(entailment, contradiction)
    score — i.e. whichever snippet takes the strongest stance on the claim,
    regardless of direction.  This surfaces the most informative evidence rather
    than defaulting to the first result.

    Verdict logic:
        contradiction_score >= NLI_CONTRADICT_THRESH  -> "contradicted"
            (checked first — contradictions are flagged loudly)
        entailment_score    >= NLI_ENTAIL_THRESH       -> "supported"
        otherwise                                       -> "unverified"

    When both scores exceed their thresholds simultaneously, contradiction wins
    (conservative: a contradiction is a harder signal than support).
    """
    if not evidence:
        return ClaimResult(
            claim=claim, verdict="unverified",
            entailment_score=0.0, contradiction_score=0.0,
            neutral_score=1.0, similarity_score=0.0,
            search_ran=True,
        )

    best_nli:    dict[str, float] = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
    best_source: Optional[dict]   = None
    best_snippet_text: str        = ""

    for e in evidence:
        snippet = f"{e['title']} {e['body']}"[:400].strip()
        if not snippet:
            continue

        try:
            nli_scores = _nli_score_pair(claim, snippet)
        except Exception:
            continue

        # Pick the snippet with the strongest stance (highest max score)
        if max(nli_scores["entailment"], nli_scores["contradiction"]) > \
           max(best_nli["entailment"],   best_nli["contradiction"]):
            best_nli          = nli_scores
            best_source       = e
            best_snippet_text = snippet

    # ── Determine verdict ─────────────────────────────────────────────────────
    # Contradiction checked first — it is the higher-risk finding.
    if best_nli["contradiction"] >= NLI_CONTRADICT_THRESH:
        verdict = "contradicted"
    elif best_nli["entailment"] >= NLI_ENTAIL_THRESH:
        verdict = "supported"
    else:
        verdict = "unverified"

    return ClaimResult(
        claim                = claim,
        verdict              = verdict,
        entailment_score     = best_nli["entailment"],
        contradiction_score  = best_nli["contradiction"],
        neutral_score        = best_nli["neutral"],
        similarity_score     = best_nli["entailment"],   # back-compat
        source_title         = best_source["title"] if best_source else None,
        source_url           = best_source["url"]   if best_source else None,
        source_body          = (best_source["body"] or "")[:200] if best_source else None,
        best_snippet         = best_snippet_text,
        search_ran           = True,
        nli_model            = NLI_MODEL_ID,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def verify_claims(text: str, nlp) -> ClaimReport:
    """
    Full pipeline: extract claims → search DuckDuckGo → NLI scoring.

    The interface is identical to v1/v2 — app.py requires no changes.

    Dependency check:
        sentence-transformers must be installed for the cross-encoder.
        duckduckgo-search is required for evidence retrieval.
        spaCy is required for claim extraction.

    Returns:
        ClaimReport with .error set if any dependency is missing.
    """
    report = ClaimReport()

    # Dependency guard
    missing: list[str] = []
    try:
        from sentence_transformers import CrossEncoder   # noqa
    except ImportError:
        missing.append("sentence-transformers")
    try:
        from duckduckgo_search import DDGS              # noqa
    except ImportError:
        missing.append("duckduckgo-search")

    if missing:
        report.error = (
            f"Missing dependencies: {', '.join(missing)}. "
            f"Run: pip install {' '.join(missing)}"
        )
        return report

    if nlp is None:
        report.error = "spaCy not available — needed for claim extraction."
        return report

    claims = extract_claims(text, nlp)
    if not claims:
        report.summary = "No verifiable claims found."
        return report

    for i, claim in enumerate(claims):
        if i > 0:
            time.sleep(SEARCH_DELAY)

        evidence = search_evidence(claim)
        result   = score_claim_against_evidence(claim, evidence)
        report.claims.append(result)

        if result.verdict == "supported":
            report.supported_count    += 1
        elif result.verdict == "contradicted":
            report.contradicted_count += 1
        else:
            report.unverified_count   += 1

    # Composite claim score — contradictions penalised 1.5×
    total = max(len(report.claims), 1)
    report.claim_score = round(
        50.0 + (report.supported_count - report.contradicted_count * 1.5) / total * 25,
        1,
    )
    report.claim_score = max(0.0, min(100.0, report.claim_score))

    report.summary = (
        f"{report.supported_count} supported, "
        f"{report.contradicted_count} contradicted, "
        f"{report.unverified_count} unverified · "
        f"Claim score: {report.claim_score}/100 "
        f"(NLI model: {NLI_MODEL_ID})"
    )
    return report