"""
claim_verifier.py  v3.1

Fixes over v3
─────────────
  1. Python 3.8 compatibility — replaced bare `list[str]` / `list[dict]`
     generics with `typing.List` / `typing.Dict` throughout.

  2. _nli_score_pair — label2id lookup now validates that ALL THREE expected
     keys are present before trusting the config; falls back to MNLI order
     (contradiction=0, entailment=1, neutral=2) on any mismatch instead of
     silently returning wrong probabilities. Prints a visible warning so
     misconfiguration is never hidden.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
MAX_CLAIMS    = 5      # max claims to extract per article
MAX_RESULTS   = 3      # DuckDuckGo results per claim
SEARCH_DELAY  = 0.8    # seconds between DDG requests

NLI_MODEL_ID  = "cross-encoder/nli-deberta-v3-small"

NLI_ENTAIL_THRESH      = 0.40
NLI_CONTRADICT_THRESH  = 0.40

_nli_model = None


def _get_nli_model():
    """
    Lazy-load the cross-encoder NLI model (~300 MB, cached after first call).
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
    claim:   str
    verdict: str                    # supported | contradicted | unverified

    entailment_score:    float = 0.0
    contradiction_score: float = 0.0
    neutral_score:       float = 0.0
    similarity_score:    float = 0.0   # backward-compat — equals entailment_score

    source_title: Optional[str] = None
    source_url:   Optional[str] = None
    source_body:  Optional[str] = None
    search_ran:   bool  = False
    nli_model:    str   = NLI_MODEL_ID
    best_snippet: str   = ""


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
# Step 1 — Claim extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_claims(text: str, nlp, max_claims: int = MAX_CLAIMS) -> List[str]:
    """
    Extract sentences that are likely factual claims:
      - contain at least one named entity  (PERSON, ORG, GPE, DATE, NORP, FAC)
      - contain at least one verb
      - are between 8 and 45 words long
    """
    ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "NORP", "FAC", "LOC"}
    doc    = nlp(text[:4000])
    claims: List[str] = []

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
# Step 2 — Evidence search
# ──────────────────────────────────────────────────────────────────────────────

def search_evidence(claim: str, max_results: int = MAX_RESULTS) -> List[Dict]:
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
# Step 3 — Cross-Encoder NLI scoring
# ──────────────────────────────────────────────────────────────────────────────

def _nli_score_pair(claim: str, snippet: str) -> Dict[str, float]:
    """
    Score one (claim, evidence-snippet) pair using the cross-encoder NLI model.

    MNLI default label order: contradiction=0, entailment=1, neutral=2.
    The model's label2id config is used when present and fully valid
    (all three required keys found). A visible warning is printed on any
    config mismatch so misconfiguration is never silently swallowed.
    """
    import numpy as np

    nli    = _get_nli_model()
    logits = nli.predict([(claim, snippet)])  # shape (1, 3)
    logit  = logits[0]

    # Numerically stable softmax
    logit_stable = logit - logit.max()
    exp_logits   = np.exp(logit_stable)
    probs        = exp_logits / exp_logits.sum()

    # ── Resolve label → index mapping, with full validation ──────────────────
    # Default MNLI convention: contradiction=0, entailment=1, neutral=2
    contradiction_idx, entailment_idx, neutral_idx = 0, 1, 2

    try:
        raw_label2id = nli.model.config.label2id
        # Normalise keys: strip whitespace and lowercase
        label2id = {k.strip().lower(): int(v) for k, v in raw_label2id.items()}
        required = {"contradiction", "entailment", "neutral"}

        if required.issubset(label2id.keys()):
            contradiction_idx = label2id["contradiction"]
            entailment_idx    = label2id["entailment"]
            neutral_idx       = label2id["neutral"]
        else:
            missing_keys = required - label2id.keys()
            print(
                f"[claim_verifier] WARNING: model label2id is missing keys {missing_keys}. "
                f"Falling back to MNLI default order (contradiction=0, entailment=1, neutral=2)."
            )
    except Exception as exc:
        print(
            f"[claim_verifier] WARNING: could not read label2id ({exc}). "
            f"Using MNLI default order."
        )

    return {
        "entailment":    round(float(probs[entailment_idx]),    4),
        "contradiction": round(float(probs[contradiction_idx]), 4),
        "neutral":       round(float(probs[neutral_idx]),       4),
    }


def score_claim_against_evidence(claim: str, evidence: List[Dict]) -> ClaimResult:
    """
    Run the Cross-Encoder NLI model on each (claim, evidence-snippet) pair
    and return the ClaimResult for the best-scoring piece of evidence.

    "Best" = snippet with highest max(entailment, contradiction).

    Verdict logic (contradiction checked first — higher risk):
        contradiction_score >= NLI_CONTRADICT_THRESH  -> "contradicted"
        entailment_score    >= NLI_ENTAIL_THRESH       -> "supported"
        otherwise                                       -> "unverified"
    """
    if not evidence:
        return ClaimResult(
            claim=claim, verdict="unverified",
            entailment_score=0.0, contradiction_score=0.0,
            neutral_score=1.0, similarity_score=0.0,
            search_ran=True,
        )

    best_nli: Dict[str, float]  = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}
    best_source: Optional[Dict] = None
    best_snippet_text: str      = ""

    for e in evidence:
        snippet = f"{e['title']} {e['body']}"[:400].strip()
        if not snippet:
            continue

        try:
            nli_scores = _nli_score_pair(claim, snippet)
        except Exception:
            continue

        if max(nli_scores["entailment"], nli_scores["contradiction"]) > \
           max(best_nli["entailment"],   best_nli["contradiction"]):
            best_nli          = nli_scores
            best_source       = e
            best_snippet_text = snippet

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
        similarity_score     = best_nli["entailment"],
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
    Interface identical to v1/v2/v3.
    """
    report = ClaimReport()

    missing: List[str] = []
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