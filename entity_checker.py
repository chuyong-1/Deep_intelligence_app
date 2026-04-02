"""
entity_checker.py

Named Entity Recognition + Wikipedia fact-checking.
Uses spaCy for NER and the free Wikipedia REST API — no API key needed.

Pipeline:
  1. Extract PERSON, ORG, GPE entities from article text using spaCy
  2. Query Wikipedia for each entity
  3. Return structured results: verified / ambiguous / not_found
  4. Compute an entity credibility score (0-100)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional
import requests

# --------------------------------------------------
# Config
# --------------------------------------------------
WIKI_API  = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS   = {"User-Agent": "NewsCredibilityChecker/1.0 (academic project)"}
TIMEOUT   = 5       # seconds per request
MAX_ENT   = 8       # max entities to look up (avoid slow UI)
DELAY     = 0.15    # seconds between requests (be polite to Wikipedia)

ENTITY_TYPES = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC"}

# --------------------------------------------------
# Data classes
# --------------------------------------------------
@dataclass
class EntityResult:
    name:        str
    entity_type: str
    status:      str           # "verified" | "not_found" | "ambiguous" | "error"
    summary:     Optional[str] = None
    wiki_url:    Optional[str] = None
    score_impact: float = 0.0  # positive = credibility boost, negative = penalty


@dataclass
class EntityCheckReport:
    entities:          list[EntityResult] = field(default_factory=list)
    entity_score:      float = 50.0   # 0-100 credibility contribution
    verified_count:    int   = 0
    not_found_count:   int   = 0
    ambiguous_count:   int   = 0
    total_checked:     int   = 0
    summary:           str   = ""
    error:             Optional[str] = None


# --------------------------------------------------
# Wikipedia lookup
# --------------------------------------------------
def _query_wikipedia(entity: str) -> dict:
    """
    Query Wikipedia REST API for an entity.
    Returns dict with keys: status, summary, url
    """
    clean = entity.strip().title()
    try:
        url  = WIKI_API + requests.utils.quote(clean)
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

        if resp.status_code == 200:
            data = resp.json()
            page_type = data.get("type", "")

            if page_type == "disambiguation":
                return {
                    "status":  "ambiguous",
                    "summary": f"Multiple Wikipedia pages match '{entity}'.",
                    "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }

            summary = data.get("extract", "")
            # Truncate to 2 sentences for display
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            short_summary = " ".join(sentences[:2]) if sentences else summary

            return {
                "status":  "verified",
                "summary": short_summary,
                "url":     data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }

        elif resp.status_code == 404:
            return {"status": "not_found", "summary": None, "url": None}
        else:
            return {"status": "error", "summary": f"HTTP {resp.status_code}", "url": None}

    except requests.Timeout:
        return {"status": "error", "summary": "Request timed out.", "url": None}
    except Exception as e:
        return {"status": "error", "summary": str(e), "url": None}


# --------------------------------------------------
# NER extraction
# --------------------------------------------------
def _extract_entities(text: str, nlp) -> list[tuple[str, str]]:
    """
    Extract unique named entities from text using spaCy.
    Returns list of (entity_text, entity_type) tuples.
    """
    doc = nlp(text[:5000])  # limit for speed

    seen  = set()
    ents  = []
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        name = ent.text.strip()
        # Skip short/generic tokens
        if len(name) < 3 or name.lower() in {"the", "a", "an", "this", "that"}:
            continue
        # Skip pure numbers
        if re.fullmatch(r'[\d\s,\.]+', name):
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            ents.append((name, ent.label_))

    return ents[:MAX_ENT]


# --------------------------------------------------
# Scoring
# --------------------------------------------------
def _compute_entity_score(results: list[EntityResult], total_checked: int) -> float:
    """
    Compute a 0-100 entity credibility score.

    Logic:
    - Start at 50 (neutral)
    - Each verified entity: +8 (real people/orgs boost credibility)
    - Each not_found entity: -15 (fabricated entities are strong fake signal)
    - Each ambiguous: -3 (minor penalty)
    - Clamp to [0, 100]
    """
    if total_checked == 0:
        return 50.0

    score = 50.0
    for r in results:
        if r.status == "verified":
            score += 8.0
            r.score_impact = +8.0
        elif r.status == "not_found":
            score -= 15.0
            r.score_impact = -15.0
        elif r.status == "ambiguous":
            score -= 3.0
            r.score_impact = -3.0

    return round(max(0.0, min(100.0, score)), 1)


# --------------------------------------------------
# Main entry point
# --------------------------------------------------
def check_entities(text: str, nlp=None) -> EntityCheckReport:
    """
    Run the full entity check pipeline on article text.

    Args:
        text: raw article text
        nlp:  loaded spaCy model (pass in to avoid reloading)

    Returns:
        EntityCheckReport with all results and a credibility score
    """
    report = EntityCheckReport()

    # Load spaCy if not provided
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            report.error = (
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            return report
        except ImportError:
            report.error = "spaCy not installed. Run: pip install spacy"
            return report

    # Extract entities
    raw_ents = _extract_entities(text, nlp)

    if not raw_ents:
        report.summary      = "No named entities found in this article."
        report.entity_score = 50.0
        return report

    report.total_checked = len(raw_ents)

    # Look up each entity
    for name, ent_type in raw_ents:
        wiki = _query_wikipedia(name)
        time.sleep(DELAY)

        result = EntityResult(
            name        = name,
            entity_type = ent_type,
            status      = wiki["status"],
            summary     = wiki.get("summary"),
            wiki_url    = wiki.get("url"),
        )
        report.entities.append(result)

        if wiki["status"] == "verified":
            report.verified_count += 1
        elif wiki["status"] == "not_found":
            report.not_found_count += 1
        elif wiki["status"] == "ambiguous":
            report.ambiguous_count += 1

    # Compute score
    report.entity_score = _compute_entity_score(report.entities, report.total_checked)

    # Generate summary sentence
    parts = []
    if report.verified_count:
        parts.append(f"{report.verified_count} verified")
    if report.not_found_count:
        parts.append(f"{report.not_found_count} not found on Wikipedia")
    if report.ambiguous_count:
        parts.append(f"{report.ambiguous_count} ambiguous")

    report.summary = (
        f"Checked {report.total_checked} entities: {', '.join(parts)}. "
        f"Entity credibility score: {report.entity_score}/100."
    )

    return report
