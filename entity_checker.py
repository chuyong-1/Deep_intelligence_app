"""
entity_checker.py  v2.1

Fixes over v2
─────────────
  1. _wiki_lookup — preserve original casing for the Wikipedia API URL.
     v2 passed `key` (lowercased) to `.title()` which converts acronyms
     like "NATO" → "nato" → "Nato", causing a Wikipedia 404. The fix
     passes the original entity name directly into the URL so "NATO" stays
     "NATO" and the lookup succeeds. The LRU cache still uses the lowercased
     key for deduplication.

  2. All other logic, public API, and signatures are unchanged.
"""

import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, List, Tuple

# --------------------------------------------------
# Config
# --------------------------------------------------
WIKI_API      = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKIDATA_API  = "https://www.wikidata.org/w/api.php"
HEADERS       = {"User-Agent": "NewsCredibilityChecker/2.0 (educational project)"}
TIMEOUT       = 6
MAX_ENTITIES  = 12
MAX_WORKERS   = 6
MAX_RETRIES   = 2
RETRY_DELAY   = 0.4

ENTITY_TYPES  = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC"}

ENTITY_WEIGHT = {
    "PERSON": 1.5,
    "ORG":    1.3,
    "NORP":   1.0,
    "GPE":    0.8,
    "FAC":    0.8,
    "LOC":    0.7,
}

# --------------------------------------------------
# Scoring constants  (single source of truth — imported by app.py)
# --------------------------------------------------
SCORE_VERIFIED  = +8
SCORE_NOT_FOUND = -15
SCORE_AMBIGUOUS = -3


# --------------------------------------------------
# Data classes
# --------------------------------------------------
@dataclass
class EntityResult:
    name: str
    entity_type: str
    status: str                  # verified | not_found | ambiguous | error
    frequency: int = 1
    confidence: float = 1.0
    summary: Optional[str] = None
    wiki_url: Optional[str] = None
    wikidata_found: bool = False
    score_impact: float = 0.0


@dataclass
class EntityCheckReport:
    entities: List[EntityResult] = field(default_factory=list)
    entity_score: float = 50.0
    verified_count: int = 0
    not_found_count: int = 0
    ambiguous_count: int = 0
    total_checked: int = 0
    summary: str = ""
    error: Optional[str] = None


# --------------------------------------------------
# Wikipedia REST API  (LRU cache, max 512 entries)
# --------------------------------------------------
@lru_cache(maxsize=512)
def _wiki_lookup(key: str, original: str) -> tuple:
    """
    key      = lowercased entity name (used only for LRU cache deduplication).
    original = entity name in its original casing, used for the API URL.

    FIX (v2.1): v2 derived the URL name via `key.title()`, which corrupts
    acronyms — "nato".title() == "Nato" → Wikipedia 404.  We now use
    `original` directly so "NATO" stays "NATO".

    Returns (status, summary, url).
    Retries up to MAX_RETRIES on 5xx / connection errors.
    """
    url = WIKI_API + requests.utils.quote(original)

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("type") == "disambiguation":
                    return (
                        "ambiguous",
                        f"Multiple Wikipedia pages match '{original}'.",
                        data.get("content_urls", {}).get("desktop", {}).get("page"),
                    )
                extract   = data.get("extract", "")
                sentences = re.split(r'(?<=[.!?])\s+', extract)
                short     = " ".join(sentences[:2])
                return (
                    "verified",
                    short,
                    data.get("content_urls", {}).get("desktop", {}).get("page"),
                )

            if resp.status_code == 404:
                return ("not_found", None, None)

            if resp.status_code >= 500 and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue

            return ("error", f"HTTP {resp.status_code}", None)

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
                continue
            return ("error", "Request timed out", None)
        except Exception as exc:
            return ("error", str(exc), None)

    return ("error", "Max retries exceeded", None)


# --------------------------------------------------
# Wikidata fallback for not_found entities
# --------------------------------------------------
@lru_cache(maxsize=256)
def _wikidata_exists(entity: str) -> bool:
    """
    Quick Wikidata search — returns True if at least one result exists.
    """
    try:
        params = {
            "action":   "wbsearchentities",
            "search":   entity,
            "language": "en",
            "format":   "json",
            "limit":    1,
        }
        resp = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code == 200:
            results = resp.json().get("search", [])
            return len(results) > 0
    except Exception:
        pass
    return False


def query_entity(entity: str) -> dict:
    """
    Full lookup: Wikipedia first, Wikidata fallback for not_found.
    Returns dict with status, summary, url, wikidata_found.
    """
    key = entity.lower().strip()
    # Pass original name so the API URL preserves casing (fixes acronyms)
    status, summary, url = _wiki_lookup(key, entity)

    wikidata_found = False
    if status == "not_found":
        if _wikidata_exists(entity):
            status         = "ambiguous"
            summary        = f"'{entity}' found on Wikidata but lacks a Wikipedia summary page."
            wikidata_found = True

    return {
        "status":         status,
        "summary":        summary,
        "url":            url,
        "wikidata_found": wikidata_found,
    }


# --------------------------------------------------
# Entity Extraction
# --------------------------------------------------
def extract_entities(text: str, nlp) -> List[Tuple[str, str, int]]:
    """
    Returns list of (canonical_name, label, frequency).
    Deduplicates by normalised key, preserves most-common casing.
    """
    doc = nlp(text[:5000])

    freq: dict      = {}
    canonical: dict = {}

    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        name = ent.text.strip()
        if len(name) < 3 or re.fullmatch(r'[\d\s,\.]+', name):
            continue
        key = name.lower()
        freq[key]      = freq.get(key, 0) + 1
        if key not in canonical:
            canonical[key] = (name, ent.label_)

    sorted_keys = sorted(freq, key=lambda k: -freq[k])[:MAX_ENTITIES]
    return [(canonical[k][0], canonical[k][1], freq[k]) for k in sorted_keys]


# --------------------------------------------------
# Scoring  (frequency + type weighted)
# --------------------------------------------------
def compute_entity_score(results: List[EntityResult]) -> float:
    if not results:
        return 50.0

    score = 50.0
    for r in results:
        weight = ENTITY_WEIGHT.get(r.entity_type, 1.0) * min(r.frequency, 3)

        if r.status == "verified":
            delta = SCORE_VERIFIED * weight
        elif r.status == "not_found":
            delta = SCORE_NOT_FOUND * weight
        elif r.status == "ambiguous":
            delta = SCORE_AMBIGUOUS * weight
        else:
            delta = 0.0

        r.score_impact = round(delta, 1)
        score         += delta

    return max(0.0, min(100.0, round(score, 1)))


# --------------------------------------------------
# Main Pipeline  (concurrent lookups)
# --------------------------------------------------
def check_entities(text: str, nlp=None) -> EntityCheckReport:
    report = EntityCheckReport()

    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            report.error = "spaCy not installed or model missing"
            return report

    raw_entities = extract_entities(text, nlp)

    if not raw_entities:
        report.summary = "No named entities found."
        return report

    report.total_checked = len(raw_entities)

    results_map: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_name = {
            pool.submit(query_entity, name): (name, label, freq)
            for name, label, freq in raw_entities
        }
        for future in as_completed(future_to_name):
            name, label, freq = future_to_name[future]
            try:
                wiki = future.result()
            except Exception as exc:
                wiki = {"status": "error", "summary": str(exc),
                        "url": None, "wikidata_found": False}
            results_map[name] = (label, freq, wiki)

    for name, label, freq in raw_entities:
        label, freq, wiki = results_map[name]
        result = EntityResult(
            name           = name,
            entity_type    = label,
            frequency      = freq,
            confidence     = ENTITY_WEIGHT.get(label, 1.0),
            status         = wiki["status"],
            summary        = wiki.get("summary"),
            wiki_url       = wiki.get("url"),
            wikidata_found = wiki.get("wikidata_found", False),
        )
        report.entities.append(result)

        if result.status == "verified":
            report.verified_count  += 1
        elif result.status == "not_found":
            report.not_found_count += 1
        elif result.status == "ambiguous":
            report.ambiguous_count += 1

    report.entity_score = compute_entity_score(report.entities)
    report.summary = (
        f"{report.verified_count} verified, "
        f"{report.not_found_count} not found, "
        f"{report.ambiguous_count} ambiguous · "
        f"Entity score: {report.entity_score}/100"
    )
    return report