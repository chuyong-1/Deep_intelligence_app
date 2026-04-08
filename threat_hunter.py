"""
threat_hunter.py  v1

Proactive Threat Hunter — background agent for the Verity OSINT Dashboard.

What it does
────────────
1. Uses duckduckgo-search to find trending articles matching "Zero-Day"
   keywords (deepfake, misinformation, propaganda, synthetic media, etc.).
2. Fetches the full article text via requests / trafilatura.
3. Runs the Verity AI detector pipeline (stylistic features + ML model).
4. Saves articles whose verdict is "Uncertain" (score 35–65) into
   staging_data/<ISO-date>/<slug>.json for the Analyst Review tab.
5. Optionally writes every scanned article into the Neo4j graph.

Run modes
─────────
  # One-shot scan (CI / cron job):
  python threat_hunter.py --once

  # Continuous loop (every N minutes):
  python threat_hunter.py --interval 30

  # Dry-run (print matches, no files written):
  python threat_hunter.py --once --dry-run

Configuration
─────────────
  All knobs live in the CONFIG section below. Override at runtime via
  environment variables — e.g. HUNTER_INTERVAL=60 python threat_hunter.py.

Dependencies
────────────
  pip install duckduckgo-search trafilatura requests joblib scikit-learn
  Optional: neo4j  (set HUNTER_NEO4J_URI to enable graph writes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# ── Third-party (graceful fallbacks where possible) ───────────────────────────
try:
    from duckduckgo_search import DDGS
    _DDG_OK = True
except ImportError:
    _DDG_OK = False
    print("[threat_hunter] WARNING: duckduckgo-search not installed.  "
          "pip install duckduckgo-search")

try:
    import trafilatura
    _TRAFILATURA_OK = True
except ImportError:
    _TRAFILATURA_OK = False

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    import joblib
    import numpy as np
    _ML_OK = True
except ImportError:
    _ML_OK = False

# ── Project imports ───────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

try:
    from features import stylistic_features, STYLE_DIM
    _FEATURES_OK = True
except ImportError:
    _FEATURES_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (all overridable via env vars)
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Staging output directory
    "STAGING_DIR": Path(os.environ.get("HUNTER_STAGING_DIR", "staging_data")),

    # ML artefacts produced by train_model.py
    "MODEL_PATH":  Path(os.environ.get("HUNTER_MODEL",  "model/model.pkl")),
    "SCALER_PATH": Path(os.environ.get("HUNTER_SCALER", "model/scaler.pkl")),

    # Neo4j (optional — leave empty to skip graph writes)
    "NEO4J_URI":  os.environ.get("HUNTER_NEO4J_URI",  ""),
    "NEO4J_USER": os.environ.get("HUNTER_NEO4J_USER", "neo4j"),
    "NEO4J_PASS": os.environ.get("HUNTER_NEO4J_PASS", ""),

    # Scan behaviour
    "MAX_RESULTS_PER_QUERY": int(os.environ.get("HUNTER_MAX_RESULTS", "5")),
    "MIN_TEXT_LEN":          int(os.environ.get("HUNTER_MIN_TEXT",    "300")),
    "REQUEST_TIMEOUT":       int(os.environ.get("HUNTER_TIMEOUT",     "10")),
    "SLEEP_BETWEEN_FETCHES": float(os.environ.get("HUNTER_SLEEP",    "1.5")),
    "INTERVAL_MINUTES":      int(os.environ.get("HUNTER_INTERVAL",   "30")),

    # Score band saved to staging (0–100 scale)
    "UNCERTAIN_LO": float(os.environ.get("HUNTER_UNCERTAIN_LO", "35")),
    "UNCERTAIN_HI": float(os.environ.get("HUNTER_UNCERTAIN_HI", "65")),
}

# ── Zero-Day keyword queries ───────────────────────────────────────────────────
HUNT_QUERIES = [
    "deepfake news misinformation",
    "AI-generated disinformation article",
    "synthetic media propaganda",
    "fake news viral social media",
    "coordinated inauthentic behaviour",
    "misinformation health vaccine",
    "election interference fake account",
    "manipulated video deepfake politician",
    "fabricated quote celebrity",
    "state-sponsored disinformation campaign",
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [threat_hunter] %(levelname)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("threat_hunter")


# ─────────────────────────────────────────────────────────────────────────────
# Model loader  (cached — loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL:  object = None
_SCALER: object = None


def _load_model() -> tuple:
    global _MODEL, _SCALER
    if _MODEL is None:
        mp, sp = CONFIG["MODEL_PATH"], CONFIG["SCALER_PATH"]
        if mp.exists() and sp.exists() and _ML_OK:
            _MODEL  = joblib.load(mp)
            _SCALER = joblib.load(sp)
            log.info("ML model loaded from %s", mp)
        else:
            log.warning("ML model not found at %s — using heuristic scorer.", mp)
    return _MODEL, _SCALER


# ─────────────────────────────────────────────────────────────────────────────
# Score an article
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_score(text: str) -> float:
    """
    Fallback scorer (no ML model) — mirrors the stylistic signal logic
    in features.py but returns a 0–100 credibility score directly.
    Lower = more suspicious.
    """
    if not _FEATURES_OK:
        return 50.0

    feats = stylistic_features(text)
    # Rough mapping:  alarm high → low score,  source high → high score
    alarm   = feats[6]   # index 6 = alarm_score
    source  = feats[5]   # index 5 = source_score
    caps    = feats[0]   # index 0 = caps_ratio
    exclaim = feats[2]   # index 2 = exclaim_ratio

    raw = 50.0 - (alarm * 8) + (source * 5) - (caps * 30) - (exclaim * 20)
    return max(0.0, min(100.0, raw))


def score_text(text: str) -> tuple[float, str]:
    """
    Returns (credibility_score 0–100, verdict).
    Uses ML model when available, heuristic otherwise.
    """
    model, scaler = _load_model()

    if model is not None and _FEATURES_OK and _ML_OK:
        feats = stylistic_features(text).reshape(1, -1)
        # The trained model only uses stylistic features when no BERT embeddings
        # are available at inference time — pad to the model's expected width.
        try:
            feats_scaled = scaler.transform(feats)
            # Model predicts P(fake) — convert to credibility score
            p_fake = model.predict_proba(feats_scaled)[0][1]
            score  = round((1 - p_fake) * 100, 1)
        except Exception as exc:
            log.debug("Model inference error: %s", exc)
            score = _heuristic_score(text)
    else:
        score = _heuristic_score(text)

    lo, hi = CONFIG["UNCERTAIN_LO"], CONFIG["UNCERTAIN_HI"]
    if score >= hi:
        verdict = "Real"
    elif score <= lo:
        verdict = "Fake"
    else:
        verdict = "Uncertain"

    return score, verdict


# ─────────────────────────────────────────────────────────────────────────────
# Article fetching
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_text(url: str) -> Optional[str]:
    """
    Fetch and extract article text.
    Prefers trafilatura (best boilerplate removal); falls back to requests.
    """
    if _TRAFILATURA_OK:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False,
                                           include_tables=False)
                if text and len(text) >= CONFIG["MIN_TEXT_LEN"]:
                    return text
        except Exception as exc:
            log.debug("trafilatura failed for %s: %s", url, exc)

    if _REQUESTS_OK:
        try:
            headers = {"User-Agent": "VerityThreatHunter/1.0 (research bot)"}
            resp = requests.get(url, timeout=CONFIG["REQUEST_TIMEOUT"],
                                headers=headers)
            resp.raise_for_status()
            # Very naive extractor — strips HTML tags
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= CONFIG["MIN_TEXT_LEN"]:
                return text[:20_000]
        except Exception as exc:
            log.debug("requests failed for %s: %s", url, exc)

    return None


def _slug(url: str) -> str:
    """URL → safe filename slug."""
    parsed = urlparse(url)
    path   = re.sub(r"[^a-z0-9]+", "-", (parsed.path + parsed.query).lower())
    return (path.strip("-") or "article")[:60]


def _article_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Staging output
# ─────────────────────────────────────────────────────────────────────────────

def _save_to_staging(record: dict) -> Path:
    date_dir = CONFIG["STAGING_DIR"] / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    fname    = f"{_article_hash(record['url'])}_{_slug(record['url'])}.json"
    out_path = date_dir / fname

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return out_path


def _already_staged(url: str) -> bool:
    """Avoid re-staging the same URL within the same day."""
    date_dir = CONFIG["STAGING_DIR"] / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fname    = f"{_article_hash(url)}_*.json"
    return bool(list(date_dir.glob(f"{_article_hash(url)}_*.json"))) \
        if date_dir.exists() else False


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j writer  (optional)
# ─────────────────────────────────────────────────────────────────────────────

_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None and CONFIG["NEO4J_URI"]:
        try:
            from neo4j_graph import Neo4jGraph
            _GRAPH = Neo4jGraph(
                CONFIG["NEO4J_URI"],
                CONFIG["NEO4J_USER"],
                CONFIG["NEO4J_PASS"],
            )
            log.info("Connected to Neo4j at %s", CONFIG["NEO4J_URI"])
        except Exception as exc:
            log.warning("Neo4j connection failed: %s", exc)
    return _GRAPH


def _write_to_graph(record: dict) -> None:
    graph = _get_graph()
    if graph is None:
        return
    try:
        domain = urlparse(record["url"]).netloc.lstrip("www.")
        graph.upsert_scan_result(
            snippet     = record.get("snippet", record["text"][:80]),
            verdict     = record["verdict"],
            final_score = record["score"],
            domain      = domain,
            modality    = "Text",
        )
    except Exception as exc:
        log.warning("Graph write failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Core scan loop
# ─────────────────────────────────────────────────────────────────────────────

def run_hunt(dry_run: bool = False) -> list[dict]:
    """
    Execute one full hunt cycle.  Returns list of staged records.
    """
    if not _DDG_OK:
        log.error("duckduckgo-search not installed — cannot hunt.")
        return []

    staged  : list[dict] = []
    seen_urls: set[str]  = set()

    log.info("🎯 Threat Hunter starting — %d queries", len(HUNT_QUERIES))

    with DDGS() as ddg:
        for query in HUNT_QUERIES:
            log.info("  🔍 Searching: %r", query)
            try:
                results = list(ddg.news(
                    keywords = query,
                    max_results = CONFIG["MAX_RESULTS_PER_QUERY"],
                ))
            except Exception as exc:
                log.warning("DDG search failed for %r: %s", query, exc)
                continue

            for item in results:
                url   = item.get("url") or item.get("link", "")
                title = item.get("title", "")

                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                if _already_staged(url):
                    log.debug("  ⏭  Already staged: %s", url)
                    continue

                log.info("  📰 Fetching: %s", url)
                time.sleep(CONFIG["SLEEP_BETWEEN_FETCHES"])

                text = _fetch_text(url)
                if not text:
                    log.debug("  ✗  Could not extract text from %s", url)
                    continue

                score, verdict = score_text(text)
                snippet        = text.replace("\n", " ")[:80]

                log.info(
                    "  %-12s  score=%.1f  verdict=%s  (%s)",
                    verdict, score, title[:50], url[:60],
                )

                record = {
                    "url":         url,
                    "title":       title,
                    "text":        text[:5000],
                    "snippet":     snippet,
                    "score":       score,
                    "verdict":     verdict,
                    "query":       query,
                    "fetched_at":  datetime.now(timezone.utc).isoformat(),
                    "source":      "threat_hunter",
                }

                # Always write to Neo4j (all verdicts, for graph completeness)
                if not dry_run:
                    _write_to_graph(record)

                # Only stage Uncertain articles for analyst review
                lo, hi = CONFIG["UNCERTAIN_LO"], CONFIG["UNCERTAIN_HI"]
                if lo <= score <= hi:
                    if not dry_run:
                        out = _save_to_staging(record)
                        log.info("  💾 Staged → %s", out)
                    else:
                        log.info("  [DRY-RUN] Would stage: %s", url)
                    staged.append(record)

    log.info(
        "✅ Hunt complete — %d URLs scanned, %d staged as Uncertain",
        len(seen_urls), len(staged),
    )
    return staged


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verity Proactive Threat Hunter")
    p.add_argument("--once",     action="store_true",
                   help="Run one scan cycle and exit")
    p.add_argument("--interval", type=int, default=CONFIG["INTERVAL_MINUTES"],
                   help="Minutes between scans in continuous mode (default: 30)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print matches but do not write files or update graph")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.once:
        run_hunt(dry_run=args.dry_run)
    else:
        interval = args.interval
        log.info("🔄 Continuous mode — scanning every %d minutes. Ctrl-C to stop.", interval)
        while True:
            run_hunt(dry_run=args.dry_run)
            log.info("💤 Sleeping %d minutes…", interval)
            time.sleep(interval * 60)