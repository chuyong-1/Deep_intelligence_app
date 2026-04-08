"""
database.py  v1

Persistent SQLite storage layer for the Verity OSINT Dashboard.

Replaces st.session_state.history with a proper database so analysis
history survives page refreshes, browser restarts, and server reboots.

Thread safety
─────────────
Streamlit runs each user interaction in its own thread. SQLite's default
mode serialises writes with a process-level GIL, which is safe for a
single-user local deployment. For multi-user / Streamlit Cloud use the
module creates one connection per thread via threading.local() — every
call gets its own connection object, eliminating "objects created in a
thread can only be used in that same thread" SQLite errors.

Schema
──────
Table: history
  id          INTEGER  PRIMARY KEY AUTOINCREMENT
  timestamp   TEXT     ISO-8601 UTC  e.g. "2024-05-01T14:32:07"
  modality    TEXT     "Text" | "Image"
  snippet     TEXT     ≤80 chars of article text or image filename
  final_score REAL     0.0 – 100.0
  verdict     TEXT     "Real" | "Uncertain" | "Fake"
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

# The database file sits next to this module. Override by setting the env var
# VERITY_DB_PATH before starting Streamlit.
import os
_DEFAULT_DB = Path(__file__).parent / "verity_history.db"
DB_PATH: str = os.environ.get("VERITY_DB_PATH", str(_DEFAULT_DB))

# ── Thread-local connection pool ──────────────────────────────────────────────
# Each thread gets its own sqlite3.Connection so we never share connection
# objects across threads (SQLite's Python driver forbids this by default).

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """
    Return the sqlite3 connection for the current thread, creating it if needed.
    check_same_thread=False is set as a belt-and-suspenders safety valve; the
    per-thread pool makes it unnecessary in practice but suppresses the warning.
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(
            DB_PATH,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
            timeout=10,           # wait up to 10 s if the DB is locked
        )
        # Use WAL mode: readers don't block writers and vice-versa.
        # This is the single most important SQLite pragma for concurrent use.
        _local.conn.execute("PRAGMA journal_mode=WAL;")
        # Foreign keys off by default in SQLite — enable for future-proofing.
        _local.conn.execute("PRAGMA foreign_keys=ON;")
        _local.conn.row_factory = sqlite3.Row   # rows behave like dicts
    return _local.conn


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the `history` table if it does not already exist.

    Safe to call multiple times (CREATE TABLE IF NOT EXISTS).
    Call once at the very top of app.py, before any Streamlit widgets render.
    """
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            modality    TEXT    NOT NULL CHECK(modality IN ('Text', 'Image')),
            snippet     TEXT    NOT NULL,
            final_score REAL    NOT NULL,
            verdict     TEXT    NOT NULL CHECK(verdict IN ('Real', 'Uncertain', 'Fake'))
        );
    """)
    # Index speeds up the ORDER BY timestamp DESC query used by get_recent_records.
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_timestamp
        ON history (timestamp DESC);
    """)
    conn.commit()


def insert_record(
    modality:    str,
    snippet:     str,
    final_score: float,
    verdict:     str,
) -> None:
    """
    Insert one analysis record into the database.

    Args:
        modality:    "Text" or "Image"
        snippet:     Up to 80 characters of article text or the image filename.
                     Truncation is applied here so callers don't need to worry.
        final_score: Credibility score 0.0–100.0
        verdict:     "Real" | "Uncertain" | "Fake"

    The timestamp is generated server-side in UTC ISO-8601 format so records
    are always comparable regardless of the user's local timezone.
    """
    # Normalise inputs
    modality    = modality.strip().title()          # "text" → "Text"
    verdict     = verdict.strip().title()           # "fake" → "Fake"
    snippet     = snippet.replace("\n", " ").strip()[:80]
    final_score = round(float(final_score), 2)
    timestamp   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    # Guard against invalid enum values silently corrupting the DB
    if modality not in ("Text", "Image"):
        modality = "Text"
    if verdict not in ("Real", "Uncertain", "Fake"):
        verdict = "Uncertain"

    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO history (timestamp, modality, snippet, final_score, verdict)
        VALUES (?, ?, ?, ?, ?);
        """,
        (timestamp, modality, snippet, final_score, verdict),
    )
    conn.commit()


def get_recent_records(limit: int = 20) -> list[dict]:
    """
    Return the most recent `limit` records, newest first.

    Returns:
        List of dicts with keys:
            id, timestamp, modality, snippet, final_score, verdict
        Returns an empty list if the table is empty or on any DB error.
    """
    try:
        conn = _get_conn()
        cursor = conn.execute(
            """
            SELECT id, timestamp, modality, snippet, final_score, verdict
            FROM   history
            ORDER  BY id DESC
            LIMIT  ?;
            """,
            (limit,),
        )
        # sqlite3.Row supports dict(row) but we return plain dicts for
        # compatibility with any code that does item["key"] access.
        return [dict(row) for row in cursor.fetchall()]
    except Exception:
        return []


def clear_history() -> None:
    """
    Delete every row in the history table.

    Uses DELETE (not DROP TABLE) so the schema is preserved and
    new records can be inserted immediately afterwards without
    calling init_db() again.
    """
    conn = _get_conn()
    conn.execute("DELETE FROM history;")
    # Reset the autoincrement counter so IDs restart from 1 after a clear.
    conn.execute("DELETE FROM sqlite_sequence WHERE name='history';")
    conn.commit()


def get_stats() -> dict:
    """
    Return aggregate statistics for the sidebar display.

    Returns dict with keys: total, real_count, fake_count, uncertain_count,
    avg_score.  All values default to 0 / 0.0 if the table is empty.
    """
    try:
        conn   = _get_conn()
        cursor = conn.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN verdict='Real'      THEN 1 ELSE 0 END) AS real_count,
                SUM(CASE WHEN verdict='Fake'      THEN 1 ELSE 0 END) AS fake_count,
                SUM(CASE WHEN verdict='Uncertain' THEN 1 ELSE 0 END) AS uncertain_count,
                ROUND(AVG(final_score), 1)                      AS avg_score
            FROM history;
        """)
        row = cursor.fetchone()
        if row and row["total"]:
            return dict(row)
    except Exception:
        pass
    return {
        "total": 0, "real_count": 0, "fake_count": 0,
        "uncertain_count": 0, "avg_score": 0.0,
    }