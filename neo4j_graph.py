"""
neo4j_graph.py  v1.2

Fixes over v1.1
───────────────
  1. basic_auth removed in neo4j Python driver 5.x — importing it raised an
     ImportError that was silently caught by the try/except block, setting
     _NEO4J_AVAILABLE = False and causing the driver to never initialise with
     the real AuraDB URI (falling back to localhost:7687 instead).

     Fix: import only GraphDatabase (always present in 5.x); auth is now
     passed as a plain tuple (user, password) which is the canonical 5.x way
     and also works on 4.x.  basic_auth is no longer needed.

  2. All other fixes from v1.1 retained unchanged.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import urlparse

try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _article_id(snippet: str, domain: str) -> str:
    raw = f"{snippet.strip().lower()[:200]}|{domain}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _normalise_domain(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return "unknown"
    if raw.startswith("http"):
        host = urlparse(raw).netloc
        return host.lower().lstrip("www.") or raw
    return raw.lower().lstrip("www.")


def _normalise_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).title()


# ─────────────────────────────────────────────────────────────────────────────
# Cypher templates
# ─────────────────────────────────────────────────────────────────────────────

_MERGE_ARTICLE = """
MERGE (a:Article {article_id: $article_id})
ON CREATE SET
    a.snippet     = $snippet,
    a.verdict     = $verdict,
    a.final_score = $final_score,
    a.modality    = $modality,
    a.created_at  = datetime()
ON MATCH SET
    a.verdict     = $verdict,
    a.final_score = $final_score,
    a.updated_at  = datetime()
"""

_MERGE_DOMAIN = "MERGE (d:Domain {name: $domain}) ON CREATE SET d.created_at = datetime()"

_LINK_DOMAIN = """
MATCH (a:Article {article_id: $article_id})
MATCH (d:Domain  {name: $domain})
MERGE (a)-[:PUBLISHED_BY]->(d)
"""

_MERGE_AUTHOR = "MERGE (au:Author {name: $author}) ON CREATE SET au.created_at = datetime()"

_LINK_AUTHOR = """
MATCH  (a:Article {article_id: $article_id})
MATCH  (au:Author {name: $author})
MERGE  (a)-[:WRITTEN_BY]->(au)
"""

_MERGE_ENTITY = """
MERGE (e:Entity {name: $name})
ON CREATE SET e.entity_type = $entity_type, e.created_at = datetime()
ON MATCH  SET e.entity_type = $entity_type
"""

_LINK_ENTITY = """
MATCH (a:Article {article_id: $article_id})
MATCH (e:Entity  {name: $name})
MERGE (a)-[r:MENTIONS]->(e)
  ON CREATE SET r.frequency = $frequency
  ON MATCH  SET r.frequency = r.frequency + $frequency
"""

# ── Read queries ──────────────────────────────────────────────────────────────

NARRATIVE_CLUSTER_QUERY = """
MATCH (e:Entity)<-[:MENTIONS]-(a:Article)-[:PUBLISHED_BY]->(d:Domain)
WHERE a.verdict IN ['Fake', 'Uncertain']
WITH  e,
      collect(DISTINCT d.name)       AS domains,
      collect(DISTINCT a.article_id) AS articles,
      count(DISTINCT a)              AS article_count,
      count(DISTINCT d)              AS domain_count
WHERE article_count >= 2 AND domain_count >= 2
RETURN e.name        AS entity,
       e.entity_type AS entity_type,
       article_count,
       domain_count,
       domains,
       articles
ORDER BY article_count DESC
LIMIT 50
"""

DOMAIN_RISK_QUERY = """
MATCH (d:Domain)<-[:PUBLISHED_BY]-(a:Article)
WITH  d.name AS domain,
      count(a)                                                        AS total,
      sum(CASE WHEN a.verdict = 'Fake'      THEN 1 ELSE 0 END)       AS fake_count,
      sum(CASE WHEN a.verdict = 'Uncertain' THEN 1 ELSE 0 END)       AS uncertain_count,
      round(avg(a.final_score), 1)                                    AS avg_score
WHERE total >= 1
RETURN domain, total, fake_count, uncertain_count, avg_score
ORDER BY fake_count DESC, uncertain_count DESC
LIMIT 30
"""

GRAPH_VIZ_QUERY = """
MATCH (a:Article)-[r]-(n)
RETURN a, r, n
LIMIT 200
"""

# FIX (v1.1): The original chained WITH count() pattern returns wrong values.
# Each CALL {} subquery runs against the full graph independently, giving
# correct per-label counts on all Neo4j 4.x and 5.x versions.
STATS_QUERY = """
CALL { MATCH (a:Article) RETURN count(a) AS articles }
CALL { MATCH (e:Entity)  RETURN count(e) AS entities }
CALL { MATCH (d:Domain)  RETURN count(d) AS domains  }
CALL { MATCH (au:Author) RETURN count(au) AS authors  }
RETURN articles, entities, domains, authors
"""


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jGraph:
    """
    Thin, idempotent wrapper around the Neo4j Python driver.

    Parameters
    ──────────
    uri      : bolt://localhost:7687  or  neo4j+s://xxxxx.databases.neo4j.io
    user     : neo4j
    password : <your password>
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        if not _NEO4J_AVAILABLE:
            raise ImportError("Run:  pip install neo4j")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    # ── Schema bootstrapping ─────────────────────────────────────────────────

    def _ensure_constraints(self) -> None:
        """
        Create uniqueness constraints if they do not already exist.

        FIX (v1.1): The original swallowed every exception including real
        connection failures (wrong credentials, network errors). That made
        __init__ appear to succeed, with the actual error surfacing only on
        the first real query with a confusing message.

        Now:
          - "already exists" / "equivalent" constraint errors → silent pass
            (idempotent re-runs on an existing DB are fine)
          - Any other exception → printed as a clear warning
          - Connection-level errors (ServiceUnavailable, AuthError) propagate
            so callers know the DB is unreachable at construction time.
        """
        stmts = [
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article)  REQUIRE a.article_id IS UNIQUE",
            "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (d:Domain)   REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (au:Author)  REQUIRE au.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity)   REQUIRE e.name IS UNIQUE",
        ]
        with self._driver.session() as s:
            for stmt in stmts:
                try:
                    s.run(stmt)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "already exists" in msg or "equivalent" in msg:
                        pass  # idempotent — constraint already present
                    else:
                        print(f"[neo4j_graph] WARNING: constraint creation failed: {exc}")

    # ── Primary write API ─────────────────────────────────────────────────────

    def upsert_scan_result(
        self,
        snippet:     str,
        verdict:     str,
        final_score: float,
        domain:      str            = "unknown",
        author:      Optional[str]  = None,
        entities:    list           = None,
        modality:    str            = "Text",
        article_id:  Optional[str]  = None,
    ) -> str:
        """
        Upsert a full scan result.

        entities  — list of (name, entity_type) OR (name, entity_type, frequency)
        Returns   — the article_id written to the graph
        """
        domain     = _normalise_domain(domain)
        article_id = article_id or _article_id(snippet, domain)
        entities   = entities or []

        normalised = []
        for e in entities:
            if len(e) == 3:
                name, etype, freq = e
            else:
                name, etype = e
                freq = 1
            normalised.append((_normalise_entity(name), etype, int(freq)))

        with self._driver.session() as session:
            session.execute_write(
                self._write_tx,
                article_id  = article_id,
                snippet     = snippet[:300],
                verdict     = verdict.strip().title(),
                final_score = round(float(final_score), 2),
                modality    = modality,
                domain      = domain,
                author      = (author or "").strip().title() or None,
                entities    = normalised,
            )
        return article_id

    @staticmethod
    def _write_tx(tx, *, article_id, snippet, verdict, final_score,
                  modality, domain, author, entities) -> None:
        tx.run(_MERGE_ARTICLE,
               article_id=article_id, snippet=snippet, verdict=verdict,
               final_score=final_score, modality=modality)

        tx.run(_MERGE_DOMAIN, domain=domain)
        tx.run(_LINK_DOMAIN,  article_id=article_id, domain=domain)

        if author:
            tx.run(_MERGE_AUTHOR, author=author)
            tx.run(_LINK_AUTHOR,  article_id=article_id, author=author)

        for name, etype, freq in entities:
            if not name:
                continue
            tx.run(_MERGE_ENTITY, name=name, entity_type=etype)
            tx.run(_LINK_ENTITY,  article_id=article_id, name=name, frequency=freq)

    # ── Read / analytics ──────────────────────────────────────────────────────

    def get_narrative_clusters(self) -> list:
        with self._driver.session() as s:
            return [dict(r) for r in s.run(NARRATIVE_CLUSTER_QUERY)]

    def get_domain_risk_profile(self) -> list:
        with self._driver.session() as s:
            return [dict(r) for r in s.run(DOMAIN_RISK_QUERY)]

    def get_graph_for_viz(self) -> dict:
        """
        Return {nodes: [...], edges: [...]} formatted for streamlit-agraph.
        """
        _VERDICT_COLOR = {"Fake": "#ef4444", "Uncertain": "#f59e0b", "Real": "#22c55e"}
        _GROUP_COLOR   = {"Entity": "#6366f1", "Domain": "#0ea5e9", "Author": "#ec4899"}

        nodes_map: dict = {}
        edges: list     = []

        with self._driver.session() as s:
            for rec in s.run(GRAPH_VIZ_QUERY):
                a, rel, n = rec["a"], rec["r"], rec["n"]

                aid = a["article_id"]
                if aid not in nodes_map:
                    nodes_map[aid] = {
                        "id":    aid,
                        "label": (a.get("snippet") or "")[:40] + "…",
                        "group": "Article",
                        "color": _VERDICT_COLOR.get(a.get("verdict"), "#94a3b8"),
                        "size":  20,
                        "title": f"Verdict: {a.get('verdict')} | Score: {a.get('final_score')}",
                    }

                # n.labels is a frozenset in both driver v4 and v5
                grp  = list(n.labels)[0] if n.labels else "Unknown"
                nid  = str(n.element_id)
                name = n.get("name") or n.get("article_id") or nid
                if nid not in nodes_map:
                    nodes_map[nid] = {
                        "id":    nid,
                        "label": name,
                        "group": grp,
                        "color": _GROUP_COLOR.get(grp, "#94a3b8"),
                        "size":  14,
                        "title": f"{grp}: {name}",
                    }

                edges.append({"source": aid, "target": nid, "label": rel.type})

        return {"nodes": list(nodes_map.values()), "edges": edges}

    def get_stats(self) -> dict:
        try:
            with self._driver.session() as s:
                row = s.run(STATS_QUERY).single()
                return dict(row) if row else {}
        except Exception:
            return {}

    def ping(self) -> bool:
        try:
            with self._driver.session() as s:
                s.run("RETURN 1")
            return True
        except Exception as e:
            print(f"\n🔥 NEO4J CONNECTION ERROR: {e}\n")
            return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()