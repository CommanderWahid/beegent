"""Cross-run memory: what previous runs tried, and the links they verified."""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Protocol

from beegent import config
from beegent.embedding import calibrate, dot, from_blob, make_embedder, to_blob
from beegent.schemas import DiscoveryRun
from beegent.web_tools import normalize_url

_log = logging.getLogger(__name__)

#: One embedding call per chunk. A model change re-embeds everything, so this is the batch.
EMBED_BATCH = 256

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_uid         TEXT PRIMARY KEY,
  ts              TEXT NOT NULL,
  country         TEXT NOT NULL,
  use_case        TEXT NOT NULL,
  status          TEXT NOT NULL,
  critic_decision TEXT,
  critic_note     TEXT,
  tried           TEXT NOT NULL,
  tokens          INTEGER NOT NULL,
  -- The use_case vector, so a later run can ask for the runs that match ITS request
  -- rather than merely the country's most recent. Nullable exactly like links.embedding.
  embedding       BLOB,
  embed_model     TEXT
);
CREATE INDEX IF NOT EXISTS runs_country_ts ON runs(country, ts DESC);

CREATE TABLE IF NOT EXISTS links (
  link_uid          TEXT PRIMARY KEY,
  run_uid           TEXT NOT NULL REFERENCES runs(run_uid),
  country           TEXT NOT NULL,
  dataset           TEXT NOT NULL,
  url               TEXT NOT NULL,
  resource_url      TEXT NOT NULL,
  resource_url_norm TEXT NOT NULL,
  verification      TEXT NOT NULL,
  claim             TEXT NOT NULL,
  confidence        REAL,
  -- Nullable on purpose: NULL means not embedded yet, and a link with no vector is
  -- still a verified link. Vectors compare only within one model, so the name rides along.
  embedding         BLOB,
  embed_model       TEXT,
  last_verified     TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS links_country ON links(country);
CREATE INDEX IF NOT EXISTS links_stale   ON links(last_verified);
CREATE INDEX IF NOT EXISTS links_norm    ON links(resource_url_norm);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Store(Protocol):
    """What run.py needs from a store - the only seam between the pipeline and persistence."""

    def record_run(self, run: DiscoveryRun, tried: list[dict]) -> None: ...

    def prior_runs(self, country: str, limit: int, use_case: str = "") -> list[dict]: ...

    def verified_links(self, country: str) -> list[dict]: ...

    def mark_link(self, link_uid: str, status: str) -> None: ...

    def sync_embeddings(self) -> dict: ...

    def measure_relevance(self) -> dict: ...


class NullStore:
    """The default: no persistence, so the CLI stays standalone and tests stay offline."""

    def record_run(self, run: DiscoveryRun, tried: list[dict]) -> None:
        return None

    def prior_runs(self, country: str, limit: int, use_case: str = "") -> list[dict]:
        return []

    def verified_links(self, country: str) -> list[dict]:
        return []

    def mark_link(self, link_uid: str, status: str) -> None:
        return None

    def sync_embeddings(self) -> dict:
        return {}

    def measure_relevance(self) -> dict:
        return {}


class SqliteStore:
    """One file, no server. sqlite3 is stdlib, so this adds no dependency."""

    def __init__(self, path: str, embed=None):
        self.path = path
        # Injected like WebTools' transport: None writes NULL vectors, which is what keeps
        # the suite offline and what a machine without the extra installed does anyway.
        self.embed = embed
        # SQLite will not create parent directories, and setting BEEGENT_DB is deliberate:
        # a missing folder should not silently downgrade the whole store to a NullStore.
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _vectors(self, texts: list[str]) -> list:
        """One batched call per run. A failure costs the vectors, never the links."""
        if not self.embed or not texts:
            return [None] * len(texts)
        try:
            return self.embed(texts)
        except Exception as exc:
            _log.info(f"[embed] not stored ({type(exc).__name__}: {exc})")
            return [None] * len(texts)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
        return db

    def record_run(self, run: DiscoveryRun, tried: list[dict]) -> None:
        """One transaction: the run, then every link it verified."""
        run_uid = str(uuid.uuid4())
        ts = _now()
        # links is a log of DISCOVERIES: re-recording a cache hit would reset its
        # freshness, and the window would then never expire.
        linkable = [c for c in run.candidates
                    if c.resource_url and c.source != "catalog"]
        texts = [c.dataset or c.title for c in linkable]
        # One batched call for the run and every link it verified, not two.
        run_vec, *vectors = self._vectors([run.use_case] + texts)
        model = getattr(self.embed, "name", None) if self.embed else None
        with self._connect() as db:
            db.execute(
                "INSERT INTO runs (run_uid, ts, country, use_case, status, critic_decision, "
                "critic_note, tried, tokens, embedding, embed_model) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_uid, ts, run.country, run.use_case, run.status,
                 run.critic_decision, run.critic_note, json.dumps(tried, ensure_ascii=False),
                 run.totals.get("total_tokens", 0),
                 to_blob(run_vec) if run_vec else None, model if run_vec else None),
            )
            for cand, text, vec in zip(linkable, texts, vectors):
                db.execute(
                    "INSERT INTO links (link_uid, run_uid, country, dataset, url, "
                    "resource_url, resource_url_norm, verification, claim, confidence, "
                    "embedding, embed_model, last_verified, status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'ok')",
                    (str(uuid.uuid4()), run_uid, run.country, text, cand.url,
                     cand.resource_url, normalize_url(cand.resource_url),
                     json.dumps(cand.verification or {}, ensure_ascii=False),
                     json.dumps(cand.claim, ensure_ascii=False), cand.confidence,
                     to_blob(vec) if vec else None, model if vec else None, ts),
                )

    def _relevant(self, rows: list, use_case: str) -> list:
        """Relevance FILTERS; recency still ORDERS, so rows arrive newest-first and stay so."""
        model = getattr(self.embed, "name", None) if self.embed else None
        if not (model and use_case and rows):
            return rows  # cannot filter at all: pure recency, exactly as before
        try:
            query = self.embed([use_case])[0]
        except Exception as exc:
            _log.info(f"[embed] no relevance filter ({type(exc).__name__}: {exc})")
            return rows
        # Being ABLE to filter but finding nothing comparable is not the same as not being able
        # to: an unmatchable row is excluded, never scored, exactly as query_catalogs does.
        usable = [r for r in rows if r["embedding"] and r["embed_model"] == model]
        if not usable:
            _log.info(f"[memory] {len(rows)} run(s) stored, none embedded with {model!r}")
        return [r for r in usable
                if dot(query, from_blob(r["embedding"])) >= config.MEMORY_MIN_RELEVANCE]

    def prior_runs(self, country: str, limit: int, use_case: str = "") -> list[dict]:
        """The country's runs that match THIS use case - not merely its most recent ones."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT ts, use_case, status, critic_decision, critic_note, tried, embedding, "
                # rowid breaks a tie: two runs in the same instant must still order.
                "embed_model FROM runs WHERE country = ? ORDER BY ts DESC, rowid DESC LIMIT ?",
                # Scan well past what is returned, so the filter has room to reject.
                (country, limit * 10),
            ).fetchall()
        return [{k: v for k, v in dict(r).items() if k not in ("embedding", "embed_model")}
                | {"tried": json.loads(r["tried"])}
                for r in self._relevant(rows, use_case)[:limit]]

    def verified_links(self, country: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT link_uid, dataset, url, resource_url, claim, confidence, "
                "embedding, embed_model, last_verified FROM (SELECT *, ROW_NUMBER() OVER ("
                "  PARTITION BY resource_url ORDER BY last_verified DESC, rowid DESC) AS rn "
                "  FROM links WHERE country = ?) "
                # rn = 1 is the newest row for that URL; only then does its status count.
                "WHERE rn = 1 AND status = 'ok' ORDER BY last_verified DESC",
                (country,),
            ).fetchall()
        return [dict(r) | {"claim": json.loads(r["claim"])} for r in rows]

    def catalog(self, country: str = "") -> list[dict]:
        """Every link's NEWEST row, rot included - for READING, not for matching.

        verified_links() is the matching input: it drops rot and carries the vectors. This
        carries `verification` and `status` instead, which is what a reader needs to see.
        """
        where = "WHERE country = ?" if country else ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT link_uid, country, dataset, url, resource_url, verification, claim, "
                "confidence, last_verified, status FROM (SELECT *, ROW_NUMBER() OVER ("
                "  PARTITION BY resource_url ORDER BY last_verified DESC, rowid DESC) AS rn "
                f"  FROM links {where}) "
                # Same newest-row-wins rule; the status is shown rather than filtered on.
                "WHERE rn = 1 ORDER BY last_verified DESC",
                (country,) if country else (),
            ).fetchall()
        return [dict(r) | {"claim": json.loads(r["claim"]),
                           "verification": json.loads(r["verification"])} for r in rows]

    def embedded_links(self, model: str) -> list[dict]:
        """Every link comparable against THIS model - filtered in SQL, never in Python."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT country, dataset, embedding FROM links "
                "WHERE embedding IS NOT NULL AND embed_model = ?", (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def embedded_runs(self, model: str) -> list[dict]:
        """Every run comparable against THIS model - filtered in SQL, never in Python."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT use_case, embedding FROM runs "
                "WHERE embedding IS NOT NULL AND embed_model = ?", (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def measure_relevance(self) -> dict:
        """What the STORED data implies each threshold should be. Measures, never applies."""
        model = getattr(self.embed, "name", None) if self.embed else None
        if not model:
            return {}
        # The queries are what this user actually asks for, never a hardcoded list.
        runs = self.embedded_runs(model)
        queries = [from_blob(r["embedding"]) for r in runs]
        if not queries:
            return {}

        links = [from_blob(h["embedding"]) for h in self.embedded_links(model)]
        # A run scores 1.0 against itself by construction, so every band would start there.
        memory = [[dot(q, o) for j, o in enumerate(queries) if j != i]
                  for i, q in enumerate(queries)]
        out = {}
        for name, scored in (("catalog", [[dot(q, v) for v in links] for q in queries]),
                             ("memory", memory)):
            # band() needs two scores to find a gap between them; one is not a boundary.
            if scored and len(scored[0]) >= 2 and (agreed := calibrate(scored)):
                out[name] = agreed
        return out

    def links_needing_embedding(self, model: str) -> list[dict]:
        """Never embedded, or embedded with another model - both are unmatchable today."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT link_uid, dataset FROM links "
                "WHERE embedding IS NULL OR embed_model IS NOT ?", (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def runs_needing_embedding(self, model: str) -> list[dict]:
        """Changing EMBED_MODEL invalidates every vector at once; this is the recovery."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT run_uid, use_case FROM runs "
                "WHERE embedding IS NULL OR embed_model IS NOT ?", (model,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_run_embedding(self, run_uid: str, vec: list, model: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE runs SET embedding = ?, embed_model = ? WHERE run_uid = ?",
                       (to_blob(vec), model, run_uid))

    def set_embedding(self, link_uid: str, vec: list, model: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE links SET embedding = ?, embed_model = ? WHERE link_uid = ?",
                       (to_blob(vec), model, link_uid))

    def sync_embeddings(self) -> dict:
        """Re-embed whatever the CURRENT model cannot compare. No model means no change at all."""
        model = getattr(self.embed, "name", None) if self.embed else None
        if not model:
            return {}  # nothing to compare against: leave every stored vector exactly as it is
        done = {}
        for kind, pending, text, uid, write in (
            ("runs", self.runs_needing_embedding, "use_case", "run_uid", self.set_run_embedding),
            ("links", self.links_needing_embedding, "dataset", "link_uid", self.set_embedding),
        ):
            todo = pending(model)
            for start in range(0, len(todo), EMBED_BATCH):
                chunk = todo[start:start + EMBED_BATCH]
                for row, vec in zip(chunk, self.embed([r[text] for r in chunk])):
                    write(row[uid], vec, model)
            if todo:
                done[kind] = len(todo)
        return done

    def mark_link(self, link_uid: str, status: str) -> None:
        """Rot, recorded where it was found - the re-probe is the only thing that can see it."""
        with self._connect() as db:
            db.execute("UPDATE links SET status = ?, last_verified = ? WHERE link_uid = ?",
                       (status, _now(), link_uid))


def make_store(path: str | None = None, embed=None) -> Store:
    """A real store only when a path is configured - persistence is opt-in, never a surprise."""
    path = path if path is not None else config.BEEGENT_DB
    if not path:
        return NullStore()
    try:
        return SqliteStore(path, embed if embed is not None else make_embedder())
    except Exception as exc:  # a broken store must not cost a run
        _log.info(f"[store] disabled ({type(exc).__name__}: {exc})")
        return NullStore()
