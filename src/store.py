"""SQLite storage layer holding facts, lifecycle, provenance, the conversation
log, and the entity registry.

Vectors are stored as float32 BLOBs; similarity search is numpy brute-force over
the active set, which is sub-millisecond at this scale. All operations are ACID
via the stdlib sqlite3 module.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable

import numpy as np

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,
    subject        TEXT NOT NULL,
    predicate      TEXT,
    object         TEXT,
    content        TEXT NOT NULL,
    embedding      BLOB,
    confidence     REAL DEFAULT 1.0,
    salience       REAL DEFAULT 0.5,
    status         TEXT NOT NULL DEFAULT 'active',   -- active | superseded | expired
    superseded_by  INTEGER REFERENCES memory(id),
    supersede_reason TEXT,
    source_turn    INTEGER,
    polarity       TEXT DEFAULT 'affirm',            -- affirm | negate
    temporal       TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    last_accessed_at TEXT,
    access_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memory_subject ON memory(subject);
CREATE INDEX IF NOT EXISTS idx_memory_status  ON memory(status);
CREATE INDEX IF NOT EXISTS idx_memory_kind    ON memory(kind);

CREATE TABLE IF NOT EXISTS turns (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    role      TEXT NOT NULL,            -- user | assistant
    content   TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entities (
    subject   TEXT PRIMARY KEY,         -- canonical: user, user.sister, persona
    label     TEXT,                     -- human label
    aliases   TEXT,                     -- JSON list of surface forms
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# --- Conversation log (working-memory source and provenance) -------------
def log_turn(conn: sqlite3.Connection, role: str, content: str) -> int:
    cur = conn.execute(
        "INSERT INTO turns(role, content) VALUES (?, ?)", (role, content)
    )
    conn.commit()
    return cur.lastrowid


def recent_turns(conn: sqlite3.Connection, n: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT * FROM turns ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return list(reversed(rows))


def turn_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM turns").fetchone()["c"]


# --- Memory CRUD ---------------------------------------------------------
def _vec_to_blob(vec: Iterable[float] | None) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(list(vec), dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def add_memory(conn: sqlite3.Connection, mem: dict[str, Any]) -> int:
    cur = conn.execute(
        """INSERT INTO memory
           (kind, subject, predicate, object, content, embedding,
            confidence, salience, status, source_turn, polarity, temporal)
           VALUES (:kind, :subject, :predicate, :object, :content, :embedding,
                   :confidence, :salience, :status, :source_turn, :polarity, :temporal)""",
        {
            "kind": mem["kind"],
            "subject": mem["subject"],
            "predicate": mem.get("predicate"),
            "object": mem.get("object"),
            "content": mem["content"],
            "embedding": _vec_to_blob(mem.get("embedding")),
            "confidence": mem.get("confidence", 1.0),
            "salience": mem.get("salience", 0.5),
            "status": mem.get("status", "active"),
            "source_turn": mem.get("source_turn"),
            "polarity": mem.get("polarity", "affirm"),
            "temporal": mem.get("temporal"),
        },
    )
    conn.commit()
    return cur.lastrowid


def active_memories(
    conn: sqlite3.Connection,
    *,
    kinds: tuple[str, ...] | None = None,
    subject: str | None = None,
) -> list[sqlite3.Row]:
    q = "SELECT * FROM memory WHERE status = 'active'"
    params: list[Any] = []
    if kinds:
        q += f" AND kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    if subject:
        q += " AND subject = ?"
        params.append(subject)
    return list(conn.execute(q, params).fetchall())


def supersede(
    conn: sqlite3.Connection, old_id: int, new_id: int, reason: str
) -> None:
    conn.execute(
        """UPDATE memory
           SET status='superseded', superseded_by=?, supersede_reason=?,
               updated_at=datetime('now')
           WHERE id=?""",
        (new_id, reason, old_id),
    )
    conn.commit()


def update_memory(
    conn: sqlite3.Connection,
    mem_id: int,
    *,
    content: str | None = None,
    predicate: str | None = None,
    object: str | None = None,
    embedding: Iterable[float] | None = None,
    confidence: float | None = None,
) -> None:
    """Refine an existing memory in place."""
    sets, params = [], []
    if content is not None:
        sets.append("content=?"); params.append(content)
    if predicate is not None:
        sets.append("predicate=?"); params.append(predicate)
    if object is not None:
        sets.append("object=?"); params.append(object)
    if embedding is not None:
        sets.append("embedding=?"); params.append(_vec_to_blob(embedding))
    if confidence is not None:
        sets.append("confidence=?"); params.append(confidence)
    if not sets:
        return
    sets.append("updated_at=datetime('now')")
    params.append(mem_id)
    conn.execute(f"UPDATE memory SET {', '.join(sets)} WHERE id=?", params)
    conn.commit()


def reinforce(conn: sqlite3.Connection, mem_id: int, delta: float = 0.1) -> None:
    """Bump salience (capped) and access count when a fact is re-stated."""
    conn.execute(
        """UPDATE memory
           SET salience = MIN(1.0, salience + ?),
               access_count = access_count + 1,
               last_accessed_at = datetime('now'),
               updated_at = datetime('now')
           WHERE id=?""",
        (delta, mem_id),
    )
    conn.commit()


def expire(conn: sqlite3.Connection, mem_id: int, reason: str = "expired") -> None:
    conn.execute(
        "UPDATE memory SET status='expired', supersede_reason=?, updated_at=datetime('now') WHERE id=?",
        (reason, mem_id),
    )
    conn.commit()


def touch_access(conn: sqlite3.Connection, ids: Iterable[int]) -> None:
    ids = list(ids)
    if not ids:
        return
    conn.executemany(
        """UPDATE memory
           SET access_count = access_count + 1,
               last_accessed_at = datetime('now')
           WHERE id = ?""",
        [(i,) for i in ids],
    )
    conn.commit()


def all_memories(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM memory ORDER BY id").fetchall())


def expire_past_episodic(conn: sqlite3.Connection, today: str) -> int:
    """Retire dated events whose date has passed.

    `temporal` is free-form, so only values that parse as an ISO date
    (YYYY-MM-DD) strictly before `today` are expired; ambiguous values are left
    untouched.
    """
    rows = conn.execute(
        """SELECT id, temporal FROM memory
           WHERE status='active' AND kind='user_episodic' AND temporal IS NOT NULL"""
    ).fetchall()
    n = 0
    for r in rows:
        t = (r["temporal"] or "")[:10]
        if len(t) == 10 and t[4] == "-" and t[7] == "-" and t < today:
            expire(conn, r["id"], reason=f"event date {t} passed")
            n += 1
    return n


# --- meta helpers (e.g. record which persona.yaml version is seeded) ------
def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
