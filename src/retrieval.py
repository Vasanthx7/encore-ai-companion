"""Hybrid memory retrieval.

Every active user memory is scored by a linear combination of semantic
similarity, entity match, recency, salience, and confidence, then gated to keep
the injected set relevant without dropping clearly-relevant facts.

At this scale (hundreds of facts) the whole active set is brute-force scored each
turn. Filtering on `status='active'` ensures superseded facts never surface.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

import config
from src import embeddings, store


@dataclass
class Scored:
    row: sqlite3.Row
    score: float
    cosine: float
    entity_match: bool

    @property
    def id(self) -> int:
        return self.row["id"]


def _age_days(created_at: str) -> float:
    try:
        t = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - t).total_seconds() / 86400.0)


def _recency(kind: str, created_at: str) -> float:
    half = config.HALF_LIFE_DAYS.get(kind, 365)
    return math.exp(-_age_days(created_at) / half)


def retrieve(
    conn: sqlite3.Connection,
    query: str,
    entities_in_turn: list[str],
    *,
    kinds: tuple[str, ...] = config.USER_KINDS,
    top_k: int | None = None,
) -> list[Scored]:
    """Return the gated, ranked memories to inject for this turn."""
    top_k = top_k or config.RETRIEVAL_TOP_K
    rows = store.active_memories(conn, kinds=kinds)
    if not rows:
        return []

    qvec = embeddings.embed_one(query) if query.strip() else None
    ent_set = set(entities_in_turn or [])
    w = config.RANK_WEIGHTS

    scored: list[Scored] = []
    for r in rows:
        mvec = store.blob_to_vec(r["embedding"])
        cosine = float(np.dot(qvec, mvec)) if (qvec is not None and mvec is not None) else 0.0
        entity_match = r["subject"] in ent_set
        score = (
            w["cosine"] * cosine
            + w["entity_match"] * (1.0 if entity_match else 0.0)
            + w["recency"] * _recency(r["kind"], r["created_at"])
            + w["salience"] * (r["salience"] or 0.0)
            + w["confidence"] * (r["confidence"] or 0.0)
        )
        scored.append(Scored(r, score, cosine, entity_match))

    # Gate 1: similarity floor, bypassed on an entity match so an exact-entity
    # hit with a weak vector is still retained. Skipped entirely when no query
    # vector exists (embeddings disabled/unreachable) — every row's cosine is 0
    # in that case, so the floor would otherwise filter out everything.
    if qvec is None:
        gated = list(scored)
    else:
        gated = [
            s for s in scored
            if s.cosine >= config.SIMILARITY_FLOOR or s.entity_match
        ]
    gated.sort(key=lambda s: s.score, reverse=True)

    # Gate 2: per-kind budget (cap volatile kinds) + Gate 3: global top-k.
    kept: list[Scored] = []
    per_kind: dict[str, int] = {}
    for s in gated:
        cap = config.PER_KIND_BUDGET.get(s.row["kind"])
        used = per_kind.get(s.row["kind"], 0)
        if cap is not None and used >= cap:
            continue
        kept.append(s)
        per_kind[s.row["kind"]] = used + 1
        if len(kept) >= top_k:
            break

    # Recalled facts gain access count, reinforcing salience over time.
    store.touch_access(conn, [s.id for s in kept])
    return kept
