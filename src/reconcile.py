"""Decide what a new fact means relative to existing memories.

For each new fact, gather a small set of candidate neighbours (same canonical
subject, plus the nearest semantic matches) and ask the model a narrow question:
how does this new fact relate to each? The model reasons only over the ≤5
pre-fetched neighbours, never the whole store.

Four outcomes:
  - duplicate → don't insert; reinforce the existing memory
  - supersede → retire the old (status=superseded, superseded_by, reason); insert new
  - refine    → update the existing memory in place with the richer detail
  - novel     → insert as a new memory

A negation with a supersede target (e.g. "I quit my job") retires the old fact and
inserts nothing new. Supersession is soft: retired rows stay in the DB for
auditability.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

import config
from src import store

RelationSchema = {
    "type": "object",
    "properties": {
        "relation": {
            "type": "string",
            "enum": ["duplicate", "supersede", "refine", "novel"],
        },
        "target_id": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["relation", "target_id", "reason"],
}

_SYSTEM = """You maintain a memory store about a user. Given ONE new fact and a numbered \
list of existing facts, decide how the new fact relates to them. Choose exactly one relation.

CRITICAL RULE: relations apply ONLY between facts about the SAME topic/attribute. A fact \
about a job NEVER supersedes a fact about a relationship. A fact about a sister NEVER \
supersedes a fact about a job. If the new fact is about a DIFFERENT topic than every \
existing fact, the answer is "novel" — even if they are about the same person.

- "novel": the new fact is about a different topic than all existing facts (THIS IS THE \
DEFAULT — choose it unless one existing fact is clearly about the same attribute).
- "duplicate": an existing fact already states the same thing (no new information).
- "supersede": the new fact makes an existing fact about the SAME attribute FALSE \
(e.g. new job replaces OLD job; "broke up with Alex" replaces "dating Alex"; "moved to \
Delhi" replaces old city). The old fact must be about the same attribute.
- "refine": the new fact ADDS DETAIL to an existing fact about the SAME thing without \
contradicting it (e.g. "has a sister" -> "sister is named Priya").

Examples:
  NEW: "The user works at Google"  EXISTING: [5] "The user works at Amazon"  -> supersede 5
  NEW: "The user works at Google"  EXISTING: [5] "The user is dating Alex"    -> novel (0)
  NEW: "The user broke up with Alex" EXISTING: [5] "The user is dating Alex"  -> supersede 5
  NEW: "The user broke up with Alex" EXISTING: [5] "The user works at Amazon" -> novel (0)
  NEW: "Her name is Priya"          EXISTING: [5] "The user has a sister"     -> refine 5

Set target_id to the relevant existing fact id, or 0 for novel. One-line reason. JSON only."""


@dataclass
class Decision:
    relation: str
    target_id: int
    reason: str


def _neighbors(
    conn: sqlite3.Connection,
    subject: str,
    vector: np.ndarray | None,
    limit: int,
) -> list[sqlite3.Row]:
    """Same-subject active memories first, then fill with nearest semantic matches."""
    same = store.active_memories(conn, subject=subject)
    picked = {r["id"]: r for r in same}

    if vector is not None:
        pool = store.active_memories(conn, kinds=config.USER_KINDS)
        scored = []
        for r in pool:
            if r["id"] in picked:
                continue
            mv = store.blob_to_vec(r["embedding"])
            if mv is None:
                continue
            scored.append((float(np.dot(vector, mv)), r))
        scored.sort(key=lambda t: t[0], reverse=True)
        for sim, r in scored:
            if len(picked) >= limit:
                break
            if sim >= config.RECONCILE_SIM_FLOOR:  # stricter than retrieval floor
                picked[r["id"]] = r

    return list(picked.values())[:limit]


def classify(
    conn: sqlite3.Connection,
    *,
    new_content: str,
    polarity: str,
    neighbors: list[sqlite3.Row],
) -> Decision:
    """One narrow structured call over the pre-fetched neighbours."""
    if not neighbors:
        return Decision("novel", 0, "no related facts")

    listing = "\n".join(f"[{r['id']}] {r['content']}" for r in neighbors)
    neg = " (this is a NEGATION / retraction)" if polarity == "negate" else ""
    user = (
        f"NEW FACT{neg}: {new_content}\n\n"
        f"EXISTING FACTS:\n{listing}\n\n"
        "How does the new fact relate?"
    )
    from src import llm  # local import to avoid cycle at module load

    try:
        raw = llm.structured(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            RelationSchema,
            schema_name="relation",
        )
        rel = str(raw.get("relation", "novel"))
        if rel not in ("duplicate", "supersede", "refine", "novel"):
            rel = "novel"
        tid = int(raw.get("target_id", 0) or 0)
        # Guard: target must be one of the offered neighbours.
        valid_ids = {r["id"] for r in neighbors}
        if rel != "novel" and tid not in valid_ids:
            tid = neighbors[0]["id"]  # fall back to the closest neighbour
        return Decision(rel, tid, str(raw.get("reason", "")))
    except Exception:  # noqa: BLE001 - never break ingestion on a model error
        return Decision("novel", 0, "reconcile failed; stored as novel")
