"""Extract structured facts and a retrieval plan from a user turn.

Returns memory-worthy facts and the retrieval plan (query + entities) in a single
call, so query-planning adds no latency. Extraction is grounded (only what the
utterance states or strongly implies), deterministic (temperature 0), and
shape-guaranteed via native structured output.

Storage (embed + insert) lives in `ingest()`, which reconciles each surviving
candidate against existing memories (duplicate/supersede/refine) before insert.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

import config
from src import embeddings, entities, llm, store

# --- Validation model ----------------------------------------------------
class Candidate(BaseModel):
    kind: str
    subject: str
    predicate: str = ""
    object: str = ""
    content: str
    confidence: float = 0.7
    salience: float = 0.5
    temporal: str | None = None
    polarity: str = "affirm"

    @field_validator("confidence", "salience")
    @classmethod
    def _clamp(cls, v: float) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @field_validator("kind")
    @classmethod
    def _valid_kind(cls, v: str) -> str:
        return v if v in config.USER_KINDS else "user_semantic"

    @field_validator("polarity")
    @classmethod
    def _valid_polarity(cls, v: str) -> str:
        return "negate" if str(v).lower().startswith("neg") else "affirm"


class Extraction(BaseModel):
    memories: list[Candidate] = Field(default_factory=list)
    retrieval_query: str = ""
    retrieval_entities: list[str] = Field(default_factory=list)


# JSON schema handed to the model (constrained decoding). Kept permissive;
# Pydantic does the real validation/coercion after parsing.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(config.USER_KINDS)},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "content": {"type": "string"},
                    "confidence": {"type": "number"},
                    "salience": {"type": "number"},
                    "temporal": {"type": "string"},
                    "polarity": {"type": "string", "enum": ["affirm", "negate"]},
                },
                "required": ["kind", "subject", "content", "confidence", "salience"],
            },
        },
        "retrieval_query": {"type": "string"},
        "retrieval_entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["memories", "retrieval_query", "retrieval_entities"],
}

_SYSTEM = """You extract durable, memory-worthy facts about the USER from their message, \
and plan memory retrieval for the reply. You are a precise information extractor, not a chatbot.

WHAT COUNTS AS MEMORY-WORTHY (capture): stable attributes (job, location, relationships, \
health), preferences and opinions, ongoing situations and plans, significant dated events, \
and recurring emotional patterns. Assign:
- kind: user_semantic (stable facts) | user_preference (likes/dislikes/opinions) | \
user_episodic (dated events) | user_state (volatile mood/situation)
- content: a COMPLETE standalone sentence that names its subject, e.g. "The user's \
sister Priya is in medical school" — NEVER a bare value like "Priya" or "Bangalore". \
Refer to the person as "the user".
- confidence: how certain the fact is (0..1)
- salience: how important/reusable it is (0..1)
- polarity: "negate" for retractions/negations ("I don't...", "not anymore"), else "affirm"
- temporal: an event date/time if stated, else omit

WHAT TO IGNORE: pleasantries, meta ("thanks", "can you repeat"), throwaway remarks with no \
reuse value, and anything the assistant said. When unsure about a low-value remark, DROP it. \
Ground every fact in what the user actually said — never infer or invent.

CRITICAL — QUESTIONS AND REQUESTS ARE NOT FACTS. If the message is a question or request \
directed at the companion — including questions about the user's OWN information ("where do I \
work?", "what do you remember about my sister?") or about the companion ("do you like scary \
movies?") — extract NO memories (return an empty list), and only fill in the retrieval plan. \
NEVER invent "absence" facts like "the user's sister is not mentioned". Examples:
  MESSAGE: "where do I work now?"                 -> memories: []
  MESSAGE: "do you like scary movies?"            -> memories: []
  MESSAGE: "what do you remember about my sister?"-> memories: []
  MESSAGE: "I just started a job at Google"        -> memories: [that fact]

ENTITY SUBJECTS: reuse the EXISTING canonical subjects below when the message refers to them. \
Only mint a new dotted subject (e.g. user.sister, user.employer) when none fits. The user \
themself is "user".

RETRIEVAL PLAN: also produce
- retrieval_query: a standalone, de-anaphorized paraphrase of what the user is really talking \
about (resolve "that", "it", "still" using context) — used to fetch relevant past memories
- retrieval_entities: canonical subjects relevant to this turn

Return JSON only."""


def _user_prompt(conn, recent: list[sqlite3.Row], user_text: str) -> str:
    known = entities.describe_known(conn)
    ctx = "\n".join(f"{r['role']}: {r['content']}" for r in recent) or "(none)"
    return (
        f"EXISTING CANONICAL SUBJECTS: {known}\n\n"
        f"RECENT CONVERSATION (for context resolution only):\n{ctx}\n\n"
        f"USER MESSAGE TO EXTRACT FROM:\n{user_text}"
    )


def extract(conn, user_text: str) -> Extraction:
    """Run the structured pass. Never raises — returns an empty Extraction on
    any model/parse failure so the chat loop stays alive."""
    recent = store.recent_turns(conn, config.WORKING_MEMORY_TURNS)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _user_prompt(conn, recent, user_text)},
    ]
    try:
        raw = llm.structured(messages, _SCHEMA, schema_name="extraction")
        extr = Extraction.model_validate(raw)
        extr.retrieval_entities = [canon_subject(e) for e in extr.retrieval_entities]
        return extr
    except (ValidationError, ValueError, KeyError, Exception):  # noqa: BLE001
        return Extraction()


def canon_subject(subject: str) -> str:
    """Fold surface variants onto canonical subjects so same-subject grouping
    (used by retrieval entity-match and contradiction detection) is stable."""
    s = (subject or "").strip().lower().replace("’", "'")
    s = s.replace("the user's ", "user.").replace("user's ", "user.")
    s = s.replace(" ", "_")
    # Strip a leading article now that spaces are underscores.
    if s.startswith("the_"):
        s = s[4:]
    if s in ("", "user", "users", "me", "i", "myself", "user_self"):
        return "user"
    if s in ("you", "kai", "persona", "assistant"):
        return "persona"
    # Guard against the model dumping a whole clause as a subject
    # (e.g. "user.friend_john_is_a_firefighter_in_chicago"). Keep the namespace
    # plus a single token; fall back to "user" if there's nothing sane left.
    if len(s) > 40 or s.count("_") > 3:
        if "." in s:
            ns, tail = s.split(".", 1)
            first = tail.split("_")[0]
            return f"{ns}.{first}" if first else ns
        return "user"
    return s


# Absence / non-fact phrasings the model sometimes emits from questions — these
# must never become memories (they otherwise trigger bogus supersessions).
_ABSENCE_MARKERS = (
    "not mentioned", "no longer mentioned", "not specified", "is unknown",
    "not stated", "not provided", "no information", "does not mention",
    "is not mentioned", "unclear", "not sure what",
)


def _looks_like_non_fact(content: str) -> bool:
    c = content.lower()
    return any(m in c for m in _ABSENCE_MARKERS)


def _readable_subject(subject: str) -> str:
    if subject == "user":
        return "The user"
    if subject.startswith("user."):
        return "The user's " + subject.split(".", 1)[1].replace("_", " ")
    if subject == "persona":
        return "You"
    return subject.replace("_", " ")


def _normalize_content(c: Candidate) -> str:
    """Ensure `content` is a full sentence; terse values like 'Priya' embed and
    read poorly."""
    content = (c.content or "").strip()
    is_terse = len(content.split()) < 3 or content.lower() == (c.object or "").strip().lower()
    if is_terse and (c.predicate or c.object):
        pred = (c.predicate or "").replace("_", " ").strip()
        parts = [_readable_subject(c.subject), pred, (c.object or "").strip()]
        content = " ".join(p for p in parts if p).strip()
    return content


def _exact_dup_exists(conn, content: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM memory WHERE status='active' AND content=? LIMIT 1",
        (content.strip(),),
    ).fetchone()
    return row is not None


@dataclass
class IngestResult:
    inserted: list[int] = field(default_factory=list)
    superseded: list[int] = field(default_factory=list)
    refined: list[int] = field(default_factory=list)
    reinforced: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> int:
        return len(self.inserted) + len(self.superseded) + len(self.refined)


def _insert(conn, c: "Candidate", vec, source_turn: int) -> int:
    return store.add_memory(
        conn,
        {
            "kind": c.kind,
            "subject": c.subject,
            "predicate": c.predicate or None,
            "object": c.object or None,
            "content": c.content.strip(),
            "embedding": vec,
            "confidence": c.confidence,
            "salience": c.salience,
            "status": "active",
            "source_turn": source_turn,
            "polarity": c.polarity,
            "temporal": c.temporal,
        },
    )


def ingest(conn, extraction: Extraction, source_turn: int) -> IngestResult:
    """Reconcile and store surviving candidates.

    Each candidate is classified against its neighbours and applied as
    duplicate / supersede / refine / novel, updating and retiring existing
    memories rather than accumulating duplicates.
    """
    from src import reconcile  # local import to avoid import cycle

    result = IngestResult()

    keepers: list[Candidate] = []
    for c in extraction.memories:
        if c.confidence < config.CONFIDENCE_FLOOR or c.salience < config.SALIENCE_FLOOR:
            continue
        c.subject = canon_subject(c.subject)
        c.content = _normalize_content(c)
        if not c.content.strip():
            continue
        if _looks_like_non_fact(c.content):  # drop bogus "not mentioned" facts
            continue
        keepers.append(c)

    entities.ensure(
        conn, [c.subject for c in keepers] + list(extraction.retrieval_entities)
    )
    if not keepers:
        return result

    vecs = embeddings.embed([c.content for c in keepers])
    for c, vec in zip(keepers, vecs):
        neighbors = reconcile._neighbors(conn, c.subject, vec, config.RECONCILE_NEIGHBORS)
        decision = reconcile.classify(
            conn, new_content=c.content, polarity=c.polarity, neighbors=neighbors
        )

        # Cross-subject guard. Subjects sometimes drift for the same real-world
        # attribute ("user" vs "user.employer"), so identical subjects can't be
        # required. But cross-subject refine/duplicate is almost always a classifier
        # error that would corrupt an unrelated fact in place. For a different-subject
        # target, allow only supersede (safe: the old row is kept as an audit trail)
        # and only above a coarse similarity floor; everything else downgrades to novel.
        if decision.relation != "novel" and decision.target_id:
            tgt = next((n for n in neighbors if n["id"] == decision.target_id), None)
            if tgt is not None and tgt["subject"] != c.subject:
                import numpy as _np

                tv = store.blob_to_vec(tgt["embedding"])
                # No vector on either side (embeddings disabled/unreachable) means
                # similarity can't be verified; sim=0.0 conservatively downgrades
                # to novel below rather than risking a wrong cross-subject supersede.
                sim = float(_np.dot(vec, tv)) if (vec is not None and tv is not None) else 0.0
                if decision.relation != "supersede" or sim < config.RECONCILE_CROSS_SUBJECT_MIN:
                    decision.relation = "novel"
                    decision.target_id = 0

        if decision.relation == "duplicate" and decision.target_id:
            store.reinforce(conn, decision.target_id)
            result.reinforced.append(decision.target_id)

        elif decision.relation == "supersede" and decision.target_id:
            if c.polarity == "negate":
                # Retraction with no replacement: retire the old fact, add nothing.
                store.supersede(conn, decision.target_id, decision.target_id, decision.reason)
                result.superseded.append(decision.target_id)
                result.notes.append(f"retired #{decision.target_id}: {decision.reason}")
            else:
                new_id = _insert(conn, c, vec, source_turn)
                store.supersede(conn, decision.target_id, new_id, decision.reason)
                result.inserted.append(new_id)
                result.superseded.append(decision.target_id)
                result.notes.append(
                    f"#{decision.target_id} → #{new_id}: {decision.reason}"
                )

        elif decision.relation == "refine" and decision.target_id:
            store.update_memory(
                conn,
                decision.target_id,
                content=c.content.strip(),
                predicate=c.predicate or None,
                object=c.object or None,
                embedding=vec,
                confidence=max(c.confidence, 0.0),
            )
            result.refined.append(decision.target_id)

        else:  # novel (or duplicate/supersede/refine with no valid target)
            new_id = _insert(conn, c, vec, source_turn)
            result.inserted.append(new_id)

    return result
