"""Canonical entity registry.

Contradiction detection gathers memories that share a subject, so subjects must
be stable across turns ("her" / "Priya" / "my sister" -> one canonical
`user.sister`). Rather than a separate resolver, the known-subject list is fed
into the extraction prompt so the model reuses existing canonical names. This
module persists that registry and records new subjects the extractor introduces.

Subjects are dotted and namespaced: `user`, `user.sister`, `user.employer`,
`user.ex_partner`, `persona`.
"""

from __future__ import annotations

import json
import sqlite3


def known_subjects(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT subject FROM entities ORDER BY subject").fetchall()
    return [r["subject"] for r in rows]


def describe_known(conn: sqlite3.Connection) -> str:
    """Compact listing for the extraction prompt: 'subject (label)'."""
    rows = conn.execute(
        "SELECT subject, label FROM entities ORDER BY subject"
    ).fetchall()
    if not rows:
        return "(none yet)"
    return ", ".join(
        f"{r['subject']}" + (f" ({r['label']})" if r["label"] else "") for r in rows
    )


def register(
    conn: sqlite3.Connection,
    subject: str,
    label: str | None = None,
    alias: str | None = None,
) -> None:
    """Insert a subject if new, or fold in a fresh alias/label if seen before."""
    row = conn.execute(
        "SELECT subject, label, aliases FROM entities WHERE subject=?", (subject,)
    ).fetchone()
    if row is None:
        aliases = json.dumps([alias]) if alias else "[]"
        conn.execute(
            "INSERT INTO entities(subject, label, aliases) VALUES (?, ?, ?)",
            (subject, label, aliases),
        )
    else:
        aliases = json.loads(row["aliases"] or "[]")
        changed = False
        if alias and alias not in aliases:
            aliases.append(alias)
            changed = True
        new_label = label or row["label"]
        if changed or new_label != row["label"]:
            conn.execute(
                "UPDATE entities SET label=?, aliases=? WHERE subject=?",
                (new_label, json.dumps(aliases), subject),
            )
    conn.commit()


def ensure(conn: sqlite3.Connection, subjects: list[str]) -> None:
    """Register any subjects introduced by extraction that we haven't seen."""
    existing = set(known_subjects(conn))
    for s in subjects:
        if s and s not in existing:
            register(conn, s)
            existing.add(s)
