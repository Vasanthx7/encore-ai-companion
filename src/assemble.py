"""Prompt assembly — the three tiers.

1. System spine    — persona canon + behavioural rules (always pinned).
2. Long-term memory — the gated, retrieved facts, as compact bullets.
3. Working memory   — the last N turns verbatim (added by the caller from the DB).

Recent context is kept verbatim while older context is retrieved as facts, rather
than packing all history into the window. The memory block is deliberately lean to
favour relevance.
"""

from __future__ import annotations

import sqlite3

import config
from src import retrieval, store

_BEHAVIOUR = (
    "\n\nUsing your memory: the notes below are what you remember about the user. "
    "Weave relevant ones in naturally as a friend would — do NOT recite them as a "
    "list or announce that you 'remember'. If nothing is relevant, just talk."
)


def _memory_block(scored: list[retrieval.Scored]) -> str:
    if not scored:
        return ""
    lines = ["\n\nWHAT YOU REMEMBER ABOUT THE USER:"]
    for s in scored:
        turn = s.row["source_turn"]
        prov = f" (learned turn {turn})" if turn else ""
        lines.append(f"- {s.row['content']}{prov}")
    return "\n".join(lines)


def _persona_block(persona_stated: list[retrieval.Scored]) -> str:
    """Inject the persona's prior improvised opinions to prevent self-contradiction.

    Canon already lives in the spine and is omitted here.
    """
    if not persona_stated:
        return ""
    lines = ["\n\nYOU HAVE ALREADY TOLD THE USER (stay consistent with these):"]
    lines.extend(f"- {s.row['content']}" for s in persona_stated)
    return "\n".join(lines)


def build(
    conn: sqlite3.Connection,
    spine: str,
    scored: list[retrieval.Scored],
    persona_stated: list[retrieval.Scored] | None = None,
) -> list[dict[str, str]]:
    """Assemble the full message list for a chat completion."""
    system = spine + _BEHAVIOUR + _memory_block(scored) + _persona_block(persona_stated or [])
    messages = [{"role": "system", "content": system}]
    for row in store.recent_turns(conn, config.WORKING_MEMORY_TURNS):
        messages.append({"role": row["role"], "content": row["content"]})
    return messages
