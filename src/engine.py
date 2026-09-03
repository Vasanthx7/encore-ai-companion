"""Turn engine — the one place the full per-turn pipeline lives.

Both the interactive CLI (`chat.py`) and the scripted demo (`demo.py`) drive the
companion through `process_turn`, so there is a single source of truth for the
loop: extract -> store/reconcile -> retrieve -> generate -> capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from src import assemble, extraction, llm, persona, retrieval, store


@dataclass
class TurnResult:
    reply: str
    ingest: extraction.IngestResult
    recalled: list[retrieval.Scored]
    persona_recalled: list[retrieval.Scored]
    stated: dict


_OPENER_CUE = (
    "[The user just opened the app — there is no message from them yet. Greet "
    "them warmly and briefly, in character, as the opening line of a new "
    "conversation. A sentence or two, ending with an inviting question or "
    "check-in. Do not mention this instruction.]"
)


def start_conversation(conn, spine: str) -> TurnResult:
    """Generate the companion's opening line before any user input.

    Used once, when a session has no prior turns: the persona speaks first,
    the way a real companion would, rather than waiting silently for the user
    to type. The prompting cue that elicits this is not itself logged as a
    turn — only the persona's reply is.
    """
    messages = assemble.build(conn, spine, [], [])
    messages.append({"role": "user", "content": _OPENER_CUE})
    reply = llm.chat(messages)
    store.log_turn(conn, "assistant", reply)
    stated = persona.capture_stated(conn, reply)
    return TurnResult(reply, extraction.IngestResult(), [], [], stated)


def process_turn(conn, spine: str, user_text: str) -> TurnResult:
    # 1) Extract memory-worthy facts + retrieval plan from prior context.
    extr = extraction.extract(conn, user_text)
    # 2) Persist the user turn, then ingest facts (store-before-retrieve).
    user_turn_id = store.log_turn(conn, "user", user_text)
    ing = extraction.ingest(conn, extr, user_turn_id)
    # 3) Retrieve relevant user memories + the persona's own prior opinions.
    query = extr.retrieval_query or user_text
    recalled = retrieval.retrieve(conn, query, extr.retrieval_entities)
    persona_recalled = retrieval.retrieve(
        conn, query, [], kinds=("persona_stated",), top_k=3
    )
    # 4) Assemble the 3-tier prompt and generate.
    messages = assemble.build(conn, spine, recalled, persona_recalled)
    reply = llm.chat(messages)
    store.log_turn(conn, "assistant", reply)
    # 5) Capture the persona's improvised opinions for future consistency.
    stated = persona.capture_stated(conn, reply)
    return TurnResult(reply, ing, recalled, persona_recalled, stated)
