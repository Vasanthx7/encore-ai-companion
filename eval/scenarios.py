"""Synthetic test conversations that deliberately exercise the memory + persona
requirements. Each turn may carry `checks` evaluated after the companion replies.

A check is a dict with a `kind`:
  Deterministic (query the DB — trustworthy, judge-independent):
    - db_active_has     {text}   an ACTIVE memory's content contains text (case-insensitive)
    - db_active_missing {text}   NO active memory contains text (e.g. stale fact retired)
    - db_retired_has    {text}   a superseded/expired memory contains text
  Judged (LLM-as-judge — subjective, see judge.py):
    - reply_recall      {truth}  the reply correctly conveys the ground-truth fact
    - reply_no_leak     {stale}  the reply does NOT treat `stale` as still true
    - reply_persona     {canon}  the reply does NOT contradict this canon opinion
    - reply_tone        {}       the reply stays in-character (not a generic assistant)

Each check maps to a scoring category via CATEGORY.
"""

from __future__ import annotations

# Split by trustworthiness: "(DB)" checks are deterministic and judge-independent;
# "(judged)" checks rely on the LLM-as-judge and are only indicative.
CATEGORY = {
    "db_active_has": "recall (DB, deterministic)",
    "db_active_missing": "contradiction (DB, deterministic)",
    "db_retired_has": "contradiction (DB, deterministic)",
    "reply_recall": "recall (judged)",
    "reply_no_leak": "no-leak (judged)",
    "reply_persona": "persona (judged)",
    "reply_tone": "tone (judged)",
}

# Small talk used to create distance between planting a fact and probing it.
_FILLER = [
    "anyway, the weather here has been really nice lately",
    "had a pretty long day today, kind of tired",
    "i tried a new recipe last night, turned out okay",
    "the traffic this morning was rough",
    "been listening to a lot of music this week",
    "not much else going on, just the usual",
]


def _filler(n: int) -> list[dict]:
    return [{"user": _FILLER[i % len(_FILLER)]} for i in range(n)]


SCENARIOS: list[dict] = [
    {
        "name": "recall_and_job_contradiction",
        "description": "Plant facts, probe long-range recall, then contradict the job and re-probe.",
        "turns": [
            {"user": "hey kai, i'm dhanush and i work at amazon as a data engineer"},
            {"user": "my sister priya just started med school"},
            {"user": "i love hiking on weekends but i can't stand crowded places"},
            *_filler(6),
            {"user": "quick check — where did i say i work?",
             "checks": [{"kind": "reply_recall", "truth": "The user works at Amazon."},
                        {"kind": "db_active_has", "text": "amazon"}]},
            {"user": "actually big news, i switched jobs — i'm at google now",
             "checks": [{"kind": "db_active_has", "text": "google"},
                        {"kind": "db_active_missing", "text": "amazon"},
                        {"kind": "db_retired_has", "text": "amazon"}]},
            *_filler(2),
            {"user": "remind me where i work these days?",
             "checks": [{"kind": "reply_recall", "truth": "The user now works at Google."},
                        {"kind": "reply_no_leak", "stale": "The user still works at Amazon."}]},
            {"user": "and what's my sister up to again?",
             "checks": [{"kind": "reply_recall", "truth": "The user's sister Priya is in medical school."}]},
        ],
    },
    {
        "name": "relationship_contradiction",
        "description": "Plant a relationship, contradict it (breakup), ensure it isn't recalled as current.",
        "turns": [
            {"user": "i've been dating someone named alex for about two years now"},
            *_filler(5),
            {"user": "alex and i broke up last week, been feeling pretty low",
             "checks": [{"kind": "db_retired_has", "text": "alex"}]},
            *_filler(2),
            {"user": "am i seeing anyone right now?",
             "checks": [{"kind": "reply_no_leak", "stale": "The user is currently in a relationship with Alex."}]},
        ],
    },
    {
        "name": "persona_consistency_under_pressure",
        "description": "Probe persona opinions early and late, with a technical-pressure turn between.",
        "turns": [
            {"user": "what kind of movies are you into, kai?",
             "checks": [{"kind": "reply_persona", "canon": "Kai dislikes horror films."},
                        {"kind": "reply_tone"}]},
            *_filler(4),
            {"user": "random question — can you explain how a vector database does similarity search?",
             "checks": [{"kind": "reply_tone"}]},
            *_filler(3),
            {"user": "so would you ever watch a horror film with me?",
             "checks": [{"kind": "reply_persona", "canon": "Kai dislikes horror films."}]},
            {"user": "beach trip or a mountain trip — which are you picking?",
             "checks": [{"kind": "reply_persona", "canon": "Kai prefers mountains to beaches."}]},
            {"user": "hot take: pineapple belongs on pizza, right?",
             "checks": [{"kind": "reply_persona", "canon": "Kai dislikes pineapple on pizza."}]},
        ],
    },
]
