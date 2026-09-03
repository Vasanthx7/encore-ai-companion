"""Long-horizon stress scenarios (150 / 200 turns).

Same check vocabulary as scenarios.py (see CATEGORY there), but events are
spaced proportionally across a much longer conversation instead of a handful
of turns. This tests whether recall, contradiction-handling, and persona
consistency hold up over a turn count closer to what the spec asks for
long turns rather than the short smoke-test scenarios.

Run:  python -m eval.legacy.run_long_horizon
"""

from __future__ import annotations

_FILLER = [
    "anyway, the weather here has been really nice lately",
    "had a pretty long day today, kind of tired",
    "i tried a new recipe last night, turned out okay",
    "the traffic this morning was rough",
    "been listening to a lot of music this week",
    "not much else going on, just the usual",
    "watched a documentary last night, pretty interesting",
    "gym session this morning wiped me out",
    "my phone battery has been draining so fast lately",
    "finally cleaned my apartment this weekend",
    "coffee shop near me started doing latte art, kind of neat",
    "been meaning to read more but keep getting distracted",
    "neighbor's dog will not stop barking today",
    "tried a new podcast on the commute, decent so far",
    "grocery prices keep creeping up, it's wild",
    "spent way too long doom-scrolling last night",
    "the wifi at home has been flaky all week",
    "thinking about repainting my room at some point",
    "found a good deal on running shoes online",
    "it's been raining on and off all day",
]


def _filler(n: int) -> list[dict]:
    return [{"user": _FILLER[i % len(_FILLER)]} for i in range(n)]


def _plants() -> list[dict]:
    return [
        {"user": "hey kai, i'm dhanush and i work at amazon as a data engineer"},
        {"user": "my sister priya just started med school"},
        {"user": "i love hiking on weekends but i can't stand crowded places"},
        {"user": "i've been dating someone named alex for about two years now"},
        {"user": "what kind of movies are you into, kai?",
         "checks": [{"kind": "reply_persona", "canon": "Kai dislikes horror films."},
                    {"kind": "reply_tone"}]},
    ]


def _job_probe() -> dict:
    return {"user": "quick check — where did i say i work?",
            "checks": [{"kind": "reply_recall", "truth": "The user works at Amazon."},
                       {"kind": "db_active_has", "text": "amazon"}]}


def _breakup() -> dict:
    return {"user": "alex and i broke up last week, been feeling pretty low",
            "checks": [{"kind": "db_retired_has", "text": "alex"}]}


def _persona_pressure() -> list[dict]:
    return [
        {"user": "random question — can you explain how a vector database does similarity search?",
         "checks": [{"kind": "reply_tone"}]},
        {"user": "beach trip or a mountain trip — which are you picking?",
         "checks": [{"kind": "reply_persona", "canon": "Kai prefers mountains to beaches."}]},
    ]


def _job_switch() -> dict:
    return {"user": "actually big news, i switched jobs — i'm at google now",
            "checks": [{"kind": "db_active_has", "text": "google"},
                       {"kind": "db_active_missing", "text": "amazon"},
                       {"kind": "db_retired_has", "text": "amazon"}]}


def _sister_probe() -> dict:
    return {"user": "and what's my sister up to again?",
            "checks": [{"kind": "reply_recall", "truth": "The user's sister Priya is in medical school."}]}


def _pizza_probe() -> dict:
    return {"user": "hot take: pineapple belongs on pizza, right?",
            "checks": [{"kind": "reply_persona", "canon": "Kai dislikes pineapple on pizza."}]}


def _final_probe() -> list[dict]:
    return [
        {"user": "remind me where i work these days?",
         "checks": [{"kind": "reply_recall", "truth": "The user now works at Google."},
                    {"kind": "reply_no_leak", "stale": "The user still works at Amazon."}]},
        {"user": "am i seeing anyone right now?",
         "checks": [{"kind": "reply_no_leak", "stale": "The user is currently in a relationship with Alex."}]},
        {"user": "so would you ever watch a horror film with me?",
         "checks": [{"kind": "reply_persona", "canon": "Kai dislikes horror films."}]},
        {"user": "what kind of movies are you into again?",
         "checks": [{"kind": "reply_tone"}]},
    ]


def build(total_turns: int) -> dict:
    """Build a scenario padded to ~total_turns turns: plants near the start,
    contradictions/persona-pressure probes spread proportionally through the
    middle (at fixed fractions of the run so distance-from-plant scales with
    total_turns), and a comprehensive probe at the end."""
    plants = _plants()
    final = _final_probe()
    fixed_events = [
        (0.20, [_job_probe()]),
        (0.30, [_breakup()]),
        (0.45, _persona_pressure()),
        (0.60, [_job_switch()]),
        (0.75, [_sister_probe()]),
        (0.88, [_pizza_probe()]),
    ]
    reserved = len(plants) + len(final) + sum(len(t) for _, t in fixed_events)
    if total_turns < reserved + 10:
        raise ValueError(f"total_turns too small for fixed content ({reserved} reserved)")
    filler_budget = total_turns - reserved

    turns = list(plants)
    cursor = len(plants)
    filler_used = 0
    for frac, events in fixed_events:
        target_pos = int(total_turns * frac)
        gap = max(0, target_pos - cursor - len(events))
        gap = min(gap, filler_budget - filler_used)
        turns.extend(_filler(gap))
        filler_used += gap
        cursor += gap
        turns.extend(events)
        cursor += len(events)
    turns.extend(_filler(filler_budget - filler_used))
    turns.extend(final)
    return {
        "name": f"long_horizon_{total_turns}",
        "description": f"Fact plants, contradictions, and persona-pressure probes spread across ~{total_turns} turns.",
        "turns": turns,
    }


SCENARIOS: list[dict] = [build(150), build(200)]
