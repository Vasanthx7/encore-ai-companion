"""Scripted, reproducible demo of the companion's memory and persona features.

Runs a fixed 11-turn conversation through the full pipeline in one command,
covering each core behaviour:
  - extraction and storage    (facts stored as they are disclosed)
  - relevant retrieval        (the right memory recalled at the right time)
  - contradiction handling    (job switch and breakup retire the old facts)
  - persona consistency       (Kai's horror opinion holds across the run)
  - long-range recall         (early facts recalled many turns later)

Uses a throwaway DB so it never touches the real companion.db.

Run:  python demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile

# Point at a throwaway DB before importing anything that reads config.
_DB = os.path.join(tempfile.gettempdir(), "companion_demo.db")
if os.path.exists(_DB):
    os.remove(_DB)
os.environ["COMPANION_DB"] = _DB

import config  # noqa: E402
from src import engine, persona, store  # noqa: E402

SCRIPT = [
    "hi kai! i'm dhanush, i work at amazon as a data engineer",
    "my sister priya just started med school, i'm really proud of her",
    "i'm seeing someone named alex, it's going well",
    "honestly i love hiking on weekends but i can't stand crowded places",
    "what kind of movies do you like, kai?",              # persona: dislikes horror
    "work has been really stressful this week with deadlines",
    "actually, big news — i switched jobs, i'm at google now",   # supersede: amazon->google
    "alex and i broke up last week, feeling low about it",       # supersede: dating->ended
    "remind me, where do i work now?",                    # recall: google (not amazon)
    "do you like scary movies? asking again",             # persona consistency check
    "what do you remember about my sister?",              # long-range recall: priya
]


def _ops(res) -> str:
    ing = res.ingest
    bits = []
    if ing.inserted:
        bits.append(f"+{len(ing.inserted)} stored")
    if ing.superseded:
        bits.append(f"{len(ing.superseded)} RETIRED")
    if ing.refined:
        bits.append(f"{len(ing.refined)} refined")
    bits.append(f"recalled {len(res.recalled)}")
    line = "   [mem: " + ", ".join(bits) + "]"
    for note in ing.notes:
        line += f"\n        contradiction: {note}"
    for fl in res.stated.get("flagged", []):
        line += f"\n        persona-flag: {fl}"
    return line


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    conn = store.connect()
    store.init_db(conn)
    p = persona.load()
    spine = persona.render_spine(p)
    n = persona.seed_canon(conn, p)
    print(f"=== DEMO: {p['identity']['name']} — seeded {n} canon memories, model {config.CHAT_MODEL} ===\n")

    for i, user_text in enumerate(SCRIPT, 1):
        print(f"[{i:>2}] you › {user_text}")
        res = engine.process_turn(conn, spine, user_text)
        print(_ops(res))
        print(f"     kai › {res.reply}\n")

    print("=" * 70)
    print("FINAL MEMORY STATE\n")
    mems = store.active_memories(conn)
    by_kind: dict[str, list] = {}
    for m in mems:
        by_kind.setdefault(m["kind"], []).append(m)
    for kind in config.ALL_KINDS:
        rows = by_kind.get(kind, [])
        if not rows or kind == "persona_canon":
            continue
        print(f"  {kind}:")
        for m in rows:
            print(f"    - {m['content']}")
    retired = [m for m in store.all_memories(conn) if m["status"] == "superseded"]
    if retired:
        print("  RETIRED (kept for audit):")
        for m in retired:
            print(f"    - {m['content']}  →  #{m['superseded_by']}")
    print(f"\n  (persona_canon: {len(by_kind.get('persona_canon', []))} rows, always in the spine)")
    print("\nDB persisted at:", _DB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
