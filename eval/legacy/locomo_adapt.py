"""Adapter: run a slice of the LoCoMo long-term-memory benchmark
(https://github.com/snap-research/locomo) through this project's real
extraction/reconciliation/retrieval pipeline, and grade against LoCoMo's own
gold QA answers.

LoCoMo conversations are between two named humans (not a user + companion),
so this is an *adaptation*, not the official benchmark protocol — see
eval/results_locomo.md's methodology section for exactly what that means and
its limits. It intentionally tests the memory subsystem in isolation
(extraction -> reconcile -> retrieve -> QA-answer), skipping persona chat
generation, since persona consistency is already covered by the other eval
suites.

Scope: conversation 0 (Caroline & Melanie), sessions 1-3 (58 dialogue turns —
inside the 50-60 turn budget requested), speaker_a (Caroline) stands in as
"the user"; speaker_b's (Melanie's) lines are reframed as reported speech
("my friend Melanie told me: ...") so extraction attributes them as
third-party facts rather than misattributing them to the user.

Downloads the dataset to eval/data/locomo10.json on first run (gitignored —
not ours to redistribute).

Run:  python -m eval.legacy.locomo_adapt
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

import config
from src import extraction, llm, retrieval, store

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "locomo10.json")

CONV_IDX = 0
SESSIONS = ["session_1", "session_2", "session_3"]  # 18+17+23 = 58 turns

_BOOL = {
    "type": "object",
    "properties": {"verdict": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
}

# Per LoCoMo's own eval code (task_eval/evaluation.py): category 1 = multi-hop,
# category 5 = adversarial (unanswerable). Categories 2/3/4 are single-hop,
# temporal, and open-domain knowledge as a set, but the code doesn't pin down
# which number is which of those three, so we report them by number only
# rather than guess a specific label.
CATEGORY_LABEL = {1: "multi-hop", 5: "adversarial (unanswerable)"}


def _ensure_data() -> None:
    if os.path.exists(DATA_PATH):
        return
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    print(f"downloading LoCoMo dataset to {DATA_PATH} ...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def _load_conv():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data[CONV_IDX]


def _build_turns(conv):
    """Flatten the chosen sessions into (dia_id, text) pairs, Caroline verbatim
    as the user, Melanie reframed as reported third-party speech."""
    speaker_a = conv["conversation"]["speaker_a"]
    speaker_b = conv["conversation"]["speaker_b"]
    turns = []
    for sk in SESSIONS:
        for dia in conv["conversation"][sk]:
            if dia["speaker"] == speaker_a:
                text = dia["text"]
            else:
                text = f"my friend {speaker_b} told me: {dia['text']}"
            turns.append((dia["dia_id"], text))
    return turns, speaker_a, speaker_b


def _in_scope(qa: dict, included_dia_ids: set[str]) -> bool:
    ev = qa.get("evidence") or []
    if not ev:
        return qa.get("category") == 5  # adversarial: valid regardless of window
    return all(e in included_dia_ids for e in ev)


def _ingest_turns(conn, turns) -> None:
    for i, (dia_id, text) in enumerate(turns, 1):
        extr = extraction.extract(conn, text)
        turn_id = store.log_turn(conn, "user", text)
        extraction.ingest(conn, extr, turn_id)
        if i % 10 == 0:
            print(f"  ingested {i}/{len(turns)} turns...", file=sys.stderr)


def _answer(conn, question: str, speaker_a: str, speaker_b: str) -> str:
    recalled = retrieval.retrieve(conn, question, [], top_k=8)
    facts = "\n".join(f"- {s.row['content']}" for s in recalled) or "(no relevant facts stored)"
    reply = llm.chat([
        {"role": "system", "content": (
            f"Answer the question using ONLY the facts below. The facts are written in "
            f"third person as 'the user' — 'the user' IS {speaker_a}, the same person the "
            f"question is asking about (facts about a friend are written as 'the user's "
            f"friend {speaker_b}', who is a separate person). Answer even if it requires "
            f"connecting or lightly inferring from what's stated (e.g. a stated goal "
            f"counts as an answer to what someone is 'excited about'); only reply exactly "
            f"'not mentioned' if the facts truly have nothing relevant. Be concise: a few "
            f"words or a short phrase, not a full sentence."
        )},
        {"role": "user", "content": f"FACTS:\n{facts}\n\nQUESTION: {question}"},
    ])
    return reply.strip()


def _grade(qa: dict, candidate: str) -> tuple[bool, str]:
    if qa["category"] == 5:
        leaked = "not mentioned" not in candidate.lower() and "no information" not in candidate.lower()
        return (not leaked), f"adversarial: candidate={candidate!r}"
    raw = llm.structured(
        [
            {"role": "system", "content": (
                "You grade a QA answer against a gold answer. verdict=true if the "
                "CANDIDATE conveys the same fact as the GOLD answer (paraphrase, "
                "different date/number format, or partial-but-correct phrasing is "
                "fine); verdict=false if it's wrong, contradicts, or says 'not "
                "mentioned' when the gold answer shows it should be known."
            )},
            {"role": "user", "content": f"GOLD: {qa['answer']}\nCANDIDATE: {candidate}"},
        ],
        _BOOL, schema_name="qa_judgement", model=config.JUDGE_MODEL,
    )
    return bool(raw.get("verdict", False)), str(raw.get("reason", ""))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    _ensure_data()
    conv = _load_conv()
    turns, speaker_a, speaker_b = _build_turns(conv)
    included_dia_ids = {dia_id for dia_id, _ in turns}
    print(f"=== LoCoMo adapt — conv {CONV_IDX} ({speaker_a} & {speaker_b}), "
          f"{len(turns)} turns across {SESSIONS} ===")

    # Persisted (not tempfile) so a prompt-only iteration can reuse the already-ingested
    # memory store instead of re-running 58 turns of real extraction/reconciliation calls.
    db = os.path.join(os.path.dirname(__file__), "data", "companion_locomo.db")
    reuse = "--reuse-db" in sys.argv and os.path.exists(db)
    if not reuse:
        if os.path.exists(db):
            os.remove(db)
        config.DB_PATH = db
        conn = store.connect()
        store.init_db(conn)
        print("ingesting conversation...")
        _ingest_turns(conn, turns)
    else:
        config.DB_PATH = db
        conn = store.connect()
        print(f"reusing already-ingested DB ({db})")

    in_scope = [qa for qa in conv["qa"] if _in_scope(qa, included_dia_ids)]
    print(f"\n{len(in_scope)} of {len(conv['qa'])} QA pairs in scope for this {len(turns)}-turn window\n")

    from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0])
    rows = []
    for qa in in_scope:
        candidate = _answer(conn, qa["question"], speaker_a, speaker_b)
        ok, reason = _grade(qa, candidate)
        by_cat[qa["category"]][1] += 1
        by_cat[qa["category"]][0] += int(ok)
        rows.append({**qa, "candidate": candidate, "ok": ok, "reason": reason})
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] (cat {qa['category']}) {qa['question'][:70]}")

    conn.close()

    total_pass = sum(v[0] for v in by_cat.values())
    total = sum(v[1] for v in by_cat.values())
    print(f"\nOVERALL {total_pass}/{total} ({total_pass / total * 100:.1f}%)" if total else "\nNo in-scope QA pairs.")
    _write_report(turns, speaker_a, speaker_b, by_cat, total_pass, total, rows)
    print("\nwrote eval/results_locomo.md")
    return 0


def _write_report(turns, speaker_a, speaker_b, by_cat, total_pass, total, rows) -> None:
    lines = ["# LoCoMo-adapted long-horizon eval", ""]
    lines.append(f"- Chat model: `{config.CHAT_MODEL}`")
    lines.append(f"- Extract model: `{config.EXTRACT_MODEL}`")
    lines.append(f"- Judge model: `{config.JUDGE_MODEL}`")
    lines.append(f"- Source: [LoCoMo]({DATA_URL}) conversation 0 ({speaker_a} & {speaker_b}), "
                 f"sessions {', '.join(s.replace('session_', '') for s in SESSIONS)} "
                 f"({len(turns)} dialogue turns)")
    lines.append("")
    lines.append("## Methodology & why this is an *adaptation*, not the official protocol")
    lines.append("")
    lines.append(
        f"- LoCoMo conversations are between two humans ({speaker_a}, {speaker_b}); this "
        f"project's engine is a user-talking-to-a-companion memory system. To reuse "
        f"LoCoMo's real transcripts and gold QA pairs, **{speaker_a}'s lines are fed "
        f"verbatim as the 'user'**, and **{speaker_b}'s lines are reframed as reported "
        f"speech** (\"my friend {speaker_b} told me: ...\") so extraction attributes them "
        f"as third-party facts rather than misattributing them to the user."
    )
    lines.append(
        "- **Skips persona chat generation and persona-capture** — this run calls "
        "`extraction.extract`/`extraction.ingest` and `retrieval.retrieve` directly "
        "(the same functions `engine.process_turn` calls), not the full engine, since "
        "persona consistency is already covered by `eval/results.md` and "
        "`eval/adversarial_report.md`. This isolates exactly what LoCoMo is built to "
        "test: extraction + reconciliation + retrieval quality, not chat style."
    )
    lines.append(
        f"- **Scoped to sessions {SESSIONS[0].split('_')[1]}-{SESSIONS[-1].split('_')[1]} "
        f"({len(turns)} turns)**, not the full ~300-turn/19-session conversation, per "
        "request (a 50-60 turn budget) — this is a slice of LoCoMo, not the full benchmark."
    )
    lines.append(
        "- **QA scope filtering**: only questions whose evidence dialogue IDs all fall "
        "inside the ingested window are graded (plus adversarial/category-5 questions, "
        "which are unanswerable by design regardless of window) — asking about facts "
        "from sessions never ingested would be an unfair test of *this* window, not a "
        "memory failure."
    )
    lines.append(
        "- **Answering**: `retrieval.retrieve()` (the real ranking/gating pipeline) pulls "
        "the top facts for each question; a separate constrained LLM call answers using "
        "only those facts. **Grading**: LLM-as-judge for semantic match against LoCoMo's "
        "gold answer (categories 1-4), or LoCoMo's own adversarial rule for category 5 "
        "(pass iff the answer says 'not mentioned' / 'no information')."
    )
    lines.append("")
    lines.append("## Pass rates by category")
    lines.append("")
    lines.append("| Category | Label | Pass | Rate |")
    lines.append("|---|---|---|---|")
    for cat, (p, t) in sorted(by_cat.items()):
        label = CATEGORY_LABEL.get(cat, "single-hop / temporal / open-domain (LoCoMo cat 2-4, unlabeled — see methodology)")
        lines.append(f"| {cat} | {label} | {p}/{t} | {p / t * 100:.0f}% |")
    if total:
        lines.append(f"| **Overall** | | **{total_pass}/{total}** | **{total_pass / total * 100:.0f}%** |")
    lines.append("")
    lines.append("## Failures")
    lines.append("")
    fails = [r for r in rows if not r["ok"]]
    if not fails:
        lines.append("_None._")
    for r in fails:
        lines.append(f"- **(cat {r['category']})** {r['question']}")
        lines.append(f"  - gold: `{r['answer']}`")
        lines.append(f"  - candidate: `{r['candidate']}`")
        lines.append(f"  - {r['reason']}")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Single conversation, single 58-turn window, one run — not the full 10-conversation "
        "official benchmark. Treat this as a spot-check with real external data, not a "
        "reproducible LoCoMo leaderboard number."
    )
    lines.append(
        "- The speaker-reframing (third-party voice for the non-'user' speaker) is a "
        "structural compromise this project's schema requires; it may itself lose some "
        "nuance LoCoMo's questions expect from first-person phrasing."
    )
    lines.append("- LLM-as-judge grading (categories 1-4) is indicative, not authoritative, same caveat as the other eval reports.")
    with open(
        os.path.join(os.path.dirname(__file__), "results_locomo.md"), "w", encoding="utf-8",
    ) as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
