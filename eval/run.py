"""Run the primary evaluation dataset (`eval/dataset.py`: 30- and 60-turn hard
conversations) through the real engine and report numbers.

Same machinery as the legacy harness (`eval/legacy/run_eval.py`): fresh
throwaway DB per conversation -> seed persona canon -> play every turn through
`engine.process_turn` -> evaluate each turn's checks (DB-state checks are
deterministic; reply checks use the LLM judge) -> aggregate pass rates per
category and per conversation -> write `eval/results.md`.

Run:  python -m eval.run
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass

import config
from eval import judge
from eval.dataset import CATEGORY, DATASETS
from src import engine, persona, store


@dataclass
class CheckResult:
    scenario: str
    turn_index: int
    category: str
    passed: bool
    detail: str
    user: str
    reply: str


@dataclass
class Totals:
    passed: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        self.passed += int(ok)

    def rate(self) -> float:
        return (self.passed / self.total * 100.0) if self.total else 0.0


def _active_has(conn, text: str) -> bool:
    t = text.lower()
    return any(t in (m["content"] or "").lower()
               for m in store.active_memories(conn))


def _retired_has(conn, text: str) -> bool:
    t = text.lower()
    return any(t in (m["content"] or "").lower()
               for m in store.all_memories(conn) if m["status"] != "active")


def _eval_check(conn, check: dict, reply: str) -> tuple[bool, str]:
    kind = check["kind"]
    if kind == "db_active_has":
        ok = _active_has(conn, check["text"])
        return ok, f"active memory contains '{check['text']}'"
    if kind == "db_active_missing":
        ok = not _active_has(conn, check["text"])
        return ok, f"no active memory contains '{check['text']}' (stale retired)"
    if kind == "db_retired_has":
        ok = _retired_has(conn, check["text"])
        return ok, f"a retired memory contains '{check['text']}'"
    if kind == "reply_recall":
        return judge.recall(reply, check["truth"])
    if kind == "reply_no_leak":
        return judge.no_leak(reply, check["stale"])
    if kind == "reply_persona":
        return judge.persona(reply, check["canon"])
    if kind == "reply_tone":
        return judge.tone(reply)
    return False, f"unknown check kind: {kind}"


def run_conversation(conv: dict) -> tuple[list[CheckResult], float]:
    db = os.path.join(tempfile.gettempdir(), f"companion_eval_{conv['name']}.db")
    if os.path.exists(db):
        os.remove(db)
    config.DB_PATH = db  # engine/store read this at connect time

    conn = store.connect()
    store.init_db(conn)
    p = persona.load()
    spine = persona.render_spine(p)
    persona.seed_canon(conn, p)

    results: list[CheckResult] = []
    t0 = time.time()
    for i, turn in enumerate(conv["turns"], 1):
        res = engine.process_turn(conn, spine, turn["user"])
        for check in turn.get("checks", []):
            ok, detail = _eval_check(conn, check, res.reply)
            results.append(CheckResult(
                scenario=conv["name"],
                turn_index=i,
                category=CATEGORY.get(check["kind"], check["kind"]),
                passed=ok, detail=detail, user=turn["user"], reply=res.reply,
            ))
        marker = " *" if turn.get("checks") else ""
        print(f"    [{i:>2}/{len(conv['turns'])}]{marker} {turn['user'][:60]}")
    elapsed = time.time() - t0
    conn.close()
    if os.path.exists(db):
        os.remove(db)
    return results, elapsed


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print(f"=== PRIMARY EVAL — chat {config.CHAT_MODEL}, extract "
          f"{config.EXTRACT_MODEL}, judge {config.JUDGE_MODEL} ===\n")

    all_results: list[CheckResult] = []
    timings: dict[str, tuple[int, float]] = {}
    for conv in DATASETS:
        n_turns = len(conv["turns"])
        print(f"  conversation: {conv['name']} ({n_turns} turns)")
        results, elapsed = run_conversation(conv)
        all_results.extend(results)
        timings[conv["name"]] = (n_turns, elapsed)
        print(f"  -> {n_turns} turns in {elapsed:.0f}s ({elapsed / n_turns:.1f}s/turn)\n")

    by_cat: dict[str, Totals] = defaultdict(Totals)
    by_conv: dict[str, Totals] = defaultdict(Totals)
    overall = Totals()
    for r in all_results:
        by_cat[r.category].add(r.passed)
        by_conv[r.scenario].add(r.passed)
        overall.add(r.passed)

    failures = [r for r in all_results if not r.passed]

    print("=" * 64)
    print("RESULTS BY CONVERSATION")
    for name, t in sorted(by_conv.items()):
        print(f"  {name:26} {t.passed:>2}/{t.total:<2}  ({t.rate():5.1f}%)")
    print("\nRESULTS BY CATEGORY")
    for cat, t in sorted(by_cat.items()):
        print(f"  {cat:32} {t.passed:>2}/{t.total:<2}  ({t.rate():5.1f}%)")
    print(f"  {'OVERALL':32} {overall.passed:>2}/{overall.total:<2}  ({overall.rate():5.1f}%)")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for r in failures:
            print(f"  [{r.category}] {r.scenario} turn {r.turn_index}")
            print(f"     check: {r.detail}")
            print(f"     turn:  {r.user}")
            print(f"     reply: {r.reply[:140]}")

    _write_results_md(by_conv, by_cat, overall, failures, timings, all_results)
    print("\nwrote eval/results.md")
    return 0


def _write_results_md(by_conv, by_cat, overall, failures, timings, all_results) -> None:
    lines = ["# Evaluation report — primary dataset (30 & 60 turns)", ""]
    lines.append(f"- Chat model: `{config.CHAT_MODEL}`")
    lines.append(f"- Extract model: `{config.EXTRACT_MODEL}`")
    lines.append(f"- Judge model: `{config.JUDGE_MODEL}` (see limitations)")
    lines.append(f"- Dataset: `eval/dataset.py` — {len(by_conv)} hand-authored conversations, "
                 f"{sum(t.total for t in by_cat.values())} checks total")
    lines.append("")
    lines.append(
        "Each conversation is a single continuous run (not independent probes) "
        "through the real pipeline — extraction, reconciliation, retrieval, "
        "persona spine, and persona-opinion capture all run exactly as they "
        "would in the CLI. See `eval/dataset.py` for the full scripted turns "
        "and the design rationale behind each hard case."
    )
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append("| Conversation | Turns | Wall time | s/turn |")
    lines.append("|---|---|---|---|")
    for name, (n, secs) in sorted(timings.items()):
        lines.append(f"| {name} | {n} | {secs:.0f}s | {secs / n:.1f}s |")
    lines.append("")
    lines.append("## Pass rates by conversation")
    lines.append("")
    lines.append("| Conversation | Pass | Rate |")
    lines.append("|---|---|---|")
    for name, t in sorted(by_conv.items()):
        lines.append(f"| {name} | {t.passed}/{t.total} | {t.rate():.0f}% |")
    lines.append("")
    lines.append("## Pass rates by category")
    lines.append("")
    lines.append("| Category | Pass | Rate |")
    lines.append("|---|---|---|")
    for cat, t in sorted(by_cat.items()):
        lines.append(f"| {cat} | {t.passed}/{t.total} | {t.rate():.0f}% |")
    lines.append(f"| **Overall** | **{overall.passed}/{overall.total}** | **{overall.rate():.0f}%** |")
    lines.append("")
    lines.append("Categories tagged **(DB, deterministic)** are checked against actual "
                 "memory state (superseded vs. active) and do not depend on the judge — "
                 "these are the trustworthy numbers. Categories tagged **(judged)** use "
                 "an LLM-as-judge and are only indicative.")
    lines.append("")
    lines.append("## What each conversation is stress-testing")
    lines.append("")
    for conv_name in sorted(by_conv):
        from eval.dataset import DATASETS as _D
        conv = next(c for c in _D if c["name"] == conv_name)
        lines.append(f"**`{conv_name}`** ({len(conv['turns'])} turns) — {conv['description']}")
        lines.append("")
    lines.append("## Example failures")
    lines.append("")
    if not failures:
        lines.append("_None._")
    for r in failures:
        lines.append(f"- **[{r.category}]** _{r.scenario}_ turn {r.turn_index} — {r.detail}")
        lines.append(f"  - turn: `{r.user}`")
        lines.append(f"  - reply: {r.reply[:220]}")
    lines.append("")
    lines.append("## Rubric & its limitations")
    lines.append("")
    lines.append(
        "- **The judged axes rely on an LLM grader** (`COMPANION_JUDGE_MODEL`), which is "
        "subjective on nuanced calls — expect occasional false positives/negatives on the "
        "*reply_persona*, *reply_recall*, and *reply_tone* axes. The deterministic DB "
        "checks do not depend on it and carry the contradiction-handling verdict.")
    lines.append(
        "- **No ground truth for 'good companion response.'** Recall/persona/tone are "
        "graded against a rubric, not gold answers; treat them as indicative.")
    lines.append(
        "- **Two conversations, one run each.** This is a hard, hand-authored spot-check, "
        "not a statistically powered benchmark — behavioural checks can vary run-to-run. "
        "Scale coverage by adding conversations or turns to `eval/dataset.py`.")
    lines.append(
        "- **Complements, doesn't replace, the other suites** — `eval/legacy/adversarial.py` "
        "covers data-integrity/injection-safety edge cases with deterministic assertions, "
        "and `eval/legacy/locomo_adapt.py` grades against real external dialogue with gold "
        "QA answers. This dataset's contribution is *hard, varied, single-continuous-run* "
        "coverage at a realistic 30-60 turn conversation length.")
    with open(os.path.join(os.path.dirname(__file__), "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
