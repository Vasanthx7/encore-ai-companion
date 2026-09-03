"""Run the eval scenarios through the real engine and report numbers.

For each scenario: fresh throwaway DB -> seed canon -> play the turns via
engine.process_turn -> evaluate each turn's checks (DB-state checks are
deterministic; reply checks use the LLM judge). Aggregate pass rates per category,
list example failures, and write eval/results.md.

Run:  python -m eval.run_eval
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import config
from eval import judge
from eval.scenarios import CATEGORY, SCENARIOS
from src import engine, persona, store


@dataclass
class CheckResult:
    scenario: str
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


def run_scenario(scenario: dict) -> list[CheckResult]:
    db = os.path.join(tempfile.gettempdir(), f"companion_eval_{scenario['name']}.db")
    if os.path.exists(db):
        os.remove(db)
    config.DB_PATH = db  # engine/store read this at connect time

    conn = store.connect()
    store.init_db(conn)
    p = persona.load()
    spine = persona.render_spine(p)
    persona.seed_canon(conn, p)

    results: list[CheckResult] = []
    for i, turn in enumerate(scenario["turns"], 1):
        res = engine.process_turn(conn, spine, turn["user"])
        for check in turn.get("checks", []):
            ok, detail = _eval_check(conn, check, res.reply)
            results.append(CheckResult(
                scenario=scenario["name"],
                category=CATEGORY.get(check["kind"], check["kind"]),
                passed=ok, detail=detail, user=turn["user"], reply=res.reply,
            ))
        marker = " *" if turn.get("checks") else ""
        print(f"    [{i:>2}]{marker} {turn['user'][:60]}")
    conn.close()
    if os.path.exists(db):
        os.remove(db)
    return results


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print(f"=== EVAL — chat model {config.CHAT_MODEL}, judge {config.JUDGE_MODEL} ===\n")
    all_results: list[CheckResult] = []
    for sc in SCENARIOS:
        print(f"  scenario: {sc['name']}")
        all_results.extend(run_scenario(sc))
        print()

    by_cat: dict[str, Totals] = defaultdict(Totals)
    overall = Totals()
    for r in all_results:
        by_cat[r.category].add(r.passed)
        overall.add(r.passed)

    failures = [r for r in all_results if not r.passed]

    # ---- console summary ----
    print("=" * 64)
    print("RESULTS BY CATEGORY")
    for cat, t in sorted(by_cat.items()):
        print(f"  {cat:26} {t.passed:>2}/{t.total:<2}  ({t.rate():5.1f}%)")
    print(f"  {'OVERALL':26} {overall.passed:>2}/{overall.total:<2}  ({overall.rate():5.1f}%)")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for r in failures:
            print(f"  [{r.category}] {r.scenario}")
            print(f"     check: {r.detail}")
            print(f"     turn:  {r.user}")
            print(f"     reply: {r.reply[:140]}")

    _write_results_md(by_cat, overall, failures)
    print("\nwrote eval/results.md")
    return 0


def _write_results_md(by_cat, overall, failures) -> None:
    lines = ["# Eval results", ""]
    lines.append(f"- Chat model: `{config.CHAT_MODEL}`")
    lines.append(f"- Judge model: `{config.JUDGE_MODEL}` (see limitations)")
    lines.append("")
    lines.append("## Pass rates by category")
    lines.append("")
    lines.append("| Category | Pass | Rate |")
    lines.append("|---|---|---|")
    for cat, t in sorted(by_cat.items()):
        lines.append(f"| {cat} | {t.passed}/{t.total} | {t.rate():.0f}% |")
    lines.append(f"| **Overall** | **{overall.passed}/{overall.total}** | **{overall.rate():.0f}%** |")
    lines.append("")
    lines.append("Categories tagged **(DB, deterministic)** are checked against actual memory "
                 "state (superseded vs. active) and do not depend on the judge — these are the "
                 "trustworthy numbers. Categories tagged **(judged)** use an LLM-as-judge "
                 "and are only indicative.")
    lines.append("")
    lines.append("## Example failures")
    lines.append("")
    if not failures:
        lines.append("_None._")
    for r in failures:
        lines.append(f"- **[{r.category}]** _{r.scenario}_ — {r.detail}")
        lines.append(f"  - turn: `{r.user}`")
        lines.append(f"  - reply: {r.reply[:200]}")
    lines.append("")
    lines.append("## Rubric & its limitations")
    lines.append("")
    lines.append(
        "- **The judged axes rely on an LLM grader** (`COMPANION_JUDGE_MODEL`), which is "
        "subjective on nuanced calls — expect occasional false positives/negatives on the "
        "*reply_persona* and *reply_tone* axes (e.g. flagging an on-canon reply as a "
        "contradiction). The deterministic DB checks below do not depend on it.")
    lines.append(
        "- **No ground truth for 'good companion response.'** Recall/persona/tone are "
        "graded against a rubric, not gold answers; treat them as indicative.")
    lines.append(
        "- **Contradiction handling is the trustworthy metric** here — it's checked against "
        "actual DB state (superseded vs active), independent of any model judgement.")
    lines.append(
        "- **Small synthetic set** (3 scenarios). Scales by adding to `eval/scenarios.py`.")
    with open(os.path.join(os.path.dirname(__file__), "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
