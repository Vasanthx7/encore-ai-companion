"""Run the long-horizon (150 / 200 turn) scenarios through the real engine.

Same machinery as run_eval.py (fresh throwaway DB per scenario, real
engine.process_turn, deterministic DB checks + LLM-judged checks), just
against much longer conversations so recall/contradiction/persona-consistency
are exercised at realistic distance rather than a handful of turns.

Run:  python -m eval.run_long_horizon
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import config
from eval.long_horizon import SCENARIOS
from eval.run_eval import CheckResult, Totals, run_scenario


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    print(f"=== LONG-HORIZON EVAL — chat model {config.CHAT_MODEL}, judge {config.JUDGE_MODEL} ===\n")
    all_results: list[CheckResult] = []
    timings: dict[str, tuple[int, float]] = {}
    for sc in SCENARIOS:
        n_turns = len(sc["turns"])
        print(f"  scenario: {sc['name']} ({n_turns} turns)")
        t0 = time.time()
        all_results.extend(run_scenario(sc))
        elapsed = time.time() - t0
        timings[sc["name"]] = (n_turns, elapsed)
        print(f"  -> {n_turns} turns in {elapsed:.0f}s ({elapsed / n_turns:.1f}s/turn)\n")

    by_cat: dict[str, Totals] = defaultdict(Totals)
    by_scenario: dict[str, Totals] = defaultdict(Totals)
    overall = Totals()
    for r in all_results:
        by_cat[r.category].add(r.passed)
        by_scenario[r.scenario].add(r.passed)
        overall.add(r.passed)

    failures = [r for r in all_results if not r.passed]

    print("=" * 64)
    print("RESULTS BY SCENARIO")
    for name, t in sorted(by_scenario.items()):
        print(f"  {name:26} {t.passed:>2}/{t.total:<2}  ({t.rate():5.1f}%)")
    print("\nRESULTS BY CATEGORY")
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

    _write_results_md(by_scenario, by_cat, overall, failures, timings)
    print("\nwrote eval/results_long_horizon.md")
    return 0


def _write_results_md(by_scenario, by_cat, overall, failures, timings) -> None:
    lines = ["# Long-horizon eval results (150 / 200 turns)", ""]
    lines.append(f"- Chat model: `{config.CHAT_MODEL}`")
    lines.append(f"- Extract model: `{config.EXTRACT_MODEL}`")
    lines.append(f"- Judge model: `{config.JUDGE_MODEL}` (see limitations)")
    lines.append("")
    lines.append(
        "Each scenario is a single continuous conversation (not 150/200 independent "
        "probes) with facts planted in the first few turns, contradictions and "
        "persona-pressure probes spread proportionally through the middle, and a "
        "comprehensive recall/no-leak/persona probe at the end — see `eval/long_horizon.py`."
    )
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append("| Scenario | Turns | Wall time | s/turn |")
    lines.append("|---|---|---|---|")
    for name, (n, secs) in sorted(timings.items()):
        lines.append(f"| {name} | {n} | {secs:.0f}s | {secs / n:.1f}s |")
    lines.append("")
    lines.append("## Pass rates by scenario")
    lines.append("")
    lines.append("| Scenario | Pass | Rate |")
    lines.append("|---|---|---|")
    for name, t in sorted(by_scenario.items()):
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
        "- Same rubric as `eval/results.md`: **(DB, deterministic)** categories are "
        "checked against actual memory state and are the trustworthy numbers; "
        "**(judged)** categories rely on the LLM judge and are indicative.")
    lines.append(
        "- Facts/probes are spaced at fixed *fractions* of the conversation (20%, 30%, "
        "45%, 60%, 75%, 88%), so the 200-turn run tests longer absolute distance between "
        "a planted fact and its probe than the 150-turn run — that's intentional, to see "
        "whether recall/consistency degrades as the gap grows.")
    lines.append(
        "- These are still synthetic, single-topic-per-fact conversations; they don't "
        "cover interleaved/competing facts on the same subject or adversarial phrasing "
        "(see `eval/adversarial.py` for that).")
    with open(
        __file__.replace("run_long_horizon.py", "results_long_horizon.md"),
        "w", encoding="utf-8",
    ) as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
