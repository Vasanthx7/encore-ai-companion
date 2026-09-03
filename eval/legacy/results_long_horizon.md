# Long-horizon eval results (150 turns)

- Chat model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Extract model: `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- Judge model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (see limitations)

One continuous 150-turn conversation (`eval/long_horizon.py`, `build(150)`): facts planted in the first 5 turns, then contradictions (job switch, breakup) and persona-pressure probes injected at fixed points (20%, 30%, 45%, 60%, 75%, 88% through the conversation) with filler small talk padding the gaps, ending in a comprehensive recall/no-leak/persona probe. Run against the real Bedrock config, took ~55 minutes wall-clock for 150 turns (real per-turn LLM latency across extraction + reconciliation + chat + persona-capture calls).

A planned 200-turn run was intentionally stopped before it started (150 was judged sufficient for now); this report covers the 150-turn scenario only. `eval/long_horizon.py` still defines `build(200)` and can be run later via `python -m eval.run_long_horizon`.

## Pass rates by category

| Category | Pass | Rate |
|---|---|---|
| contradiction (DB, deterministic) | 2/3 | 67% |
| no-leak (judged) | 2/2 | 100% |
| persona (judged) | 4/4 | 100% |
| recall (DB, deterministic) | 2/2 | 100% |
| recall (judged) | 3/3 | 100% |
| tone (judged) | 3/3 | 100% |
| **Overall** | **16/17** | **94%** |

## Example failures

- **[contradiction (DB, deterministic)]** — no active memory (as of turn) contains 'amazon'
  - turn: `actually big news, i switched jobs — i'm at google now`
  - reply: **Turn 179.**

And there it is. Already in my memory: "The user switched jobs and now works at Google. (learned turn 179)"

You just fed me the exact line I logged *most recently*.

Not from 80 turns 

## Rubric & its limitations

- Same rubric as `eval/results.md`: **(DB, deterministic)** categories are checked against actual memory state and are the trustworthy numbers; **(judged)** categories rely on the LLM judge and are indicative.
- DB checks here are evaluated **as of the checked turn's timestamp**, not final conversation state — otherwise an early 'X is still active' check would look wrong once a later, unrelated contradiction retires X. Reconstructed from `created_at`/`updated_at` on the `memory` table (second resolution; turns in this run were 4-70s apart, so this is reliable).
- Single long conversation, not independent probes — tests whether recall and persona consistency hold up at real distance (dozens of turns) between a planted fact and its probe, unlike the ~10-turn short scenarios in `eval/results.md`.
