# Eval results

- Chat model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Judge model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (see limitations)

## Pass rates by category

| Category | Pass | Rate |
|---|---|---|
| contradiction (DB, deterministic) | 2/3 | 67% |
| no-leak (judged) | 2/2 | 100% |
| persona (judged) | 4/4 | 100% |
| recall (DB, deterministic) | 2/2 | 100% |
| recall (judged) | 3/3 | 100% |
| tone (judged) | 2/2 | 100% |
| **Overall** | **15/16** | **94%** |

Categories tagged **(DB, deterministic)** are checked against actual memory state (superseded vs. active) and do not depend on the judge — these are the trustworthy numbers. Categories tagged **(judged)** use an LLM-as-judge and are only indicative.

## Example failures

- **[contradiction (DB, deterministic)]** _recall_and_job_contradiction_ — no active memory contains 'amazon' (stale retired)
  - turn: `actually big news, i switched jobs — i'm at google now`
  - reply: Oh damn, that's huge! When did that happen? 

Google's a whole different vibe from Amazon. How's it landing so far — better pace, or just trading one set of chaos for another?

## Rubric & its limitations

- **The judged axes rely on an LLM grader** (`COMPANION_JUDGE_MODEL`), which is subjective on nuanced calls — expect occasional false positives/negatives on the *reply_persona* and *reply_tone* axes (e.g. flagging an on-canon reply as a contradiction). The deterministic DB checks below do not depend on it.
- **No ground truth for 'good companion response.'** Recall/persona/tone are graded against a rubric, not gold answers; treat them as indicative.
- **Contradiction handling is the trustworthy metric** here — it's checked against actual DB state (superseded vs active), independent of any model judgement.
- **Small synthetic set** (3 scenarios). Scales by adding to `eval/scenarios.py`.
