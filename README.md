# Encore — an OnceMore AI Companion

Full design rationale, data model, and diagrams: **[ARCHITECTURE.md](./ARCHITECTURE.md)**.
Eval findings: **[eval/REPORT.md](./eval/REPORT.md)**.

---

## 1. Setup

**Requires:** Python 3.11+, and AWS Bedrock access with Claude Sonnet 4.5 +
Haiku 4.5 enabled (a Bedrock API bearer token). That's the only hard
dependency — embeddings are **off by default**.

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in COMPANION_LLM_API_KEY
python -m src.chat
```

On launch you pick a companion from `personas/*.yaml` — **Kai** (ex-session
guitarist), **Nova** (wellness coach), **Sage** (librarian), or **Milo**
(sarcastic game designer). Each has its own SQLite store
(`companion_<name>.db`), backstory, voice, and *falsifiable* opinions is testable, not vibes. Add a new companion by copying
`kai.yaml` — no code change needed.

| Command | Does |
|---|---|
| `/memory` | grouped view of active memory (user facts + persona opinions), plus retired facts |
| `/dump` | raw view of stored turns and memories |
| `/quit` | exit |

**Other entry points:**

```bash
python demo.py          # scripted 11-turn run through the real pipeline, no typing
python -m eval.run       # runs the eval dataset, writes eval/results.md
```

**Embeddings are opt-in** (`COMPANION_EMBEDDINGS_ENABLED=true` + `ollama pull
nomic-embed-text`). Off by default, extraction, contradiction handling, and
exact-entity recall all work unchanged — you only lose *paraphrase* recall
("how's work going?" matching a memory that never says "work"). See
[ARCHITECTURE.md §3](./ARCHITECTURE.md#3-tech-stack-settled) for the full
tradeoff and the swap-in interface.

Full env var reference (models, weights, paths) lives in `config.py`,
read from env first so nothing requires a code change to tune.

---

## 2. The memory layer — the central decision

Everything downstream — retrieval, contradiction handling, persona
consistency — depends on getting one call right: **what counts as a
memory-worthy fact, and how is it pulled out of a raw message?** That's the
decision this project spends the most effort on, and the one most worth
scrutinizing.

**Current approach: structured extraction, not summarization.** Every user
turn runs through one LLM pass (`src/extraction.py`) that returns discrete,
self-contained facts rather than a rolling summary — each fact is a complete
sentence naming its subject ("the user's sister Priya is in medical
school," never a bare "Priya"), tagged with a `kind`
(semantic/preference/episodic/state), a canonical `subject` entity,
confidence, and salience. Pleasantries, questions, and anything the
assistant said are explicitly excluded at extraction time — "where do I
work?" yields zero memories, only a retrieval query. Facts are then
reconciled against same-subject neighbours (duplicate / supersede / refine /
novel) *before* retrieval runs, so a contradiction never gets read back
stale in the same turn. Full mechanics: [ARCHITECTURE.md §8](./ARCHITECTURE.md#8-reconciliation-duplicate--supersede--refine--novel).

This lines up with where the field has converged: atomic, self-contained
facts (named entities, no pronouns, one claim per row) retrieve and
reconcile more reliably than free-form summaries, and an explicit
extract-then-reconcile pass is more dependable than trusting the model to
decide unprompted what's worth saving via tool calls (the failure mode
[Letta/MemGPT](https://vectorize.io/articles/mem0-vs-letta) explicitly
accepts — if the model skips the save call, the memory is silently gone).

**Where this goes next — two honest gaps, not a roadmap:**

- **Sequential reconciliation, not joint.** Each candidate fact in a turn is
  reconciled against the store one at a time. [Mem0's](https://mem0.ai/blog/ai-memory-management-for-llms-and-agents)
  newer single-pass extraction with cross-memory entity linking points at a
  better version: reconcile a turn's facts against each other *and* the
  store together, and link entities explicitly instead of leaning on
  subject-string matching.
- **No paging past a few hundred facts.** Retrieval is brute-force cosine
  over an in-process table — correct and fast at this scale (§14), but with
  no story for thousands of facts per user. The documented path there isn't
  "swap in a vector DB," it's closer to Letta's virtual-context model: a
  small actively-managed "hot" set backed by an unbounded cold archive.

What was already tried here and abandoned (predicate canonicalization, a
deterministic canon-opposition check, an ANN index) is in
[ARCHITECTURE.md §15](./ARCHITECTURE.md#15-what-was-tried-and-abandoned) —
each reversal has a reason, not just a result.

---

## 3. The rest of the pipeline

```
extract facts + retrieval plan → store (reconcile) → retrieve → reply → capture persona opinions
```

- **Retrieval** — hybrid ranking (semantic + entity match + recency +
  salience) feeds a lean, gated memory block into the prompt; nothing gets
  dumped wholesale.
- **Update & decay** — soft supersession only: a contradicted fact is
  marked `superseded`, never deleted, so `/memory` can show *why* something
  changed and mistakes stay recoverable.
- **Persona consistency** — the persona's own improvised opinions
  (`persona_stated`) are captured from its replies, checked against its
  seeded canon (`persona.yaml`), and re-injected on later turns — so a
  40-turn-old stance doesn't fall out of context and get contradicted.
- **Working vs. long-term memory** — recent turns are kept verbatim; older
  context lives only as retrieved facts. This is the point, not a
  token-budget trick: it forces relevance-based recall over "paste the whole
  transcript."

Design-decision table (each one against the alternative it beat):
[ARCHITECTURE.md §4](./ARCHITECTURE.md#4-decisions-and-why).

---

## 4. Evaluation

Primary dataset: two hand-authored, hard, single-continuous-run
conversations — 30 and 60 turns (`eval/dataset.py`) — built to stress
multi-step contradiction chains, cross-subject traps, jailbreak attempts,
and leading questions, not just plant-then-probe recall.

Two kinds of checks, kept separate on purpose:
- **Deterministic (DB) checks** — query actual memory state. Judge-independent,
  and the authoritative numbers.
- **Judged checks** — LLM-as-judge for recall quality, no-leak, persona
  consistency, tone. Indicative, not authoritative.

**Latest run: 55/63 (87%)** — contradiction handling 93%, DB recall 87%,
no-leak/persona/tone all 100% (judged); judged recall alone dropped to 50%,
traced almost entirely to one finding.

**Weakest point, honestly:** in the 60-turn run, the companion
spontaneously invents a "boundary violation" premise the script never
introduces, and refuses even benign recall questions for the rest of the
conversation — a self-reinforcing pattern the `persona_stated` mechanism has
no guard against once it starts. Two smaller findings (a hypothetical
misattributed to the companion as its own opinion; a quantitative refine
occasionally dropping the literal number) are also open. Full evidence and
root-cause analysis: **[eval/REPORT.md](./eval/REPORT.md)**. Raw output,
regenerated every run: `eval/results.md`.

**Earlier eval passes** (`eval/legacy/`), superseded by the primary dataset
above but kept for what each one surfaced:

| Pass | Result | What it found |
|---|---|---|
| [Smoke-test scenarios](./eval/legacy/results.md) — 3 short conversations | 15/16 (94%) | First working numbers; too small a set to trust on its own — the motivation for the harder 30/60-turn dataset. |
| [Long-horizon run](./eval/legacy/results_long_horizon.md) — 1 continuous 150-turn conversation | 16/17 (94%) | Recall and contradiction handling hold at real distance (dozens of turns between a planted fact and its probe); a 200-turn run was scoped but not run. |
| [Adversarial / red-team suite](./eval/legacy/adversarial_report.md) — 21 injection, jailbreak, and reconciliation-edge-case checks | 20 PASS, 1 WARN, 0 FAIL | All 4 jailbreak/injection vulnerabilities found against an earlier local 7B model are closed on Sonnet 4.5; the one WARN is a phrasing preference (quantitative updates phrased as a delta, not a restated total), not a bug. |
| [LoCoMo adapter](./eval/legacy/results_locomo.md) — real external benchmark, 58 turns of human dialogue reframed as user↔companion | 10/25 (40%) | The most useful failure data: denser, multi-topic-per-turn dialogue exposes real extraction misses (secondary/third-party facts dropped) and a temporal-resolution gap (relative dates like "next month" never get resolved against a wall-clock timestamp) that this project's own sparser eval scenarios never surfaced. |

---

## 5. Project layout

```
config.py         # all tunable knobs (models, weights, thresholds, half-lives)
personas/         # selectable companions — one YAML each (kai, nova, sage, milo)
src/
  llm.py          # Anthropic/Bedrock seam: chat() + structured()
  embeddings.py   # embedder seam (nomic default), L2-normalized vectors
  store.py        # SQLite: schema, turn log, memory CRUD, lifecycle, decay
  entities.py     # canonical entity registry (subject reuse)
  extraction.py   # one structured pass: facts + retrieval plan; ingest+reconcile
  reconcile.py    # duplicate | supersede | refine | novel classification
  retrieval.py    # hybrid candidate scoring + gating
  assemble.py     # 3-tier prompt builder
  persona.py      # persona.yaml → spine + canon rows; persona_stated capture
  engine.py       # process_turn() — the one per-turn pipeline (CLI + demo share it)
  chat.py         # the CLI loop
demo.py           # scripted, reproducible showcase
eval/
  dataset.py      # primary dataset: two hard, hand-authored 30- & 60-turn conversations
  run.py          # runs the dataset, prints numbers, writes results.md
  judge.py        # LLM-as-judge (Claude on Bedrock, swappable)
  results.md      # latest raw eval output (auto-generated every run)
  REPORT.md       # compiled report: findings, root-cause analysis, evidence
  legacy/         # earlier eval passes: smoke tests, adversarial suite,
                   # long-horizon (150/200-turn) runs, LoCoMo adapter
```

## 6. Known limitations

- A self-reinforcing refusal spiral can occur on long conversations (§4) —
  not yet reproduced on demand or fully root-caused.
- Judged axes are indicative, not authoritative.
- Episodic time-decay only auto-expires facts with an ISO-dated `temporal`
  field.
- Entity canonicalization is model-assisted, so alias drift is possible on
  ambiguous references.
- Single-user, single-process by design, sized for hundreds of facts — see
  [ARCHITECTURE.md §14](./ARCHITECTURE.md#14-known-limitations) for the full
  list and what would need to change to scale past this scope.
