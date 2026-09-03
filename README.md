# Companion-AI Core Loop — Memory & Personality Consistency

A CLI companion ("Kai") backed by a real memory architecture: extract → store →
retrieve → update/decay with contradiction handling, plus a persona that stays
consistent across long conversations. Memory is a first-class subsystem here, not
a static system prompt.

Full design rationale: **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

## How it works

The system is a persistent, hybrid memory architecture wrapped around a
self-consistent persona. Each user turn flows through one pipeline that extracts
and stores grounded facts, retrieves the relevant ones, handles contradictions,
decays stale memories over time, and keeps the companion in character. An
evaluation harness exercises and scores these behaviours.

Capabilities:

- **Persona spine, LLM seam, SQLite persistence, CLI loop** — the durable core.
- **Memory extraction & storage** — grounded and entity-canonicalized.
- **Relevant retrieval** — hybrid ranking + gating feeding a 3-tier prompt.
- **Contradiction handling** — soft-supersession, refine, and decay.
- **Persona consistency** — canon rows + captured opinions + a consistency guard.
- **Time-decay** — episodic expiry, with a `/memory` inspector for introspection.
- **Evaluation harness** — scenario-driven metrics with an LLM-as-judge.

### What each turn does

```
user turn
 → extract facts + retrieval plan (1 structured call)
 → store: reconcile each fact (duplicate / supersede / refine / novel)
 → retrieve: hybrid-ranked, gated user memories + persona's own opinions
 → reply: persona spine + memory block + verbatim window
 → capture the persona's improvised opinions (consistency guard vs canon)
```

## Requirements

- **Python 3.11+**
- **AWS Bedrock access** with Claude Sonnet 4.5 and Claude Haiku 4.5 enabled, plus
  a Bedrock API key (a bearer token). Set `COMPANION_LLM_API_KEY` and, if needed,
  `COMPANION_AWS_REGION` — see `.env.example`.
- **[Ollama](https://ollama.com)** running locally for embeddings *(optional —
  see below)*, with the embedding model pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama serve            # if not already running as a service
  ```

Embeddings are optional. If you can't run a local model (no Ollama, restricted
environment, etc.), set `COMPANION_EMBEDDINGS_ENABLED=false` and skip the Ollama
step entirely — retrieval and reconciliation fall back to entity/subject
matching instead of semantic similarity (see `src/embeddings.py`). The endpoint
and model are also fully configurable, so you can point at any
OpenAI-embeddings-compatible provider instead of Ollama via
`COMPANION_EMBED_BASE_URL` / `COMPANION_EMBED_API_KEY` / `COMPANION_EMBED_MODEL`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in COMPANION_LLM_API_KEY
```

## Run

```bash
python -m src.chat
```

On launch you pick a companion from the persona library (`personas/*.yaml`) —
**Kai** (dry-witted ex-guitarist), **Nova** (bright wellness coach), **Sage**
(quiet bookish librarian), or **Milo** (sarcastic game designer). Each has its
own backstory, voice, and set of *falsifiable* opinions, so "stay in character"
stays testable. Everything you say is persisted to SQLite and survives restarts.

Each companion gets **its own memory store** (`companion_<name>.db`) — switching
companions doesn't leak one's facts or opinions into another. Pin a single
persona and skip the picker with `COMPANION_PERSONA=personas/nova.yaml`; add your
own by dropping a new YAML into `personas/` (copy `kai.yaml` as the template).

Commands inside the loop:
- `/memory` — grouped view of active memory (user facts + persona opinions) and retired facts
- `/dump` — raw view of stored turns and memories
- `/quit` — exit

A dim line after each turn shows memory ops, e.g. `· +1 · ~1 retired · recalled 3`.

### Provider / model overrides (env)

Every model call flows through a single seam (`src/llm.py`), so models,
credentials, and paths are set by environment variable alone — no code changes.
Chat, extraction, and judging run on Claude via AWS Bedrock; embeddings run on a
local OpenAI-compatible endpoint (Ollama by default).

| Env var | Default | Purpose |
|---|---|---|
| `COMPANION_LLM_API_KEY` | _(none)_ | AWS Bedrock API key (bearer token) |
| `COMPANION_AWS_REGION` | `us-east-1` | Bedrock region with Claude access |
| `COMPANION_CHAT_MODEL` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | conversation + judge model |
| `COMPANION_EXTRACT_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | extraction model |
| `COMPANION_JUDGE_MODEL` | _(chat model)_ | eval judge model (override on its own) |
| `COMPANION_EMBEDDINGS_ENABLED` | `true` | set `false` to skip embeddings entirely (no Ollama needed); retrieval/reconciliation fall back to entity/subject matching |
| `COMPANION_EMBED_BASE_URL` | `http://localhost:11434/v1` | embeddings endpoint (any OpenAI-embeddings-compatible provider) |
| `COMPANION_EMBED_API_KEY` | `ollama` | embeddings endpoint API key |
| `COMPANION_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `COMPANION_EMBED_DIM` | `768` | expected embedding vector size (match your model) |
| `COMPANION_DB` | `./companion_<persona>.db` | SQLite path (per-persona by default) |
| `COMPANION_PERSONA` | _(picker)_ | pin one persona file; unset shows the chooser |
| `COMPANION_PERSONAS_DIR` | `./personas` | where the selectable persona YAMLs live |

The eval judge defaults to the chat model (Sonnet 4.5); point
`COMPANION_JUDGE_MODEL` at another model to swap it independently.

## Layout

```
config.py         # all tunable knobs (models, weights, thresholds, half-lives)
personas/         # selectable companions — one YAML each (kai, nova, sage, milo)
  kai.yaml        # a persona's canon: identity, voice, falsifiable opinions
src/
  llm.py          # Anthropic/Bedrock seam: chat() + structured()
  embeddings.py   # Embedder seam (nomic default), L2-normalized vectors
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
  scenarios.py    # synthetic multi-turn tests + checks
  judge.py        # LLM-as-judge (Claude on Bedrock, swappable)
  run_eval.py     # runs scenarios, prints numbers, writes results.md
  results.md      # latest eval output
```

## Reproducible demo

```bash
python demo.py
```

Runs a fixed ~11-turn conversation through the real pipeline (throwaway DB, no
typing) and prints memory ops + Kai's replies, exercising every core behaviour:
extraction, relevant recall, **contradiction handling** (a job switch and a breakup
visibly *retire* the old facts), **persona consistency** (Kai's horror opinion holds
across the conversation), and **long-range recall** (early facts recalled 10 turns
later). Ends with a grouped dump of active vs. retired memory.

## Evaluation

```bash
python -m eval.run_eval        # runs 3 scenarios, writes eval/results.md
```

Scenarios (`eval/scenarios.py`) plant facts, add distance with filler turns, then
probe recall, contradict facts (job switch, breakup), and pressure-test persona
opinions. Two kinds of checks:

- **Deterministic (DB) checks** — query actual memory state (superseded vs. active).
  Judge-independent, so these are the **authoritative** numbers and carry the
  contradiction-handling verdict.
- **Judged checks** — LLM-as-judge for recall quality, no-leak, persona
  consistency, and tone. Indicative rather than authoritative.

The two axes are separated deliberately: memory correctness (what gets stored,
superseded, and recalled) is verified deterministically and does not depend on
the judge, while the subjective axes lean on the judge and are reported as
indicative signal.

**Where it's weakest (the honest part):**

1. **Judged axes are indicative, not authoritative.** The subjective checks can
   flip pass/fail between runs on borderline replies, so we treat them as signal
   and trust the deterministic memory metrics for correctness.
2. **Persona consistency bends under leading questions.** Asked "pineapple belongs
   on pizza, right?", the companion can partly endorse it despite canon.
   Canon-in-spine prevents most drift, but a strongly-lead question can still tip a
   reply. A pre-response canon-check on opinion questions would harden this further.
3. **Memory recall & contradiction handling are the strengths** — deterministically
   verified on these scenarios.

See `eval/results.md` for the latest raw numbers, example failures, and full rubric
limitations.

## What was tried and abandoned

Honest record of decisions reversed during the build — the reasoning matters more
than the final value:

- **Predicate canonicalization → subject-only.** Originally planned a controlled
  predicate vocabulary for contradiction detection. Abandoned once we realized the
  *model* judges contradictions over pre-fetched neighbours — canonicalization only
  needs to gather the right neighbours, so only **subjects** must be canonical.
- **ANN vector index (chroma/sqlite-vec) → numpy brute-force.** At hundreds of
  facts an index is pure ceremony; brute-force cosine is sub-ms and has no
  native-extension risk on Windows. Documented upgrade path past ~10k facts.
- **Deterministic canon-opposition check (embedding sim + polarity) → narrow LLM
  judge.** Short first-person opinions embed too similarly (shared "Kai …"
  structure), so a cosine+polarity rule produced false "contradictions" (pineapple
  "opposing" horror). Replaced with a narrow binary judge — far more reliable, and
  swappable for any model.
- **Cross-subject supersede gate 0.72 → supersede-only above 0.60.** Same-attribute
  paraphrases ("works at Amazon" vs "employer is now Google") and unrelated pairs
  *overlap* in cosine, so no clean threshold exists. We trust the strengthened
  classifier and restrict cross-subject actions to **supersede only** (never refine,
  which overwrites in place) with a coarse similarity floor.
- **Blanket "canon-topic → skip storing" → contradiction-gated skip.** Blocking
  storage by canon similarity wrongly dropped novel opinions that were merely
  music-adjacent. Now we only skip on an actual (best-effort) contradiction verdict.
- **Questions-as-facts** (found via `demo.py`): the extractor was turning questions
  like "what do you remember about my sister?" into bogus facts that *retired real
  memories*. Fixed with explicit prompt examples + an absence-phrasing post-filter.

## Known limitations

See [ARCHITECTURE.md §14](./ARCHITECTURE.md). Highlights: the subjective judged
axes are indicative rather than authoritative, so nuanced persona-contradiction
flagging is best-effort (correctness of what gets *stored* does not depend on the
judge); episodic time-decay only acts on ISO-dated events; and the store is
single-user / single-process by design.
