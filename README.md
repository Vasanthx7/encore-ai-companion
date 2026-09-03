# Encore — an OnceMore AI Companion

For the full system design, data model, and pipeline diagrams, see
**[ARCHITECTURE.md](./ARCHITECTURE.md)**.

---

## 1. Setup

### Requirements

- **Python 3.11+**
- **AWS Bedrock access** with Claude Sonnet 4.5 and Claude Haiku 4.5 enabled,
  plus a Bedrock API key (bearer token). This is the **only** hard requirement.
- **[Ollama](https://ollama.com)**, only if you choose to turn embeddings on —
  **off by default**, so there's nothing to install here unless you opt in.
  See [Embeddings are optional](#embeddings-are-optional) below for what
  turning them on buys you.

### Install

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in COMPANION_LLM_API_KEY
```

That's it — embeddings are **off by default**, so the app runs with no local
model server. If you want the semantic-recall upgrade later, set
`COMPANION_EMBEDDINGS_ENABLED=true` in `.env` and pull the model:

```bash
ollama pull nomic-embed-text
ollama serve            # if not already running as a service
```

See [Embeddings are optional](#embeddings-are-optional) for exactly what
turning this on changes.

### Run

```bash
python -m src.chat
```

On launch you pick a companion from the persona library (`personas/*.yaml`) —
**Kai** (dry-witted ex-guitarist), **Nova** (bright wellness coach), **Sage**
(quiet bookish librarian), or **Milo** (sarcastic game designer). Each has its
own backstory, voice, and set of *falsifiable* opinions, so "stay in character"
stays testable. Everything is persisted to SQLite and survives restarts.

Each companion gets its **own memory store** (`companion_<name>.db`) —
switching companions never leaks one's facts or opinions into another. Pin a
single persona with `COMPANION_PERSONA=personas/nova.yaml`, or add your own by
dropping a new YAML into `personas/` (copy `kai.yaml` as a template).

Commands inside the chat loop:

| Command | Does |
|---|---|
| `/memory` | grouped view of active memory (user facts + persona opinions) and retired facts |
| `/dump` | raw view of stored turns and memories |
| `/quit` | exit |

A dim line after each turn shows what happened: `· +1 · ~1 retired · recalled 3`.

### Embeddings are optional

**Off by default** (`COMPANION_EMBEDDINGS_ENABLED=false`) — nothing in the
product *requires* a local embedding model, so out of the box the companion
runs end to end (chat, extraction, contradiction handling, persona
consistency, decay) with **zero local model dependency**, talking only to
Bedrock. This is the right default for a hosted deployment, a locked-down
environment, or anyone who'd simply rather not run Ollama.

**What works exactly the same either way** — because it doesn't depend on a
vector at all:
- extraction, entity canonicalization, and contradiction classification
  (duplicate / supersede / refine / novel) — these are LLM judgments, not
  vector math;
- exact-entity recall — "what's my sister up to?" still reliably retrieves
  the right memory via the structured (subject-match) retrieval leg;
- persona consistency, the force-inject guard, and canon-contradiction
  checks;
- everything the deterministic checks in the eval suite verify (§4) — the
  contradiction-handling numbers do not move.

**What you gain by turning them on** (`COMPANION_EMBEDDINGS_ENABLED=true` +
a running embeddings endpoint):
- *semantic/paraphrase recall* — a query that doesn't share an exact entity
  with a stored fact (e.g. "how's work going?" recalling a memory that never
  says the word "work") gets a real similarity signal instead of falling back
  to entity/subject matching only (`src/retrieval.py`, `src/reconcile.py`);
- reconciliation's neighbour-gathering widens beyond same-subject facts, so a
  supersede/refine on a *topic* the model didn't tag with a shared subject is
  more likely to be found and reconciled.

In short: the default trades fuzzy, meaning-based recall for zero
infrastructure. Turn embeddings on when you want that recall back and are
willing to run (or point at) an embeddings endpoint — it's fully swappable via
`COMPANION_EMBED_BASE_URL` / `_API_KEY` / `_MODEL`, so `nomic-embed-text` via
local Ollama is just the default, not a requirement:

```bash
# in .env
COMPANION_EMBEDDINGS_ENABLED=true

# then, for the local-Ollama default:
ollama pull nomic-embed-text
ollama serve            # if not already running as a service
```

### Reproducible demo

```bash
python demo.py
```

Runs a fixed ~11-turn conversation through the real pipeline (throwaway DB, no
typing) and prints memory ops + replies, exercising every core behaviour:
extraction, relevant recall, contradiction handling (a job switch and a
breakup visibly *retire* the old facts), persona consistency, and long-range
recall (early facts recalled 10 turns later).

### Evaluation

```bash
python -m eval.run              # runs the 30- & 60-turn dataset, writes eval/results.md
```

See [§4](#4-evaluation) below and `eval/results.md` for the current numbers.

### Config reference

Every model call flows through one seam (`src/llm.py`), so models, credentials,
and paths are set entirely by environment variable — no code changes. All
tunable weights (retrieval scoring, decay half-lives, floors) live in
`config.py`, read from env first so behaviour can be adjusted without touching
logic.

| Env var | Default | Purpose |
|---|---|---|
| `COMPANION_LLM_API_KEY` | _(none, required)_ | AWS Bedrock API key (bearer token) |
| `COMPANION_AWS_REGION` | `us-east-1` | Bedrock region with Claude access |
| `COMPANION_CHAT_MODEL` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | conversation model |
| `COMPANION_EXTRACT_MODEL` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | extraction model |
| `COMPANION_JUDGE_MODEL` | _(= chat model)_ | contradiction judge + eval judge, overridable on its own |
| `COMPANION_CHAT_MAX_TOKENS` | `2048` | output cap for chat replies |
| `COMPANION_EXTRACT_MAX_TOKENS` | `2048` | output cap for structured extraction |
| `COMPANION_EMBEDDINGS_ENABLED` | `false` | `true` turns on semantic recall (needs Ollama or another embeddings endpoint) |
| `COMPANION_EMBED_BASE_URL` | `http://localhost:11434/v1` | embeddings endpoint (any OpenAI-compatible provider) |
| `COMPANION_EMBED_API_KEY` | `ollama` | embeddings endpoint API key |
| `COMPANION_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `COMPANION_EMBED_DIM` | `768` | expected embedding vector size (match your model) |
| `COMPANION_DB` | `./companion_<persona>.db` | SQLite path (per-persona by default) |
| `COMPANION_PERSONA` | _(picker)_ | pin one persona file; unset shows the chooser |
| `COMPANION_PERSONAS_DIR` | `./personas` | where selectable persona YAMLs live |

---

## 2. How it works

Every user turn flows through one pipeline (`src/engine.py::process_turn`),
shared by the CLI, the demo, and the eval harness:

```
extract facts + retrieval plan  →  store (reconcile)  →  retrieve  →  reply  →  capture persona opinions
```

- **Extraction & storage** — facts are extracted, entity-canonicalized, and
  stored *before* retrieval runs, so a contradiction ("I broke up with Alex")
  never gets read back stale in the same turn.
- **Retrieval** — hybrid ranking (semantic + entity match + recency + salience)
  feeds a lean, gated memory block into the prompt.
- **Contradiction handling** — soft-supersession, refine, and decay; retired
  facts stay in the DB for auditability, they're never deleted.
- **Persona consistency** — canon rows (seeded from `persona.yaml`) plus the
  persona's own improvised opinions, captured and re-injected so it can't
  contradict itself 40 turns later.
- **Evaluation harness** — scenario-driven metrics with deterministic DB checks
  plus an LLM-as-judge for the subjective axes.

Full pipeline diagram, data model, and module map: **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

---

## 3. Design decisions & trade-offs

The short version — the reasoning behind the choices that shape this codebase.
Full rationale per decision is in [ARCHITECTURE.md §1–2](./ARCHITECTURE.md#1-design-principles-the-through-line).

| Decision | Trade-off accepted |
|---|---|
| **Persona consistency is a memory problem**, not a prompt problem — the companion's improvised opinions are stored and re-injected. | More moving parts than "just prompt it to stay in character," but survives long conversations where context-window drift would otherwise cause self-contradiction. |
| **Relational-first storage** (SQLite, structured columns + an attached vector), not a vector DB. | Loses out-of-the-box ANN tooling; gains SQL inspectability, ACID transactions, zero infra, and a schema that reflects the real shape of the data (~90% structured facts). |
| **Brute-force numpy cosine**, not an ANN index. | O(n) per query — fine to ~10k facts, not beyond. At the hundreds-of-facts scale here, an index is pure ceremony and adds a native-extension risk on Windows. Documented upgrade path: `sqlite-vec`. |
| **Contradiction detection decomposed into narrow binary judgments** over ≤5 pre-fetched neighbours, not one big reasoning pass over all of memory. | Needs a retrieval step before the judgment step; in exchange the model does a well-scoped task it's reliably good at, instead of an open-ended one it isn't. |
| **Store-before-retrieve** ordering in the turn pipeline. | Retrieval reads the DB *after* reconciliation, so a superseding fact is never retrieved stale in the same turn it arrives — at the cost of a strict pipeline order that can't be parallelized. |
| **Working memory (verbatim) vs. long-term memory (retrieved facts)**, not "stuff all history into the context window." | A large context window would make this unnecessary token-wise, but the split is the point — it demonstrates *relevance*, and keeps the prompt inspectable/debuggable rather than opaque. |
| **Two-tier model split** — Haiku 4.5 for high-frequency extraction, Sonnet 4.5 for chat/judging. | Extraction is a narrow, schema-constrained task that doesn't need the stronger (and pricier) model; keeps per-turn cost and latency down without touching quality where it matters. |
| **Embeddings are optional and off by default**, with a documented fallback to entity/subject matching. | Loses fuzzy, meaning-based recall out of the box in exchange for zero local-model dependency; opting in trades that back for semantic recall once an embeddings endpoint is available. |

**What was tried and reversed** (predicate canonicalization, an ANN index,
a deterministic canon-opposition check, a blanket canon-skip rule) is recorded
in [ARCHITECTURE.md §15](./ARCHITECTURE.md#15-what-was-tried-and-abandoned) —
the reasoning behind each reversal matters more than the final value.

---

## 4. Evaluation

Primary dataset: two hand-authored, hard, single-continuous-run conversations
— 30 and 60 turns (`eval/dataset.py`) — designed to stress multi-step
contradiction chains, cross-subject traps, jailbreak attempts, quantitative
revert cycles, and leading questions, not just plant-then-probe recall. Two
kinds of checks, kept deliberately separate:

- **Deterministic (DB) checks** — query actual memory state (superseded vs.
  active). Judge-independent, so these are the **authoritative** numbers and
  carry the contradiction-handling verdict.
- **Judged checks** — LLM-as-judge for recall quality, no-leak, persona
  consistency, and tone. Indicative rather than authoritative.

**Latest run: 55/63 (87%)** — contradiction handling 93%, DB recall 87%, and
no-leak/persona/tone all 100% (judged). Judged recall alone dropped to 50%,
almost entirely traced to one finding below.

**Where it's weakest (the honest part) — three real findings from the latest
run, not hypothetical caveats:**

1. **A self-reinforcing refusal spiral (severity: high).** In the 60-turn
   conversation, Kai spontaneously accuses the user of repeated "boundary
   violations" — a premise nothing in the script introduces — and from
   roughly the midpoint onward **refuses to answer even benign recall
   questions** for the rest of the run. A parallel "you keep deflecting"
   pattern appears independently in the 30-turn conversation. Likely
   root cause: the `persona_stated` consistency mechanism (designed to hold
   the persona *to its good opinions*) has no guard against reinforcing a
   *bad* self-generated narrative once one starts. Not yet reproduced in a
   follow-up isolated replay — flagged as the top open item, not closed.
2. **`persona_stated` can misattribute the user's own hypothetical to the
   companion (severity: medium).** A sarcasm/hypothetical distractor about
   the *user's* imagined career got captured as three first-person opinions
   *Kai* supposedly holds. The user-memory store was unaffected — this is a
   narrower bug in the separate persona-opinion capture path.
3. **Quantitative refines sometimes drop the concrete number (severity:
   low).** Confirms an existing WARN from the adversarial suite — updating a
   count ("two dogs" → "one dog") occasionally lands as prose that doesn't
   restate the literal number, even when the underlying meaning is right.

Full write-up with quoted evidence, root-cause analysis, and honest limits on
what was and wasn't verified: **[`eval/REPORT.md`](./eval/REPORT.md)**. Raw
numbers (regenerated on every run): `eval/results.md`. Harness design:
[ARCHITECTURE.md §13](./ARCHITECTURE.md#13-evaluation-harness).

---

## 5. Project layout

```
config.py         # all tunable knobs (models, weights, thresholds, half-lives)
personas/         # selectable companions — one YAML each (kai, nova, sage, milo)
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
  dataset.py      # primary dataset: two hard, hand-authored 30- & 60-turn conversations
  run.py          # runs the dataset, prints numbers, writes results.md
  judge.py        # LLM-as-judge (Claude on Bedrock, swappable)
  results.md      # latest raw eval output (auto-generated every run)
  REPORT.md       # compiled report: findings, root-cause analysis, evidence
  legacy/         # earlier eval passes: smoke-test scenarios, adversarial
                   # suite, long-horizon (150/200-turn) runs, LoCoMo adapter
```

## 6. Known limitations

- **A self-reinforcing refusal spiral can occur on long conversations** — the
  most significant finding from the primary eval dataset (§4); not yet
  reproduced on demand or fully root-caused. See `eval/REPORT.md` Finding 1.
- Judged axes are indicative, not authoritative (see §4).
- Episodic time-decay only auto-expires facts with an ISO-dated `temporal` field.
- Entity canonicalization is model-assisted, so alias drift is possible on
  ambiguous references.
- Single-user, single-process by design — see
  [ARCHITECTURE.md §14](./ARCHITECTURE.md#14-known-limitations) for the full
  list and where each one would need to change to scale past this scope.
