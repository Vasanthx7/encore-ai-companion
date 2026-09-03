# Companion-AI — Memory & Persona Architecture

**System:** Companion-AI Core Loop — a CLI chat companion backed by a real memory architecture: extract, store, retrieve, and update/decay with contradiction handling, plus a persona that stays consistent across long conversations.
**Evaluation harness:** an included component that scores memory recall, contradiction handling, and persona consistency (see §13).

---

## 1. Design principles (the through-line)

Every decision below traces back to four commitments:

1. **The data is relational-first.** ~90% of what we store is structured facts (subject/predicate/object, lifecycle, provenance) with a vector *attached* — not vectors with metadata attached. The storage engine reflects that.
2. **Persona consistency is a memory problem, not a prompt problem.** The companion's own improvised opinions are captured as first-class memories and re-injected, so it can't contradict itself 40 turns later. This is the central design decision.
3. **Decompose hard reasoning into narrow, verifiable judgments.** Rather than asking the model to reason over all of memory at once, contradiction detection is decomposed into narrow binary judgments over pre-fetched neighbors — a well-scoped task the model performs reliably and cheaply.
4. **Simplest thing that is correct at this scale.** ~50–500 facts, single user, single process. No ANN index, no server DB, no infra. Effort goes into retrieval/lifecycle logic, not ceremony.

---

## 2. Tech stack (settled)

| Layer | Choice | Reasoning |
|---|---|---|
| Language | **Python 3.13** | Richest ecosystem for this task. |
| LLM serving | **AWS Bedrock** behind a thin client | Provider abstraction — model ids are config, so upgrading or swapping models needs no code change. |
| Chat + judging model | **Claude Sonnet 4.5** (via Bedrock) | Strong instruction-following, reliable structured output, and the nuanced reasoning the persona/contradiction path depends on. |
| Extraction model | **Claude Haiku 4.5** (via Bedrock) | Fast, low-cost, and reliable at the narrowly-scoped extraction + retrieval-plan pass run on every turn. |
| Structured output | Native **tool-use / JSON-schema output** | Malformed JSON becomes *impossible* — the model only has to get content right, never format. No hand-rolled retry/parse layer. |
| Embeddings | **`nomic-embed-text`** via **Ollama**, behind an `Embedder` interface | Provider-swappable embedding layer. Documented fallback: `bge-base-en-v1.5` via `fastembed` if retrieval tests underperform. |
| Storage | **SQLite** (single file, `sqlite3` stdlib) | Relational + ACID + SQL + zero-install + inspectable at any time. Matches the problem's real shape (relational-first, single-process, zero-infra, auditable). |
| Vector search | **BLOB-in-SQLite + numpy brute-force cosine** | At hundreds of vectors an ANN index is pure ceremony; brute-force is sub-ms and trivially correct, no native-extension risk on Windows. Upgrade path: `sqlite-vec` past ~10k vectors. |

**Where we'd switch (documented, not defaults):** concurrent/multi-user → Postgres; millions of memories → pgvector/Qdrant; shared network service → a server DB. All are outside the current scope.

---

## 3. The turn pipeline

```
user turn
  │
  ├─▶ [1 structured LLM pass]  extract candidate facts  +  plan retrieval (query + entities)
  │
  ├─▶ reconcile & store facts           ← BEFORE retrieval, so retrieval sees updated state
  │        (duplicate | supersede | refine | novel)
  │
  ├─▶ retrieve relevant memories        ← hybrid: structured + semantic, gated
  │
  ├─▶ assemble prompt                   ← persona spine + long-term memory block + working memory
  │
  ├─▶ generate reply                    (warm temperature, in persona voice)
  │
  └─▶ [post-reply] extract persona_stated memories from the assistant turn
```

**Store-before-retrieve** is deliberate: "I broke up with my ex" supersedes the old relationship fact *first*, so the reply is never generated against stale memory.

---

## 4. Data model

Single unified `memory` table with a `kind` discriminator spanning both memory domains.

```sql
memory(
  id            INTEGER PRIMARY KEY,
  kind          TEXT,     -- user_semantic | user_preference | user_episodic | user_state
                          --   | persona_canon | persona_stated
  subject       TEXT,     -- canonical entity: "user", "user.sister", "user.ex_partner", "persona"
  predicate     TEXT,     -- soft relation hint: "dislikes", "works_as", "has_sibling"
  object        TEXT,     -- value: "horror films", "nurse", "Priya"
  content       TEXT,     -- NL rendering — what we embed and display
  embedding     BLOB,     -- nomic 768-dim, float32
  confidence    REAL,     -- extractor certainty 0..1
  salience      REAL,     -- importance weight, feeds ranking + decay
  status        TEXT,     -- active | superseded | expired
  superseded_by INTEGER,  -- FK → the memory that replaced it
  supersede_reason TEXT,  -- human-readable audit trail
  source_turn   INTEGER,  -- provenance
  polarity      TEXT,     -- affirm | negate
  temporal      TEXT,     -- resolved event date, or NULL
  created_at, updated_at, last_accessed_at, access_count
)
```

Supporting: an **entity registry** (canonical subjects + aliases) and a `turns` log (raw conversation, for working-memory window + provenance).

**Why each column earns its place:** `(subject,predicate)` → contradiction candidate gathering; `content`+`embedding` → semantic retrieval; `status`/`superseded_by` → lifecycle; `salience`+timestamps+`access_count` → decay/ranking; `source_turn` → auditability; `polarity` → retirements without a replacement ("I quit my job").

---

## 5. Memory taxonomy (two domains)

**User memory** — sub-typed because they decay/contradict differently:
- `user_semantic` — stable ("sister is Priya", "is a nurse") — slow decay
- `user_preference` — contradiction-prone ("hates horror") — medium
- `user_episodic` — timestamped events ("dentist appointment next Tuesday") — auto-`expired` when past-dated
- `user_state` — volatile ("stressed this week") — short half-life

**Persona memory** — the consistency engine:
- `persona_canon` — seeded from `persona.yaml`, **protected** (user extraction can never supersede it)
- `persona_stated` — opinions the persona *improvises* mid-conversation, captured so it's held to them

---

## 6. Extraction pipeline (ingestion)

**Memory-worthiness policy** (precision-biased): capture durable, person-specific, reusable disclosures — attributes, preferences, plans, significant events, recurring emotional patterns. Ignore pleasantries/meta, throwaway remarks, the user's own questions, and the assistant's suggestions. A **salience + confidence floor** (≈0.6) gates storage.

**Extraction schema** (per item, native structured output, temp 0):
```json
{ "kind","subject","predicate","object","content",
  "confidence","salience","temporal","polarity" }
```
The user-turn call returns `{ memories:[...], retrieval_query, retrieval_entities }` — extraction **and** the retrieval plan in one pass.

**Entity canonicalization** (what makes contradiction detection work): we canonicalize **subjects**, not predicates — because the *model* judges contradictions; canonicalization only needs to gather the right neighbors. The known-entity list is **fed into the extraction prompt** so the model reuses canonical subjects ("her"/"Priya"/"my sister" → `user.sister`). No separate resolver.

**Reconciliation** (new fact meets existing): gather ≤5 active neighbors (same subject ± semantically near), one structured call classifies:
- **duplicate** → skip insert, **reinforce** (bump salience/access)
- **supersede** → retire old (`status=superseded`, `superseded_by`, reason), insert new; also fires on `polarity=negate` with no replacement
- **refine** → merge added detail into existing
- **novel** → insert

**Guardrails:** grounding (extract only what's stated/strongly implied, temp 0, no inference-hallucination); confidence/salience floors; reconciliation as dedup guard.

---

## 7. Retrieval pipeline

**Query construction:** piggyback on the extraction call — it emits a de-anaphorized `retrieval_query` ("yeah, still dealing with that" → resolved) + `retrieval_entities`. No extra LLM call.

**Candidate generation (hybrid union):**
- *Structured leg* — active memories whose `subject` ∈ entities (reliable exact-entity recall)
- *Semantic leg* — numpy cosine KNN over active-memory vectors vs the query
- union + dedupe

**Ranking (transparent linear score):**
```
score = 0.55·cosine + 0.20·entity_match + 0.10·recency + 0.10·salience + 0.05·confidence
recency = exp(−Δt / half_life[kind])      # 'state' fast, 'semantic' slow
```
No cross-encoder reranker — overkill at this scale, and an opaque reranker isn't justified when the linear score is inspectable. Weights are config constants, tuned by eval.

**Gating (the "don't dump / don't miss" dial):**
1. `status=active` only — superseded/expired *never* surface (this is how contradiction handling shows up at retrieval)
2. similarity floor (τ≈0.35) — *unless* the candidate arrived via strong entity match (never miss an exact-entity hit)
3. top-k cap (k≈6–8) — the anti-dump limit
4. per-kind budget — cap volatile `state` to ~2, reserve slots for high-salience `semantic`

On retrieval we bump `last_accessed`/`access_count` — recalled facts gain salience (reinforcement counterweight to decay).

---

## 8. Contradiction & decay lifecycle

- **Contradiction/update** — handled in reconciliation (§6). Soft-supersession only: retired facts stay in the DB (`status=superseded`) for auditability and inspection (e.g. an ex-partner fact is retired, not deleted, when the relationship fact changes).
- **Decay = graceful de-prioritization, never deletion** — a recency+salience term in the ranking, shorter half-life for volatile kinds, auto-`expired` for past-dated episodic events, optional confidence decay for stale unconfirmed facts.

---

## 9. Persona subsystem

**Authoring:** a single versionable **`persona.yaml`** → at init, (1) render a system-prompt **spine** (identity + voice + always-on canon), (2) upsert opinions/backstory as **`persona_canon` rows**. Dual representation: spine keeps it always-present; rows make its commitments queryable/checkable.

**Schema:**
```yaml
identity:     { name, age, pronouns, concept }
backstory:    [ ... ]                              # → canon rows
personality:  { warmth, humor_style, directness, curiosity }
voice:        { register, rhythm, verbal_tics, emoji_policy, never_sounds_like }
opinions:     [ { topic, stance, polarity, strength } ]   # → canon rows (falsifiable)
values:       [ ... ]
boundaries:   { tone: warm-platonic, refusals, deflection_style }
relationship: "a friend who remembers you — not a therapist, not an assistant"
```

**Falsifiable opinions (the testability engine):** the schema *mandates* ~5–8 concrete, checkable stances ("dislikes horror", "prefers mountains to beaches"). Double duty: consistency anchors (force-injected via the guard below) **and** ready-made eval probe targets.

**Two consistency mechanisms:**
1. **Always-inject canon** — the spine is pinned in every prompt, so the persona structurally can't fall out of context (the usual cause of drift as history grows).
2. **`persona_stated` capture + force-inject guard** — improvised opinions are stored; when the current turn touches a subject the persona has opined on, that memory is force-injected *even if below top-k*, so it can't contradict itself. New `persona_stated` facts are also checked against canon; contradictions are flagged.

**Anti-flattening** (the "resets to generic assistant" failure): explicit negative instruction ("you are *not* a generic AI; never say 'As an AI…'; stay in-voice under hard topics"), 2–3 pinned in-character style exemplars, and the pinned canon.

**Reference persona:** "Kai" — 34, warm + dry-witted, ex-session-guitarist turned music teacher, hiker, strong coffee opinions — chosen because it yields testable stances. The schema holds regardless of character.

---

## 10. Prompt assembly (three tiers)

1. **System** — persona canon spine + behavioral rules ("stay in character, weave memories in naturally, don't recite them").
2. **Long-term memory block** — gated top-k user memories + relevant `persona_stated`, compact bullets with light provenance ("learned turn 12"), score-ordered.
3. **Working memory** — last ~8 turns *verbatim* (sliding window). Older turns are *not* kept verbatim — their content lives in extracted memories.

The working/long-term split **is** the thesis: recent-verbatim + old-retrieved-as-facts, not "stuff all history into context." A large context window means we keep the memory block lean *on purpose* — to demonstrate relevance, not because we're token-starved.

---

## 11. Module layout

```
ai_companion/
├── README.md
├── ARCHITECTURE.md                # this file
├── persona.yaml                   # authored persona canon
├── requirements.txt               # boto3, numpy, pyyaml, pydantic
├── config.py                      # models, weights, thresholds, k, half-lives
├── src/
│   ├── llm.py                     # Bedrock client; chat() + structured()
│   ├── embeddings.py              # Embedder interface (nomic via Ollama, default)
│   ├── store.py                   # SQLite: schema, CRUD, lifecycle, vector search
│   ├── entities.py                # canonical entity registry + aliasing
│   ├── extraction.py              # user-turn extract + retrieval-plan (one pass)
│   ├── reconcile.py               # duplicate|supersede|refine|novel
│   ├── retrieval.py               # hybrid candidates, ranking, gating
│   ├── persona.py                 # load persona.yaml → spine + canon rows; guards
│   ├── assemble.py                # 3-tier prompt builder
│   └── chat.py                    # the CLI loop wiring it all together
└── eval/                          # evaluation harness
    ├── scenarios.py               # synthetic multi-turn test conversations
    ├── judge.py                   # LLM-as-judge rubric (Claude Sonnet 4.5 via Bedrock)
    └── run_eval.py                # metrics + example failures
```

---

## 12. Implementation status

The core loop and the evaluation harness are implemented. Capabilities in place:

- **Skeleton & persistence** — `config`, `llm.py`, `store.py` schema, and the CLI loop; the SQLite store persists across restarts.
- **Extraction & storage** — fact extraction with subject-level entity canonicalization.
- **Retrieval & assembly** — hybrid (structured + semantic) retrieval with the three-tier prompt builder.
- **Reconciliation** — duplicate / supersede / refine classification, giving contradiction handling and update semantics.
- **Persona subsystem** — canon rows, `persona_stated` capture, and the force-inject consistency guard for long-conversation coherence.
- **Time-decay & tooling** — episodic expiry, a `/memory` inspector command, and the README.
- **Evaluation harness** — scenarios, judge, and reported numbers.

Restart-persistence is validated throughout (kill the process, reload, confirm memory survives).

**Engineering note — reconciliation guards:** two guards were added beyond the original design after testing revealed the model over-superseding across unrelated subjects — (a) a code-level cross-subject guard that refuses to supersede/refine unless the two facts are near-identical in meaning, and (b) a swappable `JUDGE_MODEL` for the persona contradiction check, since nuanced contradiction judgement is the hardest part of the reasoning. Correctness of what gets *stored* does not depend on the judge.

---

## 13. Evaluation harness

A lean, honest evaluation of the memory and persona behavior:
- **Scenarios** — synthetic multi-turn conversations that plant a fact, revisit it 40 turns later, contradict it, and probe persona opinions. Deterministic, scripted.
- **Automated detection** — LLM-as-judge (Claude Sonnet 4.5 via Bedrock) scoring: memory recall (correct/forgotten/wrong), contradiction handling (retired vs duplicated), persona consistency (opinion contradiction), tone flattening.
- **Numbers + example failures** — pass/fail rates per axis, concrete failure transcripts, and our own read on the weakest area.
- **Optional oracle** — a strong model given the *full* memory store, asked for the ideal recall, as an upper-bound baseline.
- **Rubric limitations stated** — no ground truth for "good companion response"; the judge is itself an LLM; the synthetic set is small.

---

## 14. Known limitations

- The models can miss **subtle multi-hop contradictions**; sarcasm/hypotheticals can be **mis-extracted as facts**.
- Entity canonicalization is model-assisted, so **alias drift** is possible on ambiguous references.
- Numpy brute-force is correct but **O(n) per query** — fine to ~10k facts, not beyond.
- Decay is heuristic (half-lives are hand-tuned, then eval-tuned) — not learned.
- Single-user, single-process by design.

---

## Open items

1. **Bedrock access** — the chat, extraction, and judge paths require AWS credentials with permission for the configured Claude models; the embedding path runs locally via Ollama.
2. **Persona selection** — "Kai" is the reference persona; the schema supports any character without code change.
