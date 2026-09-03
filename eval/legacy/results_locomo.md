# LoCoMo-adapted long-horizon eval

- Chat model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Extract model: `us.anthropic.claude-haiku-4-5-20251001-v1:0`
- Judge model: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
- Source: [LoCoMo](https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json) conversation 0 (Caroline & Melanie), sessions 1, 2, 3 (58 dialogue turns)

## Methodology & why this is an *adaptation*, not the official protocol

- LoCoMo conversations are between two humans (Caroline, Melanie); this project's engine is a user-talking-to-a-companion memory system. To reuse LoCoMo's real transcripts and gold QA pairs, **Caroline's lines are fed verbatim as the 'user'**, and **Melanie's lines are reframed as reported speech** ("my friend Melanie told me: ...") so extraction attributes them as third-party facts rather than misattributing them to the user.
- **Skips persona chat generation and persona-capture** — this run calls `extraction.extract`/`extraction.ingest` and `retrieval.retrieve` directly (the same functions `engine.process_turn` calls), not the full engine, since persona consistency is already covered by `eval/results.md` and `eval/adversarial_report.md`. This isolates exactly what LoCoMo is built to test: extraction + reconciliation + retrieval quality, not chat style.
- **Scoped to sessions 1-3 (58 turns)**, not the full ~300-turn/19-session conversation, per request (a 50-60 turn budget) — this is a slice of LoCoMo, not the full benchmark.
- **QA scope filtering**: only questions whose evidence dialogue IDs all fall inside the ingested window are graded (plus adversarial/category-5 questions, which are unanswerable by design regardless of window) — asking about facts from sessions never ingested would be an unfair test of *this* window, not a memory failure.
- **Answering**: `retrieval.retrieve()` (the real ranking/gating pipeline) pulls the top facts for each question; a separate constrained LLM call answers using only those facts. **Grading**: LLM-as-judge for semantic match against LoCoMo's gold answer (categories 1-4), or LoCoMo's own adversarial rule for category 5 (pass iff the answer says 'not mentioned' / 'no information').

## Diagnosis — what actually drives the score

The first run scored **6/25 (24%)**. Investigating the failures (not just reporting the
number) surfaced one adapter bug and let real system limitations be separated from
adapter artifacts:

1. **Adapter bug, fixed (24% → 40%):** memory content is stored in third person ("the
   user did X") but the first QA-answering prompt never told the grader "the user" IS
   Caroline. One candidate reply said so outright: *"the facts refer to 'the user' and
   'Melanie,' but do not mention anyone named Caroline."* Fixing the prompt to state that
   equivalence explicitly moved the score to **10/25 (40%)** with the *same* underlying
   memory store — proof the first number understated what was actually retrievable.
2. **Verified against the raw DB, not just re-running:** for the remaining failures, I
   queried `eval/data/companion_locomo.db` directly (kept on disk via `--reuse-db`
   instead of being deleted after the run) to check whether each fact was ever extracted
   at all, separately from whether retrieval surfaced it. This splits the remaining 15
   failures into three distinct causes:
   - **(a) Temporal / absolute-date resolution — ~6 failures.** This adapter feeds only
     dialogue text, never each session's real-world `date_time` from the LoCoMo data.
     Extraction correctly stores *relative* time language as stated ("next month", "last
     week") but has no way to resolve it to an absolute date like the gold answers expect
     ("June 2023", "the week before 9 June 2023") — it was never given the date to resolve
     against. This is an adapter gap, not necessarily a core-system one; a real deployment
     would pass the wall-clock turn timestamp, which this synthetic-scenario-less slice
     doesn't reproduce.
   - **(b) Genuine extraction misses on secondary/third-party topics — ~6 failures.**
     Direct queries for `charity`, `research`, `agency` returned **zero** memory rows —
     Melanie's charity race (an entire sub-thread, reframed as third-party reported
     speech) and Caroline's specific adoption-agency reasoning ("researching agencies",
     "chose it for LGBTQ+ inclusivity") were never extracted, despite being present in the
     ingested turns. Likewise a directly-stated relationship-status fact ("Single") never
     became a discrete row (only an inferential-adjacent "planning to become a single
     parent" fact exists). This is a real recall gap surfaced by LoCoMo's denser,
     multi-topic-per-turn dialogue — this project's own eval scenarios are comparatively
     sparse (~1 fact per turn), so this gap wasn't visible before this run.
   - **(c) Extraction paraphrases toward gist, losing enumerable specifics — ~2-3
     failures.** Melanie's self-care fact *was* extracted, but as "a way to stay present
     for her family" (the motivation) rather than the gold answer's concrete list
     (running, reading, playing violin). Retrieval and the QA step both worked correctly
     here; extraction's summarization simply dropped the enumerable detail.
   - Retrieval itself was **not** a bottleneck in the cases checked: for one failing
     question the exact right fact ("The user's core goal is to create a family for
     children who need one through adoption") was retrieval's #1-ranked hit (cosine 0.60)
     — the QA-answering step was just initially too conservative to connect a stated goal
     to "what are they excited about"; loosening that prompt flipped it to a pass.

## Pass rates by category

| Category | Label | Pass | Rate |
|---|---|---|---|
| 1 | multi-hop | 1/3 | 33% |
| 2 | single-hop / temporal / open-domain (LoCoMo cat 2-4, unlabeled — see methodology) | 1/7 | 14% |
| 3 | single-hop / temporal / open-domain (LoCoMo cat 2-4, unlabeled — see methodology) | 1/1 | 100% |
| 4 | single-hop / temporal / open-domain (LoCoMo cat 2-4, unlabeled — see methodology) | 2/9 | 22% |
| 5 | adversarial (unanswerable) | 5/5 | 100% |
| **Overall** | | **10/25** | **40%** |

## Failures

- **(cat 2)** When did Caroline go to the LGBTQ support group?
  - gold: `7 May 2023`
  - candidate: `not mentioned`
  - The candidate claims the information is 'not mentioned' when the gold answer shows a specific date (7 May 2023) was provided. This is incorrect.
- **(cat 2)** When did Melanie paint a sunrise?
  - gold: `2022`
  - candidate: `not mentioned`
  - The candidate claims the information is 'not mentioned' when the gold answer shows a specific year (2022) was provided.
- **(cat 1)** What did Caroline research?
  - gold: `Adoption agencies`
  - candidate: `Not mentioned.`
  - The candidate claims the information is 'Not mentioned', but the gold answer shows 'Adoption agencies' is the correct answer that should have been found in the source material.
- **(cat 2)** When did Melanie run a charity race?
  - gold: `The sunday before 25 May 2023`
  - candidate: `not mentioned`
  - The candidate claims the information is 'not mentioned', but the gold answer provides a specific date reference ('The sunday before 25 May 2023', which would be May 21, 2023). This indicates the information was actually present in the source material, making the candidate's answer incorrect.
- **(cat 2)** When is Melanie planning on going camping?
  - gold: `June 2023`
  - candidate: `Next month`
  - The candidate answer 'Next month' is a relative time reference that cannot be verified as equivalent to 'June 2023' without knowing the current date context. The gold answer provides a specific absolute date while the candidate is vague and context-dependent.
- **(cat 1)** What is Caroline's relationship status?
  - gold: `Single`
  - candidate: `Not mentioned`
  - The candidate claims the information is not mentioned, but the gold answer shows it is 'Single', indicating the information was present in the source material.
- **(cat 2)** When did Caroline give a speech at a school?
  - gold: `The week before 9 June 2023`
  - candidate: `Last week`
  - The candidate answer 'Last week' is relative and ambiguous without knowing the current date, whereas the gold answer provides a specific absolute time reference ('The week before 9 June 2023'). Without context about when 'last week' was stated, we cannot verify they refer to the same time period.
- **(cat 2)** When did Caroline meet up with her friends, family, and mentors?
  - gold: `The week before 9 June 2023`
  - candidate: `Last week (met up with Melanie); other meetings not mentioned`
  - The candidate provides a vague timeframe ('last week') without specifying the actual date range, and adds uncertainty by stating 'other meetings not mentioned' which contradicts the definitive gold answer of 'the week before 9 June 2023'. The candidate does not convey the same specific factual information as the gold answer.
- **(cat 4)** What did the charity race raise awareness for?
  - gold: `mental health`
  - candidate: `Not mentioned`
  - The candidate claims the information is 'Not mentioned', but the gold answer shows it should be 'mental health'. This is a direct contradiction - the candidate failed to identify information that was present.
- **(cat 4)** What did Melanie realize after the charity race?
  - gold: `self-care is important`
  - candidate: `Not mentioned`
  - The candidate claims the information is 'Not mentioned' when the gold answer indicates that self-care being important should be found in the source material.
- **(cat 4)** How does Melanie prioritize self-care?
  - gold: `by carving out some me-time each day for activities like running, reading, or playing the violin`
  - candidate: `As a way to stay present and attentive for her family.`
  - The candidate answer describes a general purpose or goal ('stay present and attentive for her family') rather than the specific activities mentioned in the gold answer (carving out me-time for running, reading, or playing the violin). These are fundamentally different types of answers - one lists concrete activities while the other states a motivation.
- **(cat 4)** What are Caroline's plans for the summer?
  - gold: `researching adoption agencies`
  - candidate: `Not mentioned`
  - The CANDIDATE claims the information is not mentioned, but the GOLD answer shows it should be known as 'researching adoption agencies'.
- **(cat 4)** What type of individuals does the adoption agency Caroline is considering support?
  - gold: `LGBTQ+ individuals`
  - candidate: `Not mentioned`
  - The candidate claims the information is not mentioned, but the gold answer indicates that LGBTQ+ individuals is the correct answer that should have been found in the source material.
- **(cat 4)** Why did Caroline choose the adoption agency?
  - gold: `because of their inclusivity and support for LGBTQ+ individuals`
  - candidate: `Not mentioned`
  - The candidate claims the information is 'Not mentioned' when the gold answer indicates a specific reason (inclusivity and LGBTQ+ support) was actually provided in the source material.
- **(cat 4)** What does Melanie think about Caroline's decision to adopt?
  - gold: `she thinks Caroline is doing something amazing and will be an awesome mom`
  - candidate: `Melanie has been supportive, offering supportive words about Caroline's adoption journey.`
  - The candidate is too vague and generic, stating only that Melanie has been 'supportive' without conveying the specific facts in the gold answer: that she thinks Caroline is doing something amazing and will be an awesome mom. The gold answer contains specific positive sentiments that are not captured in the candidate's general statement about being supportive.

## Limitations

- Single conversation, single 58-turn window, one run — not the full 10-conversation official benchmark. Treat this as a spot-check with real external data, not a reproducible LoCoMo leaderboard number.
- The speaker-reframing (third-party voice for the non-'user' speaker) is a structural compromise this project's schema requires; it may itself lose some nuance LoCoMo's questions expect from first-person phrasing.
- LLM-as-judge grading (categories 1-4) is indicative, not authoritative, same caveat as the other eval reports.
