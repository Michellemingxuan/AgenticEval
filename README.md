# AgenticEval

Standalone, black-box comparison framework for two versions of an agentic Q&A
system. The evaluator does not import either system under test.

Check a config without starting anything, then run one end to end:

```bash
python -m agentic_eval validate \
  --config experiments/templates/compare_versions.template.yaml

pip install -e .                              # puts `agentic-eval` on PATH
bin/compare --config experiments/configs/series_abcd.yaml --scope smoke
```

`bin/compare` is the whole chain — run both systems, judge the answers, build
the page. [Running a comparison](#running-a-comparison) covers its flags;
`--scope smoke` is the cheap end-to-end check to start from.

`experiment.repeats` is the shared repetition count `k`. Each system receives
the same question set `k` times. The resulting physical runs are reused for all
four analyses: latency/resources, orchestration consistency, memory use, and
content quality. There is no separate content-repeat setting that could
accidentally evaluate a different sample.

For every system/question, latency output contains raw run values plus mean,
median, standard deviation, min, p95, maximum, and Tukey-IQR outliers — the
last exposed only at k≥4, since with three observations an "outlier" is an
artifact of the method. Token counts, LLM-call counts, and retry counts receive
the same distribution summary; retry rate is the percentage of instrumented
runs with at least one retry.

Consistency is reported at three levels:

- team construction: `team_exact_consistency` and `team_pairwise_jaccard`;
- tool usage: `tool_exact_consistency` over names, plus
  `tool_call_pairwise_multiset_jaccard` over normalized calls — arguments and
  repeated calls included;
- subqueries: `subquery_pairwise_similarity`, mean per-specialist lexical-token
  Jaccard.

Tool-call signatures are normalized before comparison, or the same work counts
as a difference: nested JSON arguments are canonicalised, a batch call is
expanded into the N single calls it stands for
(`batch_summarize_trend` → N × `summarize_trend`), equivalent month bounds are
widened to one form, and column aliases are resolved from the result payloads.
`tool_call_runs_not_comparable` marks the pairs where that could not be done,
rather than scoring them as disagreement.

Memory evaluation is explicitly annotated in the question set:

```yaml
- name: payment_returns_followup
  text: When did those returned payments occur?
  memory_required: true
```

For each required question, `memory_used` is `true` when the run records a
memory-use signal and `false` otherwise. `memory_hit_rate` is the percentage of
required runs where `memory_used` is true. Questions marked false or left
unannotated are excluded. See
[the memory question template](experiments/templates/questions.memory.template.yaml).
(`text:` and `question:` are accepted interchangeably for the prompt.)

See [the comparison framework guide](docs/agentic-eval-comparison-framework.md)
for configuration, experiment design, metrics, and review workflow.

Content-quality evaluation is specified separately in
[the content evaluation design](docs/content-evaluation-design.md), including
atomic facts, evidence coverage, numeric traceability, must-have judging,
logical verification, and the blind human-review viewer.
The rendered version is available as
[content-evaluation-design.pdf](docs/content-evaluation-design.pdf).

## Module layout

Each evaluation dimension is its own module, so one can be read, tested, or run
without the others.

```
agentic_eval/
  common/          coerce.py  stats.py  io.py        shared primitives
  adapters/        agenticsys_sse.py  base.py        black-box system access
  dimensions/      consistency.py  content.py        one file per dimension,
                   latency.py  memory.py              each exposing section(rows)
  content/         document -> claims -> evidence    the content cascade
                   -> numeric -> verify -> oracles
                   -> metrics -> aggregate -> verdicts
  render/          page.py  markdown.py              the viewer and the reports
                   markers.py  run_summary.py
  config.py        one config object, paths resolved
  layout.py        every path in a run folder, in one place
  cases.py         which cases a run covers
  workers.py       per-worker ports, logs and trace DBs
  memory_store.py  snapshot/restore of the system's memory store
  toolcalls.py     tool-call payloads, outcomes and counts
  process.py       launching and stopping the systems
  review.py        blinded human-review aggregation
  scoring.py       composes dimensions/, compares systems
  runner.py        drives the systems; writes runs.jsonl
  cli.py           thin dispatcher
```

`agentic_eval.content` re-exports its public surface, so callers import from
the package, not its internals. `agentic_eval.dimensions.EVAL_MODULES` is the
one registry a dimension is selected from; `scoring.SECTION_BUILDERS` derives
from it. `layout.RunLayout` owns every filename a run writes, so the folder
shape can be changed without grepping for string literals.

The four dimensions are `consistency`, `content`, `latency`, and `memory`.
Per-invocation overrides let one YAML serve a whole sweep, without templating a
config file per case:

```bash
python -m agentic_eval run --config experiments/configs/compare_versions.yaml \
  --case-id 366132845011 --repeats 10 --mode stateful \
  --eval-module content,latency          # comma-separated,
  # --eval-module content --eval-module latency   # repeated, or
  # --eval-module all                             # everything
```

Selections resolve in registry order, so summaries stay comparable across
invocations, and an unknown name raises rather than silently dropping a
dimension. `validate` accepts the same flags, so a sweep can be checked before
it spends anything. Omitting `--eval-module` computes every dimension.

The `content` module has two layers. Its `section` is free — provenance
completeness and a structural score read off the run record. The evidence-bound
cascade in `agentic_eval.content` costs LLM calls, so it runs only when
`content` is named explicitly; `--eval-module all` still defers to
`content_evaluation.auto_run`, so a broad sweep cannot start spending by
omission.

## Experiment inputs

`experiments/` separates inputs by role — see
[its README](experiments/README.md):

```
configs/     what to run       questions/  what to ask
oracles/     what is true      templates/  starting points to copy
results/  traces/              output, gitignored
```

Relative paths inside a config resolve from **that config file**, so moving one
between folders means re-checking its `questions_file`, `rubric_file`,
`output_dir`, `oracle_cwd`, `process.cwd`, and `trace_db`.

## Question sets

A question set is owned by AgenticEval, not by the system under test. Sourcing
it from the candidate's own test directory couples the two: the baseline
checkout carries its own copy, and if the copies drift the systems are
answering different questions while the report still calls it a comparison.

The working suite is
[`experiments/configs/series_abcd.yaml`](experiments/configs/series_abcd.yaml)
— **17 questions in four sets**, each set its own file and its own session:

| set | questions | settled by |
|---|---|---|
| `series_a` | 7 | an oracle script — no judge is consulted |
| `series_b` | 8 | a rubric, as one eight-turn conversation |
| `series_c` | 1 | a rubric: summarise the report, or say there is none |
| `series_d` | 1 | series B's last question, asked cold |

At the config's `repeats: 3` over two cases that is 204 records for two
systems. [`questions/simple.yaml`](experiments/questions/simple.yaml) is a
smaller six-question oracle-only suite, run with
`--config experiments/configs/simple_questions.yaml`.

Question, expectations, and oracle are one unit: each question carries its own
`evaluation` block inline, so there is no name-matching step to get wrong.

`content_evaluation.rubric_file` still exists for swapping a stricter rubric
onto a fixed question set. It merges **by name**, so it now raises on an entry
matching no question — a dead check that still looks configured is worse than a
missing one, because the suite runs, the expectation never fires, and the report
shows a blank rather than a failure.

Questions run in question-set order. In `cold` mode every turn is preceded by a
reset, so the questions are independent; in `stateful` mode one reset opens the
repeat and the questions share a session, each recording its
`sequence_position`. `both` runs cold first, then stateful. Set
`experiment.mode`, or override per invocation with `--mode`.

A follow-up question only has a referent in `stateful`, so a suite containing
one is measuring different things in the two modes: cold probes how the system
handles a dangling reference, stateful measures whether it carried the prior
turn. `memory_hit_rate` is only meaningful in the latter. System order is
shuffled from the seeded RNG at every step, so machine load or cache warmth
cannot systematically favour one side.

## Content evaluation

Keep content judging outside the timed system runs. First run the comparison,
then point the evaluator at the resulting `runs.jsonl`:

```bash
python -m agentic_eval evaluate-content \
  --config experiments/configs/series_abcd.yaml \
  --runs experiments/results/<run-folder>/runs.jsonl
```

Calibrate before spending a full pass: `--limit 2` judges the first two
eligible answers, and `--question <name>` judges one question across every
repeat and case. Use `--resume` after an interruption to keep completed judge
results and evaluate only missing answers.

Re-judging a subset **preserves the questions it did not touch** — an earlier
version truncated the file to what it had just judged, which silently discarded
eight completed evaluations. The command writes, inside `<run>/`:

- `content/evaluations.jsonl`: claims, evidence links, verdicts, and judge-call
  telemetry for every answer;
- `content/summary.json`: per-system/mode/question means, sample standard
  deviations, min/max values, and values by run index across all `k` runs;
- `content/comparison.md`: human-readable baseline/candidate scorecard;
- `content/walkthrough.md`: raw answer → atomic facts → numeric verdicts, one
  marked-up section per answer, so a rate can be traced back to the span that
  produced it;
- `content/answer_comparison.html`: the self-contained viewer — metrics over
  every repeat, and the answers for **one sampled repeat and case** side by
  side, on two tabs;
- `review/answers.csv`, `review/answers.key.csv`, `review/evidence.jsonl`,
  `review/evidence.key.csv`: blinded human-review packets, with the system
  identity held in a separate key file.

Regenerate the viewer alone, choosing which repeat to sample:

```bash
python -m agentic_eval compare-answers \
  --evaluations experiments/results/<run>/content/evaluations.jsonl
```

Everything else is defaulted. `--baseline` and `--candidate` come from the run's
`manifest.json`, which is the only correct source: inferring them from the
systems present means sorting names, and `sorted(["current", "previous"])`
assigns the roles backwards so every delta carries the wrong sign. `--mode`
takes the manifest's mode when the records actually carry it — a run recorded as
`both` is not itself a filter — and `--run-index` takes the first repeat
present, so regenerating shows the same sample rather than silently changing
which run is on screen. Override any of them:

```bash
  ... --baseline previous --candidate current --mode stateful --run-index 2
```

A repeat is shown, never an average: averaging answers is meaningless, and
reading one real pair is how a rate gets sanity-checked. The command prints
which systems it picked and where they came from.

Regenerate the walkthrough alone, without re-judging:

```bash
python -m agentic_eval walkthrough \
  --evaluations experiments/results/<run-folder>/content/evaluations.jsonl
```

Numeric content is reported as a three-step cascade, each step's denominator
being the previous step's yes-branch:

1. **supported by numbers** — asked of every atomic factual claim, not only the
   ones already carrying a number;
2. **numeric traceability** — the value is located in a real tool output *and*
   correctly obtained from it, by direct read or by a recomputed derivation
   (`max`, `min`, `mean`, `sum`, `count`, `difference`, `product`, `ratio`,
   `percent_change`), so "the peak TSR" is checked as `max(series)`;
3. **correct tool usage** → **actually correct**.

A claim that states no number can still assert a checkable relation. For
threshold, comparison, and ranking claims the judge returns `relations`
(`{left, operator, right}`, both sides addressing real tool values) and Python
evaluates them: "TSR was below its risk threshold" is verified as
`8.4 < 20` with both operands read from the tool output. A relational claim with
no relation supplied is `relation_not_supplied`, not exempt.

Claim extraction stays with the LLM because the unit of an atomic fact depends
on stance: a figure the answer reports in order to refute it belongs to the same
claim as the correction, and splitting it invents an assertion the answer never
made. Such figures are marked `quoted` and excluded from the numeric funnel.
Hedged values carry a comparator (`~28+` is checked as `>= 28`), and numbers
that merely name a metric (`30+ DPD`) are immaterial.

**Hallucination** spans steps 2 and 3: a value locatable in no tool output and
derivable from none, or a materially wrong tool, so a real number answers the
wrong question. A located-but-mismatched value is an arithmetic defect, not a
hallucination; a run with no captured provenance is missing instrumentation.
`failure_counts` decomposes the failing branch by cause, and a handful of
causes are **disclosed but never charged** — `not_a_quantity` and
`stated_constant` are judge-side errors, not the system's, so they appear in
the breakdown and stay out of the numerator.

For questions a script can answer outright, no judge is consulted. Add
`expected_answers` to a question's rubric with either a literal `value` or a
`command`; Python computes the truth and compares it against the numbers the
answer states, reporting `expected_answer_accuracy_rate`. `kind: boolean`
checks a yes/no ground truth against rubric-supplied `affirmative_patterns` and
`negative_patterns`, with negation taking precedence when both match; numeric
items accept `accept_patterns` for values normally written in words, such as a
count of zero. See
[the simple-question suite](experiments/questions/simple.yaml) and
[its oracles](experiments/oracles/case_facts.py).

Claims without numbers get their own measurement, because the numeric funnel is
`not_applicable` for them and would otherwise leave them scored on the judge's
verdict alone. Every factual claim gets a `grounding_kind`, and these are the
markers the page shows:

| kind | marker | means |
|---|---|---|
| `factual` | ◆ | a route to operations on specific tables, *and* that route answers the question asked |
| `report` | ◇ | curated report material that resolves in the ledger |
| `none` | ○ | neither — the page prints the reason underneath |

`factual_grounded_rate` and `report_grounded_rate` are the two shares;
`grounded_rate` is their sum. Which one *should* be high depends on the
question: series C asks for a summary of the report, so ◇ is the target
there — but on a case with no curated report, analysing the live tables and
saying so is the correct answer, and ◆ is right. What is always a defect is
misattribution: presenting live analysis as the report's account, or citing a
report that does not say it.

Separately, `evidence_resolution` records whether the cited ids were found at
all (`resolved`, `unresolved`, `none`), and restatements — claims repeating an
earlier claim in the same answer — are counted once and not re-verified.

### How rates are averaged

Per-question judgements — accuracy and must-have coverage — are averaged **at
each level**: a question's own rate over its repeats and cases first, then the
mean of those rates over questions for a set, then over sets for the overview.
Pooling instead would let a question asked about more cases count for more, and
the aggregate denominator would read as questions while counting answers: with
one oracle-bearing question the pooled figure showed `3/4`, which looks like
three of four questions and is three of four answers to a single one. Aggregate
levels label their denominator `(N questions)`; at a single question the level
collapses and the answer counts are shown instead.

Must-haves score **1, 0.5 or 0** per applicable point, with **no weights** —
every point counts once — then average over repeats, questions and cases the
same way. Rates are shown to the precision they have: `87.5%`, not `88%`.

Everything else — claim counts, grounding shares, tool-call totals — pools over
questions and repeats, because those are properties of answers rather than
judgements about questions. Consistency is the exception in the other
direction: it is computed *within* a case across that case's repeats, then
averaged over cases, since two customers legitimately draw different
specialists and different tables.

One call is one ledger entry. The SSE `agent_completed` event and the trace row
capture the same call, so entries are merged on `call_id`, provenance is
unioned, and the dropped ID is kept as an alias so either still resolves.

Python resolves the judge-provided evidence paths, recomputes direct or derived
values, checks tool-call success, and applies deterministic rubric constraints.
The LLM judges whether the selected tool, fields, filters, window, population,
denominator, and aggregation are semantically appropriate. Missing trace
instrumentation is `N/A`, not hallucination. Must-haves come only from the
configured rubric; the LLM judge is not allowed to invent them.

`content_evaluation.llm` uses an independent JSON-mode client with pinned
model, temperature, timeout, and retry settings. It does not import either
AgenticSys checkout. The normal `OPENAI_API_KEY` environment variable is used
unless `api_key_env`, `api_key`, or `base_url` is configured explicitly; see
[the judge transports](#the-judge-transport) for the private environment's
gateway.

The cascade is three judge calls per answer, and the split between the last two
is the one that matters:

```
extract      the question and the answer      -> claims
evidence     claims + ledger + the run trace  -> pointers, routes, must-haves
eligibility  routes + briefs + earlier turns  -> one verdict per claim
```

A judge that has already decided a claim is sound will describe a route that
justifies it, so the route description is written before any verdict exists.
The ledger is roughly 13k tokens and travels exactly once. A fourth call,
`memory_leverage`, runs only when the turn was actually offered memory —
asking it about an empty set would spend a call to learn nothing.

## Running a comparison

`bin/compare` does the whole chain — run both systems, score the answers, build
the page — and every knob is a flag, so a sweep never needs a new config file.

```bash
bin/compare --config experiments/configs/series_abcd.yaml \
  --question b2_tsr_cdss_reaction --question b3_bureau_during_reaction \
  --repeats 1 --case-id 366132845011
```

| flag | changes |
|---|---|
| `--scope` | a saved selection from `experiment.scopes`, e.g. `smoke` |
| `--question-scope` | a saved question selection from `experiment.question_scopes` |
| `--question` | which questions run; repeatable or comma-separated |
| `--repeats` | k |
| `--mode` | `cold` (reset each turn) or `stateful` (one session per repeat) |
| `--case-id` | the case both systems analyse; repeat it to cover several |
| `--cases-from` | read the case id list from a data directory; run them all |
| `--workers` | how many sessions run at once (ceiling 8) |
| `--baseline-cwd` / `--candidate-cwd` | which two checkouts are compared |
| `--env-file` | where `OPENAI_API_KEY` is read from |
| `--skip-run` + `--runs` | re-judge answers already on disk, starting no system |

### Smoke run

Before spending the judge budget on a full pass, check the whole chain on a
fraction of it:

```bash
bin/compare --config experiments/configs/series_abcd.yaml --scope smoke
```

`--scope` applies a **run scope** — a saved preset that pins the whole shape of
a run: questions, k, cases, workers. `smoke` is four questions across three
sets at k=2 over both cases: **32 records against the full run's 216**, and
still enough to exercise the per-set metrics tables and both the case and
repeat selectors.

Read a smoke run as a check that the chain works, not as a measurement — k=2
over four questions is far too little to compare two systems on.

### Question scopes

A **question scope** narrows what is asked and nothing else. k, cases and
workers stay exactly as the config has them, so every rate rests on the repeats
the config intends:

```bash
bin/compare --config experiments/configs/series_abcd.yaml --question-scope series_b
```

| | `--scope` | `--question-scope` |
|---|---|---|
| questions | pinned | pinned |
| k, cases, workers | pinned | **from the config** |
| for | spending little | measuring a subset |

`series_abcd.yaml` defines `series_a`…`series_d` and `cold_vs_warm` (B8 and D1
— the same question with and without the eight turns before it, in two sets so
they stay two sessions). A question scope carrying `repeats` or `cases` is
refused rather than honoured: changing k there would hand back rates that look
like the config's and are not, and nothing downstream would say so.

Both flags are alternatives — a run scope already pins the questions.

### Selecting questions checks the parent chain

A follow-up asked without the turn it refers to has no referent — `b3` asks
about "the reacting period" that `b2` established. The system answers something
vague, every metric on it reads badly, and the conclusion drawn is about the
selection rather than the system. So naming `b4` alone is refused, and the
error names the whole chain:

```
follow-up question(s) selected without the turn they refer to:
b3 needs b2, b4 needs b3. Add --question b2 --question b3
```

### Rescoring without re-running

Metric artifacts are written once, by `run`. When a scoring bug is fixed
afterwards the answers are still good — only the numbers derived from them are
stale, and re-running both systems to correct arithmetic wastes the run and
changes the sample, so the two readings stop being comparable.

```bash
python -m agentic_eval rescore --runs experiments/results/<run>/runs.jsonl
```

Recomputes `metrics/summary.json`, `comparison.json` and `comparison.md` in
place from the answers already on disk, reading baseline/candidate from the
run's `manifest.json`. Follow it with `compare-answers` to refresh the page.

### Several cases in one run

Every question is asked about every case, so a difference between the two
systems can be told apart from a quirk of one customer's data — a candidate
that wins on one case and loses on two has not won.

```bash
bin/compare --config experiments/configs/series_abcd.yaml \
  --cases-from ../AgenticSys_v2/data_tables/real
```

`--cases-from` takes only the **id list** from that directory. It is not a data
source: AgenticEval never reads a table, and each system answers about those
ids using its own checkout's `data_tables/`. Listing from AgenticSys_v2 just
asks "which cases exist?" — the baseline and the candidate still run on their
own data.

Metrics pool over cases and repeats alike; consistency is the exception, and is
measured *within* each case before averaging, since two customers legitimately
draw different specialists and different tables. The page gets a case selector
above the repeat selector — together they pick which answers are on screen,
while every metric stays totalled over all of them.

### A set is a conversation

Each `questions_file` is a **set**, and a set is a conversation: in stateful
mode it gets its own session, so questions in different sets never see each
other's turns. Records carry `question_set`, and the page grows a metrics
section per set alongside the overall one.

This is what makes `series_d` work. D1 asks series B's final question ("Any
model opportunities?") with none of B's context — run in one flat session it
followed B8, the identical question, two turns later, so it measured the QA
cache rather than cold discovery. As separate sets, D1 is turn 1 of its own
conversation, and the judge's prior-turn context stops at the set boundary too.

Set a `question_set:` key at the top of a questions file to name it something
other than the filename.

### Running sessions concurrently

`--workers N` (or `experiment.workers`) runs N sessions at once. The unit of
parallelism is the **session** — one (case, set, repeat) — so the questions
inside one still go in the configured order, on one server, and a follow-up
still lands after the turn it refers to.

**A worker owns whole cases, not an arbitrary slice of sessions.** Two workers
on one case would collide even with separate servers, because the memory store
is shared and `/rewind` purges it *by case id, across processes*: one worker
opening a session would delete the memories the other is mid-way through
writing. Owning the case end to end closes that window, and it is what makes
a case's repeats strictly sequential — which is what makes them independent.
With more workers than cases the extras stay idle, and the run says so.

**Each worker starts its own server instance per system.** That is not a
performance choice, it is required for correctness: the system under test keeps
its data gateway and catalog as process-globals and re-scopes them to a case at
the start of every turn, so two sessions sharing a process can execute against
each other's tables. `server.py` documents the race itself:

> turns on ONE case are serialized by `sess.turn_lock`, but two different
> cases' turns can still interleave on these shared globals

Worker *w* takes each system's configured port plus *w*, and shifts
`config.base_url`, `process.env.PORT`, the stdout log **and the trace DB**
together — a mismatch is refused rather than silently health-checking against
another worker's server, or reading another worker's traces. Space your
systems' ports at least `workers` apart; a collision is caught before anything
starts. The ceiling is 8, because N workers means N full servers per system.

Two consequences worth knowing:

- **Latency is measured under contention.** N servers share one machine, so the
  numbers are comparable within the run but not against a serial one. The
  manifest records `latency_measured_concurrently` so a later reader knows.
- **Record order stays deterministic.** Results are reassembled in session
  order, and each session's baseline/candidate ordering is seeded from its own
  identity rather than drawn from one shared generator — so the same config and
  seed produce the same `runs.jsonl` whatever order the workers finish in.

### Leaving the memory store as the run found it

The system writes a `qa_turn` memory to a **real** store on every turn, and
`/rewind` purges only the case it clears — so whatever the last session of each
case wrote outlives the run, and is still there for the next one. Measured on
one afternoon of small runs: thirteen memories left behind, from both systems,
each readable by the other for the same case. A test that changes the
environment it measures is not repeatable, and the residue is
indistinguishable from real operating history.

```yaml
experiment:
  memory_store:
    url: "${AMEM_STORE_URL:-http://127.0.0.1:6333}"
    collection: "${AMEM_COLLECTION_NAME:-amem_memories}"
    restore_after_run: true
```

With this set, the runner snapshots every point — **payload and vector** —
before anything starts, and restores exactly that afterwards: same ids, same
payloads, same vectors. Re-inserting payloads alone would leave memories that
exist but can never be retrieved, which is worse than deleting them, because
the store then looks intact.

A store that cannot be read is reported and **left alone**. "No snapshot" and
"snapshot of nothing" must never look alike, or a failed read would authorise
wiping everything. The run prints the count before and the footprint after:

```
  memory store: 0 memories before the run
  memory store: removed 24 written by this run, reinstated 0; back to 0
```

Repeat independence is a separate mechanism — worker case ownership, above —
because restoring at the end of a run says nothing about what one repeat can
see of another during it.

Case ids are the data directory names **exactly** as they appear on disk, and
one of the real ones ends in a space. Quote it (`--case-id '11854808010 '`), or
let `--cases-from` find it: the system under test keys on the raw folder name,
so a stripped id matches no case and the run completes with every answer empty.

### The interpreter the systems need

The system under test needs an interpreter carrying its own dependencies. Point
`AGENTIC_SYS_PYTHON` at it — the configs fall back to `python3`, which only
works if that already has them:

```bash
export AGENTIC_SYS_PYTHON=~/.pyenv/versions/3.11.13/envs/autoAI/bin/python
```

### The judge transport

The judge is separate from the systems under test and has two transports:
`openai` (default) and `safechain` for the private environment. Both present
`.chat.completions.create` and both honour `response_format`, so JSON mode and
the judging code are identical either way — set
`content_evaluation.llm.backend`, or `LLM_BACKEND=safechain`. SafeChain is
constructed by AgenticEval rather than borrowed from the system being
evaluated: a judge is not independent if a change to its subject can change it.

**`LLM_BACKEND` overrides the config file** — the only setting that does. The
transport is a fact about *where the run happens*, and the same checkout runs
in both places; every shipped config pins `backend: openai` because that is
what dev has. Confirm which one a run will use before spending a judge pass:

```bash
LLM_BACKEND=safechain agentic-eval validate --config experiments/configs/series_abcd.yaml \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['content_evaluation'])"
# {'enabled': True, 'auto_run': False, 'model': 'gpt-4.1', 'backend': 'safechain'}
```

SafeChain reads its own configuration through `ee_config`, which wants
`CONFIG_PATH` — so the judge needs the same environment the system under test
runs in, not just an API key. `run`, `validate` and `evaluate-content` fill
gaps in the environment from a `KEY=value` file:

```bash
agentic-eval evaluate-content --config <cfg> --runs <runs.jsonl> --env-file .env
```

Without the flag they look at `$AGENTIC_EVAL_ENV`, then at this repo's own
`.env`. **Already-exported variables always win** — the file fills gaps and
never overrules something a caller set on purpose, which matters when one of
those values decides where judging traffic goes. The path and the count are
printed to stderr, so a file that changes the backend leaves a trace. A file
named by flag or variable and not found is an error, not a shrug.

`bin/safechain-doctor` checks the whole transport before a run depends on it,
in escalating stages:

```bash
bin/safechain-doctor            # environment, imports, ee_config — no network
bin/safechain-doctor --build    # also await amodel(), which acquires a token
bin/safechain-doctor --call     # also one real JSON-mode round trip
```

The default stage is offline and answers the question that costs a run to
discover otherwise: whether `CONFIG_PATH` is set, whether it points at a file
that exists, and — when `ee_config` is importable — which variables and default
filenames the package itself reads. Secrets are reported as `set (N chars)`,
never echoed, because this output gets pasted into tickets.

`--call` goes through `agentic_eval.llm_judge`, not a hand-rolled chain: a
probe with its own code path can pass while the judge fails, which is the one
outcome a probe must not have. Exit status is 0 only if every stage that ran
passed.

The gateway also needs `nest-asyncio` (`pip install -e '.[safechain]'`), plus
`safechain` and `langchain-core` from the private index — those two are not
declared as dependencies, since naming them would make the package
uninstallable anywhere else. `SAFECHAIN_MODEL` overrides the model id if the
gateway names it differently. If the compliance template cannot be imported
the judge refuses to run rather than invoking the model bare.

Anything not passed falls back to the config. Names are validated *before*
either system starts, so a typo costs a second rather than a ten-minute run —
`agentic-eval validate --config <cfg> --question <name>` prints the plan alone.

## What a run leaves behind

```
manifest.json                   what was run: systems, mode, repeats, seed, and
                                whether latency was measured concurrently
runs.jsonl                      one record per answer: the answer, the team, the
                                evidence ledger, tokens, latency, memory offered
content/evaluations.jsonl       the same answers scored: claims, per-claim grounding
                                and routes, must-haves, oracles, metrics
content/answer_comparison.html  the page — metrics first, then answers side by side
                                with their claims and markers
content/walkthrough.md          the same as text: answer -> claims -> numbers
content/comparison.md           one scorecard table, both systems, every question
content/summary.json            aggregated metrics, for a script to read
metrics/                        summary.json, comparison.json, comparison.md —
                                consistency, memory and latency, per question
review/                         blinded review packets and their separate keys
logs/                           each system's server log
```

`runs.jsonl` is the only irreplaceable artifact; everything else is derived
from it and can be rebuilt. `rescore` recomputes `metrics/` and
`evaluate-content` recomputes `content/`, both without touching the systems.

The HTML page is built automatically at the end of `evaluate-content` — there is
no separate step. `agentic-eval compare-answers --evaluations <file>` rebuilds it
alone, which is what to use when only the rendering changed.

Reading the page: it opens on **Metrics** — the overview, then a section per
question set, with each set's questions nested under it. **Answers & claims**
carries the case and repeat selectors, which switch *which answers are shown*
and nothing else: every metric stays totalled over all cases and repeats.

Claim markers: **◆ factual** means the run recorded a route to operations on
specific tables *and* that route answers the question asked; **◇ report** means
the claim relays curated report material that resolves; **○** is neither, and
the red line beneath it says why, with the detail on hover. A row with no
marker is a restatement — counted once, not re-verified, and hidden by default.
