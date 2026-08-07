# AgenticEval

Standalone, black-box comparison framework for two versions of an agentic Q&A
system. The evaluator does not import either system under test.

Start with:

```bash
python -m agentic_eval validate \
  --config experiments/examples/compare_versions.example.yaml

python -m agentic_eval run \
  --config experiments/examples/compare_versions.example.yaml
```

`experiment.repeats` is the shared repetition count `k`. Each system receives
the same question set `k` times. The resulting physical runs are reused for all
three analyses: latency/resources, orchestration consistency, and content
quality. There is no separate content-repeat setting that could accidentally
evaluate a different sample.

For every system/question, latency output contains raw run values plus mean,
median, standard deviation, p95, maximum, and Tukey-IQR outliers. Token counts,
LLM-call counts, and retry counts receive the same distribution summary; retry
rate is the percentage of instrumented runs with at least one retry.

Consistency is reported at three levels:

- team construction: exact modal-team consistency and pairwise team Jaccard;
- tool usage: tool-name consistency plus normalized tool-call consistency,
  including arguments and repeated calls;
- subqueries: mean pairwise, per-specialist lexical-token Jaccard.

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
[the memory question example](experiments/examples/questions.memory.example.yaml).

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
  adapters/        agenticsys_sse.py                 black-box system access
  modules/         consistency.py  content.py        one file per dimension,
                   latency.py  memory.py              each exposing section(rows)
  content/         document -> claims -> evidence    the content cascade
                   -> numeric -> verify -> oracles
                   -> metrics -> aggregate -> report
  scoring.py       composes modules/, compares systems
  runner.py        drives the systems; writes runs.jsonl
  cli.py           thin dispatcher
```

`agentic_eval.content` re-exports its public surface, so callers import from
the package, not its internals. `agentic_eval.modules.EVAL_MODULES` is the one
registry a dimension is selected from; `scoring.SECTION_BUILDERS` derives from
it.

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
oracles/     what is true      examples/   templates
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

[`experiments/questions/simple.yaml`](experiments/questions/simple.yaml) is the
simple-question suite — six questions whose right answer a Python script can
compute, so correctness never depends on a judge. Run it with:

```bash
python -m agentic_eval run --config experiments/configs/simple_questions.yaml
```

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
  --config experiments/examples/compare_versions.example.yaml \
  --runs experiments/results/<run-folder>/runs.jsonl
```

Use `--limit 2` for a low-cost prompt/evidence calibration pass before judging
all repetitions. Use `--resume` after an interruption to keep completed judge
results and evaluate only missing answers. The command writes:

- `content_evaluations.jsonl`: claims, evidence links, verdicts, and judge-call
  telemetry for every answer;
- `content_summary.json`: per-system/mode/question means, sample standard
  deviations, min/max values, and values by run index across all `k` runs;
- `content_comparison.md`: human-readable baseline/candidate scorecard.
- `content_walkthrough.md`: raw answer → atomic facts → numeric verdicts, one
  marked-up section per answer, so a rate can be traced back to the span that
  produced it;

- `answer_comparison.html`: a self-contained viewer for **one sampled repeat** —
  both versions' raw answers side by side, their atomic facts with the cascade
  markers, and the per-question metrics, on three tabs;

all inside `<run>/content` by default, rather than beside the run's own outputs.

Regenerate the viewer alone, choosing which repeat to sample:

```bash
python -m agentic_eval compare-answers \
  --evaluations experiments/results/<run>/content/content_evaluations.jsonl
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

- `evidence_review.jsonl` and `evidence_review_key.csv`: blinded Phase-B
  claim/evidence review packets with the system identity kept in a separate key.

Regenerate the walkthrough alone, without re-judging:

```bash
python -m agentic_eval walkthrough \
  --evaluations experiments/results/<run-folder>/content_evaluations.jsonl
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
`trace_failure_counts` decomposes the failing branch by cause.

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
verdict alone. Every factual claim receives an `evidence_grounding` tier:

- `primary`: cited to a structured tool result or a canonical fact;
- `secondary`: cited only to another agent's findings, or to a prose blob such
  as a report file the system itself wrote earlier in the run;
- `unresolved`: every cited evidence ID is absent from the ledger;
- `none`: nothing cited.

`qualitative_grounding_rate` is the share of non-numeric factual claims that
resolve at any tier; `qualitative_primary_grounding_rate` is the strict share
backed by an actual measurement. The scorecard shows the strict one.

One call is one ledger entry. The SSE `agent_completed` event and the trace row
capture the same call, so entries are merged on `call_id`, provenance is
unioned, and the dropped ID is kept as an alias so either still resolves.

Python resolves the judge-provided evidence paths, recomputes direct or derived
values, checks tool-call success, and applies deterministic rubric constraints.
The LLM judges whether the selected tool, fields, filters, window, population,
denominator, and aggregation are semantically appropriate. Missing trace
instrumentation is `N/A`, not hallucination. Must-haves come only from the
configured rubric; the LLM judge is not allowed to invent them.

`content_evaluation.llm` uses an independent OpenAI-compatible JSON-mode client
with pinned model, temperature, timeout, and retry settings. It does not import
either AgenticSys checkout. The normal `OPENAI_API_KEY` environment variable is
used unless `api_key_env`, `api_key`, or `base_url` is configured explicitly.
