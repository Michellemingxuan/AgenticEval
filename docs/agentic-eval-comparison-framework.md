# Decoupled AgenticSys version comparison

`agentic_eval` is a standalone black-box benchmark, kept outside the
AgenticSys checkouts. It does not import either version under test. A target
adapter converts that system's public request/response protocol into a stable
normalized run record.

The claim-level content layer is defined in
[`content-evaluation-design.md`](content-evaluation-design.md). Its rubric
shape is illustrated by
[`experiments/examples/content_rubric.example.yaml`](../experiments/examples/content_rubric.example.yaml).

For AgenticSys, the built-in `agenticsys_sse` adapter:

1. posts the question to `/api/cases/<case_id>/turn`;
2. reads the matching turn from `/api/cases/<case_id>/stream`;
3. extracts answer, outcome, team, sub-questions, `scope`, and
   `measured_over`;
4. optionally reads that target's trace SQLite file for tokens, LLM calls,
   retries, QA-cache hits, and KB-context exposure.

All target-specific settings live in YAML. Switching a saved version means
changing its `cwd`, command/port, URL, and trace path.

If an older saved version uses different route names, set `healthcheck_path`,
`turn_path`, `stream_path`, `reset_path`, or `question_field` under that
target's `config`; the evaluator core remains unchanged.

## Configuration

Start from
[`experiments/examples/compare_versions.example.yaml`](../experiments/examples/compare_versions.example.yaml).
Each target can either be evaluator-managed:

```yaml
systems:
  previous:
    adapter: agenticsys_sse
    process:
      cwd: ../../AgenticSys_v1
      command: [python, server.py]
      env: {PORT: "49102", TRACE_VIEWER_DISABLE: "1"}
    config:
      base_url: http://127.0.0.1:49102
      case_id: "366132845011"
      trace_db: ./traces/previous.db
```

or already running, by omitting `process`:

```yaml
systems:
  previous:
    adapter: agenticsys_sse
    config:
      base_url: http://127.0.0.1:49102
      case_id: "366132845011"
      trace_db: ./traces/previous.db
```

Paths resolve relative to the YAML file. Environment variables in values are
expanded. The evaluator automatically passes the configured `trace_db` to a
managed AgenticSys process as `NODE_TRACE_DB`, keeping the writer and reader
aligned.

Validate without starting either version:

```bash
python -m agentic_eval validate \
  --config experiments/examples/compare_versions.example.yaml
```

Run the comparison:

```bash
python -m agentic_eval run \
  --config experiments/examples/compare_versions.example.yaml
```

## Experimental convention

Run two configurations rather than mixing interpretations:

- `mode: cold`, `repeats: 10`: every question/version/repeat is reset. Measures
  team, tool, and sub-question consistency; latency/outliers; tokens; calls;
  retries; provenance; and content quality.
- `mode: stateful`, typically `repeats: 3`: each version is reset once per
  sequence, then receives seed/repeat/paraphrase/follow-up questions in order.
  Measures answer-cache and specialist-memory behavior.

`repeats` is the shared `k` for the entire experiment. For a given mode, both
versions answer the identical question set at run indices `1..k`. Latency,
resource use, orchestration consistency, and content quality are computed from
these same stored runs. Content judging happens later, outside the system timer,
but it does not generate a new sample of system answers.

For each system/mode/question group, resource metrics are:

- latency: raw `k` values, mean, median, sample standard deviation, minimum,
  p95, maximum, and Tukey 1.5*IQR outlier count/rate;
- tokens: prompt, completion, and total-token distributions;
- LLM calls: call-count distribution;
- retries: percentage of instrumented runs with at least one retry, total retry
  attempts, and retry-count distribution.

Outlier labeling is enabled only for `k >= 4`. With smaller samples the report
still discloses every value, p95, and maximum but leaves the outlier rate N/A.

Consistency is evaluated only among repetitions of the same question:

- team construction: modal-team exact-match rate, number of unique team
  variants, and mean pairwise Jaccard;
- tool usage: the same metrics for tool names, plus a stricter normalized call
  signature containing the tool name and arguments and preserving repeat calls;
- subqueries: mean pairwise lexical-token Jaccard for each matched specialist,
  with a missing specialist/subquery scored as zero.

Content evaluation scores every one of the `k` answers independently. The
content summary reports the mean, sample standard deviation, minimum, maximum,
and value by `run_index` for every content metric. This prevents one unusually
good or bad generation from being hidden inside a single mean.

### Memory utilization

Memory is evaluated against an explicit question-level requirement rather than
assuming that every later turn should use memory:

```yaml
- name: seed_question
  text: Did the customer have payment returns?
  memory_required: false

- name: followup_question
  text: When did those occur?
  memory_required: true
```

The metric is:

\[
\text{memory hit rate} =
\frac{\text{memory-required runs where memory was used}}
{\text{memory-required runs}}
\]

For each run, `memory_used` is `true` when the adapter observes any memory-use
signal and `false` otherwise. Questions marked `memory_required: false` or left
unannotated do not enter this metric. No memory-quality or memory-correctness
judgment is added here; this dimension measures usage only.

Within each paired trial, baseline/candidate execution order is randomized
using the configured seed. The systems are not run simultaneously, avoiding
one version changing the other's backend load during a latency measurement.
The report also computes paired candidate-minus-baseline deltas, deterministic
bootstrap 95% confidence intervals for the mean delta, and candidate
win/tie/loss counts. Pairing removes much of the question/repeat-level variance
that would be hidden by comparing two unrelated averages.

The runner writes each result immediately. A completed run contains:

- `runs.jsonl`: normalized raw records;
- `summary.json`: per-version/per-question statistics;
- `comparison.json` and `comparison.md`: candidate-minus-baseline deltas;
- `blind_review.csv`: answers with version identity hidden;
- `review_key.csv`: version, trace provenance, and run identity revealed later;
- target server logs and a reproducibility manifest.

The manifest records each managed checkout's Git revision and dirty/clean
state. Process environment values—which may contain secrets—are not copied.

Positive deltas are better for quality/consistency. Negative deltas are better
for latency, token consumption, LLM calls, and retries. Missing telemetry is
reported as `—`, never silently converted to zero.

## Adding another Q&A system

Implement `agentic_eval.adapters.base.SystemAdapter` with three methods:

- `healthcheck()`
- `reset()`
- `run(request, timeout_s) -> AdapterResult`

Then set:

```yaml
adapter: your_package.your_adapter:YourAdapter
```

The scheduler, scoring, paired statistics, persistence, and review workflow do
not change.

After filling the blind sheet:

```bash
python -m agentic_eval score-reviews \
  --review experiments/results/<run>/blind_review.csv \
  --key experiments/results/<run>/review_key.csv
```
