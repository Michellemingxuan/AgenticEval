# experiments

Inputs are versioned; outputs are not.

```
configs/     what to run   — one experiment definition per file
questions/   what to ask   — question sets, each carrying its own expectations
oracles/     what is true  — Python ground truth, no LLM involved
examples/    templates to copy, not to run against real systems
results/     run output    (gitignored)
traces/      system trace databases, written by the systems under test (gitignored)
```

Every relative path inside a config resolves from **that config file**, not from
your shell. Moving a config between folders therefore means re-checking its
`questions_file`, `rubric_file`, `output_dir`, `oracle_cwd`, `process.cwd`, and
`trace_db`.

## Configs

| file | question set | purpose |
|---|---|---|
| `smoke.yaml` | AgenticSys consistency suite | fastest end-to-end check |
| `compare_versions.yaml` | AgenticSys consistency suite | baseline vs candidate |
| `content_smoke.yaml` | AgenticSys consistency suite | content cascade, `tsr_cdss_trend` rubric |
| `simple_questions.yaml` | `questions/simple.yaml` | Python-verifiable questions |

```bash
python -m agentic_eval validate --config experiments/configs/simple_questions.yaml
python -m agentic_eval run      --config experiments/configs/simple_questions.yaml
```

## Question sets

`questions/simple.yaml` carries its expectations inline — scope, oracles, and
memory annotations live beside the question text, so a rename cannot unwire a
check.

`questions/tsr_cdss.rubric.yaml` is the other form: a rubric merged **by name**
onto a question set owned elsewhere (AgenticSys's consistency suite). It is
loaded via `content_evaluation.rubric_file`, and an entry matching no question
is an error rather than a silently dead check.

## Results

One folder per run, named `<experiment.name>_<timestamp>`, all gitignored.

```
<run>/
  manifest.json                what was run: systems, mode, repeats, seed
  runs.jsonl                   raw records — the only irreplaceable artifact
  metrics/
    summary.json               per-question aggregates, all four modules
    comparison.json  comparison.md
  content/
    evaluations.jsonl          claims, verdicts, judge telemetry
    summary.json  comparison.md
    walkthrough.md             answer -> atomic facts -> numeric verdicts
    answer_comparison.html     side-by-side viewer, one sampled repeat
  review/
    answers.csv  answers.key.csv        phase A: blind answer review
    evidence.jsonl  evidence.key.csv    phase B: blind claim/evidence review
  logs/
    <system>.server.log
```

`agentic_eval/layout.py` owns this shape. Four writers contribute to a run — the
runner, the content cascade, the review packet builders, and the viewer — and
when each picks its own filenames the result is a flat pile where nothing says
which artifact came from where.

The two review keys stay separate files: the phases are independently blinded
and written by different commands at different times, so merging them would
make `evaluate-content` rewrite a file that `run` owns. They sit side by side
and join on `turn_id`.

`evaluate-content` writes to `<run>/content` by default. Re-evaluating the same
run overwrites it, so move the previous one into `archive/` first if it is
worth keeping, or pass `--output-dir` to write a named variant:

```bash
python -m agentic_eval evaluate-content \
  --config experiments/configs/content_smoke.yaml \
  --runs experiments/results/<run>/runs.jsonl \
  --output-dir experiments/results/<run>/archive/content_strict
```

A run assembled by hand (rather than by `run`) will lack `manifest.json`,
`summary.json`, and `comparison.*`; it still carries `runs.jsonl`, which is all
`evaluate-content` needs.

## Oracles

`oracles/case_facts.py` reads the case CSVs directly and prints
`{"value": ..., "detail": {...}}`. It is the only evaluation input not derived
from the system under test, which is what lets it catch a fabrication the
system wrote into its own report and then read back as evidence.

```bash
python3 experiments/oracles/case_facts.py --fact latest_fico_score
```
