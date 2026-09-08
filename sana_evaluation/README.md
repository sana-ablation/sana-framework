# sana_evaluation

Runtime package for SANA benchmark execution.

## Contents

- `cli.py`: the evaluation entry point — resolves the run and calls the
  orchestrator directly.
- `runner/orchestration.py` and `runner/reporting.py`: task discovery,
  per-directory orchestration, and the CSV/JSONL writers it uses.
- `benchmarks.py` and `preflight.py`: benchmark paths, artifact discovery, and
  readiness checks.
- `profiles.py`: loader and process state for a task's runtime profile.
- `helper/`: shared runtime helpers for prompts, results, logging, and
  sandbox handling.
- `instrumentation/`: plugins for traces, loop metadata, read traces, costs,
  and search-call budgets.
- `llm/`: model factory and cached OpenAI model adapter.
- `prompts/`: baseline, managed, and search-mode prompt templates.
- `tools/`: the data-lake tools, the search backends and their wrapper
  (`tools/search/`), the gold-data oracles (`tools/oracle/`), and the
  standalone `fetch` and `plan` tools.

Prefer invoking this package with `python -m sana_evaluation.<module>` from the
repo root so relative benchmark and result paths resolve consistently.

## Presets

```bash
python -m sana_evaluation.cli [smoke|full] [options]
```

A preset is a default set, nothing more: explicit flags always win, and
omitting the preset reproduces the raw evaluator defaults (`--search standard
--results rich --plan standard --compute standard`, non-verbose, no resume).

`smoke` runs one small task bucket under `test_logs/` and `test_results/`;
`full` runs the maintained task set, always pooled into one worker pool, and
resumes it via `--only-new` (skip task files already recorded in
`eval_results.csv`). Both shift the four axes to `ideal`, turn on verbose
logging, and move the output roots to `log-kramabench/` and
`results-kramabench/` under `--benchmark kramabench`. Use `--plan` (or its
`--plans` alias) to override the planning axis.
