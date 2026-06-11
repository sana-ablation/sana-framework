# sana_evaluation

Runtime package for SANA benchmark execution.

## Contents

- `setup_run.py`: builds configured evaluation runs.
- `run_eval.py` and `run_mode_eval.py`: run evaluation entry points.
- `artifacts.py` and `preflight.py`: artifact discovery and readiness checks.
- `helper/`: shared runtime helpers for prompts, metrics, results, logging, and
  sandbox handling.
- `instrumentation/`: plugins for traces, loop metadata, read traces, costs,
  and search-call budgets.
- `llm/`: model factory and cached OpenAI model adapter.
- `prompts/`: baseline, managed, and search-mode prompt templates.
- `tools/`: agent tools plus external and helper tool wrappers.

Prefer invoking this package with `python -m sana_evaluation.<module>` from the
repo root so relative benchmark and result paths resolve consistently.

## setup_run Defaults

`setup_run.py` is the friendly wrapper for `run_mode_eval.py`. Use:

```bash
python -m sana_evaluation.setup_run smoke|full [options]
```

The wrapper defaults to ideal search results, ideal planning/profile mode,
ideal compute mode, verbose logging, and resume mode for `full`. Use `--plans`
or the compatible `--profile` flag to override the planning axis, and use
`--no-continue` when a full run should rerun every task.
