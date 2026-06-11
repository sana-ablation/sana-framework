"""Diagnostic: verify the agent can see the reasoning chain / authored plan.

Loads the agent in ``--mode {ideal,standard}``, points it at a real task, and
asks it to recite what it has access to.

- ``--mode ideal``:  the gold reasoning chain is pre-injected into the system
  prompt under a ``## GOLD REASONING CHAIN`` section. We ask the agent to
  recite that section verbatim.
- ``--mode standard``: no chain is pre-injected; the agent has the ``plan``
  tool. We ask it to author a brief plan and then submit that plan's text.

Writes a single log under ``test_logs/plan_recitation_<mode>_<ts>.log`` that
captures: the chosen task id, the expected reasoning-chain text (ideal mode
only), the recitation prompt sent to the agent, and the agent's submitted
answer + reasoning. Makes a real LLM call — requires ``.env`` credentials for
the chosen model.

Usage (from repo root):

    PYTHONPATH=. python test/test_plan_recitation.py --mode ideal
    PYTHONPATH=. python test/test_plan_recitation.py --mode standard \\
        --model-name openai/gpt-5-mini
    PYTHONPATH=. python test/test_plan_recitation.py --mode ideal \\
        --task-file benchmarks/lakeqa/tasks-mini/tasks/k-2-d-3/task_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation.agent_with_mode import DataLakeAgent
from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.preflight import run_preflight
from sana_evaluation.tools.external.ideal import runtime_profile_store

_IDEAL_PROMPT = (
    "This is a diagnostic. Do NOT attempt to answer the underlying research "
    "question. Look at your current system prompt — there should be a section "
    "titled '## GOLD REASONING CHAIN'. Copy the text of that section verbatim, "
    "preserving line breaks and numbering. Immediately call submit_answer with "
    "that exact text as your final answer. Do not call any other tool."
)

_STANDARD_PROMPT = (
    "This is a diagnostic. Do NOT attempt to answer the underlying research "
    "question. You have a `plan` tool — call it exactly once with a short, "
    "3-5 step execution plan (any generic approach is fine). Then call "
    "submit_answer with the exact plan text you just wrote as your final "
    "answer. Do not call any other tool."
)


def _load_task(task_file: str) -> Dict[str, Any]:
    path = Path(task_file)
    if not path.exists():
        raise FileNotFoundError(f"task file not found: {task_file}")
    with path.open() as f:
        task = json.load(f)
    task_id = task.get("id") or task_file
    task["_task_id"] = task_id
    task["_task_file"] = task_file
    return task


def _build_run_config(
    mode: str,
    *,
    db_path: str,
    trace_dir: Path,
    results_dir: Path,
    logs_dir: Path,
) -> RunConfig:
    return RunConfig(
        results_output_dir=str(results_dir),
        logs_output_dir=str(logs_dir),
        max_tool_calls=8,
        sliding_window_k=20,
        timeout_seconds=180,
        search_db_path=db_path,
        search_tool_mode=mode,
        search_results_mode="ideal" if mode == "ideal" else "naive",
        profile_mode=mode,
        condition_config=ConditionConfig(
            condition=f"diagnostic/plan_recitation_{mode}",
            base_condition="baseline",
            trace_output_dir=str(trace_dir),
        ),
    )


def _expected_chain(task_file: str) -> Optional[str]:
    try:
        plan = runtime_profile_store.load_runtime_profile_for_task(task_file)
    except Exception:
        return None
    return plan.reasoning_chain_text


def _dump(log, title: str, body: str) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)
    print(body)
    log.write(f"{bar}\n{title}\n{bar}\n{body}\n")


def run_recitation(
    *,
    mode: str,
    task_file: str,
    model_name: str,
    db_path: str,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        task = _load_task(task_file)
        task_id = task["_task_id"]

        _dump(log, "RUN CONFIG", json.dumps({
            "mode": mode,
            "task_file": task_file,
            "task_id": task_id,
            "model_name": model_name,
            "db_path": db_path,
            "timestamp": datetime.now().isoformat(),
        }, indent=2))

        expected = _expected_chain(task_file) if mode == "ideal" else None
        if mode == "ideal":
            _dump(log, "EXPECTED REASONING CHAIN (from runtime-profiles)", expected or "<NONE>")

        trace_dir = Path("test_results/diagnostic_plan_recitation/traces") / mode
        results_dir = Path("test_results/diagnostic_plan_recitation/results") / mode
        logs_dir = Path("test_logs/diagnostic_plan_recitation") / mode
        run_config = _build_run_config(
            mode,
            db_path=db_path,
            trace_dir=trace_dir,
            results_dir=results_dir,
            logs_dir=logs_dir,
        )
        agent_config = AgentConfig(model_name=model_name)

        try:
            run_preflight(run_config, [task_file])
        except Exception as exc:
            _dump(log, "PREFLIGHT FAILED", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return

        question = _IDEAL_PROMPT if mode == "ideal" else _STANDARD_PROMPT
        _dump(log, "RECITATION PROMPT SENT TO AGENT", question)

        task_context = {
            "task_id": task_id,
            "datasets_used": task.get("datasets_used", []),
            "reasoning_chain": task.get("reasoning_chain", []),
        }
        if mode == "ideal":
            import sana_evaluation.tools.external.ideal.search_ideal as _si
            _si.set_task_context(task_context)

        da = DataLakeAgent(agent_config, run_config)
        try:
            result = da.run(question, task_context=task_context)
        except Exception as exc:
            _dump(log, "AGENT RUN RAISED", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return

        _dump(log, "AGENT SUBMITTED ANSWER", (result.answer or "<EMPTY>").strip())
        _dump(
            log,
            "AGENT REASONING (final turn text)",
            (getattr(result, "reasoning", None) or "<EMPTY>").strip(),
        )
        _dump(
            log,
            "SOURCES SEEN",
            json.dumps(getattr(result, "sources", []), indent=2, default=str),
        )

        if mode == "ideal" and expected:
            normalize = lambda s: " ".join(s.split())
            match = normalize(result.answer or "") == normalize(expected)
            _dump(log, "MATCHES EXPECTED (normalized whitespace)", str(match))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("ideal", "standard"), default="ideal")
    parser.add_argument("--task-file", default="benchmarks/lakeqa/tasks-mini/tasks/k-5-d-4/task_1.json")
    parser.add_argument("--model-name", default="bedrock/claude-sonnet-4.5")
    parser.add_argument("--db", default="lance_data")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    log_path = Path(args.out) if args.out else (
        Path("test_logs") / f"plan_recitation_{args.mode}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    )

    run_recitation(
        mode=args.mode,
        task_file=args.task_file,
        model_name=args.model_name,
        db_path=args.db,
        log_path=log_path,
    )
    print(f"\nWrote diagnostic log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
