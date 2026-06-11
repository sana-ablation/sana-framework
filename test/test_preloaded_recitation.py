"""Diagnostic: verify the preloaded mode exposes datasets, reasoning chain, and skills.

Creates a real agent in ``search_tool=preloaded`` mode, asks it to recite what
is already present in its system state, and writes a detailed log capturing:

- the expected preloaded dataset block derived from ``source_sequence``
- the expected gold reasoning chain (when ``--management ideal``)
- the configured skill paths and skill file contents
- the exact prompt sent to the agent
- the agent's submitted answer and final reasoning

This makes a real LLM call and requires working model credentials.
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

from sana_evaluation.agent_with_mode import DataLakeAgent, build_mode_bundle
from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.helper.prompting import compose_preloaded_block, skill_paths_for_modes
from sana_evaluation.preflight import run_preflight
from sana_evaluation.tools.external.ideal.runtime_profile_store import load_runtime_profile_for_task

_RECITATION_PROMPT = (
    "This is a diagnostic. Do NOT solve the underlying benchmark question. "
    "Read your current system instructions and then immediately call submit_answer. "
    "Put the full diagnostic content in the `answer` field of submit_answer, not in `reasoning`. "
    "Keep `reasoning` empty or very brief. "
    "In the submitted answer, include exactly these sections in this order:\n"
    "1. PRELOADED_DATASETS: recite the full '## PRELOADED DATASETS' block you can see.\n"
    "2. GOLD_REASONING_CHAIN: recite the full '## GOLD REASONING CHAIN' section if present; otherwise write NONE.\n"
    "3. SKILLS_VISIBLE: list the skill names or instruction blocks you can see.\n"
    "Do not call any tool other than submit_answer."
)


def _dump(log, title: str, body: str) -> None:
    bar = "=" * 72
    print(bar)
    print(title)
    print(bar)
    print(body)
    log.write(f"{bar}\n{title}\n{bar}\n{body}\n")


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


def _load_skill_bodies(skill_paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for relpath in skill_paths:
        skill_file = Path(relpath) / "SKILL.md"
        out[relpath] = skill_file.read_text() if skill_file.is_file() else "<MISSING SKILL FILE>"
    return out


def _build_run_config(
    *,
    management_mode: str,
    db_path: str,
    trace_dir: Path,
    results_dir: Path,
    logs_dir: Path,
) -> RunConfig:
    return RunConfig(
        results_output_dir=str(results_dir),
        logs_output_dir=str(logs_dir),
        max_tool_calls=4,
        sliding_window_k=20,
        timeout_seconds=180,
        search_db_path=db_path,
        search_tool_mode="preloaded",
        search_results_mode="ideal",
        profile_mode=management_mode,
        condition_config=ConditionConfig(
            condition=f"diagnostic/preloaded_recitation_{management_mode}",
            base_condition="baseline",
            trace_output_dir=str(trace_dir),
        ),
    )


def run_recitation(
    *,
    task_file: str,
    management_mode: str,
    model_name: str,
    db_path: str,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        task = _load_task(task_file)
        task_id = task["_task_id"]
        ideal_plan = load_runtime_profile_for_task(task_file)
        preloaded_block = compose_preloaded_block(ideal_plan.source_sequence)
        expected_chain = ideal_plan.reasoning_chain_text if management_mode == "ideal" else None
        skill_paths = skill_paths_for_modes("preloaded", management_mode)
        skill_bodies = _load_skill_bodies(skill_paths)

        _dump(
            log,
            "RUN CONFIG",
            json.dumps(
                {
                    "task_file": task_file,
                    "task_id": task_id,
                    "management_mode": management_mode,
                    "model_name": model_name,
                    "db_path": db_path,
                    "timestamp": datetime.now().isoformat(),
                },
                indent=2,
            ),
        )
        _dump(log, "EXPECTED PRELOADED DATASETS BLOCK", preloaded_block)
        _dump(log, "EXPECTED GOLD REASONING CHAIN", expected_chain or "<NONE>")
        _dump(log, "CONFIGURED SKILL PATHS", json.dumps(skill_paths, indent=2))
        _dump(
            log,
            "CONFIGURED SKILL FILES",
            json.dumps(skill_bodies, indent=2),
        )

        trace_dir = Path("test_results/diagnostic_preloaded_recitation/traces") / management_mode
        results_dir = Path("test_results/diagnostic_preloaded_recitation/results") / management_mode
        logs_dir = Path("test_logs/diagnostic_preloaded_recitation") / management_mode
        run_config = _build_run_config(
            management_mode=management_mode,
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

        task_context = {
            "task_id": task_id,
            "datasets_used": task.get("datasets_used", []),
            "reasoning_chain": task.get("reasoning_chain", []),
        }
        bundle = build_mode_bundle(run_config, data_tools=[], task_context=task_context)
        _dump(log, "COMPOSED SYSTEM PROMPT", bundle.system_prompt)
        _dump(log, "RECITATION PROMPT SENT TO AGENT", _RECITATION_PROMPT)

        da = DataLakeAgent(agent_config, run_config)
        try:
            result = da.run(_RECITATION_PROMPT, task_context=task_context)
        except Exception as exc:
            _dump(log, "AGENT RUN RAISED", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            return

        _dump(log, "AGENT SUBMITTED ANSWER", (result.answer or "<EMPTY>").strip())
        _dump(log, "AGENT REASONING (final turn text)", (getattr(result, "reasoning", None) or "<EMPTY>").strip())
        _dump(log, "SOURCES SEEN", json.dumps(getattr(result, "sources", []), indent=2, default=str))

        combined_text = "\n".join(part for part in [result.answer or "", result.reasoning or ""] if part)
        checks = {
            "contains_preloaded_header": "PRELOADED_DATASETS:" in combined_text,
            "contains_dataset_id": ideal_plan.dataset_sequence[0] in combined_text,
            "contains_reasoning_chain_section": ("GOLD_REASONING_CHAIN:" in combined_text) if expected_chain else True,
            "mentions_query_data_skill": "query-data" in combined_text,
            "answer_field_nonempty": bool((result.answer or "").strip()),
        }
        _dump(log, "BASIC CHECKS", json.dumps(checks, indent=2))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-file", default="benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_2.json")
    parser.add_argument("--management", choices=("standard", "ideal"), default="ideal")
    parser.add_argument("--model-name", default="openai/gpt-5-mini")
    parser.add_argument("--db", default="lance_data")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    log_path = Path(args.out) if args.out else (
        Path("test_logs") / f"preloaded_recitation_{args.management}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    )

    run_recitation(
        task_file=args.task_file,
        management_mode=args.management,
        model_name=args.model_name,
        db_path=args.db,
        log_path=log_path,
    )
    print(f"\nWrote diagnostic log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
