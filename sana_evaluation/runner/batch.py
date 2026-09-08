"""
Parallel batch runner for the Data Lake benchmark.

Runs many tasks through a ``runner.agent.DataLakeAgent`` (or a subclass)
across a ``ProcessPoolExecutor``, one task per worker process. Holds the
worker-process entry point (``_run_task_worker``, module-level so it can be
pickled for the pool), the worker memory cap, and ``BatchRunner`` itself.
"""

import concurrent.futures
import logging
import os
import resource
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sana_evaluation.config import AXIS_DEFAULTS, AgentConfig, RunConfig
from sana_evaluation.instrumentation import set_trace_context
from sana_evaluation.helper.logger import configure_worker_logging
from sana_evaluation.runner.agent import DataLakeAgent
from sana_evaluation.runner.modes import (
    _NAIVE_SEARCH_TOOLS_AVAILABLE,
    _STANDARD_SEARCH_TOOLS_AVAILABLE,
    _merge_sources_used,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker function (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Worker memory cap
# ---------------------------------------------------------------------------

# Three grid runs were killed by the kernel OOM killer; twice a single process
# reached ~27.5 GB while the pool workers sat at 0.58 GB each on a 31 GB box.
# The allocation is unidentified -- DuckDB's limit, result materialisation,
# execute_ideal, the artifact caches, the JSON reader and the failing SQL were
# each measured and ruled out.
#
# Capping the worker heap converts that kill into a MemoryError in one task,
# which the existing handler reports as a task error, so the grid survives. It
# also preserves the traceback naming the allocation site, which an OOM kill
# destroys -- that is the evidence every previous diagnosis lacked.
#
# DISABLED BY DEFAULT -- RLIMIT_DATA is the wrong instrument here.
#
# On Linux it counts anonymous mmap *reservations*, i.e. virtual address space,
# not resident memory. torch, DuckDB and lance reserve enormous arenas: the
# OOM-killed process showed total-vm 179 GB against 27.5 GB resident, and a
# healthy worker is not far off. An 8 GB cap therefore fires during module
# import, before the task does any work.
#
# Measured: with the cap at 8 GB, up to 13 of 20 tasks per cell failed with
# MemoryError inside `from ... import OpenAICachedUsageModel`. It corrupted
# results rather than containing anything.
#
# Bounding resident memory needs a cgroup (systemd-run -p MemoryMax=...), not an
# rlimit. Until that exists, supervise_grid.sh handles crashes instead. Set
# SANA_WORKER_MEMORY_CAP_GB explicitly to re-enable, knowing the above.
_DEFAULT_WORKER_MEMORY_CAP_GB = ""


def _apply_worker_memory_cap() -> Optional[int]:
    """Cap this process's heap. Returns the cap in bytes, or None if not applied."""
    raw = os.getenv("SANA_WORKER_MEMORY_CAP_GB", _DEFAULT_WORKER_MEMORY_CAP_GB)
    try:
        gb = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if gb <= 0:
        return None

    cap = int(gb * 1024**3)
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))
    except (OSError, ValueError, AttributeError) as exc:
        # A restricted sandbox may forbid this. Losing the cap is worse than
        # nothing, but far better than refusing to run the task.
        logger.warning("Could not apply worker memory cap: %s", exc)
        return None
    return cap


def _run_task_worker(
    task: Dict[str, Any],
    task_index: int,
    agent_config: AgentConfig,
    run_config: RunConfig,
    run_id: str,
    batch_name: Optional[str],
    agent_class: Optional[type] = None,
) -> Dict[str, Any]:
    """Run a single task in a worker process.

    `agent_class` lets callers swap in a DataLakeAgent subclass (e.g. for SANA).
    Defaults to `DataLakeAgent`.
    """
    _apply_worker_memory_cap()
    from sana_evaluation.metrics import compute_exact_match, compute_f1_score, normalize_text

    log_model_name = agent_config.model_name or agent_config.model_id
    effort = (agent_config.extra_model_kwargs or {}).get("reasoning_effort")
    if effort:
        log_model_name = f"{log_model_name}-{effort}"

    configure_worker_logging(
        run_config,
        model=log_model_name,
        condition=run_config.condition_config.condition,
        task_id=task.get("id"),
    )

    # Eagerly load search backend state once per worker process. Coalesced
    # against AXIS_DEFAULTS so the backend set up here is the one
    # build_mode_bundle will actually hand the agent.
    mode_search_tool = (
        run_config.search_tool_mode or AXIS_DEFAULTS["search_tool_mode"]
    ).strip().lower()
    mode_computation_tool = (
        run_config.computation_tool_mode or AXIS_DEFAULTS["computation_tool_mode"]
    ).strip().lower()

    if mode_search_tool == "standard" and _STANDARD_SEARCH_TOOLS_AVAILABLE:
        try:
            import sana_evaluation.tools.search.standard as _sa
            if run_config.search_db_path:
                _sa.set_db_path(run_config.search_db_path)
            _sa.setup()
        except Exception as e:
            logger.warning(f"Hybrid search setup failed: {e}")

    if mode_search_tool == "naive" and _NAIVE_SEARCH_TOOLS_AVAILABLE:
        # NOTE: do not call search_naive_tools.setup() for naive mode.
        # setup() triggers external-tools/hybrid_search/api.setup(), which eagerly
        # loads embedding/reranker models that are unnecessary for sparse-only
        # search paths and can heavily impact worker startup and memory.
        if run_config.search_db_path:
            try:
                import sana_evaluation.tools.search.naive as _sb

                _sb.set_db_path(run_config.search_db_path)
            except Exception as e:
                logger.warning(f"Sparse search path override failed: {e}")

    if mode_search_tool == "ideal":
        if run_config.search_db_path:
            import sana_evaluation.tools.oracle.search as _si

            _si.set_db_path(run_config.search_db_path)

    try:
        logger.info(f"Starting task {task_index + 1}: {task.get('question', '')[:80]}...")

        agent_cls = agent_class or DataLakeAgent
        da = agent_cls(agent_config, run_config)

        cond = run_config.condition_config
        gold_ids = task.get("datasets_used", [])
        task_id = task.get("id", str(task_index))
        set_trace_context(task_id, gold_ids, cond.trace_output_dir)
        from sana_evaluation.instrumentation import ideal_subagent_costs as _ideal_costs

        _ideal_costs.reset_stats()

        try:
            from sana_evaluation.instrumentation import delegation_subagent_costs as _deleg_costs
        except ImportError:
            _deleg_costs = None
        if _deleg_costs is not None:
            _deleg_costs.reset_stats()

        task_context = {
            "task_id": task_id,
            "datasets_used": gold_ids,
            "reasoning_chain": task.get("reasoning_chain", []),
        }
        if mode_search_tool == "ideal":
            import sana_evaluation.tools.oracle.search as _si

            _si.set_task_context(task_context)

        result = da.run(task["question"], task_context=task_context)
        delegation_subagent_stats: Dict[str, Any] = {}
        if _deleg_costs is not None:
            delegation_subagent_stats = _deleg_costs.get_stats()

        result_dict: Dict[str, Any] = {
            "task_id": task.get("id", task_index),
            "model": agent_config.model_id,
            "question": task.get("question", ""),
            "ground_truth": task.get("answer", ""),
            "predicted_answer": result.answer,
            "reasoning": result.reasoning,
            "sources_used": _merge_sources_used(result.sources, delegation_subagent_stats),
            "time": result.elapsed_time,
            "success": result.success,
            "error": result.error,
        }

        # Tokens
        result_dict["input_tokens"]     = result.input_tokens
        result_dict["cached_input_tokens"] = result.cached_input_tokens
        result_dict["uncached_input_tokens"] = result.uncached_input_tokens
        # Priced differently from both other kinds on some models, and only
        # derivable from the other three by subtraction, which stops being
        # reconstructable the moment any of them changes meaning.
        result_dict["cache_write_input_tokens"] = result.cache_write_input_tokens
        result_dict["output_tokens"]    = result.output_tokens
        result_dict["total_tokens"]     = result.total_tokens
        result_dict["cost_usd"]         = result.cost_usd

        # Tool counts
        _, total_calls = result.get_cumulative_tool_counts()
        result_dict["tool_calls_total"] = total_calls
        result_dict["api_tool_calls"]   = result.get_api_tool_calls()

        # Cycle count
        result_dict["cycle_count"]      = result.cycle_count

        # Per-tool breakdown (for tools CSV) — List[Dict], pickle-safe
        result_dict["tool_counts"]      = result.get_tool_counts()

        if mode_computation_tool == "ideal":
            from sana_evaluation.tools.oracle import computation as _ci

            result_dict.update(_ci.get_stats())

        ideal_subagent_stats = _ideal_costs.get_stats()
        result_dict.update(ideal_subagent_stats)
        ideal_cost = float(ideal_subagent_stats.get("ideal_subagent_cost_usd", 0.0) or 0.0)
        result_dict["total_cost_with_ideal_subagents_usd"] = result.cost_usd + ideal_cost

        if _deleg_costs is not None:
            result_dict.update(delegation_subagent_stats)
        delegation_cost = float(
            delegation_subagent_stats.get("delegation_subagent_cost_usd", 0.0) or 0.0
        )
        result_dict["total_cost_with_all_subagents_usd"] = (
            result.cost_usd + ideal_cost + delegation_cost
        )

        try:
            from sana_evaluation.tools.delegation_common import clear_delegation_runtime
        except ImportError:
            pass
        else:
            clear_delegation_runtime()

        # Compute accuracy metrics if ground truth available
        if task.get("answer"):
            gt = str(task["answer"])
            pred = result.answer
            result_dict["exact_match"] = compute_exact_match(pred, gt)
            result_dict["f1_score"] = compute_f1_score(pred, gt)
            result_dict["normalized_prediction"] = normalize_text(pred)
            result_dict["normalized_ground_truth"] = normalize_text(gt)


        result_dict["task_metadata"] = {
            "num_nodes": len(task.get("nodes", {})),
            "has_reasoning_chain": "reasoning_chain" in task,
        }

        if result.success:
            logger.info(
                f"Completed task {task_index + 1}: "
                f"input={result.input_tokens} cached_input={result.cached_input_tokens} "
                f"uncached_input={result.uncached_input_tokens} output={result.output_tokens} "
                f"tokens={result.total_tokens} cost=${result.cost_usd:.4f} "
                f"tools={total_calls} cycles={result.cycle_count}"
            )
        else:
            logger.warning(
                f"Task {task_index + 1} finished with failure: {result.error}"
            )
        return result_dict

    except Exception as e:
        logger.error(
            f"Worker error for task {task_index + 1} ({type(e).__name__}): {e}",
            exc_info=True,
        )
        return {
            "task_id": task.get("id", task_index),
            "model": agent_config.model_id,
            "question": task.get("question", ""),
            "ground_truth": task.get("answer", ""),
            "predicted_answer": "",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------

class BatchRunner:
    """Run DataLakeAgent on multiple tasks with parallel ProcessPoolExecutor."""

    # Subclasses can override this to swap in a DataLakeAgent subclass.
    _AGENT_CLASS: Optional[type] = None

    def __init__(
        self,
        agent_config: AgentConfig,
        run_config: Optional[RunConfig] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.agent_config = agent_config
        self.run_config = run_config or RunConfig()
        self.max_workers = max_workers or min(6, os.cpu_count() or 1)

    def run_tasks(
        self,
        tasks: List[Dict[str, Any]],
        verbose: bool = False,
        max_workers: Optional[int] = None,
        batch_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run agent on a list of tasks in parallel."""
        num_workers = max_workers or self.max_workers
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_label = batch_name or "batch"

        if verbose:
            print(f"\nRunning {len(tasks)} tasks with {num_workers} workers...")

        results: List[tuple] = []

        import multiprocessing as _mp
        mp_ctx = _mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            future_to_index = {
                executor.submit(
                    _run_task_worker,
                    task,
                    i,
                    self.agent_config,
                    self.run_config,
                    run_id,
                    batch_label,
                    self._AGENT_CLASS,
                ): i
                for i, task in enumerate(tasks)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    if verbose:
                        em = result.get("exact_match")
                        status = f" EM={em:.2f}" if em is not None else ""
                        print(f"  Task {idx + 1}/{len(tasks)} done{status}")
                except Exception as e:
                    logger.error(
                        f"Future for task {idx + 1} raised {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    results.append((idx, {
                        "task_id": tasks[idx].get("id", idx),
                        "model": self.agent_config.model_id,
                        "question": tasks[idx].get("question", ""),
                        "ground_truth": tasks[idx].get("answer", ""),
                        "predicted_answer": "",
                        "success": False,
                        "error": f"{type(e).__name__}: {e}",
                    }))
                    if verbose:
                        print(f"  Task {idx + 1}/{len(tasks)} FAILED: {type(e).__name__}: {e}")

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def run_from_files(
        self,
        task_files: List[str],
        verbose: bool = False,
        max_workers: Optional[int] = None,
        batch_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load tasks from JSON files and run them."""
        import json

        tasks = []
        for path in task_files:
            with open(path) as f:
                task = json.load(f)
                task["id"] = path
                tasks.append(task)

        if not batch_name and task_files:
            try:
                batch_name = Path(os.path.commonpath(task_files)).name
            except Exception:
                pass

        return self.run_tasks(
            tasks, verbose=verbose, max_workers=max_workers, batch_name=batch_name
        )
