"""--only-new is the one resume path.

It is a filter *inside* ``run_evaluation`` that skips task files already
present as ``task_id`` rows in the variant's ``eval_results.csv``. Unlike the
deleted ``--task-continue`` (a separate, non-pooling, interactive top-level run
mode), it composes with ``--all-tasks``/pooled scheduling and never blocks on
stdin, so it works from an unattended restart loop.
"""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sana_evaluation import cli
from sana_evaluation.runner import orchestration as run_eval


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_only_new_parses_and_defaults_to_false():
    assert cli.parse([]).only_new is False


def test_only_new_flag_sets_true():
    assert cli.parse(["--only-new"]).only_new is True


@pytest.mark.parametrize("dead", [
    ["--task-continue"],
    ["--continue"],
    ["--no-continue"],
    ["--pool-tasks"],
])
def test_deleted_flags_are_gone(dead):
    """Same pattern as test_cli.py::test_deleted_flags_are_gone.

    Each of these used to be a bare store_true/store_false flag, so there is no
    "value lands on a positional" trap here -- but the preset is still supplied
    up front so a parser that still defines the flag parses cleanly (no
    SystemExit) rather than merely coincidentally exiting for an unrelated
    reason.
    """
    with pytest.raises(SystemExit):
        cli.parse(["smoke"] + dead)


def test_no_run_continue_attribute():
    assert not hasattr(run_eval, "_run_continue")


# ---------------------------------------------------------------------------
# run_evaluation(only_new=True)
# ---------------------------------------------------------------------------

class _FakeBatchRunner:
    """Records what it was asked to run, so the filter is observable."""

    calls = []

    def __init__(self, agent_config=None, run_config=None, max_workers=None):
        _FakeBatchRunner.calls.append({"max_workers": max_workers, "files": None})

    def run_from_files(self, task_files, verbose=False):
        _FakeBatchRunner.calls[-1]["files"] = list(task_files)
        return [
            {"task_id": p, "exact_match": 0, "f1_score": 0.0, "time": 1.0,
             "cost_usd": 0.0, "tool_calls_total": 0, "success": True}
            for p in task_files
        ]


class OnlyNewFilterTests(unittest.TestCase):
    def setUp(self):
        _FakeBatchRunner.calls = []
        self._tmp = tempfile.TemporaryDirectory()
        self.task_dir = Path(self._tmp.name) / "k-1-d-1"
        self.task_dir.mkdir(parents=True)
        self.files = []
        for name in ("task_1", "task_2", "task_3"):
            p = self.task_dir / f"{name}.json"
            p.write_text(json.dumps({"question": "q", "answer": "a"}))
            self.files.append(str(p))
        self.out = tempfile.mkdtemp()

    def tearDown(self):
        self._tmp.cleanup()

    def _cfg(self):
        agent = MagicMock()
        agent.model_id = "m"
        agent.model_name = "m"
        run = MagicMock()
        run.condition_config = MagicMock(condition="c")
        return agent, run

    def _patched(self):
        return patch.multiple(
            run_eval,
            _display_name=MagicMock(return_value="model"),
            _results_dir=MagicMock(return_value=self.out),
        )

    def _write_csv(self, task_ids):
        csv_path = os.path.join(self.out, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id"])
            writer.writeheader()
            for tid in task_ids:
                writer.writerow({"task_id": tid})

    def test_only_new_skips_recorded_tasks_and_runs_the_rest(self):
        self._write_csv([self.files[0]])
        agent, run = self._cfg()
        with self._patched():
            run_eval.run_evaluation(
                str(self.task_dir), agent, run, batch_runner_cls=_FakeBatchRunner,
                only_new=True, parallel=2,
            )

        self.assertEqual(len(_FakeBatchRunner.calls), 1, "the runner still runs once")
        ran = _FakeBatchRunner.calls[0]["files"]
        self.assertEqual(sorted(ran), sorted(self.files[1:]))
        self.assertNotIn(self.files[0], ran)

    def test_without_only_new_everything_runs_regardless_of_the_csv(self):
        self._write_csv([self.files[0]])
        agent, run = self._cfg()
        with self._patched():
            run_eval.run_evaluation(
                str(self.task_dir), agent, run, batch_runner_cls=_FakeBatchRunner,
                only_new=False, parallel=2,
            )

        ran = _FakeBatchRunner.calls[0]["files"]
        self.assertEqual(sorted(ran), sorted(self.files))

    def test_only_new_with_everything_recorded_skips_the_runner_entirely(self):
        self._write_csv(self.files)
        agent, run = self._cfg()
        with self._patched():
            result = run_eval.run_evaluation(
                str(self.task_dir), agent, run, batch_runner_cls=_FakeBatchRunner,
                only_new=True, parallel=2,
            )

        self.assertEqual(_FakeBatchRunner.calls, [], "the runner must not be invoked")
        summary = result["m"]["summary"]
        self.assertEqual(summary["total_tasks"], 0)
        self.assertEqual(result["m"]["results"], [])


# ---------------------------------------------------------------------------
# --only-new composes with --all-tasks / pooled scheduling
# ---------------------------------------------------------------------------

class OnlyNewComposesWithPoolingTests(unittest.TestCase):
    """The property --task-continue lacked: pooling and resume together."""

    def test_run_all_tasks_pooled_forwards_only_new_into_run_evaluation(self):
        seen = {}

        def fake_run_evaluation(task_dir, agent_config, run_config, **kw):
            seen["only_new"] = kw.get("only_new")
            seen["task_files"] = kw.get("task_files")
            return {}

        dirs = ["/t/k-3-d-2", "/t/k-4-d-3"]
        files = {"/t/k-3-d-2": ["/t/k-3-d-2/task_6.json"],
                 "/t/k-4-d-3": ["/t/k-4-d-3/task_1.json"]}

        with patch.object(run_eval, "find_all_task_dirs", return_value=dirs), \
             patch.object(run_eval, "run_evaluation", side_effect=fake_run_evaluation), \
             patch.object(run_eval.glob, "glob", side_effect=lambda pat: sorted(files[os.path.dirname(pat)])), \
             patch.object(run_eval, "print_comparison_table", lambda *a, **k: None):
            run_eval._run_all_tasks_pooled(
                task_set="/t", agent_config=object(), run_config=object(),
                verbose=False, only_new=True, parallel=8, tasks_per_dir=None,
                batch_runner_cls=_FakeBatchRunner,
            )

        self.assertIs(seen["only_new"], True)
        self.assertEqual(len(seen["task_files"]), 2, "still pools across every directory")


if __name__ == "__main__":
    unittest.main()
