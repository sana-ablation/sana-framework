"""Pooled task scheduling: one worker pool across all task directories.

`--all-tasks` calls `run_evaluation` once per `k-*-d-*` directory, and each call
builds its own BatchRunner pool. Directories run sequentially, so effective
concurrency is capped by the largest directory, not by `--parallel`. On
subset20b (20 tasks over 11 directories sized 4,3,3,2,2,1,1,1,1,1,1) that means
~1.3-way concurrency against `--parallel 8`, which made a 7-cell grid take hours
longer than the worker count suggests.

Flattening the tasks into one directory is not an option: runtime-profile lookup
keys off the path suffix after `benchmarks/<bench>/tasks-mini/tasks`
(`runtime_profile_store._profile_location_from_task`), and subset20b has six
colliding basenames (`task_6` appears three times), so a flat tree would both
mis-resolve profiles and silently drop 8 of 20 tasks.

So the fix is to pool the *files* while leaving the tree alone.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation import run_eval


class _FakeBatchRunner:
    """Records what it was asked to run, so scheduling is observable."""

    calls = []

    def __init__(self, agent_config=None, run_config=None, max_workers=None):
        self.max_workers = max_workers
        _FakeBatchRunner.calls.append({"max_workers": max_workers, "files": None})

    def run_from_files(self, task_files, verbose=False):
        _FakeBatchRunner.calls[-1]["files"] = list(task_files)
        return [
            {"task_id": p, "exact_match": 0, "f1_score": 0.0, "time": 1.0,
             "cost_usd": 0.0, "tool_calls_total": 0, "success": True}
            for p in task_files
        ]


class PooledSchedulingTests(unittest.TestCase):
    def setUp(self):
        import tempfile, json, os
        _FakeBatchRunner.calls = []
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "benchmarks/lakeqa/tasks-mini/tasks"
        # Mirror subset20b's shape: uneven directories, colliding basenames.
        for d, names in [("k-3-d-2", ["task_6", "task_11"]), ("k-4-d-3", ["task_6"]),
                         ("k-6-d-2", ["task_6", "task_1"])]:
            (self.root / d).mkdir(parents=True, exist_ok=True)
            for n in names:
                (self.root / d / f"{n}.json").write_text(json.dumps({"question": "q", "answer": "a"}))
        self.dirs = sorted(str(p) for p in self.root.glob("k-*-d-*"))
        self._prev = run_eval.BatchRunner
        run_eval.BatchRunner = _FakeBatchRunner
        self.out = tempfile.mkdtemp()

    def tearDown(self):
        run_eval.BatchRunner = self._prev
        self._tmp.cleanup()

    def _cfg(self):
        agent = MagicMock(); agent.model_id = "m"; agent.model_name = "m"
        run = MagicMock(); run.condition_config = MagicMock(condition="c")
        return agent, run

    def _patched(self):
        return patch.multiple(
            run_eval,
            _display_name=MagicMock(return_value="model"),
            _results_dir=MagicMock(return_value=self.out),
        )

    def test_per_directory_scheduling_builds_one_pool_per_directory(self):
        """Baseline: today's behaviour, which is what makes the grid slow."""
        agent, run = self._cfg()
        with self._patched():
            for d in self.dirs:
                run_eval.run_evaluation(d, agent, run, parallel=8)

        self.assertEqual(len(_FakeBatchRunner.calls), 3, "one pool per directory")
        self.assertEqual([len(c["files"]) for c in _FakeBatchRunner.calls], [2, 1, 2])

    def test_explicit_task_files_bypass_the_directory_glob(self):
        """run_evaluation must accept a file list so callers can pool across dirs."""
        agent, run = self._cfg()
        every = sorted(str(p) for p in self.root.rglob("*.json"))
        with self._patched():
            run_eval.run_evaluation(
                str(self.root), agent, run, parallel=8, task_files=every,
            )

        self.assertEqual(len(_FakeBatchRunner.calls), 1, "a single pool for every task")
        self.assertEqual(sorted(_FakeBatchRunner.calls[0]["files"]), every)
        self.assertEqual(_FakeBatchRunner.calls[0]["max_workers"], 8)

    def test_pooling_preserves_the_directory_path_segments(self):
        """Profile lookup keys off the path, so pooling must not rewrite paths."""
        agent, run = self._cfg()
        every = sorted(str(p) for p in self.root.rglob("*.json"))
        with self._patched():
            run_eval.run_evaluation(str(self.root), agent, run, parallel=8, task_files=every)

        ran = _FakeBatchRunner.calls[0]["files"]
        for p in ran:
            self.assertRegex(p, r"benchmarks/lakeqa/tasks-mini/tasks/k-\d+-d-\d+/task_\d+\.json")

    def test_pooling_keeps_colliding_basenames_distinct(self):
        """task_6 appears in three directories; all three must survive."""
        agent, run = self._cfg()
        every = sorted(str(p) for p in self.root.rglob("*.json"))
        with self._patched():
            run_eval.run_evaluation(str(self.root), agent, run, parallel=8, task_files=every)

        ran = _FakeBatchRunner.calls[0]["files"]
        self.assertEqual(sum(1 for p in ran if p.endswith("task_6.json")), 3)
        self.assertEqual(len(set(ran)), len(ran))


if __name__ == "__main__":
    unittest.main()


class PoolTasksFlagTests(unittest.TestCase):
    """--pool-tasks must gather every directory into a single run_evaluation call."""

    def test_flag_defaults_to_off(self):
        import sana_evaluation.run_mode_eval as rme
        parser_args = rme.main.__doc__  # touch module so import errors surface
        self.assertIsNotNone(rme)

    def test_pool_tasks_collects_every_directory_into_one_call(self):
        import sana_evaluation.run_mode_eval as rme
        seen = {}

        def fake_run_evaluation(task_dir, agent_config, run_config, **kw):
            seen.setdefault("calls", []).append({"dir": task_dir, "files": kw.get("task_files")})
            return {}

        dirs = ["/t/k-3-d-2", "/t/k-4-d-3"]
        files = {"/t/k-3-d-2": ["/t/k-3-d-2/task_6.json", "/t/k-3-d-2/task_11.json"],
                 "/t/k-4-d-3": ["/t/k-4-d-3/task_6.json"]}

        with patch.object(rme.base_eval, "find_all_task_dirs", return_value=dirs), \
             patch.object(rme.base_eval, "run_evaluation", side_effect=fake_run_evaluation), \
             patch.object(rme.glob, "glob", side_effect=lambda pat: sorted(files[os.path.dirname(pat)])), \
             patch.object(rme.base_eval, "print_comparison_table", lambda *a, **k: None):
            rme._run_all_tasks_pooled(
                task_set="/t", agent_config=object(), run_config=object(),
                verbose=False, only_new=False, parallel=8, tasks_per_dir=None,
            )

        self.assertEqual(len(seen["calls"]), 1, "one pooled call, not one per directory")
        self.assertEqual(len(seen["calls"][0]["files"]), 3)
        self.assertEqual(sum(1 for f in seen["calls"][0]["files"] if f.endswith("task_6.json")), 2)


import os  # noqa: E402  (used by the patch above)
