"""Worker memory cap: turn a box-killing OOM into a per-task error.

Three runs of the model-tier grid were killed by the kernel OOM killer. Twice a
single process reached ~27.5 GB while the four pool workers sat at 0.58 GB each
and the whole box has 31 GB. The allocation has not been identified: DuckDB's
limit, result materialisation, execute_ideal, the artifact caches, the JSON
reader and the exact failing SQL were each tested and ruled out.

Capping the worker's heap does two things. The runaway allocation raises
MemoryError in that one task, which the existing handler already reports as a
task error, so the grid survives. And the traceback names the allocation site —
which the kernel's OOM kill destroys, and which is why five rounds of inference
found nothing.

The cap must sit well above the legitimate fixed cost, measured at ~1.6 GB
(0.71 GB imports + 0.89 GB artifact caches), so it can never fire spuriously.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation.runner import batch as awm


class WorkerMemoryCapTests(unittest.TestCase):
    def test_disabled_by_default(self):
        """RLIMIT_DATA counts virtual reservations, so a cap fires during import."""
        import os
        env = dict(os.environ)
        env.pop("SANA_WORKER_MEMORY_CAP_GB", None)
        with patch.dict("os.environ", env, clear=True):
            self.assertIsNone(awm._apply_worker_memory_cap())

    def test_disabled_when_env_is_empty(self):
        with patch.dict("os.environ", {"SANA_WORKER_MEMORY_CAP_GB": ""}, clear=False):
            self.assertIsNone(awm._apply_worker_memory_cap())

    def test_disabled_when_env_is_zero(self):
        with patch.dict("os.environ", {"SANA_WORKER_MEMORY_CAP_GB": "0"}, clear=False):
            self.assertIsNone(awm._apply_worker_memory_cap())

    def test_applies_the_configured_cap(self):
        calls = []

        def fake_setrlimit(which, limits):
            calls.append((which, limits))

        with patch.dict("os.environ", {"SANA_WORKER_MEMORY_CAP_GB": "8"}, clear=False), \
             patch.object(awm.resource, "setrlimit", fake_setrlimit):
            applied = awm._apply_worker_memory_cap()

        self.assertEqual(applied, 8 * 1024**3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], awm.resource.RLIMIT_DATA)
        self.assertEqual(calls[0][1], (8 * 1024**3, 8 * 1024**3))

    def test_cap_sits_well_above_the_legitimate_fixed_cost(self):
        """~1.6 GB is real work (imports + artifact caches); the cap must clear it."""
        with patch.dict("os.environ", {"SANA_WORKER_MEMORY_CAP_GB": "8"}, clear=False), \
             patch.object(awm.resource, "setrlimit", lambda *a: None):
            applied = awm._apply_worker_memory_cap()
        self.assertGreater(applied / 1024**3, 4.0, "too close to the 1.6 GB working set")

    def test_failure_to_set_the_cap_is_not_fatal(self):
        """A sandbox may forbid setrlimit; the task must still run."""
        def boom(*a):
            raise OSError("not permitted")

        with patch.dict("os.environ", {"SANA_WORKER_MEMORY_CAP_GB": "8"}, clear=False), \
             patch.object(awm.resource, "setrlimit", boom):
            self.assertIsNone(awm._apply_worker_memory_cap())

    def test_bad_value_is_ignored_rather_than_crashing_the_worker(self):
        with patch.dict("os.environ", {"SANA_WORKER_MEMORY_CAP_GB": "not-a-number"}, clear=False):
            self.assertIsNone(awm._apply_worker_memory_cap())


if __name__ == "__main__":
    unittest.main()
