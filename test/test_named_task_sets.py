"""Named task sets: resolution, and that a set reuses the shared runtime profiles.

The second half is the one that matters. runtime_profile_store used to match the
literal segment `benchmarks/<benchmark>/tasks-mini/tasks`, so a set under any
other name did not raise -- it resolved to a *wrong* profile path and failed
much later with a confusing missing-file error.
"""
import json
from pathlib import Path

import pytest

from sana_evaluation.cli import _named_task_sets, resolve_task_set
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    _profile_location_from_task,
    runtime_profiles_root,
    set_runtime_profiles_root,
)

DEFAULT_PROFILES_ROOT = Path("benchmarks/lakeqa/tasks-mini/runtime-profiles")


@pytest.fixture(autouse=True)
def _pinned_profiles_root():
    """set_runtime_profiles_root mutates a module global that other test modules
    repoint at fixtures, so these assertions must not inherit whatever ran last."""
    previous = runtime_profiles_root()
    set_runtime_profiles_root(DEFAULT_PROFILES_ROOT)
    yield
    set_runtime_profiles_root(previous)


class TestResolution:
    def test_short_name_resolves_to_the_set_path(self):
        assert resolve_task_set("tasks_20_subset", "lakeqa") == "benchmarks/lakeqa/tasks_20_subset/tasks"

    def test_a_path_is_used_unchanged(self):
        p = "benchmarks/lakeqa/tasks_20_subset/tasks"
        assert resolve_task_set(p, "lakeqa") == p

    def test_none_falls_back_to_the_benchmark_default(self):
        assert resolve_task_set(None, "lakeqa") == "benchmarks/lakeqa/tasks-mini/tasks"
        assert resolve_task_set(None, "kramabench").startswith("benchmarks/kramabench/")

    def test_unknown_name_lists_what_is_available(self):
        with pytest.raises(ValueError, match="tasks_20_subset"):
            resolve_task_set("does-not-exist", "lakeqa")

    def test_tasks_20_subset_is_discoverable(self):
        assert "tasks_20_subset" in _named_task_sets("lakeqa")


class TestProfileSharing:
    @pytest.mark.parametrize("task_set", ["tasks-mini", "tasks_20_subset"])
    def test_every_set_resolves_to_the_same_shared_profiles(self, task_set):
        root, rel = _profile_location_from_task(
            f"benchmarks/lakeqa/{task_set}/tasks/k-3-d-2/task_11.json"
        )
        assert root == DEFAULT_PROFILES_ROOT
        assert rel == Path("k-3-d-2/task_11.json")

    def test_resolution_survives_a_path_prefix(self):
        # Sweeps run from a copied tree on a remote box; the prefix must not matter.
        root, rel = _profile_location_from_task(
            "tmp/staging/benchmarks/lakeqa/tasks_20_subset/tasks/k-5-d-3/task_13.json"
        )
        assert root == DEFAULT_PROFILES_ROOT
        assert rel == Path("k-5-d-3/task_13.json")

    def test_kramabench_still_resolves_to_its_own_profiles(self):
        root, _ = _profile_location_from_task(
            "benchmarks/kramabench/tasks-mini/tasks/legal/task_1.json"
        )
        assert root == Path("benchmarks/kramabench/tasks-mini/runtime-profiles")


class TestNano20Contents:
    ROOT = Path("benchmarks/lakeqa/tasks_20_subset")

    def test_manifest_and_tree_agree(self):
        ids = json.loads((self.ROOT / "manifest.json").read_text())
        on_disk = {
            str(p.relative_to(self.ROOT / "tasks")).removesuffix(".json")
            for p in (self.ROOT / "tasks").rglob("*.json")
        }
        assert on_disk == set(ids)
        assert len(ids) == 20

    def test_every_task_has_a_runtime_profile(self):
        ids = json.loads((self.ROOT / "manifest.json").read_text())
        for task_id in ids:
            root, rel = _profile_location_from_task(
                f"benchmarks/lakeqa/tasks_20_subset/tasks/{task_id}.json"
            )
            assert (root / rel).is_file(), f"no runtime profile for {task_id}"
