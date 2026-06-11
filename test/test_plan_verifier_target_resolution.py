import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "plan-verifier"
    / "scripts"
)


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


resolve_plan_targets = _load_module("resolve_plan_targets.py", "resolve_plan_targets")


class TestPlanVerifierTargetResolution(unittest.TestCase):
    def _seed_repo(self, root: Path) -> None:
        (root / "plans_mini").mkdir(parents=True, exist_ok=True)
        (root / "tasks_mini").mkdir(parents=True, exist_ok=True)

    def test_resolve_targets_accepts_single_plan_file(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            plan_path = root / "plans_mini" / "k-2-d-4" / "task_1.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text("{}\n")

            resolved = resolve_plan_targets.resolve_targets(plan_path)

            self.assertEqual(resolved, [plan_path.resolve()])

    def test_resolve_targets_walks_subtree_in_sorted_order(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            subtree = root / "plans_mini" / "k-2-d-4"
            (subtree / "nested").mkdir(parents=True, exist_ok=True)

            matching = [
                subtree / "task_2.json",
                subtree / "nested" / "task_1.json",
                subtree / "task_10.json",
            ]
            for path in matching:
                path.write_text("{}\n")
            (subtree / "notes.json").write_text("{}\n")

            resolved = resolve_plan_targets.resolve_targets(subtree)

            self.assertEqual(resolved, sorted(path.resolve() for path in matching))

    def test_resolve_targets_caps_subtree_batches_at_five(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            subtree = root / "plans_mini" / "k-2-d-4"
            subtree.mkdir(parents=True, exist_ok=True)

            matching = [subtree / f"task_{i}.json" for i in range(1, 8)]
            for path in matching:
                path.write_text("{}\n")

            resolved = resolve_plan_targets.resolve_targets(subtree)

            self.assertEqual(
                resolved,
                sorted(path.resolve() for path in matching)[: resolve_plan_targets.MAX_TARGETS],
            )
            self.assertEqual(len(resolved), 5)

    def test_resolve_targets_rejects_missing_path(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            missing = root / "plans_mini" / "k-2-d-4" / "task_9.json"

            with self.assertRaises(FileNotFoundError):
                resolve_plan_targets.resolve_targets(missing)


if __name__ == "__main__":
    unittest.main()
