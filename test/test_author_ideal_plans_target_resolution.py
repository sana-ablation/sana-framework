import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "author-ideal-plans"
    / "scripts"
)


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


resolve_task_targets = _load_module("resolve_task_targets.py", "resolve_task_targets")


class TestAuthorIdealPlansTargetResolution(unittest.TestCase):
    def _seed_repo(self, root: Path) -> None:
        (root / "tasks_mini").mkdir(parents=True, exist_ok=True)
        (root / "plans_mini").mkdir(parents=True, exist_ok=True)

    def test_resolve_targets_accepts_single_task_file(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            task_path = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text("{}\n")

            resolved = resolve_task_targets.resolve_targets(task_path)

            self.assertEqual(resolved, [task_path.resolve()])

    def test_resolve_targets_caps_subtree_batches_at_five(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            subtree = root / "tasks_mini" / "k-1-d-1"
            subtree.mkdir(parents=True, exist_ok=True)

            matching = [subtree / f"task_{i}.json" for i in range(1, 8)]
            for path in matching:
                path.write_text("{}\n")

            resolved = resolve_task_targets.resolve_targets(subtree)

            self.assertEqual(
                resolved,
                sorted(path.resolve() for path in matching)[: resolve_task_targets.MAX_TARGETS],
            )
            self.assertEqual(len(resolved), 5)

    def test_resolve_targets_rejects_missing_path(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._seed_repo(root)
            missing = root / "tasks_mini" / "k-1-d-1" / "task_9.json"

            with self.assertRaises(FileNotFoundError):
                resolve_task_targets.resolve_targets(missing)


if __name__ == "__main__":
    unittest.main()
