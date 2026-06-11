import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "task-quality-auditor"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPT_DIR))


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pending_files = _load_module("list_pending_task_audits.py", "task_quality_pending_files")


class TestTaskQualityPendingFiles(unittest.TestCase):
    def _write_task(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    def test_pending_task_paths_only_returns_unfinished_by_default(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_root = root / "tasks_mini"
            output_root = root / "tasks_mini_quality"

            a = tasks_root / "k-5-d-3" / "task_1.json"
            b = tasks_root / "k-5-d-3" / "task_2.json"
            self._write_task(a)
            self._write_task(b)

            mirrored_a = output_root / a.relative_to(tasks_root)
            mirrored_a.parent.mkdir(parents=True, exist_ok=True)
            mirrored_a.write_text("{}\n")

            entries = pending_files.pending_task_paths(
                tasks_root=tasks_root,
                scope_path=tasks_root,
                output_root=output_root,
            )

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["relative_task"], "k-5-d-3/task_2.json")
            self.assertEqual(entries[0]["done"], False)

    def test_pending_task_paths_support_single_directory_scope(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_root = root / "tasks_mini"
            output_root = root / "tasks_mini_quality"

            target_dir = tasks_root / "k-5-d-4"
            other_dir = tasks_root / "k-6-d-3"
            self._write_task(target_dir / "task_1.json")
            self._write_task(target_dir / "task_2.json")
            self._write_task(other_dir / "task_1.json")

            entries = pending_files.pending_task_paths(
                tasks_root=tasks_root,
                scope_path=target_dir,
                output_root=output_root,
            )

            relative_tasks = [entry["relative_task"] for entry in entries]
            self.assertEqual(relative_tasks, ["k-5-d-4/task_1.json", "k-5-d-4/task_2.json"])

    def test_render_text_numbers_entries(self):
        text = pending_files.render_text(
            [
                {"relative_task": "k-5-d-3/task_1.json", "done": False},
                {"relative_task": "k-5-d-3/task_2.json", "done": True},
            ],
            include_complete=True,
        )

        self.assertIn("1. k-5-d-3/task_1.json", text)
        self.assertIn("2. k-5-d-3/task_2.json [done]", text)


if __name__ == "__main__":
    unittest.main()
